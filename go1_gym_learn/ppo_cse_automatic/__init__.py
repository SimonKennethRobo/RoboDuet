# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import copy
import os
import os.path as osp
import pickle
import shutil
import statistics
import time
from collections import deque

import cv2
import imageio
import torch
from params_proto import PrefixProto

import wandb
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.roboduet.utils import aggregate_episode_value, apply_wbc_reward_settings
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.utils import global_switch

from .arm_ac import ArmActorCritic
from .dog_ac import DogActorCritic
from .ppo import PPO


def _load_matching_state_dict(model, checkpoint_state, *, skip_prefixes=()):
    model_state = model.state_dict()
    loadable_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            raise RuntimeError(
                f"Checkpoint tensor {key} has shape {tuple(value.shape)}, "
                f"but current model expects {tuple(model_state[key].shape)}."
            )
        loadable_state[key] = value

    incompatible = model.load_state_dict(loadable_state, strict=False)
    missing_required = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in skip_prefixes)
    ]
    if missing_required:
        raise RuntimeError(f"Checkpoint is missing required tensors: {missing_required}")
    return skipped


def _check_dog_obs_layout(ckpt_path, cfg):
    """R7.4: refuse a checkpoint from a different observation layout, loudly.

    The R7 observation change (90 -> 112 dims, history 30 -> 50) already makes
    the actor's first layer mismatch, so loading would fail anyway -- but it
    fails as a bare shape mismatch on a tensor named ``actor_body.0.weight``,
    which says nothing about the cause.  The run directory keeps a
    ``parameters.pkl`` snapshot of the config it was trained with, so the real
    answer is available: say it.

    Silently loading what happens to fit would be far worse than either.  The
    layout change is not a resize, it is a different observation vector, and a
    partially loaded policy would be reading pitch where it learned velocity.
    """
    snapshot = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "parameters.pkl")
    if not os.path.exists(snapshot):
        return
    try:
        with open(snapshot, "rb") as handle:
            parameters = pickle.load(handle)
        trained = parameters.get("Cfg", {}).get("dog", {})
    except Exception:
        return
    mismatches = []
    for key in ("dog_num_observations", "dog_num_observation_history"):
        before = trained.get(key)
        now = getattr(cfg.dog, key, None)
        if before is not None and now is not None and int(before) != int(now):
            mismatches.append(f"{key}: checkpoint {before} != current {now}")
    if mismatches:
        raise RuntimeError(
            "This checkpoint was trained with a different dog observation layout:\n  "
            + "\n  ".join(mismatches)
            + f"\n(snapshot: {snapshot})\n"
            "The observation vector changed, it was not merely resized, so the "
            "weights cannot be reused even in part -- stage-1 has to be retrained "
            "from scratch. Use this checkpoint only as an evaluation baseline."
        )


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = "PPO"
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 1000  # check for potential saves every this many iterations
    save_video_interval = 1000
    log_freq = 10
    log_video = True

    # load and resume
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model


class ArmRunnerArgs(PrefixProto, cli=False):
    ckpt_path = None


class DogRunnerArgs(PrefixProto, cli=False):
    ckpt_path = None
    stage2_freeze_loco_policy = True
    stage2_loco_learning_rate = None


def custom_decay_reward_scale(iteration, initial_scale=1.5, final_scale=0.8, max_iterations=8000):
    if iteration >= max_iterations:
        return final_scale
    x = (iteration / max_iterations) ** 2  # Using square to achieve the desired curve
    reward_scale = final_scale + (initial_scale - final_scale) * (1 - x)
    return reward_scale


def custom_increase_reward_scale(iteration, initial_scale=0.2, final_scale=0.7, max_iterations=8000):
    if iteration >= max_iterations:
        return final_scale
    x = (iteration / max_iterations) ** 2  # Using square to achieve the desired curve
    reward_scale = initial_scale + (final_scale - initial_scale) * x
    return reward_scale


class Runner:
    def __init__(self, env, device="cpu", run_name: str = None, resume=False, log_dir=None, debug=False):

        self.device = device
        self.env: HistoryWrapper = env
        self.run_name = run_name
        self.log_dir = log_dir
        self.debug = debug
        self.num_steps_per_env = RunnerArgs.num_steps_per_env
        self.stage2_loco_policy_frozen = False
        self._stage2_loco_policy_mode_applied = False
        self.arm_policy_enabled = self.env.arm_policy_enabled

        # R8.2's alarm, kept over iterations rather than judged on one: the
        # per-iteration value is a fraction over whichever envs held an
        # uninterrupted window, and a single high reading is a small sample, not
        # a miscalibrated reference model.
        diagnostics = getattr(getattr(self.env.cfg, "response", None), "diagnostics", None)
        self._ripple_warn_fraction = float(getattr(diagnostics, "ripple_warn_fraction", 0.5))
        self._ripple_history = deque(
            maxlen=max(2, int(getattr(diagnostics, "ripple_warn_iterations", 20)))
        )
        self._ripple_last_warned = None

        self.arm_model = None
        self.alg_arm = None
        if self.arm_policy_enabled:
            self.arm_model = ArmActorCritic(
                num_obs=self.env.cfg.arm.arm_num_observations,
                num_privileged_obs=self.env.cfg.arm.arm_num_privileged_obs,
                num_obs_history=self.env.cfg.arm.arm_num_obs_history,
                num_actions=self.env.cfg.arm.num_actions_arm_cd,
                use_adaptation_module=self.env.cfg.arm.use_adaptation_module,
                device=self.device,
            ).to(self.device)

        self.dog_model = DogActorCritic(
            num_obs=self.env.cfg.dog.dog_num_observations,
            num_privileged_obs=self.env.cfg.dog.dog_num_privileged_obs,
            num_obs_history=self.env.cfg.dog.dog_num_obs_history,
            num_actions=self.env.cfg.dog.dog_actions,
            use_adaptation_module=self.env.cfg.dog.use_adaptation_module,
        ).to(self.device)

        if DogRunnerArgs.ckpt_path is not None:
            _check_dog_obs_layout(DogRunnerArgs.ckpt_path, self.env.cfg)
            weights = torch.load(DogRunnerArgs.ckpt_path, map_location=self.device)
            if DogRunnerArgs.stage2_freeze_loco_policy:
                skipped = _load_matching_state_dict(self.dog_model, weights, skip_prefixes=("critic_body.",))
                print("successfully loaded dog weights without critic for frozen stage2 locomotion policy!!!")
                if skipped:
                    print(f"Skipped {len(skipped)} dog checkpoint tensors: {skipped}")
            else:
                self.dog_model.load_state_dict(state_dict=weights)
                print("successfully loaded dog weights!!!")

        if ArmRunnerArgs.ckpt_path is not None:
            if not self.arm_policy_enabled:
                raise ValueError("An arm checkpoint cannot be loaded when the arm policy is disabled for pure stage-1.")
            weights = torch.load(ArmRunnerArgs.ckpt_path, map_location=self.device)
            self.arm_model.load_state_dict(state_dict=weights)
            print("successfully loaded arm weights!!!")

        if self.arm_policy_enabled:
            self.alg_arm = PPO(self.arm_model, device=self.device)
            self.alg_arm.init_storage(
                self.env.num_train_envs,
                self.num_steps_per_env,
                [self.env.cfg.arm.arm_num_observations],
                [self.env.cfg.arm.arm_num_privileged_obs],
                [self.env.cfg.arm.arm_num_obs_history],
                [self.env.cfg.arm.num_actions_arm_cd],
                [self.env.cfg.arm.num_actions_arm_cd],
            )

        self.alg_dog = PPO(self.dog_model, device=self.device)
        self.alg_dog.init_storage(
            self.env.num_train_envs,
            self.num_steps_per_env,
            [self.env.cfg.dog.dog_num_observations],
            [self.env.cfg.dog.dog_num_privileged_obs],
            [self.env.cfg.dog.dog_num_obs_history],
            [self.env.cfg.dog.dog_actions],
            [self.env.cfg.dog.dog_actions],
        )

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        if global_switch.switch_open:
            self._apply_stage2_loco_policy_settings()

        self.env.reset()

    def _set_dog_policy_requires_grad(self, requires_grad):
        for parameter in self.dog_model.parameters():
            parameter.requires_grad_(requires_grad)

    def _apply_stage2_loco_policy_settings(self):
        if self._stage2_loco_policy_mode_applied:
            return

        if DogRunnerArgs.stage2_freeze_loco_policy:
            self._set_dog_policy_requires_grad(False)
            self.dog_model.eval()
            self.stage2_loco_policy_frozen = True
            self.env.env.disable_dog_policy_rewards = True
            print("Stage2 locomotion policy is frozen; dog PPO updates and dog rewards are disabled.")
        else:
            self._set_dog_policy_requires_grad(True)
            self.dog_model.train()
            self.stage2_loco_policy_frozen = False
            self.env.env.disable_dog_policy_rewards = False
            if DogRunnerArgs.stage2_loco_learning_rate is not None:
                self.alg_dog.set_learning_rate(float(DogRunnerArgs.stage2_loco_learning_rate))
                print(f"Stage2 locomotion policy learning rate set to {self.alg_dog.learning_rate}.")

        self._stage2_loco_policy_mode_applied = True

    def _dog_policy_trainable_this_iteration(self):
        return not (global_switch.switch_open and self.stage2_loco_policy_frozen)

    def _check_gait_frequency_ripple(self, iteration, ep_infos, episode_keys):
        """R8.2: warn when the R4.1 reward keeps rippling at the gait frequency.

        The diagnostic R8.2 prescribes, and the reason it prescribes one: a
        reference model whose posture channels were calibrated without joint
        (domain, phase) binning is achievable at average phase and not at the
        unfavourable ones, which shows up here and essentially nowhere else --
        the reward simply looks a bit lower, so it is easy to read as a weight
        problem and to spend a training run on the wrong fix.

        Only a warning.  Nothing in the loop acts on it: the response is to stop
        and re-run scripts/calibrate_reference_model.py, which is a decision for
        a person, not a schedule.
        """
        key = "perf_ref_tracking_ripple_fraction"
        if key not in episode_keys:
            return
        self._ripple_history.append(float(aggregate_episode_value(ep_infos, key)))
        if len(self._ripple_history) < self._ripple_history.maxlen:
            return
        sustained = sum(self._ripple_history) / len(self._ripple_history)
        if sustained <= self._ripple_warn_fraction:
            return
        # One warning per full window, not one per iteration: it stays true for
        # as long as the calibration is wrong, and a per-iteration print would
        # bury the rest of the log.
        span = self._ripple_history.maxlen
        if self._ripple_last_warned is not None and iteration - self._ripple_last_warned < span:
            return
        self._ripple_last_warned = iteration
        print(
            "\033[1;33m"
            f"[R8.2] ref_tracking is rippling at the gait frequency in {sustained:.0%} of "
            f"environments, averaged over the last {span} iterations.\n"
            "       That is the signature of posture channels calibrated without joint "
            "(domain, gait-phase) binning:\n"
            "       the reference model is achievable at average phase and not at the "
            "unfavourable ones.\n"
            "       Re-run scripts/calibrate_reference_model.py before reading anything "
            "into the consistency curves."
            "\033[0m"
        )

    def _advance_stage_schedule(self, iteration):
        global_switch.count += 1
        if not global_switch.switch_open:
            global_switch.stage1_count += 1

        if iteration != global_switch.pretrained_to_wbc_start or global_switch.switch_open:
            return
        if not self.arm_policy_enabled:
            raise RuntimeError("Stage schedule attempted to enable WBC without an initialized arm policy.")

        blue_bold_text = "\033[1;34m"  # bold blue
        reset_color = "\033[0m"  # reset
        print(
            blue_bold_text
            + "=" * 160
            + "\n"
            + "Multi-agents Policy Output: Pretrained model training finished, start to train WBC model."
            + "\n"
            + "=" * 160
            + reset_color
        )
        global_switch.open_switch()
        apply_wbc_reward_settings(self.env.cfg)
        self._apply_stage2_loco_policy_settings()

    def learn(
        self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, eval_expert=False, width=80, pad=35
    ):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_arm = privileged_obs_arm = obs_history_arm = None
        if self.arm_policy_enabled:
            obs_dict_arm = self.env.get_arm_observations()
            obs_arm, privileged_obs_arm, obs_history_arm = (
                obs_dict_arm["obs"].to(self.device),
                obs_dict_arm["privileged_obs"].to(self.device),
                obs_dict_arm["obs_history"].to(self.device),
            )
            self.alg_arm.actor_critic.train()
        if self._dog_policy_trainable_this_iteration():
            self.alg_dog.actor_critic.train()
        else:
            self.alg_dog.actor_critic.eval()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        pending_reward_sums = []
        pending_episode_lengths = []
        ep_infos = []

        mean_value_loss_arm, mean_surrogate_loss_arm, mean_adaptation_module_loss_arm = 0, 0, 0
        mean_value_loss_dog, mean_surrogate_loss_dog, mean_adaptation_module_loss_dog = 0, 0, 0

        tot_iter = self.current_learning_iteration + num_learning_iterations
        actions_arm = torch.zeros(
            self.env.num_envs, self.env.num_actions_arm, dtype=torch.float, device=self.device, requires_grad=False
        )
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env + 1):
                    if global_switch.switch_open:
                        actions_arm = self.alg_arm.act(
                            obs_arm[:num_train_envs],
                            privileged_obs_arm[:num_train_envs],
                            obs_history_arm[:num_train_envs],
                        )
                        if self.env.num_plan_actions > 0:
                            self.env.plan(actions_arm)

                    dog_obs_dict = self.env.get_dog_observations()

                    # initial step
                    if i == 0:
                        pass
                    else:
                        # use for compute last value
                        obs_dog, privileged_obs_dog, obs_history_dog = (
                            dog_obs_dict["obs"],
                            dog_obs_dict["privileged_obs"],
                            dog_obs_dict["obs_history"],
                        )
                        self.alg_dog.process_env_step(rewards_dog[:num_train_envs], dones[:num_train_envs], infos)
                        if i == self.num_steps_per_env:
                            break

                    # add reward
                    actions_dog = self.alg_dog.act(
                        dog_obs_dict["obs"],
                        dog_obs_dict["privileged_obs"],
                        dog_obs_dict["obs_history"],
                        deterministic=not self._dog_policy_trainable_this_iteration(),
                    )

                    actions_arm_step = (
                        actions_arm[:, : self.env.num_actions_arm]
                        if global_switch.switch_open and self.env.num_plan_actions > 0
                        else actions_arm
                    )
                    ret = self.env.step(actions_dog, actions_arm_step)
                    rewards_dog, rewards_arm, dones, infos = ret

                    if global_switch.switch_open:
                        obs_dict_arm = self.env.get_arm_observations()
                        obs_arm, privileged_obs_arm, obs_history_arm = (
                            obs_dict_arm["obs"],
                            obs_dict_arm["privileged_obs"],
                            obs_dict_arm["obs_history"],
                        )

                        obs_arm, privileged_obs_arm, obs_history_arm, rewards_dog, rewards_arm, dones = (
                            obs_arm.to(self.device),
                            privileged_obs_arm.to(self.device),
                            obs_history_arm.to(self.device),
                            rewards_dog.to(self.device),
                            rewards_arm.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg_arm.process_env_step(rewards_arm[:num_train_envs], dones[:num_train_envs], infos)

                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.env.clear_cached(env_ids)

                    if self.log_dir is not None:
                        if "train/episode" in infos:
                            ep_infos.append(infos["train/episode"])

                        cur_reward_sum += rewards_dog + rewards_arm
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        if len(new_ids_train) > 0:
                            pending_reward_sums.append(cur_reward_sum[new_ids_train].detach().clone())
                            pending_episode_lengths.append(cur_episode_length[new_ids_train].detach().clone())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if global_switch.switch_open:
                    self.alg_arm.compute_returns(obs_history_arm[:num_train_envs], privileged_obs_arm[:num_train_envs])
                dog_policy_trainable = self._dog_policy_trainable_this_iteration()
                if dog_policy_trainable:
                    self.alg_dog.compute_returns(obs_history_dog[:num_train_envs], privileged_obs_dog[:num_train_envs])
                else:
                    self.alg_dog.clear_storage()

            if global_switch.switch_open:
                (
                    mean_value_loss_arm,
                    mean_surrogate_loss_arm,
                    mean_adaptation_module_loss_arm,
                    mean_decoder_loss,
                    mean_decoder_loss_student,
                    mean_adaptation_module_test_loss,
                    mean_decoder_test_loss,
                    mean_decoder_test_loss_student,
                ) = self.alg_arm.update(un_adapt=not self.arm_model.use_adaptation_module)
            if dog_policy_trainable:
                (
                    mean_value_loss_dog,
                    mean_surrogate_loss_dog,
                    mean_adaptation_module_loss_dog,
                    mean_decoder_loss_dog,
                    mean_decoder_loss_student_dog,
                    mean_adaptation_module_test_loss_dog,
                    mean_decoder_test_loss_dog,
                    mean_decoder_test_loss_student_dog,
                ) = self.alg_dog.update()
            else:
                mean_value_loss_dog = 0.0
                mean_surrogate_loss_dog = 0.0
                mean_adaptation_module_loss_dog = 0.0
                mean_decoder_loss_dog = 0.0
                mean_decoder_loss_student_dog = 0.0
                mean_adaptation_module_test_loss_dog = 0.0
                mean_decoder_test_loss_dog = 0.0
                mean_decoder_test_loss_student_dog = 0.0
            stop = time.time()
            learn_time = stop - start

            self._advance_stage_schedule(it)

            if self.log_dir is not None:
                if pending_reward_sums:
                    rewbuffer.extend(torch.cat(pending_reward_sums).cpu().tolist())
                    lenbuffer.extend(torch.cat(pending_episode_lengths).cpu().tolist())
                    pending_reward_sums.clear()
                    pending_episode_lengths.clear()

                ep_string = f""
                wandb_dict = {}
                wandb_dict["Efficiency/collect_time"] = collection_time
                wandb_dict["Efficiency/learn_time"] = learn_time
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += learn_time + collection_time
                iteration_time = learn_time + collection_time
                fps = self.num_steps_per_env * self.env.num_envs / iteration_time

                episode_keys = dict.fromkeys(key for ep_info in ep_infos for key in ep_info)
                self._check_gait_frequency_ripple(it, ep_infos, episode_keys)
                for key in episode_keys:
                    if key == "perf_episode_count":
                        continue
                    mean = aggregate_episode_value(ep_infos, key)

                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {mean:.4f}\n"""

                    if not self.debug:
                        if key == "stage1_arm_curriculum_intensity":
                            wandb_dict["Curriculum/arm_disturbance_intensity"] = mean
                        elif key.startswith("curriculum_threshold_"):
                            name = key.replace("curriculum_threshold_", "", 1)
                            wandb_dict["Curriculum/threshold_" + name] = mean
                        elif key == "command_curriculum_weight":
                            wandb_dict["Curriculum/command_bin_weight"] = mean
                        elif key.startswith("stage2_base_unlock_"):
                            name = key.replace("stage2_base_unlock_", "", 1)
                            wandb_dict["Curriculum/stage2_base_unlock_" + name] = mean
                        elif key.startswith("reset_curriculum_"):
                            name = key.replace("reset_curriculum_", "", 1)
                            wandb_dict["Curriculum/reset_" + name] = mean
                        elif key.startswith("traj_curriculum_"):
                            name = key.replace("traj_curriculum_", "", 1)
                            wandb_dict["Curriculum/traj_" + name] = mean
                        elif key.startswith("perf_"):
                            name = key.replace("perf_", "", 1)
                            wandb_dict["Performance/" + name] = mean
                        elif key.startswith("global_switch_"):
                            name = key.replace("global_switch_", "", 1)
                            wandb_dict["Global_Switch/" + name] = mean
                        else:
                            wandb_dict["Train_Reward_episode/" + key] = mean

                arm_action_std = (
                    self.alg_arm.actor_critic.std.clone()
                    if self.arm_policy_enabled
                    else torch.empty(0, device=self.device)
                )
                dog_action_std = self.alg_dog.actor_critic.std.clone()
                if not self.debug:
                    if self.arm_policy_enabled:
                        wandb_dict["Train_Loss/mean_value_loss_arm"] = mean_value_loss_arm
                        wandb_dict["Train_Loss/mean_surrogate_loss_arm"] = mean_surrogate_loss_arm
                        wandb_dict["Train_Loss/mean_adaptation_module_loss_arm"] = mean_adaptation_module_loss_arm

                    wandb_dict["Train_Loss/mean_value_loss_dog"] = mean_value_loss_dog
                    wandb_dict["Train_Loss/mean_surrogate_loss_dog"] = mean_surrogate_loss_dog
                    wandb_dict["Train_Loss/mean_adaptation_module_loss_dog"] = mean_adaptation_module_loss_dog

                    if self.arm_policy_enabled:
                        wandb_dict["Train_std/arm_action_std"] = arm_action_std.mean()
                    wandb_dict["Train_std/dog_action_std"] = dog_action_std.mean()

                    if len(rewbuffer) > 0:
                        wandb_dict["Train_Total_Reward/mean_reward"] = statistics.mean(rewbuffer)
                        wandb_dict["Train_Total_Reward/mean_episode_length"] = statistics.mean(lenbuffer)

                    wandb.log(wandb_dict, step=it)
                str = f" \033[1m Learning iteration {it}/{tot_iter} \033[0m "

                log_string = (
                    f"""{"#" * width}\n"""
                    f"""{str.center(width, " ")}\n\n"""
                )
                log_string += ep_string
                log_string += f"""{"-" * width}\n"""
                log_string += f"""\033[1m{"run_name:":>{pad}} {self.run_name}\033[0m \n"""
                if len(rewbuffer) > 0:
                    log_string += (
                        f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                        f"""{"Arm action std:":>{pad}} {arm_action_std.cpu().tolist()}\n"""
                        f"""{"Dog action std:":>{pad}} {dog_action_std.cpu().tolist()}\n"""
                        f"""{"Arm Value function loss:":>{pad}} {mean_value_loss_arm:.8f}\n"""
                        f"""{"Arm Surrogate loss:":>{pad}} {mean_surrogate_loss_arm:.8f}\n"""
                        f"""{"Arm Adaptation loss:":>{pad}} {mean_adaptation_module_loss_arm:.8f}\n"""
                        f"""{"Dog Value function loss:":>{pad}} {mean_value_loss_dog:.8f}\n"""
                        f"""{"Dog Surrogate loss:":>{pad}} {mean_surrogate_loss_dog:.8f}\n"""
                        f"""{"Dog Adaptation loss:":>{pad}} {mean_adaptation_module_loss_dog:.8f}\n"""
                        f"""{"Mean reward (total):":>{pad}} {statistics.mean(rewbuffer):.4f}\n"""
                        f"""{"Mean episode length:":>{pad}} {statistics.mean(lenbuffer):.4f}\n"""
                    )

                else:
                    log_string = (
                        f"""{"#" * width}\n"""
                        f"""{str.center(width, " ")}\n\n"""
                        f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                        f"""{"Arm Value function loss:":>{pad}} {mean_value_loss_arm:.8f}\n"""
                        f"""{"Arm Surrogate loss:":>{pad}} {mean_surrogate_loss_arm:.8f}\n"""
                        f"""{"Arm Adaptation loss:":>{pad}} {mean_adaptation_module_loss_arm:.4f}\n"""
                        f"""{"Dog Value function loss:":>{pad}} {mean_value_loss_dog:.8f}\n"""
                        f"""{"Dog Surrogate loss:":>{pad}} {mean_surrogate_loss_dog:.8f}\n"""
                        f"""{"Dog Adaptation loss:":>{pad}} {mean_adaptation_module_loss_dog:.8f}\n"""
                    )

                curr_it = it - copy.copy(self.current_learning_iteration)
                eta = self.tot_time / (curr_it + 1) * (num_learning_iterations - curr_it)

                mins = eta // 60
                secs = eta % 60
                log_string += (
                    f"""{"-" * width}\n"""
                    f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
                    f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                    f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
                    f"""{"ETA:":>{pad}} {mins:.0f} mins {secs:.1f} s\n"""
                )
                print(log_string)

                with open(osp.join(self.log_dir, "log.txt"), "a") as f:
                    f.write(log_string)

            if RunnerArgs.save_video_interval and RunnerArgs.log_video:
                self.log_video(it)

            if not self.debug and it % RunnerArgs.save_interval == 0:
                if global_switch.switch_open:
                    self.save_arm(it)
                self.save_dog(it)

            ep_infos.clear()

        if self.arm_policy_enabled:
            self.save_arm(it)
        self.save_dog(it)

    def save_dog(self, it):
        torch.save(
            self.alg_dog.actor_critic.state_dict(), osp.join(self.log_dir, f"checkpoints_dog/ac_weights_{it:06d}.pt")
        )
        shutil.copyfile(
            osp.join(self.log_dir, f"checkpoints_dog/ac_weights_{it:06d}.pt"),
            osp.join(self.log_dir, f"checkpoints_dog/ac_weights_last_dog.pt"),
        )

        path = osp.join(self.log_dir, f"deploy_model")
        if self.alg_dog.actor_critic.adaptation_module is not None:
            adaptation_module_dog_path = f"{path}/adaptation_module_latest_dog.jit"
            adaptation_module_dog = copy.deepcopy(self.alg_dog.actor_critic.adaptation_module).to("cpu")
            traced_script_adaptation_module_dog = torch.jit.script(adaptation_module_dog)
            traced_script_adaptation_module_dog.save(adaptation_module_dog_path)
        body_dog_path = f"{path}/body_latest_dog.jit"
        body_model_dog = copy.deepcopy(self.alg_dog.actor_critic.actor_body).to("cpu")
        traced_script_body_module_dog = torch.jit.script(body_model_dog)
        traced_script_body_module_dog.save(body_dog_path)

    def save_arm(self, it):
        if not self.arm_policy_enabled:
            return
        torch.save(
            self.alg_arm.actor_critic.state_dict(), osp.join(self.log_dir, f"checkpoints_arm/ac_weights_{it:06d}.pt")
        )
        shutil.copyfile(
            osp.join(self.log_dir, f"checkpoints_arm/ac_weights_{it:06d}.pt"),
            osp.join(self.log_dir, f"checkpoints_arm/ac_weights_last_arm.pt"),
        )

        path = osp.join(self.log_dir, f"deploy_model")
        if self.alg_arm.actor_critic.adaptation_module is not None:
            adaptation_module_path = f"{path}/adaptation_module_latest_arm.jit"
            adaptation_module = copy.deepcopy(self.alg_arm.actor_critic.adaptation_module).to("cpu")
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)
        body_path = f"{path}/body_latest_arm.jit"
        body_model = copy.deepcopy(self.alg_arm.actor_critic.actor_body).to("cpu")
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(body_path)
        history_arm_path = f"{path}/history_latest_arm.jit"
        history_model_arm = copy.deepcopy(self.alg_arm.actor_critic.actor_history_encoder).to("cpu")
        traced_script_history_module_arm = torch.jit.script(history_model_arm)
        traced_script_history_module_arm.save(history_arm_path)

    def save_cv(self, frames, it):
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*"X264")
        out = cv2.VideoWriter(
            osp.join(self.log_dir, f"videos/{it:06d}.mp4"),
            fourcc,
            int(1 / self.env.dt),
            (self.env.camera_props.width, self.env.camera_props.height),
        )

        for frame in frames:
            out.write(frame[..., :3])
        out.release()

    def save_io(self, frames, it):
        frame_stride = max(1, int(getattr(self.env.cfg.env, "recording_frame_stride", 1)))
        writer = imageio.get_writer(
            osp.join(self.log_dir, f"videos/{it:06d}.mp4"), fps=max(1, int(1 / (self.env.dt * frame_stride)))
        )
        for frame in frames:
            writer.append_data(frame[..., :3])
        writer.close()

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")

            self.save_io(frames, it)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                # wandb.log({"video": wandb.Video(frames, fps=1 / self.env.dt, format='mp4')})
                # wandb.run.summary["latest_video"] = wandb.Video(frames, fps=1 / self.env.dt, format='mp4')

    def get_inference_policy(self, device=None):
        if not self.arm_policy_enabled:
            raise RuntimeError("Arm inference policy is unavailable during pure stage-1 training.")
        self.alg_arm.actor_critic.eval()
        if device is not None:
            self.alg_arm.actor_critic.to(device)
        return self.alg_arm.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        if not self.arm_policy_enabled:
            raise RuntimeError("Arm expert policy is unavailable during pure stage-1 training.")
        self.alg_arm.actor_critic.eval()
        if device is not None:
            self.alg_arm.actor_critic.to(device)
        return self.alg_arm.actor_critic.act_expert
