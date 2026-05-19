import isaacgym

assert isaacgym
import torch
import argparse

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset

import wandb
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import shutil
import pickle

from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv

from go1_gym_learn.ppo_cse_automatic import Runner
from go1_gym_learn.ppo_cse_automatic.ppo import PPO_Args
from go1_gym_learn.ppo_cse_automatic import RunnerArgs, ArmRunnerArgs, DogRunnerArgs
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC_Args
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmAC_Args


from go1_gym.utils import format_code, set_seed, global_switch

os.environ["WANDB_SILENT"] = "true"


def train_go1(arg):

    if args.debug:
        mode = "disabled"
        args.num_envs = 12
    else:
        mode = "online"

        if args.offline:
            mode = "offline"

    if args.no_wandb:
        mode = "disabled"

    if args.resume:
        args.tags.append("resume")

    args.seed = set_seed(args.seed)
    args.tags.append(f"seed{args.seed}")

    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    Cfg.commands.distributional_commands = False
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False

    Cfg.control.control_type = "M"

    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]

    Cfg.env.num_envs = args.num_envs

    Cfg.env.keep_arm_fixed = True
    Cfg.env.stage1_arm_curriculum = not args.no_stage1_arm_curriculum
    Cfg.env.stage1_arm_fixed_fraction = args.stage1_arm_fixed_fraction
    Cfg.env.stage1_arm_max_accel = args.stage1_arm_max_accel
    Cfg.env.stage1_arm_max_vel = args.stage1_arm_max_vel
    Cfg.env.stage1_arm_max_offset = args.stage1_arm_max_offset
    Cfg.env.stage1_arm_accel_resample_time_s = args.stage1_arm_accel_resample_time_s

    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
        Cfg.terrain.teleport_robots = False

    Cfg.control.update_obs_freq = 20  # Hz
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63

    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.7 * Cfg.reward_scales.tracking_lin_vel
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.5 * Cfg.reward_scales.tracking_ang_vel

    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy = -0.00004

    Cfg.reward_scales.jump = -0.00
    Cfg.rewards.terminal_body_height = 0.28
    Cfg.rewards.use_terminal_body_height = True

    DogRunnerArgs.resume = args.resume
    DogRunnerArgs.resume_path = "your_dog_ckpt_path"
    ArmRunnerArgs.resume = args.resume
    ArmRunnerArgs.resume_path = "your_arm_ckpt_path"

    global_switch.pretrained_to_hybrid_start = 2000 if args.resume else 8000  # 2000 with pretrained, 10000 from scratch

    if args.wo_two_stage:
        global_switch.pretrained_to_hybrid_start = 0

    global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 0

    if args.debug:
        if global_switch.pretrained_to_hybrid_start > 0:
            global_switch.pretrained_to_hybrid_start = 2
        global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 2
        RunnerArgs.save_interval = 2
        RunnerArgs.save_video_interval = 10

    Cfg.commands.T_force_range = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres = 0.3
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01

    Cfg.env.priv_observe_vel = False
    Cfg.commands.global_reference = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9

    Cfg.asset.render_sphere = True  # NOTE no use in headless
    Cfg.hybrid.use_vision = False
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1
    Cfg.hybrid.reward_scales.arm_dof_vel = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * Cfg.reward_scales.action_smoothness_1
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * Cfg.reward_scales.action_smoothness_2

    Cfg.use_rot6d = args.use_rot6d
    if Cfg.use_rot6d:
        Cfg.env.num_observations += 3

        Cfg.arm.arm_num_observations += 3
        Cfg.arm.arm_num_obs_history = Cfg.arm.arm_num_observations * Cfg.arm.arm_num_observation_history
        Cfg.arm.arm_num_commands += 3

        Cfg.dog.dog_num_observations += 3
        Cfg.dog.dog_num_obs_history = Cfg.dog.dog_num_observations * Cfg.dog.dog_num_observation_history

    if args.dyna_gait:
        Cfg.commands.use_dynamic_gait = True

        # Unlock gait parameter ranges (wtw_config sets wide ranges, don't re-lock)
        Cfg.commands.gait_frequency_cmd_range = [
            args.dyna_gait_min_frequency,
            Cfg.commands.gait_frequency_cmd_range[1],
        ]
        Cfg.commands.limit_gait_frequency = Cfg.commands.gait_frequency_cmd_range
        Cfg.commands.limit_footswing_height = Cfg.commands.footswing_height_range
        Cfg.commands.limit_gait_duration = Cfg.commands.gait_duration_cmd_range
        Cfg.commands.limit_stance_width = Cfg.commands.stance_width_range
        Cfg.commands.limit_stance_length = Cfg.commands.stance_length_range

        # Expand dog command dims: +5 gait params
        num_new_gait_dims = 5
        Cfg.dog.dog_num_commands += num_new_gait_dims  # 5 -> 10

        # Dog observation gets +5 extra command dims
        Cfg.dog.dog_num_observations += num_new_gait_dims
        Cfg.dog.dog_num_obs_history = Cfg.dog.dog_num_observations * Cfg.dog.dog_num_observation_history

        # Arm plan actions: +5 gait suggestions
        Cfg.arm.num_actions_arm_cd = Cfg.arm.num_actions_arm + 7  # 6 + 7 = 13

        # Arm observation includes current gait params as feedback
        Cfg.arm.arm_num_observations += num_new_gait_dims
        Cfg.arm.arm_num_obs_history = Cfg.arm.arm_num_observations * Cfg.arm.arm_num_observation_history

        # Update env observation dim (compute_observations goes from 3→10 cmd dims = +7)
        Cfg.env.num_observations += num_new_gait_dims + 2

        # Enable gait command observation
        Cfg.env.observe_gait_commands = True

        # Enable curriculum bins for gait parameters
        Cfg.commands.num_bins_gait_frequency = 11
        Cfg.commands.num_bins_footswing_height = 5
        Cfg.commands.num_bins_gait_duration = 3
        Cfg.commands.num_bins_stance_width = 3
        Cfg.commands.num_bins_stance_length = 3

    global_switch.init_sigmoid_lr()
    # global_switch.init_linear_lr()

    if args.robot == "go1":
        Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    elif args.robot == "go2":
        Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf"

    # if args.headless:
    #     RunnerArgs.log_video = False

    now = datetime.now()
    wandb.init(
        entity="simon00715",
        project="roboduet",
        group=args.run_name,
        mode=mode,
        notes=args.notes,
        name=f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}_{args.run_name}",
        tags=args.tags,
        dir=f"{MINI_GYM_ROOT_DIR}",
    )

    args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs", wandb.run.name)
    args.log_dir += f"_seed{args.seed}"

    os.makedirs(osp.join(args.log_dir, "checkpoints_arm"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "checkpoints_dog"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "deploy_model"), exist_ok=True)
    os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

    if not args.debug:
        os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)

        # save code
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/scripts/auto_train.py", f"{args.log_dir}/scripts/auto_train.py")
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot.py", f"{args.log_dir}/scripts/legged_robot.py"
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot_config.py",
            f"{args.log_dir}/scripts/legged_robot_config.py",
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/__init__.py", f"{args.log_dir}/scripts/env__init__.py"
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/go1/asset_config.py", f"{args.log_dir}/scripts/asset_config.py"
        )

        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/__init__.py",
            f"{args.log_dir}/scripts/ppo_cse_automatic__init__.py",
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py", f"{args.log_dir}/scripts/arm_ac.py"
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/dog_ac.py", f"{args.log_dir}/scripts/dog_ac.py"
        )
        shutil.copyfile(f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py", f"{args.log_dir}/scripts/ppo.py")
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym_learn/ppo_cse_automatic/rollout_storage.py",
            f"{args.log_dir}/scripts/rollout_storage.py",
        )

        wandb.run.log_code(f"{args.log_dir}/scripts")

        temp_dict = {
            "Cfg": vars(Cfg),
            "RunnerArgs": vars(RunnerArgs),
            "ArmAC_Args": vars(ArmAC_Args),
            "DogAC_Args": vars(DogAC_Args),
            "PPO_Args": vars(PPO_Args),
        }

        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            format_temp_dict = format_code(str(temp_dict))
            f.write(format_temp_dict)

        with open(osp.join(args.log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump(temp_dict, f)
        wandb.save(osp.join(args.log_dir, "parameters.pkl"), policy="now")

        wandb.log(
            {
                "Global_Switch/start": global_switch.pretrained_to_hybrid_start,
                "Global_Switch/end": global_switch.pretrained_to_hybrid_end,
            },
            step=0,
        )

    env = VelocityTrackingEasyEnv(
        sim_device=args.sim_device,
        headless=args.headless,
        cfg=Cfg,
        graphics_device_id=args.graphics_device_id,
    )
    env = HistoryWrapper(env)
    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(
        env, device=f"cuda:{gpu_id}", run_name=args.run_name, resume=args.resume, log_dir=args.log_dir, debug=args.debug
    )
    runner.learn(
        num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=None)
    parser.add_argument("--num_learning_iterations", type=int, default=100000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--robot", type=str, default="go1", choices=["go1", "go2"])
    parser.add_argument("--wo_two_stage", action="store_true", default=False)
    parser.add_argument("--use_rot6d", action="store_true", default=False)
    parser.add_argument("--dyna_gait", action="store_true", default=False)
    parser.add_argument("--no_stage1_arm_curriculum", action="store_true", default=False)
    parser.add_argument("--stage1_arm_fixed_fraction", type=float, default=0.1)
    parser.add_argument("--stage1_arm_max_accel", type=float, default=2.0)
    parser.add_argument("--stage1_arm_max_vel", type=float, default=1.0)
    parser.add_argument("--stage1_arm_max_offset", type=float, default=0.35)
    parser.add_argument("--stage1_arm_accel_resample_time_s", type=float, default=0.5)
    parser.add_argument("--dyna_gait_min_frequency", type=float, default=0.0)

    args = parser.parse_args()

    train_go1(args)
