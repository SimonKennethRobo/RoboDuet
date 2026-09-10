import copy
from contextlib import contextmanager
from pathlib import Path

import torch
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmAC_Args
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC_Args
import pickle as pkl
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.envs.config import (
    RoboDuetRuntimeOptions,
    apply_config_snapshot,
    build_roboduet_config,
    restore_dog_observation_layout,
)
from go1_gym.envs.config.wbc import ROBOT_ASSET_FILES


_MODEL_ARG_FIELDS = (
    "actor_hidden_dims",
    "critic_hidden_dims",
    "adaptation_module_branch_hidden_dims",
    "activation",
    "use_decoder",
)


def _load_run_parameters(logdir):
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        return pkl.load(file)


def _checkpoint_path(logdir, policy_name, ckpt_id):
    suffix = f"last_{policy_name}" if ckpt_id == "last" else str(ckpt_id).zfill(6)
    return Path(logdir) / f"checkpoints_{policy_name}" / f"ac_weights_{suffix}.pt"


def _linear_weight_shapes(state_dict, prefix):
    layers = []
    for key, value in state_dict.items():
        if not (key.startswith(prefix + ".") and key.endswith(".weight") and value.ndim == 2):
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        layers.append((int(parts[1]), tuple(value.shape)))
    return [shape for _, shape in sorted(layers)]


def _model_args_from_checkpoint(saved_args, checkpoint):
    """Use checkpoint tensor shapes as the source of truth for MLP widths."""
    inferred = dict(saved_args or {})
    actor_shapes = _linear_weight_shapes(checkpoint, "actor_body")
    critic_shapes = _linear_weight_shapes(checkpoint, "critic_body")
    adaptation_shapes = _linear_weight_shapes(checkpoint, "adaptation_module")
    if len(actor_shapes) >= 2:
        inferred["actor_hidden_dims"] = [shape[0] for shape in actor_shapes[:-1]]
    if len(critic_shapes) >= 2:
        inferred["critic_hidden_dims"] = [shape[0] for shape in critic_shapes[:-1]]
    if len(adaptation_shapes) >= 2:
        inferred["adaptation_module_branch_hidden_dims"] = [
            shape[0] for shape in adaptation_shapes[:-1]
        ]
    return inferred


@contextmanager
def _temporary_model_args(args_class, values):
    """Construct one checkpoint-specific model without leaking global Args."""
    previous = {
        field: copy.deepcopy(getattr(args_class, field))
        for field in _MODEL_ARG_FIELDS
        if hasattr(args_class, field)
    }
    try:
        for field in _MODEL_ARG_FIELDS:
            if field in values and hasattr(args_class, field):
                setattr(args_class, field, copy.deepcopy(values[field]))
        yield
    finally:
        for field, value in previous.items():
            setattr(args_class, field, value)


def _checkpoint_structure(checkpoint, policy_name):
    actor_shapes = _linear_weight_shapes(checkpoint, "actor_body")
    if not actor_shapes:
        raise ValueError(f"{policy_name} checkpoint has no actor_body linear weights")
    adaptation_shapes = _linear_weight_shapes(checkpoint, "adaptation_module")
    structure = {
        "actor_input": actor_shapes[0][1],
        "actor_hidden_dims": [shape[0] for shape in actor_shapes[:-1]],
        "actions": actor_shapes[-1][0],
        "uses_adaptation": bool(adaptation_shapes),
    }
    if adaptation_shapes:
        structure["adaptation_input"] = adaptation_shapes[0][1]
        structure["adaptation_output"] = adaptation_shapes[-1][0]
    if policy_name == "arm":
        history_shapes = _linear_weight_shapes(checkpoint, "actor_history_encoder")
        if not history_shapes:
            raise ValueError("arm checkpoint has no actor_history_encoder linear weights")
        structure["history_input"] = history_shapes[0][1]
        structure["history_output"] = history_shapes[-1][0]
    return structure


def _validate_checkpoint_layout(checkpoint, policy_name, cfg):
    structure = _checkpoint_structure(checkpoint, policy_name)
    mismatches = []
    if policy_name == "dog":
        history = int(cfg.dog.dog_num_obs_history)
        privileged = int(cfg.dog.dog_num_privileged_obs)
        expected_actor_input = history + privileged if structure["uses_adaptation"] else history
        expected_actions = int(cfg.dog.dog_actions)
        if structure.get("adaptation_input") not in (None, history):
            mismatches.append(
                f"adaptation input checkpoint={structure['adaptation_input']} runtime={history}"
            )
        if structure.get("adaptation_output") not in (None, privileged):
            mismatches.append(
                f"adaptation latent checkpoint={structure['adaptation_output']} runtime={privileged}"
            )
    else:
        observations = int(cfg.arm.arm_num_observations)
        history = int(cfg.arm.arm_num_obs_history)
        privileged = int(cfg.arm.arm_num_privileged_obs)
        expected_history_input = history - observations
        expected_actor_input = observations + int(structure["history_output"])
        if structure["uses_adaptation"]:
            expected_actor_input += privileged
        expected_actions = int(cfg.arm.num_actions_arm_cd)
        if structure["history_input"] != expected_history_input:
            mismatches.append(
                f"history input checkpoint={structure['history_input']} runtime={expected_history_input}"
            )
        if structure.get("adaptation_input") not in (None, history):
            mismatches.append(
                f"adaptation input checkpoint={structure['adaptation_input']} runtime={history}"
            )
        if structure.get("adaptation_output") not in (None, privileged):
            mismatches.append(
                f"adaptation latent checkpoint={structure['adaptation_output']} runtime={privileged}"
            )

    if structure["actor_input"] != expected_actor_input:
        mismatches.append(
            f"actor input checkpoint={structure['actor_input']} runtime={expected_actor_input}"
        )
    if structure["actions"] != expected_actions:
        mismatches.append(f"actions checkpoint={structure['actions']} runtime={expected_actions}")
    if mismatches:
        raise ValueError(
            f"{policy_name} checkpoint layout is incompatible with the reconstructed runtime: "
            + "; ".join(mismatches)
        )
    return structure


def _load_inference_state(model, checkpoint, policy_name):
    """Strictly load every inference tensor while ignoring critic-only state."""
    prefixes = ["actor_body."]
    if policy_name == "arm":
        prefixes.append("actor_history_encoder.")
    if any(key.startswith("adaptation_module.") for key in checkpoint):
        prefixes.append("adaptation_module.")

    model_state = model.state_dict()
    required_model_keys = {
        key for key in model_state if any(key.startswith(prefix) for prefix in prefixes)
    }
    checkpoint_keys = {
        key for key in checkpoint if any(key.startswith(prefix) for prefix in prefixes)
    }
    missing = sorted(required_model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - required_model_keys)
    mismatched = sorted(
        key
        for key in required_model_keys & checkpoint_keys
        if model_state[key].shape != checkpoint[key].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatched:
            shape_details = [
                f"{key}: checkpoint={tuple(checkpoint[key].shape)} runtime={tuple(model_state[key].shape)}"
                for key in mismatched
            ]
            details.append(f"shape mismatch={shape_details}")
        raise ValueError(
            f"{policy_name} checkpoint inference structure is incompatible: "
            + "; ".join(details)
        )

    model.load_state_dict(
        {key: checkpoint[key] for key in checkpoint_keys},
        strict=False,
    )


def _ensure_asset_file(cfg, robot=None, checkpoint_asset_file=None):
    if checkpoint_asset_file:
        return

    asset_file = getattr(cfg.asset, "file", "")
    robot = robot or "go2"
    if robot not in ROBOT_ASSET_FILES:
        if asset_file:
            return
        raise ValueError(f"Unknown robot '{robot}', expected one of {sorted(ROBOT_ASSET_FILES)}")
    cfg.asset.file = ROBOT_ASSET_FILES[robot]

    print(f"[RoboDuet] checkpoint asset file was empty; using {robot} asset: {cfg.asset.file}")


def load_dog_policy(logdir, ckpt_id, cfg):
    run_parameters = _load_run_parameters(logdir)
    ckpt_path = _checkpoint_path(logdir, "dog", ckpt_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    structure = _validate_checkpoint_layout(ckpt, "dog", cfg)
    model_args = _model_args_from_checkpoint(run_parameters.get("DogAC_Args"), ckpt)
    with _temporary_model_args(DogAC_Args, model_args):
        actor_critic = DogActorCritic(
            cfg.dog.dog_num_observations,
            cfg.dog.dog_num_privileged_obs,
            cfg.dog.dog_num_obs_history,
            cfg.dog.dog_actions,
            use_adaptation_module=structure["uses_adaptation"],
        ).to("cpu")
    _load_inference_state(actor_critic, ckpt, "dog")
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body

    def policy(obs, info=None):
        info = {} if info is None else info
        history = obs["obs_history"].to("cpu")
        actor_input = (history,)
        if adaptation_module is not None:
            latent = adaptation_module(history)
            actor_input = (history, latent)
            info["latent"] = latent
        return body(torch.cat(actor_input, dim=-1))

    return policy


def load_arm_policy(logdir, ckpt_id, cfg):
    run_parameters = _load_run_parameters(logdir)
    ckpt_path = _checkpoint_path(logdir, "arm", ckpt_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    structure = _validate_checkpoint_layout(ckpt, "arm", cfg)
    model_args = _model_args_from_checkpoint(run_parameters.get("ArmAC_Args"), ckpt)
    with _temporary_model_args(ArmAC_Args, model_args):
        actor_critic = ArmActorCritic(
            cfg.arm.arm_num_observations,
            cfg.arm.arm_num_privileged_obs,
            cfg.arm.arm_num_obs_history,
            cfg.arm.num_actions_arm_cd,
            use_adaptation_module=structure["uses_adaptation"],
        ).to("cpu")
    _load_inference_state(actor_critic, ckpt, "arm")
    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    actor_his = actor_critic.actor_history_encoder

    def policy(obs, info=None):
        info = {} if info is None else info
        history = obs["obs_history"].to("cpu")
        hist = actor_his(history[..., :-cfg.arm.arm_num_observations])
        actor_input = (obs["obs"].to("cpu"), hist)
        if adaptation_module is not None:
            latent = adaptation_module(history)
            actor_input = (obs["obs"].to("cpu"), latent, hist)
            info["latent"] = latent
        return body(torch.cat(actor_input, dim=-1))

    return policy

def load_env(logdir, wrapper, headless=False, device='cuda:0', robot=None):
    print('*'*10, logdir)
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1, robot=robot or "go2"))

    checkpoint_asset_file = None
    pkl_cfg = _load_run_parameters(logdir)
    snapshot = pkl_cfg["Cfg"]
    checkpoint_asset_file = snapshot.get("asset", {}).get("file")
    apply_config_snapshot(cfg, snapshot, drop_unknown=True)

    _ensure_asset_file(cfg, robot=robot, checkpoint_asset_file=checkpoint_asset_file)
    restore_dog_observation_layout(cfg, snapshot)
    recorded_arm_obs = snapshot.get("arm", {}).get("arm_num_observations")
    current_arm_obs = int(cfg.arm.arm_num_observations)
    if (
        bool(getattr(cfg.wbc.goal_reaching, "enabled", False))
        and recorded_arm_obs is not None
        and int(recorded_arm_obs) == current_arm_obs + 16
    ):
        # Early goal-reaching policies exposed the 16D coordination
        # diagnostics to the actor as well as the critic. Current policies
        # keep them critic-only. Select the checkpoint-era actor layout
        # explicitly instead of padding/truncating observations.
        cfg.arm.checkpoint_observation_layout = "goal_reaching_extended_v1"
        cfg.arm.arm_num_observations = int(recorded_arm_obs)
        cfg.arm.arm_num_obs_history = (
            cfg.arm.arm_num_observation_history * cfg.arm.arm_num_observations
        )
        print(
            "[RoboDuet] using checkpoint arm observation layout "
            f"goal_reaching_extended_v1 ({cfg.arm.arm_num_observations}D)"
        )
    else:
        cfg.arm.checkpoint_observation_layout = "current"

    # Every arm.action_mode has the same obs/action widths, so a checkpoint's
    # shapes carry no hint of which one it was trained with -- announce it, or
    # a waypoint policy silently gets replayed as a joint-residual one.
    print(f"[RoboDuet] arm action mode: {cfg.arm.action_mode}")

    cfg.terrain.mesh_type = "plane"
    if cfg.terrain.mesh_type == "plane":
      cfg.terrain.teleport_robots = False

    cfg.domain_rand.randomize_dog_obs_latency = False
    cfg.domain_rand.dog_obs_latency_jitter_steps = 0

    # turn off DR for evaluation script
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_com_displacement = False

    cfg.domain_rand.randomize_end_effector_force = False

    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = 1
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = False
    cfg.asset.render_sphere = True
    cfg.env.episode_length_s = 10000
    cfg.commands.resampling_time = 10000
    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    cfg.control.control_type = "M"
    cfg.rewards.use_terminal_body_height = False
    cfg.rewards.use_terminal_roll = False
    cfg.rewards.use_terminal_pitch = False
    cfg.wbc.rewards.use_terminal_body_height = False
    cfg.wbc.rewards.use_terminal_roll = False
    cfg.wbc.rewards.use_terminal_pitch = False
    # Cfg.sim.physx["num_position_iterations"] = 8
    # Cfg.sim.physx["num_velocity_iterations"] = 8


    env = wrapper(sim_device=device, headless=headless, cfg=cfg)
    env = HistoryWrapper(env)
    # load policy



    return env, cfg
