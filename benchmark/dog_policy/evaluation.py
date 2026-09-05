from __future__ import annotations

import json
import math
import os
import pickle as pkl
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import isaacgym  # noqa: F401 - must precede torch
import torch
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.config import (
    ConfigNode,
    apply_config_snapshot,
    build_roboduet_config,
    configure_privileged_obs_dims,
    recompute_observation_dims,
)
from go1_gym.envs.config.wbc import ROBODUET_OVERRIDES
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils.math_utils import quat_apply_yaw
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate_inverse
from scripts.load_policy import _ensure_asset_file

DEFAULT_CFG = build_roboduet_config()

SCENARIO_META = {
    "vel_grid": (
        "A - Velocity Grid",
        [
            ("xy_rmse", "lin_vel_xy_rmse"),
            ("vx_rmse", "lin_vel_x_rmse"),
            ("vy_rmse", "lin_vel_y_rmse"),
            ("yaw_rmse", "ang_vel_yaw_rmse"),
            ("resp_cons", "response_consistency_rmse"),
            ("lin_rew", "tracking_lin_vel_reward"),
            ("yaw_rew", "tracking_ang_vel_reward"),
            ("h_m", "base_height_mean"),
            ("fall_h_pct", "fall_rate_height"),
        ],
    ),
    "arm_sweep": (
        "B - Arm-Disturbance Sweep",
        [
            ("xy_rmse", "lin_vel_xy_rmse"),
            ("vx_rmse", "lin_vel_x_rmse"),
            ("vy_rmse", "lin_vel_y_rmse"),
            ("yaw_rmse", "ang_vel_yaw_rmse"),
            ("resp_cons", "response_consistency_rmse"),
            ("lin_rew", "tracking_lin_vel_reward"),
            ("yaw_rew", "tracking_ang_vel_reward"),
            ("fall_h_pct", "fall_rate_height"),
        ],
    ),
    "body_pose": (
        "C - Body-Pose Tracking",
        [
            ("pitch_deg_rmse", "pitch_rmse_deg"),
            ("roll_deg_rmse", "roll_rmse_deg"),
            ("orient_ctl", "orientation_control_rmse"),
            ("height_m_rmse", "height_rmse_m"),
            ("xy_rmse", "lin_vel_xy_rmse"),
            ("fall_h_pct", "fall_rate_height"),
        ],
    ),
    "gait": (
        "D - Gait-Parameter Tracking",
        [
            ("contact_f", "gait_contact_force_cost"),
            ("contact_v", "gait_contact_vel_cost"),
            ("stance_l_m", "stance_length_rmse_m"),
            ("clearance_m", "foot_clearance_rmse_m"),
            ("raibert_m", "raibert_rmse_m"),
            ("fall_h_pct", "fall_rate_height"),
        ],
    ),
    "vel_step": (
        "E - Velocity Step Response (predictable-plant)",
        [
            ("resp_cons", "response_consistency_rmse"),
            ("xy_rmse", "lin_vel_xy_rmse"),
            ("vx_rmse", "lin_vel_x_rmse"),
            ("yaw_rmse", "ang_vel_yaw_rmse"),
            ("lin_rew", "tracking_lin_vel_reward"),
            ("fall_h_pct", "fall_rate_height"),
        ],
    ),
}

# ---------------------------------------------------------------------------
# Command layout detection
# ---------------------------------------------------------------------------


class CommandLayout(NamedTuple):
    n_dims: int
    has_body_pitch: bool
    has_body_roll: bool
    has_body_height: bool
    has_dynamic_gait: bool
    has_stance_length: bool
    has_gait_duration: bool
    base_height_target: float


def detect_command_layout(cfg) -> CommandLayout:
    n = getattr(cfg.dog, "dog_num_commands", 3)
    bh = float(getattr(cfg.rewards, "base_height_target", 0.34))
    return CommandLayout(
        n_dims=n,
        has_body_pitch=n >= 4,
        has_body_roll=n >= 5,
        has_body_height=n >= 6,
        has_dynamic_gait=n >= 9,
        has_stance_length=n >= 10,
        has_gait_duration=n >= 11,
        base_height_target=bh,
    )


# ---------------------------------------------------------------------------
# Per-policy dim reading + standalone policy loader
# ---------------------------------------------------------------------------


def _dog_dims_from_cfg_dict(cfg_dict: dict) -> dict:
    dog = cfg_dict.get("dog", {})

    def _get(key, default):
        v = dog.get(key)
        if v is None:
            v = ROBODUET_OVERRIDES.get(f"dog.{key}", default)
        return v

    obs = _get("dog_num_observations", 48)
    priv = _get("dog_num_privileged_obs", 2)
    hlen = _get("dog_num_observation_history", 30)
    acts = _get("dog_actions", 12)
    use_adapt = _get("use_adaptation_module", True)

    return dict(
        dog_num_observations=obs,
        dog_num_privileged_obs=priv,
        dog_num_observation_history=hlen,
        dog_num_obs_history=hlen * obs,
        dog_actions=acts,
        use_adaptation_module=use_adapt,
    )


def _read_dog_dims(logdir: str) -> dict:
    """Extract dog network dims from a logdir's parameters.pkl.

    Does not mutate a process-global config — safe to call for every logdir.
    Falls back to the centralized WBC task defaults for any key that is absent.
    """
    with open(logdir + "/parameters.pkl", "rb") as f:
        cfg_dict = pkl.load(f)["Cfg"]
    return _dog_dims_from_cfg_dict(cfg_dict)


def load_dog_policy_for_benchmark(
    logdir: str,
    ckpt_id: str,
    env_cfg,
    expected_dims: Optional[dict] = None,
    device: str = "cuda:0",
) -> Callable:
    """Load a dog policy whose observation/action layout matches its env group.

    The policy runs on ``device`` -- the simulation's device -- so the whole
    rollout stays on the GPU. Running it on the CPU instead costs a full
    observation-history transfer off the device every step and dominates
    evaluation time.
    """
    dims = _read_dog_dims(logdir)

    if expected_dims is None:
        expected_dims = {
            "dog_num_observations": env_cfg.dog.dog_num_observations,
            "dog_num_privileged_obs": env_cfg.dog.dog_num_privileged_obs,
            "dog_num_observation_history": env_cfg.dog.dog_num_observation_history,
            "dog_num_obs_history": env_cfg.dog.dog_num_obs_history,
            "dog_actions": env_cfg.dog.dog_actions,
        }
    mismatches = [
        f"{key}: checkpoint={dims[key]} env={expected}"
        for key, expected in expected_dims.items()
        if dims[key] != expected
    ]
    if mismatches:
        raise ValueError(
            f"{logdir}: checkpoint dog policy obs/action dimensions are incompatible with its benchmark env "
            "(use_adaptation_module differences are allowed when obs/action dimensions match). "
            + "; ".join(mismatches)
        )

    actor_critic = DogActorCritic(
        dims["dog_num_observations"],
        dims["dog_num_privileged_obs"],
        dims["dog_num_obs_history"],
        dims["dog_actions"],
        use_adaptation_module=dims["use_adaptation_module"],
    ).to(device)

    ckpt_id_ = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
    ckpt = torch.load(
        logdir + f"/checkpoints_dog/ac_weights_{ckpt_id_}.pt",
        map_location="cpu",
    )
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()

    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body

    def policy(obs: dict, info: dict = {}):
        hist = obs["obs_history"].to(device)
        actor_input = (hist,)
        if adaptation_module is not None:
            latent = adaptation_module(hist)
            actor_input = (hist, latent)
            info["latent"] = latent
        return body(torch.cat(actor_input, dim=-1))

    dims_str = (
        f"obs={dims['dog_num_observations']} hist={dims['dog_num_observation_history']} acts={dims['dog_actions']}"
    )
    print(f"    dims: {dims_str}  adapt={'on' if dims['use_adaptation_module'] else 'off'}")

    return policy


# ---------------------------------------------------------------------------
# Policy handle — owns one contiguous slice of the shared env pool
# ---------------------------------------------------------------------------


@dataclass
class PolicyHandle:
    name: str
    policy: Callable  # dog_policy(obs_dict) → action tensor (sim device)
    env_start: int  # inclusive index into the shared env pool
    env_end: int  # exclusive
    envs_per_point: int  # envs averaged for one scenario point (--num_envs_per_policy)

    @property
    def n_envs(self) -> int:
        return self.env_end - self.env_start

    @property
    def points_per_batch(self) -> int:
        """Scenario points this handle's slice can evaluate simultaneously."""
        return max(1, self.n_envs // self.envs_per_point)

    def cell(self, point_index: int) -> Tuple[int, int]:
        """Env range holding one scenario point's samples for this policy."""
        start = self.env_start + point_index * self.envs_per_point
        return start, start + self.envs_per_point


class BenchmarkHistoryWrapper(HistoryWrapper):
    """History wrapper that recreates checkpoint-specific dog observations.

    The runtime environment stays on the current implementation. Only the
    policy-facing dog observation is adapted when an older checkpoint predates
    the current fixed-width tracking layout or includes the historical
    trajectory EE-pose block.
    """

    def __init__(self, env: WBCEnv, checkpoint_cfg: dict):
        super().__init__(env)
        dims = _dog_dims_from_cfg_dict(checkpoint_cfg)
        self.benchmark_dog_dims = dims
        self._checkpoint_cfg = checkpoint_cfg
        self._runtime_dog_obs_dim = int(env.cfg.dog.dog_num_observations)
        self._dog_obs_mode = self._detect_dog_obs_mode()
        self.benchmark_observation_mode = self._dog_obs_mode
        self.dog_obs_history = torch.zeros(
            env.num_envs,
            dims["dog_num_obs_history"],
            dtype=torch.float,
            device=env.device,
            requires_grad=False,
        )

    def _detect_dog_obs_mode(self) -> str:
        saved = int(self.benchmark_dog_dims["dog_num_observations"])
        if saved == self._runtime_dog_obs_dim:
            return "native"

        legacy_arm_trajectory = bool(
            self._checkpoint_cfg.get("arm", {}).get("trajectory", {}).get("enabled", False)
        )
        wbc_trajectory = bool(
            self._checkpoint_cfg.get("wbc", {}).get("trajectory", {}).get("enabled", False)
        )
        if wbc_trajectory and saved == self._runtime_dog_obs_dim + 9:
            return "native_plus_ee_pose"
        if legacy_arm_trajectory:
            return "legacy_pre_v3"
        raise ValueError(
            "Unsupported dog observation layout: "
            f"checkpoint={saved}, current_runtime={self._runtime_dog_obs_dim}. "
            "No known legacy observation adapter matches this parameters.pkl."
        )

    def _legacy_pre_v3_observation(self) -> torch.Tensor:
        b = self.env
        cfg = b.cfg
        obs = torch.cat(
            (
                b.projected_gravity,
                (b.dof_pos[:, : b.num_actions_loco] - b.default_dof_pos[:, : b.num_actions_loco])
                * b.obs_scales.dof_pos,
                b.dof_vel[:, : b.num_actions_loco] * b.obs_scales.dof_vel,
                b.actions[:, : b.num_actions_loco],
            ),
            dim=-1,
        )
        dog_commands = (b.commands_dog * b.commands_scale_dog)[:, : cfg.dog.dog_num_commands]
        legacy_vision = bool(
            self._checkpoint_cfg.get("hybrid", {}).get("use_vision", False)
        )
        if legacy_vision:
            arm_task = torch.cat(
                (
                    b.obj_obs_pose_in_ee
                    if global_switch.switch_open
                    else torch.zeros_like(b.obj_obs_pose_in_ee),
                    b.obj_obs_abg_in_ee
                    if global_switch.switch_open
                    else torch.zeros_like(b.obj_obs_abg_in_ee),
                ),
                dim=-1,
            )
        else:
            arm_commands = b.commands_arm_obs[:, : cfg.arm.arm_num_commands]
            arm_task = (
                arm_commands
                if global_switch.switch_open
                else torch.zeros_like(arm_commands)
            )
        obs = torch.cat(
            (obs, dog_commands, arm_task, b.roll.unsqueeze(1), b.pitch.unsqueeze(1)),
            dim=-1,
        )
        if cfg.env.observe_two_prev_actions:
            obs = torch.cat((obs, b.last_actions), dim=-1)
        if cfg.env.observe_timing_parameter:
            obs = torch.cat((obs, b.gait_indices.unsqueeze(1)), dim=-1)
        if cfg.env.observe_clock_inputs:
            obs = torch.cat((obs, b.clock_inputs), dim=-1)

        legacy_env = self._checkpoint_cfg.get("env", {})
        if legacy_env.get("observe_vel", False):
            lin_vel = (
                b.root_states[: b.num_envs, 7:10]
                if cfg.commands.global_reference
                else b.base_lin_vel
            )
            obs = torch.cat(
                (lin_vel * b.obs_scales.lin_vel, b.base_ang_vel * b.obs_scales.ang_vel, obs),
                dim=-1,
            )
        if legacy_env.get("observe_only_ang_vel", False):
            obs = torch.cat((b.base_ang_vel * b.obs_scales.ang_vel, obs), dim=-1)
        if legacy_env.get("observe_only_lin_vel", False):
            obs = torch.cat((b.base_lin_vel * b.obs_scales.lin_vel, obs), dim=-1)
        if legacy_env.get("observe_yaw", False):
            forward = quat_apply(b.base_quat, b.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            obs = torch.cat((obs, heading), dim=-1)
        if legacy_env.get("observe_contact_states", False):
            contacts = (b.contact_forces[:, b.feet_indices, 2] > 1.0).view(b.num_envs, -1).float()
            obs = torch.cat((obs, contacts), dim=-1)

        obs = torch.cat((obs, b.get_ee_pose_body_9d()), dim=-1)
        arm_slice = slice(b.num_actions_loco, b.num_actions_loco + b.num_actions_arm)
        arm_pos = (b.dof_pos[:, arm_slice] - b.default_dof_pos[:, arm_slice]) * b.obs_scales.dof_pos
        arm_vel = b.dof_vel[:, arm_slice] * b.obs_scales.dof_vel
        return torch.cat((obs, arm_pos, arm_vel), dim=-1)

    def get_dog_observations(self):
        if self._dog_obs_mode == "native":
            obs, privileged_obs = self.env.get_dog_observations()
        elif self._dog_obs_mode == "native_plus_ee_pose":
            obs, privileged_obs = self.env.get_dog_observations()
            arm_state_width = 2 * self.env.num_actions_arm
            obs = torch.cat(
                (obs[:, :-arm_state_width], self.env.get_ee_pose_body_9d(), obs[:, -arm_state_width:]),
                dim=-1,
            )
        else:
            obs = self._legacy_pre_v3_observation()
            privileged_obs = torch.zeros(
                self.env.num_envs,
                self.benchmark_dog_dims["dog_num_privileged_obs"],
                dtype=torch.float,
                device=self.env.device,
            )

        clip_obs = self.env.cfg.normalization.clip_observations
        obs = torch.clip(obs, -clip_obs, clip_obs)
        expected = int(self.benchmark_dog_dims["dog_num_observations"])
        if obs.shape[1] != expected:
            raise AssertionError(
                f"{self._dog_obs_mode} dog observation width {obs.shape[1]} != checkpoint width {expected}"
            )
        self.dog_obs_history = torch.cat(
            (self.dog_obs_history[:, expected:], obs),
            dim=-1,
        )
        return {
            "obs": obs,
            "privileged_obs": privileged_obs,
            "obs_history": self.dog_obs_history,
        }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    run_name: str
    scenario: str
    label: str
    cmd_x: float = 0.0
    cmd_y: float = 0.0
    cmd_yaw: float = 0.0
    cmd_pitch: float = 0.0
    cmd_roll: float = 0.0
    cmd_height_delta: float = 0.0
    cmd_gait_freq: float = 0.0
    cmd_footswing_height: float = 0.0
    cmd_stance_width: float = 0.0
    cmd_stance_length: float = 0.0
    cmd_gait_duration: float = 0.0
    arm_intensity: float = 0.0
    disturbance_seed: Optional[int] = None
    velocity_group: Optional[str] = None
    pose_axis: Optional[str] = None
    sweep_axis: Optional[str] = None
    n_env_steps: int = 0
    n_falls: int = 0
    lin_vel_x_rmse: Optional[float] = None
    lin_vel_y_rmse: Optional[float] = None
    lin_vel_xy_rmse: Optional[float] = None
    ang_vel_yaw_rmse: Optional[float] = None
    response_consistency_rmse: Optional[float] = None
    pitch_rmse_deg: Optional[float] = None
    roll_rmse_deg: Optional[float] = None
    height_rmse_m: Optional[float] = None
    gait_freq_rmse_hz: Optional[float] = None
    footswing_height_rmse_m: Optional[float] = None
    stance_width_rmse_m: Optional[float] = None
    stance_length_rmse_m: Optional[float] = None
    tracking_lin_vel_reward: Optional[float] = None
    tracking_ang_vel_reward: Optional[float] = None
    orientation_control_rmse: Optional[float] = None
    gait_contact_force_cost: Optional[float] = None
    gait_contact_vel_cost: Optional[float] = None
    foot_clearance_rmse_m: Optional[float] = None
    raibert_rmse_m: Optional[float] = None
    fall_rate: float = 0.0
    fall_rate_height: float = 0.0
    base_height_mean: float = 0.0
    base_height_std: float = 0.0
    roll_deg_rms: Optional[float] = None
    pitch_deg_rms: Optional[float] = None
    max_torque_mean: Optional[float] = None


# ---------------------------------------------------------------------------
# Running accumulator
# ---------------------------------------------------------------------------


class Accumulator:
    """Welford-lite sum / sum-of-squares accumulator on a specified device."""

    def __init__(self, num_envs: int, device: str):
        self.n = num_envs
        self.dev = device
        self._stats: Dict[str, torch.Tensor] = {}
        self.steps = torch.zeros(num_envs, device=device)

    def _get(self, key: str) -> torch.Tensor:
        if key not in self._stats:
            self._stats[key] = torch.zeros(self.n, device=self.dev)
        return self._stats[key]

    def add_sq_err(self, key: str, pred: torch.Tensor, target: torch.Tensor):
        self._get(f"{key}_sq").add_((pred - target).pow(2))

    def add_sq(self, key: str, sq: torch.Tensor):
        self._get(f"{key}_sq").add_(sq)

    def add_val(self, key: str, val: torch.Tensor):
        self._get(f"{key}_sum").add_(val)
        self._get(f"{key}_sq").add_(val.pow(2))

    def add_count(self, key: str, mask: torch.Tensor):
        self._get(key).add_(mask.float())

    def tick(self):
        self.steps.add_(1.0)

    # ---- summary methods: called once per scenario point ----
    #
    # One accumulator covers the whole env pool; ``env_slice`` selects the envs
    # belonging to a single scenario point, so per-step accumulation stays a
    # handful of full-width ops no matter how many points share the rollout.

    def rmse(self, key: str, env_slice: slice = slice(None)) -> float:
        sq = self._stats.get(f"{key}_sq")
        if sq is None:
            return 0.0
        return float((sq[env_slice] / self.steps[env_slice].clamp(min=1)).mean().sqrt().cpu())

    def mean(self, key: str, env_slice: slice = slice(None)) -> float:
        s = self._stats.get(f"{key}_sum")
        if s is None:
            return 0.0
        return float((s[env_slice] / self.steps[env_slice].clamp(min=1)).mean().cpu())

    def std(self, key: str, env_slice: slice = slice(None)) -> float:
        s = self._stats.get(f"{key}_sum")
        sq = self._stats.get(f"{key}_sq")
        if s is None or sq is None:
            return 0.0
        n = self.steps[env_slice].clamp(min=1)
        return float((sq[env_slice] / n - (s[env_slice] / n).pow(2)).clamp(min=0).mean().sqrt().cpu())

    def total_count(self, key: str, env_slice: slice = slice(None)) -> int:
        t = self._stats.get(key)
        if t is None:
            return 0
        return int(t[env_slice].sum().cpu().item())

    def view(self, start: int, end: int) -> "AccumulatorView":
        return AccumulatorView(self, start, end)

    def reset(self):
        self._stats.clear()
        self.steps.zero_()


class AccumulatorView:
    """One scenario point's envs within a pooled Accumulator.

    Exposes the same read API as Accumulator so result building does not care
    whether a rollout measured one point or many.
    """

    def __init__(self, acc: Accumulator, start: int, end: int):
        self._acc = acc
        self._slice = slice(start, end)

    @property
    def steps(self) -> torch.Tensor:
        return self._acc.steps[self._slice]

    @property
    def _stats(self) -> Dict[str, torch.Tensor]:
        return self._acc._stats

    def rmse(self, key: str) -> float:
        return self._acc.rmse(key, self._slice)

    def mean(self, key: str) -> float:
        return self._acc.mean(key, self._slice)

    def std(self, key: str) -> float:
        return self._acc.std(key, self._slice)

    def total_count(self, key: str) -> int:
        return self._acc.total_count(key, self._slice)


# ---------------------------------------------------------------------------
# Config / env loading
# ---------------------------------------------------------------------------

CRITICAL_COMPAT_CFG_PATHS = [
    "dog.dog_num_commands",
    "dog.dog_num_observations",
    "dog.dog_num_observation_history",
    "dog.dog_num_privileged_obs",
    "dog.dog_actions",
    "dog.num_actions_loco",
    "arm.arm_num_commands",
    "arm.num_actions_arm",
    "arm.trajectory.enabled",
    "wbc.trajectory.enabled",
    "asset.file",
    "commands.global_reference",
    "env.observe_two_prev_actions",
    "env.observe_timing_parameter",
    "env.observe_clock_inputs",
    "env.observe_vel",
    "env.observe_only_ang_vel",
    "env.observe_only_lin_vel",
    "env.observe_yaw",
    "env.observe_contact_states",
    "wbc.use_vision",
    "use_rot6d",
]


def _read_cfg_dict(logdir: str) -> dict:
    with open(logdir + "/parameters.pkl", "rb") as f:
        return pkl.load(f)["Cfg"]


def _nested_get_dict(d: dict, path: str):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _nested_get_attr(obj, path: str):
    cur = obj
    for key in path.split("."):
        if not hasattr(cur, key):
            return None
        cur = getattr(cur, key)
    return cur


def _cfg_value(cfg_dict: dict, path: str):
    value = _nested_get_dict(cfg_dict, path)
    if value is not None:
        return value
    if path in ROBODUET_OVERRIDES:
        return ROBODUET_OVERRIDES[path]
    return _nested_get_attr(DEFAULT_CFG, path)


def read_dog_num_commands(logdir: str) -> int:
    return int(_cfg_value(_read_cfg_dict(logdir), "dog.dog_num_commands"))


def validate_shared_env_compatibility(base_logdir: str, candidate_logdirs: List[str]):
    """Fail fast when policies do not share the same env/observation semantics."""
    base_cfg = _read_cfg_dict(base_logdir)
    errors = []
    for logdir in candidate_logdirs:
        cfg = _read_cfg_dict(logdir)
        mismatches = []
        for path in CRITICAL_COMPAT_CFG_PATHS:
            base_value = _cfg_value(base_cfg, path)
            value = _cfg_value(cfg, path)
            if value != base_value:
                mismatches.append(f"{path}: base={base_value!r} candidate={value!r}")
        if mismatches:
            errors.append(f"{logdir}:\n    " + "\n    ".join(mismatches))
    if errors:
        raise ValueError(
            "All compared policies must share the same observation/control layout when using one shared env.\n"
            + "\n".join(errors)
        )


def shared_env_compatibility_key(logdir: str) -> Tuple[object, ...]:
    """Return the resolved config signature that determines shared-env safety."""
    cfg = _read_cfg_dict(logdir)

    def _freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_freeze(item) for item in value)
        return value

    return tuple(_freeze(_cfg_value(cfg, path)) for path in CRITICAL_COMPAT_CFG_PATHS)


def group_shared_env_compatible_runs(logdirs: List[str]) -> List[List[int]]:
    """Group input run indices by compatible observation/control semantics.

    Group and member order are stable. Policies in one group can still share a
    simulator; incompatible groups must be evaluated in separate simulators.
    """
    groups: List[List[int]] = []
    key_to_group: Dict[Tuple[object, ...], int] = {}
    for index, logdir in enumerate(logdirs):
        key = shared_env_compatibility_key(logdir)
        group_index = key_to_group.get(key)
        if group_index is None:
            group_index = len(groups)
            key_to_group[key] = group_index
            groups.append([])
        groups[group_index].append(index)
    return groups


def describe_shared_env_group(logdir: str) -> Dict[str, object]:
    """Return a JSON-friendly compatibility description for metadata."""
    cfg = _read_cfg_dict(logdir)
    return {path: _cfg_value(cfg, path) for path in CRITICAL_COMPAT_CFG_PATHS}


def _load_cfg_from_pkl(logdir: str, robot: Optional[str] = None) -> ConfigNode:
    cfg = build_roboduet_config()
    checkpoint_asset_file = None
    with open(logdir + "/parameters.pkl", "rb") as f:
        pkl_cfg = pkl.load(f)
        cfg_snapshot = pkl_cfg["Cfg"]
        checkpoint_asset_file = cfg_snapshot.get("asset", {}).get("file")
        apply_config_snapshot(cfg, cfg_snapshot, drop_unknown=True)
    _ensure_asset_file(cfg, robot=robot, checkpoint_asset_file=checkpoint_asset_file)
    recompute_observation_dims(cfg)
    return cfg


def _apply_benchmark_env_overrides(cfg, total_envs: int, envs_per_policy: int):
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    for attr in [
        "push_robots",
        "randomize_friction",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
        "randomize_com_displacement",
        "randomize_end_effector_force",
    ]:
        if hasattr(cfg.domain_rand, attr):
            setattr(cfg.domain_rand, attr, False)
    cfg.env.num_envs = total_envs
    cfg.env.num_recording_envs = 0
    cfg.env.asset_bucket_cycle_length = envs_per_policy
    side = max(5, int(math.ceil(math.sqrt(total_envs))) + 1)
    cfg.terrain.num_rows = side
    cfg.terrain.num_cols = side
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.asset.render_sphere = False
    cfg.env.episode_length_s = 20.0
    cfg.commands.resampling_time = 1e9
    cfg.control.control_type = "M"
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.use_terminal_roll = True
    cfg.rewards.use_terminal_pitch = True
    cfg.wbc.rewards.use_terminal_body_height = True
    cfg.wbc.rewards.use_terminal_roll = True
    cfg.wbc.rewards.use_terminal_pitch = True
    cfg.env.stage1_arm_curriculum = True


def load_env_benchmark(
    logdir: str,
    total_envs: int,
    envs_per_policy: int,
    headless: bool = True,
    device: str = "cuda:0",
    robot: Optional[str] = None,
):
    checkpoint_cfg = _read_cfg_dict(logdir)
    cfg = _load_cfg_from_pkl(logdir, robot=robot)
    _apply_benchmark_env_overrides(cfg, total_envs, envs_per_policy)
    configure_privileged_obs_dims(cfg)
    env = WBCEnv(sim_device=device, headless=headless, cfg=cfg)
    env = BenchmarkHistoryWrapper(env, checkpoint_cfg)
    return env, cfg


def preview_command_layout(logdir: str, robot: Optional[str] = None) -> CommandLayout:
    """Resolve a run's command layout without building its simulation.

    Env pool sizing depends on which scenarios the layout enables, and that has
    to be decided before WBCEnv is constructed.
    """
    return detect_command_layout(_load_cfg_from_pkl(logdir, robot=robot))


# ---------------------------------------------------------------------------
# Stage-1 global switch
# ---------------------------------------------------------------------------


def configure_stage1(arm_intensity: float, ramp_iters: int = 1000):
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.stage1_arm_ramp_iterations = ramp_iters
    global_switch.stage1_count = int(max(0.0, min(1.0, arm_intensity)) * ramp_iters)
    global_switch.pretrained_to_wbc_start = 10_000_000
    global_switch.pretrained_to_wbc_end = 10_000_001


# ---------------------------------------------------------------------------
# Command setters (apply to all envs uniformly)
# ---------------------------------------------------------------------------

# Set by record_command_writes() while a scenario's command function runs, so
# the caller learns which command columns that scenario actually drives. Batched
# evaluation replays only those columns per env, leaving the rest as the env
# itself set them.
_CMD_WRITE_COLUMNS: Optional[set] = None


@contextmanager
def record_command_writes():
    """Collect the command column indices written inside the block."""
    global _CMD_WRITE_COLUMNS
    columns: set = set()
    _CMD_WRITE_COLUMNS = columns
    try:
        yield columns
    finally:
        _CMD_WRITE_COLUMNS = None


def _set_cmd(env: HistoryWrapper, idx: int, value: float):
    b = env.env
    if b.commands_dog.shape[1] > idx:
        b.commands_dog[:, idx] = value
        if _CMD_WRITE_COLUMNS is not None:
            _CMD_WRITE_COLUMNS.add(idx)


def set_vel_cmd(env: HistoryWrapper, x: float, y: float, yaw: float):
    b = env.env
    b.commands_dog[:, 0] = x
    b.commands_dog[:, 1] = y
    b.commands_dog[:, 2] = yaw
    if _CMD_WRITE_COLUMNS is not None:
        _CMD_WRITE_COLUMNS.update((0, 1, 2))


def set_pose_cmd(env: HistoryWrapper, pitch: float, roll: float, height_delta: float):
    _set_cmd(env, 3, pitch)
    _set_cmd(env, 4, roll)
    _set_cmd(env, 5, height_delta)


def set_gait_cmd(
    env: HistoryWrapper,
    gait_freq: Optional[float] = None,
    footswing_height: Optional[float] = None,
    stance_width: Optional[float] = None,
    stance_length: Optional[float] = None,
    gait_duration: Optional[float] = None,
):
    if gait_freq is not None:
        _set_cmd(env, 6, gait_freq)
    if footswing_height is not None:
        _set_cmd(env, 7, footswing_height)
    if stance_width is not None:
        _set_cmd(env, 8, stance_width)
    if stance_length is not None:
        _set_cmd(env, 9, stance_length)
    if gait_duration is not None:
        _set_cmd(env, 10, gait_duration)


# ---------------------------------------------------------------------------
# Batched metric helpers — all stay on GPU
# ---------------------------------------------------------------------------


def _actual_height(env: HistoryWrapper) -> torch.Tensor:
    b = env.env
    return b.base_pos[:, 2]


def _orientation_control_sq(env: HistoryWrapper, cmd: torch.Tensor) -> torch.Tensor:
    """Mirror Rewards._reward_orientation_control before reward scaling."""
    b = env.env
    axis_x = torch.tensor([1, 0, 0], device=b.device, dtype=torch.float)
    axis_y = torch.tensor([0, 1, 0], device=b.device, dtype=torch.float)
    quat_roll = quat_from_angle_axis(-cmd[:, 4], axis_x)
    quat_pitch = quat_from_angle_axis(-cmd[:, 3], axis_y)
    desired_quat = quat_mul(quat_roll, quat_pitch)
    desired_gravity = quat_rotate_inverse(desired_quat, b.gravity_vec)
    return torch.sum(torch.square(b.projected_gravity[:, :2] - desired_gravity[:, :2]), dim=1)


def _gait_training_costs(env: HistoryWrapper) -> Dict[str, torch.Tensor]:
    """Mirror the gait-related training penalties used by Rewards.

    Lower is better for all returned values. These are not command-vs-actual
    scalar RMSEs: gait frequency and stance settings enter training through the
    desired contact clock, foot-clearance target, and Raibert foot placement.
    """
    b = env.env
    desired_contact = b.desired_contact_states

    foot_forces = torch.norm(b.contact_forces[:, b.feet_indices, :], dim=-1)
    contact_force_cost = torch.zeros(b.num_envs, device=b.device)
    for i in range(4):
        contact_force_cost += (1 - desired_contact[:, i]) * (
            1 - torch.exp(-foot_forces[:, i].pow(2) / b.cfg.rewards.gait_force_sigma)
        )
    contact_force_cost = contact_force_cost / 4

    foot_velocities = torch.norm(b.foot_velocities, dim=2).view(b.num_envs, -1)
    contact_vel_cost = torch.zeros(b.num_envs, device=b.device)
    for i in range(4):
        contact_vel_cost += desired_contact[:, i] * (
            1 - torch.exp(-foot_velocities[:, i].pow(2) / b.cfg.rewards.gait_vel_sigma)
        )
    contact_vel_cost = contact_vel_cost / 4

    phases = 1 - torch.abs(1.0 - torch.clip((b.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    foot_height = b.foot_positions[:, :, 2].view(b.num_envs, -1)
    if b.cfg.commands.use_dynamic_gait:
        footswing_height = b.commands_dog[:, 7:8]
    else:
        footswing_height = 0.04
    target_height = footswing_height * phases + 0.02
    clearance_sq = torch.sum(torch.square(target_height - foot_height) * (1 - desired_contact), dim=1)

    cur_footsteps_translated = b.foot_positions - b.base_pos.unsqueeze(1)
    qexp = quat_conjugate(b.base_quat).unsqueeze(1).expand(-1, 4, -1).reshape(b.num_envs * 4, 4)
    footsteps_body = quat_apply_yaw(qexp, cur_footsteps_translated.reshape(b.num_envs * 4, 3)).view(b.num_envs, 4, 3)

    if b.cfg.commands.use_dynamic_gait:
        desired_stance_width = b.commands_dog[:, 8]
        if b.commands_dog.shape[1] > 9:
            desired_stance_length = b.commands_dog[:, 9]
        else:
            desired_stance_length = torch.full((b.num_envs,), 0.45, device=b.device)
        frequencies = torch.clamp(b.commands_dog[:, 6:7], min=0.1)
    else:
        desired_stance_width = torch.full((b.num_envs,), 0.3, device=b.device)
        desired_stance_length = torch.full((b.num_envs,), 0.45, device=b.device)
        frequencies = 3.0

    desired_ys = torch.stack(
        [
            desired_stance_width / 2,
            -desired_stance_width / 2,
            desired_stance_width / 2,
            -desired_stance_width / 2,
        ],
        dim=1,
    )
    desired_xs = torch.stack(
        [
            desired_stance_length / 2,
            desired_stance_length / 2,
            -desired_stance_length / 2,
            -desired_stance_length / 2,
        ],
        dim=1,
    )
    phases = torch.abs(1.0 - (b.foot_indices * 2.0)) - 0.5
    x_vel_des = b.commands_dog[:, 0:1]
    yaw_vel_des = b.commands_dog[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length.unsqueeze(1) / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)
    desired_ys = desired_ys + desired_ys_offset
    desired_xs = desired_xs + desired_xs_offset
    desired_steps = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)
    raibert_sq = torch.sum(torch.square(torch.abs(desired_steps - footsteps_body[:, :, 0:2])), dim=(1, 2))

    return {
        "contact_force_cost": contact_force_cost,
        "contact_vel_cost": contact_vel_cost,
        "clearance_sq": clearance_sq,
        "raibert_sq": raibert_sq,
    }


def _contact_transition_mask(env: HistoryWrapper, prev_contact: torch.Tensor, env_slice: slice) -> torch.Tensor:
    """Per-env count of leg contact-state transitions this step."""
    b = env.env
    curr = (b.contact_forces[env_slice, b.feet_indices, 2] > 1.0)
    changed = (curr != prev_contact).sum(dim=1)
    prev_contact[:] = curr
    return changed.float()


def _actual_swing_height(env: HistoryWrapper) -> torch.Tensor:
    """Mean foot height of airborne feet; 0 when all grounded."""
    b = env.env
    in_contact = (b.contact_forces[:, b.feet_indices, 2] > 1.0)
    in_swing = ~in_contact
    foot_z = b.foot_positions[:, :, 2]
    swing_z = (foot_z * in_swing.float()).sum(dim=1)
    n_swing = in_swing.float().sum(dim=1).clamp(min=1)
    return swing_z / n_swing


def _actual_stance_width(env: HistoryWrapper) -> torch.Tensor:
    """Stance width from foot Y-spread in yaw-only body frame."""
    rel_body = _feet_in_yaw_body_frame(env)
    y_max = rel_body[:, :, 1].max(dim=1).values
    y_min = rel_body[:, :, 1].min(dim=1).values
    return y_max - y_min


def _actual_stance_length(env: HistoryWrapper) -> torch.Tensor:
    """Stance length from foot X-spread in yaw-only body frame."""
    rel_body = _feet_in_yaw_body_frame(env)
    x_max = rel_body[:, :, 0].max(dim=1).values
    x_min = rel_body[:, :, 0].min(dim=1).values
    return x_max - x_min


def _feet_in_yaw_body_frame(env: HistoryWrapper) -> torch.Tensor:
    """Foot positions relative to base in a yaw-only body frame."""
    b = env.env
    foot_pos = b.foot_positions
    base_pos = b.root_states[:, :3]
    base_quat = b.base_quat
    rel = foot_pos - base_pos.unsqueeze(1)
    n_envs, n_feet, _ = rel.shape
    rel_flat = rel.reshape(n_envs * n_feet, 3)
    quat_flat = quat_conjugate(base_quat).repeat_interleave(n_feet, dim=0)
    return quat_apply_yaw(quat_flat, rel_flat).reshape(n_envs, n_feet, 3)


# ---------------------------------------------------------------------------
# Core parallel eval loop
# ---------------------------------------------------------------------------


def _eval_loop_parallel(
    env: HistoryWrapper,
    handles: List[PolicyHandle],
    layout: CommandLayout,
    n_steps: int,
    arm_intensity: float,
    device: str,
    cmd_fn: Optional[Callable] = None,
    settle_steps: int = 0,
    settle_cmd_fn: Optional[Callable] = None,
    metric_groups: Optional[List[Tuple[int, int]]] = None,
) -> List["AccumulatorView"]:
    """Step all env groups in one call; report metrics per metric group.

    Simulator tensors stay on device inside the metric loop.

    Actions are computed per handle, over that policy's whole env slice, so a
    policy runs as one batched forward pass. Metrics accumulate once over the
    whole pool -- every env carries its own command, so the maths is already
    per-point correct -- and ``metric_groups`` (``(env_start, env_end)`` ranges)
    only splits the result at the end, one view each, in the same order. It
    defaults to one group per handle; batched scenario evaluation passes one
    group per (policy, scenario point) cell so a single rollout yields every
    point in the batch without extra per-step work.

    ``settle_steps`` runs that many un-accumulated steps first, holding
    ``settle_cmd_fn`` (falling back to ``cmd_fn``). Used by the step-response
    scenario to damp the reset's hardcoded +-0.5 m/s initial base velocity to
    rest -- and bring the first-order reference model dog_vel_ref to 0 -- so the
    subsequently-held command is a clean step from rest.
    """
    configure_stage1(arm_intensity)
    env.reset()
    if cmd_fn:
        cmd_fn(env)

    base = env.env
    gpu = base.device
    if metric_groups is None:
        metric_groups = [(h.env_start, h.env_end) for h in handles]

    def _step_policies():
        with torch.no_grad():
            obs = env.get_dog_observations()
            actions = torch.zeros(base.num_envs, base.num_actions_loco, device=gpu)
            for h in handles:
                actions[h.env_start:h.env_end] = h.policy(
                    {k: v[h.env_start:h.env_end] for k, v in obs.items()}
                ).to(gpu)
        env.step(actions, env.arm_fake_actions)

    # Settle: reach steady stance at settle_cmd_fn (e.g. zero velocity) before
    # measurement, so what follows is a genuine step response, not the reset
    # transient. No accumulation here.
    for _ in range(settle_steps):
        (settle_cmd_fn or cmd_fn or (lambda _e: None))(env)
        _step_policies()

    acc = Accumulator(base.num_envs, device=gpu)
    prev_contact = (base.contact_forces[:, base.feet_indices, 2] > 1.0).clone()
    contact_transition_counts = torch.zeros(base.num_envs, device=gpu)
    R2D = 180.0 / math.pi

    for _ in range(n_steps):
        if cmd_fn:
            cmd_fn(env)

        with torch.no_grad():
            all_obs = env.get_dog_observations()  # dict; tensors shape [total_envs, ...]

            # Fill the full action buffer; each policy only sees its env slice
            all_actions = torch.zeros(base.num_envs, base.num_actions_loco, device=gpu)
            for h in handles:
                s, e = h.env_start, h.env_end
                group_obs = {k: v[s:e] for k, v in all_obs.items()}
                all_actions[s:e] = h.policy(group_obs).to(gpu)

        env.step(all_actions, env.arm_fake_actions)

        # --- shared tensors read once; accumulated full-width (no extra copies) ---
        vel = base.base_lin_vel  # [total_envs, 3]
        ang = base.base_ang_vel  # [total_envs, 3]
        cmd = base.commands_dog  # [total_envs, D]
        vel_ref = getattr(base, "dog_vel_ref", None)  # [total_envs, 2] first-order ref, or None
        pitch = base.pitch  # [total_envs]
        roll = base.roll  # [total_envs]
        height = _actual_height(env)  # [total_envs]
        orientation_sq = _orientation_control_sq(env, cmd) if layout.has_body_roll else None
        reset_b = base.reset_buf  # [total_envs] bool
        timeout = base.time_out_buf  # [total_envs] bool
        torques = base.torques  # [total_envs, dof]

        if layout.has_dynamic_gait:
            gait_costs = _gait_training_costs(env)
            swing_height = _actual_swing_height(env)
            stance_width = _actual_stance_width(env)
            stance_length = _actual_stance_length(env)

        # --- accumulation over the whole pool ---
        # Each env already holds its own scenario point's command, so every
        # error below is per-point correct without slicing; the per-point split
        # happens once at the end, in the returned views.
        acc.add_sq_err("vx", vel[:, 0], cmd[:, 0])
        acc.add_sq_err("vy", vel[:, 1], cmd[:, 1])
        acc.add_sq_err("yaw", ang[:, 2], cmd[:, 2])
        lin_vel_err = torch.sum(torch.square(cmd[:, :2] - vel[:, :2]), dim=1)
        acc.add_sq("lin_vel_xy", lin_vel_err)
        if vel_ref is not None:
            # Predictable-plant metric: deviation of the realised base velocity
            # from the first-order reference model of the command
            # (v_ref += (v_cmd - v_ref) * dt / T). Same quantity as the
            # response_consistency reward -- low = leg response tracks a
            # fixed-time-constant linear plant, which upstream v_ff needs.
            acc.add_sq("response_consistency", torch.sum(torch.square(vel[:, :2] - vel_ref), dim=1))
        yaw_err = torch.square(cmd[:, 2] - ang[:, 2])
        acc.add_val("tracking_lin_vel_reward", torch.exp(-lin_vel_err / base.cfg.rewards.tracking_sigma))
        acc.add_val("tracking_ang_vel_reward", torch.exp(-yaw_err / base.cfg.rewards.tracking_sigma_yaw))

        if layout.has_body_pitch:
            acc.add_sq_err("pitch_deg_track", pitch * R2D, cmd[:, 3] * R2D)
        if layout.has_body_roll:
            acc.add_sq_err("roll_deg_track", roll * R2D, cmd[:, 4] * R2D)
            acc.add_sq("orientation_control", orientation_sq)
        if layout.has_body_height:
            acc.add_sq_err("height_track", height, cmd[:, 5] + layout.base_height_target)

        if layout.has_dynamic_gait:
            contact_transition_counts.add_(_contact_transition_mask(env, prev_contact, slice(None)))
            acc.add_sq_err("swing_h", swing_height, cmd[:, 7])
            acc.add_sq_err("stance_w", stance_width, cmd[:, 8])
            if layout.has_stance_length and cmd.shape[1] > 9:
                acc.add_sq_err("stance_l", stance_length, cmd[:, 9])

            acc.add_val("gait_contact_force_cost", gait_costs["contact_force_cost"])
            acc.add_val("gait_contact_vel_cost", gait_costs["contact_vel_cost"])
            acc.add_sq("foot_clearance", gait_costs["clearance_sq"])
            acc.add_sq("raibert", gait_costs["raibert_sq"])

        # Stability RMS: raw value accumulation (always computed)
        acc.add_val("height", height)
        acc.add_val("pitch_deg_raw", pitch * R2D)
        acc.add_val("roll_deg_raw", roll * R2D)

        dog_t = torques[:, :12] if torques.shape[1] >= 12 else torques
        acc.add_val("max_torque", dog_t.abs().max(dim=1).values)
        acc.add_count("fell", (reset_b & ~timeout).float())
        if hasattr(base, "body_height_buf"):
            height_terminal = reset_b & base.body_height_buf
        else:
            terminal_h = float(getattr(base.cfg.rewards, "terminal_body_height", 0.17))
            height_terminal = reset_b & (height < terminal_h)
        acc.add_count("fell_height", (height_terminal & ~timeout).float())

        acc.tick()

    if layout.has_dynamic_gait:
        elapsed = max(float(n_steps) * float(base.dt), 1e-6)
        actual_freq = contact_transition_counts / (8.0 * elapsed)
        acc.add_sq(
            "gait_freq",
            torch.square(actual_freq - base.commands_dog[:, 6]) * acc.steps.clamp(min=1),
        )

    return [acc.view(s, e) for s, e in metric_groups]


# ---------------------------------------------------------------------------
# acc → ScenarioResult
# ---------------------------------------------------------------------------


def _acc_to_result(
    acc: Accumulator,
    layout: CommandLayout,
    run_name: str,
    scenario: str,
    label: str,
    n_steps: int,
    **kwargs,
) -> ScenarioResult:
    total_steps = int(acc.steps.sum().cpu().item())
    r = ScenarioResult(
        run_name=run_name,
        scenario=scenario,
        label=label,
        n_env_steps=total_steps,
        n_falls=acc.total_count("fell"),
        fall_rate=acc.total_count("fell") / max(1, total_steps),
        fall_rate_height=acc.total_count("fell_height") / max(1, total_steps),
        lin_vel_x_rmse=acc.rmse("vx"),
        lin_vel_y_rmse=acc.rmse("vy"),
        lin_vel_xy_rmse=acc.rmse("lin_vel_xy"),
        ang_vel_yaw_rmse=acc.rmse("yaw"),
        tracking_lin_vel_reward=acc.mean("tracking_lin_vel_reward"),
        tracking_ang_vel_reward=acc.mean("tracking_ang_vel_reward"),
        base_height_mean=acc.mean("height"),
        base_height_std=acc.std("height"),
        max_torque_mean=acc.mean("max_torque"),
    )
    if "response_consistency_sq" in acc._stats:
        r.response_consistency_rmse = acc.rmse("response_consistency")
    # Stability RMS: always computed from raw values
    r.pitch_deg_rms = acc.rmse("pitch_deg_raw")
    r.roll_deg_rms = acc.rmse("roll_deg_raw")
    # Tracking RMSE: only meaningful when layout commands them
    if layout.has_body_pitch:
        r.pitch_rmse_deg = acc.rmse("pitch_deg_track")
    if layout.has_body_roll:
        r.roll_rmse_deg = acc.rmse("roll_deg_track")
        r.orientation_control_rmse = acc.rmse("orientation_control")
    if layout.has_body_height:
        r.height_rmse_m = acc.rmse("height_track")
    if layout.has_dynamic_gait:
        r.gait_freq_rmse_hz = acc.rmse("gait_freq")
        r.footswing_height_rmse_m = acc.rmse("swing_h")
        r.stance_width_rmse_m = acc.rmse("stance_w")
        if layout.has_stance_length:
            r.stance_length_rmse_m = acc.rmse("stance_l")
        r.gait_contact_force_cost = acc.mean("gait_contact_force_cost")
        r.gait_contact_vel_cost = acc.mean("gait_contact_vel_cost")
        r.foot_clearance_rmse_m = acc.rmse("foot_clearance")
        r.raibert_rmse_m = acc.rmse("raibert")
    for k, v in kwargs.items():
        if hasattr(r, k):
            setattr(r, k, v)
    return r


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _fmt(v: Optional[float], fmt: str = ".4f") -> str:
    if v is None:
        return "—"
    return "—" if math.isnan(v) else f"{v:{fmt}}"


def _scenario_labels(all_results: Dict[str, Dict[str, List[ScenarioResult]]], scenario: str) -> List[str]:
    labels = []
    for run_name in all_results:
        for result in all_results[run_name].get(scenario, []):
            if result.label not in labels:
                labels.append(result.label)
    return labels


def _format_metric_value(result: Optional[ScenarioResult], attr: str) -> str:
    if result is None:
        return "—"
    value = getattr(result, attr)
    if value is None:
        return "—"
    if math.isnan(value):
        return "—"
    if attr in ("fall_rate", "fall_rate_height"):
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def print_comparison_table(all_results: Dict[str, Dict[str, List[ScenarioResult]]]):
    run_names = list(all_results.keys())
    sep = "=" * 80

    for scenario, (title, cols) in SCENARIO_META.items():
        labels = _scenario_labels(all_results, scenario)
        if not labels:
            continue

        print(f"\n{sep}\nScenario {title}\n{sep}")

        lookup = {rn: {r.label: r for r in all_results[rn].get(scenario, [])} for rn in run_names}
        col_w = max(12, max(len(cn) for cn, _ in cols) + 2)
        label_w = max(22, max(len(lb) for lb in labels) + 2)
        run_w = max(8, max(len(rn) for rn in run_names) + 2)

        header = f"{'label':<{label_w}}"
        for rn in run_names:
            for cn, _ in cols:
                header += f"  {(rn[: run_w - 2] + '/' + cn):<{col_w}}"
        print(header)
        print("-" * len(header))

        for lb in labels:
            row = f"{lb:<{label_w}}"
            for rn in run_names:
                res = lookup[rn].get(lb)
                for _, attr in cols:
                    if res is None:
                        row += f"  {'—':<{col_w}}"
                    elif attr in ("fall_rate", "fall_rate_height"):
                        v = getattr(res, attr)
                        row += f"  {v * 100:>{col_w - 2}.1f}%  "
                    else:
                        row += f"  {_fmt(getattr(res, attr)):<{col_w}}"
            print(row)

    print(f"\n{sep}")


def save_markdown_report(
    all_results: Dict[str, Dict[str, List[ScenarioResult]]],
    output_path: str,
    metadata: Optional[Dict[str, object]] = None,
):
    run_names = list(all_results.keys())
    lines = ["# RoboDuet Policy Benchmark", ""]
    if metadata:
        lines.extend(["## Run Configuration", ""])
        for key, value in metadata.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")

    for scenario, (title, cols) in SCENARIO_META.items():
        labels = _scenario_labels(all_results, scenario)
        lines.extend([f"## {title}", ""])
        if not labels:
            lines.extend(["No results. This scenario was skipped or not applicable.", ""])
            continue

        header = ["label"]
        for run_name in run_names:
            for col_name, _ in cols:
                header.append(f"{run_name}/{col_name}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        lookup = {run_name: {r.label: r for r in all_results[run_name].get(scenario, [])} for run_name in run_names}
        for label in labels:
            row = [label]
            for run_name in run_names:
                result = lookup[run_name].get(label)
                for _, attr in cols:
                    row.append(_format_metric_value(result, attr))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Benchmark] Markdown report saved → {output_path}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_results(all_results: Dict[str, Dict[str, List[ScenarioResult]]], output_path: str):
    data = {rn: {sc: [asdict(r) for r in rs] for sc, rs in scenarios.items()} for rn, scenarios in all_results.items()}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Benchmark] Results saved → {output_path}")


def save_metadata(metadata: Dict[str, object], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[Benchmark] Metadata saved → {output_path}")
