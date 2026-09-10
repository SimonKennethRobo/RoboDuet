"""Export a RoboDuet stage-1 dog policy into an rl_sar-compatible bundle.

By default, writes ``config.yaml`` and ``policy.pt`` under
``<logdir>/rl_sar/<config_name>/``, without a robot-level ``base.yaml``.

With an explicit ``--rl_sar_root``, produces under
``<rl_sar_root>/policy/<robot>/``::

    base.yaml                       robot-level, hardware joint order
    <config_name>/config.yaml       policy-level, training (policy) joint order
    <config_name>/policy.pt         TorchScript actor, forward([1, H]) -> [1, A]

Every value is derived from the checkpoint's own ``parameters.pkl`` snapshot,
never from this file's constants -- the training config changes shape depending
on the flags a run was launched with (``arm_num_commands`` is 6 or 9 depending
on ``--rot6d``, ``dog_num_commands`` is 6 or 11 depending on ``--dyna_gait``),
so hardcoding would silently drift.

Deliberately does not import IsaacGym: ``go1_gym.envs.config`` is pure Python and
``dog_ac.py`` is loaded standalone, so this runs anywhere PyTorch does.

Usage::

    python scripts/export_rl_sar.py --logdir runs/<date>/<run>

writes into ``<logdir>/rl_sar/``; pass ``--rl_sar_root`` to target an rl_sar
checkout directly instead.
"""

import argparse
import importlib.util
import pickle as pkl
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_gym.envs.config import (  # noqa: E402
    RoboDuetRuntimeOptions,
    apply_config_snapshot,
    build_roboduet_config,
    recompute_observation_dims,
)


# Bundles are written inside the run they came from, under <logdir>/RL_SAR_DIR.
# Keeping them with the checkpoint means a run directory stays self-contained:
# the exported artifact can never outlive, or drift from, the weights it was
# built from, and nothing is ever written outside runs/ unless the caller names
# an explicit --rl_sar_root.
RL_SAR_DIR = "rl_sar"


def default_rl_sar_root(logdir):
    """Where to write the bundle for `logdir`: inside the run itself.

    Contains <config_name>/config.yaml and policy.pt. Copy <config_name>/
    into an rl_sar checkout under policy/<robot>/ to deploy.
    """
    return str(Path(logdir) / RL_SAR_DIR)


def _load_dog_ac_module():
    """Import dog_ac.py without going through go1_gym_learn's package __init__,
    which transitively imports IsaacGym."""
    path = REPO_ROOT / "go1_gym_learn" / "ppo_cse_automatic" / "dog_ac.py"
    spec = importlib.util.spec_from_file_location("roboduet_dog_ac_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Joint order
# ---------------------------------------------------------------------------
# Training (IsaacGym DOF) order is URDF tree order: FL, FR, RL, RR legs, then the
# arm joints. Hardware order is the Unitree SDK motor order FR, FL, RR, RL, then
# the arm. joint_mapping[training_index] = hardware_index, which is exactly the
# indirection rl_sar applies in GetState()/SetCommand().
LEG_JOINTS_TRAINING_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
LEG_JOINTS_HARDWARE_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINTS = {
    "go2_x5": ["x5_joint1", "x5_joint2", "x5_joint3", "x5_joint4", "x5_joint5", "x5_joint6"],
    "go1": ["widow_waist", "widow_shoulder", "widow_elbow", "widow_forearm_roll",
            "widow_wrist_angle", "widow_wrist_rotate"],
}

# URDF effort limits, in training joint order. Used as rl_sar torque_limits.
LEG_TORQUE_LIMITS = [23.7, 23.7, 45.43] * 4
ARM_TORQUE_LIMITS = {"go2_x5": [27.0, 27.0, 27.0, 7.0, 7.0, 7.0]}


def build_joint_mapping(arm_joints):
    """training index -> hardware index."""
    hardware_index = {name: i for i, name in enumerate(LEG_JOINTS_HARDWARE_ORDER)}
    mapping = [hardware_index[name] for name in LEG_JOINTS_TRAINING_ORDER]
    mapping += [len(LEG_JOINTS_HARDWARE_ORDER) + i for i in range(len(arm_joints))]
    return mapping


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------
def load_runtime_cfg(logdir, robot):
    """Rebuild the exact cfg the checkpoint was trained with."""
    with open(Path(logdir) / "parameters.pkl", "rb") as handle:
        run_parameters = pkl.load(handle)
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1, robot=robot))
    apply_config_snapshot(cfg, run_parameters["Cfg"], drop_unknown=True)
    recompute_observation_dims(cfg)
    return cfg, run_parameters


def joint_order_lists(cfg, robot):
    arm_joints = ARM_JOINTS[robot]
    training_joints = LEG_JOINTS_TRAINING_ORDER + arm_joints
    hardware_joints = LEG_JOINTS_HARDWARE_ORDER + arm_joints
    num_leg = int(cfg.dog.num_actions_loco)
    num_arm = int(cfg.arm.num_actions_arm)
    if len(training_joints) != num_leg + num_arm:
        raise ValueError(
            f"joint table has {len(training_joints)} entries but cfg wants "
            f"{num_leg} leg + {num_arm} arm DoFs"
        )
    return training_joints, hardware_joints, arm_joints


def default_dof_pos(cfg, training_joints):
    angles = cfg.init_state.default_joint_angles
    missing = [name for name in training_joints if name not in angles]
    if missing:
        raise KeyError(f"init_state.default_joint_angles is missing: {missing}")
    return [float(angles[name]) for name in training_joints]


def pd_gains(cfg, training_joints, num_leg):
    """rl_kp / rl_kd in training order.

    Legs run torque control with dog.control.stiffness_leg/damping_leg; the arm
    runs position control with per-joint gains from arm.control.stiffness_arm.
    """
    leg_kp = float(list(cfg.dog.control.stiffness_leg.values())[0])
    leg_kd = float(list(cfg.dog.control.damping_leg.values())[0])
    kp = [leg_kp] * num_leg
    kd = [leg_kd] * num_leg
    for name in training_joints[num_leg:]:
        if name not in cfg.arm.control.stiffness_arm:
            raise KeyError(f"arm.control.stiffness_arm has no entry for '{name}'")
        kp.append(float(cfg.arm.control.stiffness_arm[name]))
        kd.append(float(cfg.arm.control.damping_arm[name]))
    return kp, kd


def action_scale(cfg, num_leg, num_arm):
    """Per-joint action scale, mirroring LeggedRobot._compute_torques().

    Hip joints (training indices 0, 3, 6, 9) get an extra hip_scale_reduction.
    The arm entries are 0.0: the dog policy emits no arm actions, so rl_sar's
    zero-padded action must leave the arm at default_dof_pos.
    """
    scale = float(cfg.control.action_scale)
    hip_scale = scale * float(cfg.control.hip_scale_reduction)
    per_joint = []
    for i in range(num_leg):
        per_joint.append(hip_scale if i % 3 == 0 else scale)
    return per_joint + [0.0] * num_arm


def dog_command_layout(cfg):
    """(extra command constants, full command scale vector).

    The first six dog commands are operator-driven in rl_sar
    (x/y/yaw/body_pitch/body_roll/body_height). With dynamic gait the policy
    also observes five gait commands; those are not exposed as operator inputs,
    so they are frozen at the midpoint of the range they were trained over.
    """
    obs_scales = cfg.obs_scales
    scale = [
        float(obs_scales.lin_vel), float(obs_scales.lin_vel), float(obs_scales.ang_vel),
        float(obs_scales.body_pitch_cmd), float(obs_scales.body_roll_cmd),
        float(obs_scales.body_height_cmd),
        float(obs_scales.gait_freq_cmd), float(obs_scales.footswing_height_cmd),
        float(obs_scales.stance_width_cmd), float(obs_scales.stance_length_cmd),
        float(obs_scales.gait_duration_cmd),
    ][: int(cfg.dog.dog_num_commands)]

    extra = []
    if int(cfg.dog.dog_num_commands) > 6:
        def midpoint(name):
            lo, hi = getattr(cfg.commands, name)
            return float(lo + hi) / 2.0
        extra = [
            midpoint("limit_gait_frequency"),
            0.06,  # footswing_height: the constant WBCEnv.plan() sends
            midpoint("limit_stance_width"),
            midpoint("limit_stance_length"),
            0.49,  # gait_duration: the constant WBCEnv.plan() sends
        ][: int(cfg.dog.dog_num_commands) - 6]
    return extra, scale


def gait_clock_params(cfg, dog_commands_extra):
    """Frequency / duration used by LeggedRobot._step_contact_targets().

    With use_dynamic_gait=False they are the hardcoded 3.0 / 0.5; otherwise they
    come from the (frozen) gait command slots.
    """
    if bool(cfg.commands.use_dynamic_gait) and len(dog_commands_extra) == 5:
        return dog_commands_extra[0], dog_commands_extra[4]
    return 3.0, 0.5


def observation_terms(cfg):
    """Term list mirroring WBCEnv.get_dog_observations(), in concat order."""
    terms = [
        "gravity_vec",                # projected_gravity            3
        "roboduet/leg_dof_pos",       # dog dof pos (relative)      12
        "roboduet/leg_dof_vel",       # dog dof vel                 12
        "roboduet/leg_actions",       # previous dog actions        12
        "roboduet/dog_commands",      # commands_dog * scale      6/11
        "roboduet/arm_commands",      # zero in stage 1            6/9
    ]
    if bool(cfg.env.observe_two_prev_actions):
        raise NotImplementedError("env.observe_two_prev_actions has no rl_sar term")
    if bool(cfg.env.observe_timing_parameter):
        raise NotImplementedError("env.observe_timing_parameter has no rl_sar term")
    if bool(cfg.env.observe_clock_inputs):
        terms.append("roboduet/clock_inputs")                     # 4
    terms += [
        "ang_vel",                    # base_ang_vel * scale         3
        "roboduet/base_lin_vel",      # base_lin_vel * scale         3
        "roboduet/body_pose_actual",  # [height, pitch, roll]        3
        "roboduet/body_pose_error",   # target - actual              3
        "roboduet/velocity_error",    # command - actual             3
    ]
    if bool(cfg.env.observe_yaw):
        raise NotImplementedError("env.observe_yaw has no rl_sar term")
    if bool(cfg.env.observe_contact_states):
        raise NotImplementedError("env.observe_contact_states has no rl_sar term")
    terms += ["roboduet/arm_dof_pos", "roboduet/arm_dof_vel"]      # 6, 6
    # R7.1.  The first three are the same second-order integrator the MPC
    # already runs as its nominal dynamics, so rl_sar implements it once and
    # reads three terms off it -- and the agreement between the on-robot xi and
    # the MPC's own xi becomes a free diagnostic: if they diverge, the command
    # path, the clock or the parameters are wrong, and it shows up before the
    # behaviour does.
    terms += [
        "roboduet/reference_state",       # xi                      5
        "roboduet/reference_rate",        # xi_dot / rate_limit     5
        "roboduet/reference_minus_cmd",   # xi - u                  5
        "roboduet/ee_pos_in_base",        # arm FK                  3
        "roboduet/response_deviation",    # (g-1, l) x IMU channels 4
    ]
    return terms


def expected_obs_width(cfg, terms):
    num_leg = int(cfg.dog.num_actions_loco)
    num_arm = int(cfg.arm.num_actions_arm)
    num_channels = len(cfg.response.channel_order)
    widths = {
        "gravity_vec": 3,
        "ang_vel": 3,
        "roboduet/leg_dof_pos": num_leg,
        "roboduet/leg_dof_vel": num_leg,
        "roboduet/leg_actions": num_leg,
        "roboduet/dog_commands": int(cfg.dog.dog_num_commands),
        "roboduet/arm_commands": int(cfg.arm.arm_num_commands),
        "roboduet/clock_inputs": 4,
        "roboduet/base_lin_vel": 3,
        "roboduet/body_pose_actual": 3,
        "roboduet/body_pose_error": 3,
        "roboduet/velocity_error": 3,
        "roboduet/arm_dof_pos": num_arm,
        "roboduet/arm_dof_vel": num_arm,
        "roboduet/reference_state": num_channels,
        "roboduet/reference_rate": num_channels,
        "roboduet/reference_minus_cmd": num_channels,
        "roboduet/ee_pos_in_base": 3,
        "roboduet/response_deviation": 2 * len(cfg.response.deviation.channels),
    }
    return sum(widths[name] for name in terms)


# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------
def export_policy(logdir, ckpt_id, cfg, out_path):
    dog_ac = _load_dog_ac_module()

    suffix = "last_dog" if ckpt_id == "last" else str(ckpt_id).zfill(6)
    ckpt_path = Path(logdir) / "checkpoints_dog" / f"ac_weights_{suffix}.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    uses_adaptation = any(key.startswith("adaptation_module.") for key in checkpoint)
    if uses_adaptation:
        raise NotImplementedError(
            "This checkpoint has an adaptation module, so the actor takes two "
            "inputs (obs_history, latent). rl_sar's InferenceRuntime only feeds "
            "a single tensor. Retrain with dog.use_adaptation_module=False, or "
            "extend this script to script a wrapper Module that runs the "
            "adaptation branch internally."
        )

    # Layer widths come from the checkpoint tensors, not from DogAC_Args, so an
    # old checkpoint with different hidden dims still exports correctly.
    actor_shapes = []
    for key, value in checkpoint.items():
        if key.startswith("actor_body.") and key.endswith(".weight") and value.ndim == 2:
            actor_shapes.append((int(key.split(".")[1]), tuple(value.shape)))
    actor_shapes = [shape for _, shape in sorted(actor_shapes)]
    if len(actor_shapes) < 2:
        raise ValueError(f"{ckpt_path} has no usable actor_body layers")
    dog_ac.DogAC_Args.actor_hidden_dims = [shape[0] for shape in actor_shapes[:-1]]

    num_obs_history = int(cfg.dog.dog_num_obs_history)
    if actor_shapes[0][1] != num_obs_history:
        raise ValueError(
            f"actor input is {actor_shapes[0][1]} but cfg.dog.dog_num_obs_history "
            f"is {num_obs_history}; the checkpoint and parameters.pkl disagree"
        )

    actor_critic = dog_ac.DogActorCritic(
        int(cfg.dog.dog_num_observations),
        int(cfg.dog.dog_num_privileged_obs),
        num_obs_history,
        int(cfg.dog.dog_actions),
        use_adaptation_module=False,
    )
    actor_body = actor_critic.actor_body
    actor_body.load_state_dict(
        {key[len("actor_body."):]: value
         for key, value in checkpoint.items() if key.startswith("actor_body.")}
    )
    actor_body.eval()

    scripted = torch.jit.script(actor_body)
    scripted.save(str(out_path))

    # Round-trip the saved artifact: catches a silently broken script() far more
    # reliably than checking the in-memory module.
    reloaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        probe = torch.randn(1, num_obs_history)
        if not torch.allclose(reloaded(probe), actor_body(probe), atol=1e-6):
            raise RuntimeError("TorchScript output diverges from the eager module")
    return ckpt_path, int(cfg.dog.dog_actions)


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------
def _wrap(key, rendered, per_line):
    """Render `  key: [...]`, wrapping every per_line entries and aligning the
    continuation lines under the opening bracket."""
    if not rendered:
        return f"  {key}: []"
    if len(rendered) <= per_line:
        return f"  {key}: [" + ", ".join(rendered) + "]"
    pad = " " * (len(f"  {key}: ") + 1)
    chunks = [", ".join(rendered[start:start + per_line])
              for start in range(0, len(rendered), per_line)]
    return f"  {key}: [" + (",\n" + pad).join(chunks) + "]"


def float_entry(key, values, per_line=3):
    return _wrap(key, [f"{float(value):.6g}" for value in values], per_line)


def string_entry(key, values, per_line=3):
    return _wrap(key, [f'"{value}"' for value in values], per_line)


HEADER = (
    "# Generated by RoboDuet scripts/export_rl_sar.py -- do not edit by hand.\n"
    "# Source run: {logdir}\n"
    "# Source checkpoint: {ckpt}\n"
)


def write_base_yaml(path, robot, cfg, ctx):
    controllers = [name.replace("_joint", "_controller") for name in ctx["hardware_joints"]]
    body = f"""{robot}:
  dt: {float(cfg.sim.dt):g}
  decimation: {int(cfg.control.decimation)}
  num_of_dofs: {ctx['num_dofs']}
  num_leg_dofs: {ctx['num_leg']}
  num_arm_dofs: {ctx['num_arm']}
  wheel_indices: []
  # Stiff gains used by the get-up / get-down interpolation and to hold the arm
  # while the legs are passive. Independent of the policy's rl_kp / rl_kd.
{float_entry('fixed_kp', ctx['fixed_kp'])}
{float_entry('fixed_kd', ctx['fixed_kd'])}
{float_entry('torque_limits', ctx['torque_limits'])}
  # Policy joint order (FL, FR, RL, RR, then the arm).
{float_entry('default_dof_pos', ctx['default_dof_pos'])}
  # Hardware joint order (Unitree SDK: FR, FL, RR, RL, then the arm).
{string_entry('joint_names', ctx['hardware_joints'])}
{string_entry('joint_controller_names', controllers)}
  # joint_mapping[policy_index] = hardware_index
  joint_mapping: {ctx['joint_mapping']}
  # MuJoCo only: read framelinvel + framepos appended after the gyro, standing
  # in for the FAST-LIO state estimate used on hardware.
  use_base_state_sensor: true
"""
    path.write_text(HEADER.format(**ctx['provenance']) + "\n" + body)


def write_config_yaml(path, robot, config_name, cfg, ctx):
    history = list(range(int(cfg.dog.dog_num_observation_history) - 1, -1, -1))
    clip_actions = float(cfg.normalization.clip_actions)
    terms = "\n".join(f'    - "{name}"' for name in ctx['observations'])
    body = f"""{robot}/{config_name}:
  model_name: "policy.pt"
  num_observations: {ctx['num_observations']}
  # Mirrors WBCEnv.get_dog_observations() term for term, in concat order.
  observations:
{terms}
  # HistoryWrapper keeps the oldest frame first and the newest last, so the
  # index list counts down to 0 (0 is the latest observation).
  observations_history: {history}
  observations_history_priority: "time"
  clip_obs: {float(cfg.normalization.clip_observations):g}
{float_entry('clip_actions_lower', [-clip_actions] * ctx['num_actions'])}
{float_entry('clip_actions_upper', [clip_actions] * ctx['num_actions'])}

  num_of_dofs: {ctx['num_dofs']}
  # The policy drives only the {ctx['num_leg']} leg joints. rl_sar zero-pads the
  # action to num_of_dofs; a 0.0 action_scale keeps each arm joint parked at
  # default_dof_pos under its own rl_kp / rl_kd.
{float_entry('action_scale', ctx['action_scale'])}
{float_entry('rl_kp', ctx['rl_kp'])}
{float_entry('rl_kd', ctx['rl_kd'])}
  wheel_indices: []

  lin_vel_scale: {float(cfg.obs_scales.lin_vel):g}
  ang_vel_scale: {float(cfg.obs_scales.ang_vel):g}
  dof_pos_scale: {float(cfg.obs_scales.dof_pos):g}
  dof_vel_scale: {float(cfg.obs_scales.dof_vel):g}
  body_height_cmd_scale: {float(cfg.obs_scales.body_height_cmd):g}
  body_pitch_cmd_scale: {float(cfg.obs_scales.body_pitch_cmd):g}
  body_roll_cmd_scale: {float(cfg.obs_scales.body_roll_cmd):g}

  # ---- RoboDuet-specific ----
  num_leg_dofs: {ctx['num_leg']}
  num_arm_dofs: {ctx['num_arm']}
  arm_num_commands: {int(cfg.arm.arm_num_commands)}
{float_entry('dog_commands_scale', ctx['dog_commands_scale'], 6)}
  # Gait command slots the operator does not drive, frozen at their trained
  # midpoints. Empty unless the run used dynamic gait.
{float_entry('dog_commands_extra', ctx['dog_commands_extra'], 5)}
  gait_frequency: {ctx['gait_frequency']:g}
  gait_duration: {ctx['gait_duration']:g}
  # trotting: [phases, offsets, bounds]
  gait_phases: [0.5, 0.0, 0.0]
  base_height_target: {float(cfg.rewards.base_height_target):g}
  # These mirror the dog.observe_* switches. When false the slot stays at its
  # trained width but is zero-filled, exactly as in WBCEnv.
  observe_lin_vel: {str(bool(cfg.dog.observe_lin_vel)).lower()}
  observe_pose_actual: {str(bool(cfg.dog.observe_pose_actual)).lower()}
  observe_track_error: {str(bool(cfg.dog.observe_track_error)).lower()}

  # Operator command clamps, taken from the ranges the policy trained against.
{float_entry('limit_vel_x', list(cfg.commands.limit_vel_x))}
{float_entry('limit_vel_y', list(cfg.commands.limit_vel_y))}
{float_entry('limit_vel_yaw', list(cfg.commands.limit_vel_yaw))}
{float_entry('limit_body_pitch', list(cfg.commands.limit_body_pitch))}
{float_entry('limit_body_roll', list(cfg.commands.limit_body_roll))}
{float_entry('limit_body_height', list(cfg.commands.limit_body_height))}

{float_entry('default_dof_pos', ctx['default_dof_pos'])}
  joint_mapping: {ctx['joint_mapping']}
"""
    response = response_observation_config(cfg, robot)
    body += "\n" + "\n".join(
        "  " + line for line in yaml.safe_dump(
            {"response_observation": response}, sort_keys=False
        ).splitlines()
    ) + "\n"
    path.write_text(HEADER.format(**ctx['provenance']) + "\n" + body)


def response_observation_config(cfg, robot):
    """Export the reference/estimator constants and base-to-grasp FK chain.

    Joint origins come from the training URDF and joint angles are indexed in
    policy order. No simulator installation or runtime URDF parser is needed
    by the deployment. The grasp offset is applied exactly once after ee_body.
    """
    from go1_gym.response.reference import build_channels

    channels = build_channels(cfg.response.channel_order, cfg.response.omega_n,
                              cfg.response.rate_limit)
    _, command_scales = dog_command_layout(cfg)
    training_joints, _, _ = joint_order_lists(cfg, robot)
    asset_path = Path(str(cfg.asset.file).replace("{MINI_GYM_ROOT_DIR}", str(REPO_ROOT)))
    tree = ET.parse(asset_path).getroot()
    parent_joint = {j.find("child").get("link"): j for j in tree.findall("joint")}
    link = cfg.asset.ee_body_name
    chain = []
    while link in parent_joint:
        joint = parent_joint[link]
        kind = joint.get("type")
        if kind not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"Unsupported EE chain joint type: {kind}")
        origin = joint.find("origin")
        def vector(element, key, default):
            return [float(v) for v in (element.get(key, default) if element is not None else default).split()]
        chain.append({
            "name": joint.get("name"),
            "xyz": vector(origin, "xyz", "0 0 0"),
            "rpy": vector(origin, "rpy", "0 0 0"),
            "axis": vector(joint.find("axis"), "xyz", "1 0 0"),
            "dof_index": -1 if kind == "fixed" else training_joints.index(joint.get("name")),
        })
        link = joint.find("parent").get("link")
    if not chain:
        raise ValueError(f"No kinematic chain found for EE body {cfg.asset.ee_body_name}")
    deviation = cfg.response.deviation
    return {
        "version": 1,
        "channels": [c.name for c in channels],
        "omega_n": [c.omega_n for c in channels],
        "rate_limit": [c.rate_limit for c in channels],
        "obs_scale": [float(command_scales[c.cmd_index]) for c in channels],
        "command_amplitude": [float(cfg.response.reward.calibration_amplitudes[c.name]) for c in channels],
        "deviation": {
            "channels": list(deviation.channels),
            "tau_s": float(deviation.tau_s),
            "warmup_s": float(deviation.warmup_s),
            "rate_deadband": float(deviation.rate_deadband),
            "excitation_fraction": float(deviation.excitation_fraction),
        },
        "ee_base_link": link,
        "ee_body_name": cfg.asset.ee_body_name,
        "ee_chain": list(reversed(chain)),
        "ee_local_pos": [float(v) for v in cfg.arm.ik.ee_local_pos],
    }


# ---------------------------------------------------------------------------
def robot_from_logdir(logdir, fallback="go2_x5"):
    """Infer the robot key from the checkpoint's own recorded asset path.

    The caller usually knows which robot it is playing, but not always in the
    key this script uses (play_by_key_stage1's --robot has its own, shorter
    choice list, and its default is not necessarily what the run was trained
    with). Getting it wrong would export the wrong arm joint names and torque
    limits, so read it from the snapshot instead of trusting a flag.
    """
    from go1_gym.envs.config.wbc import ROBOT_ASSET_FILES

    try:
        with open(Path(logdir) / "parameters.pkl", "rb") as handle:
            asset_file = pkl.load(handle)["Cfg"]["asset"]["file"]
    except (OSError, KeyError, pkl.UnpicklingError):
        return fallback
    for robot, template in ROBOT_ASSET_FILES.items():
        if Path(template).name == Path(str(asset_file)).name and robot in ARM_JOINTS:
            return robot
    # mount-randomization buckets rewrite the URDF filename, so also match on
    # the directory the asset lives in
    parent = Path(str(asset_file)).parent.name
    for robot, template in ROBOT_ASSET_FILES.items():
        if Path(template).parent.name == parent and robot in ARM_JOINTS:
            return robot
    return fallback


def export(logdir, rl_sar_root=None, ckpt_id="last", robot=None,
           config_name="roboduet_stage1", quiet=False):
    """Write the rl_sar bundle for one run. Returns the output directory.

    ``rl_sar_root`` defaults to ``<logdir>/rl_sar`` -- the bundle lives with the
    run that produced it, under ``<logdir>/rl_sar/<config_name>/`` without
    ``base.yaml``. An explicit root uses the rl_sar checkout layout:
    ``<rl_sar_root>/policy/<robot>/<config_name>/`` plus robot ``base.yaml``.

    Importable so callers other than this script's CLI can export -- notably
    scripts/play_by_key_stage1.py, which exports the same policy it is about
    to play so the deployed bundle can never silently lag what was inspected.
    """
    flat = rl_sar_root is None
    rl_sar_root = rl_sar_root or default_rl_sar_root(logdir)
    robot = robot or robot_from_logdir(logdir)
    log = (lambda *a: None) if quiet else print

    cfg, _ = load_runtime_cfg(logdir, robot=robot)

    training_joints, hardware_joints, arm_joints = joint_order_lists(cfg, robot)
    num_leg = int(cfg.dog.num_actions_loco)
    num_arm = int(cfg.arm.num_actions_arm)
    num_dofs = num_leg + num_arm

    rl_kp, rl_kd = pd_gains(cfg, training_joints, num_leg)
    dog_commands_extra, dog_commands_scale = dog_command_layout(cfg)
    gait_frequency, gait_duration = gait_clock_params(cfg, dog_commands_extra)
    observations = observation_terms(cfg)

    width = expected_obs_width(cfg, observations)
    if width != int(cfg.dog.dog_num_observations):
        raise ValueError(
            f"observation terms sum to {width} but the policy expects "
            f"{int(cfg.dog.dog_num_observations)}; get_dog_observations() and "
            f"observation_terms() have diverged"
        )

    out_dir = (Path(rl_sar_root) / config_name if flat else
               Path(rl_sar_root) / "policy" / robot / config_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, num_actions = export_policy(logdir, ckpt_id, cfg, out_dir / "policy.pt")

    ctx = {
        "num_dofs": num_dofs,
        "num_leg": num_leg,
        "num_arm": num_arm,
        "num_actions": num_actions,
        "num_observations": width,
        "training_joints": training_joints,
        "hardware_joints": hardware_joints,
        "joint_mapping": build_joint_mapping(arm_joints),
        "default_dof_pos": default_dof_pos(cfg, training_joints),
        "rl_kp": rl_kp,
        "rl_kd": rl_kd,
        # Get-up / get-down gains. These are a deployment choice, not a training
        # parameter: the interpolation states hold a static pose against gravity
        # rather than tracking a policy. 80/3 is what rl_sar ships for the stock
        # Go2 and leaves ~0.09 rad of calf droop at the standing pose. The arm
        # keeps its running gains -- it is holding the same pose either way.
        "fixed_kp": [80.0] * num_leg + rl_kp[num_leg:],
        "fixed_kd": [3.0] * num_leg + rl_kd[num_leg:],
        "torque_limits": LEG_TORQUE_LIMITS + ARM_TORQUE_LIMITS[robot],
        "action_scale": action_scale(cfg, num_leg, num_arm),
        "dog_commands_scale": dog_commands_scale,
        "dog_commands_extra": dog_commands_extra,
        "gait_frequency": gait_frequency,
        "gait_duration": gait_duration,
        "observations": observations,
        "provenance": {"logdir": str(Path(logdir).resolve()), "ckpt": str(ckpt_path)},
    }

    if not flat:
        base_yaml = Path(rl_sar_root) / "policy" / robot / "base.yaml"
        write_base_yaml(base_yaml, robot, cfg, ctx)
    write_config_yaml(out_dir / "config.yaml", robot, config_name, cfg, ctx)

    log(f"[export_rl_sar] robot        : {robot}  (ckpt {ckpt_path.name})")
    log(f"[export_rl_sar] observations : {width}  ({len(observations)} terms)")
    log(f"[export_rl_sar] obs history  : {int(cfg.dog.dog_num_observation_history)}"
        f" x {width} = {int(cfg.dog.dog_num_obs_history)}")
    log(f"[export_rl_sar] actions      : {num_actions} over {num_dofs} DoFs")
    if not flat:
        log(f"[export_rl_sar] wrote {base_yaml}")
    log(f"[export_rl_sar] wrote {out_dir / 'config.yaml'}")
    log(f"[export_rl_sar] wrote {out_dir / 'policy.pt'}")
    if bool(cfg.dog.observe_lin_vel) or bool(cfg.dog.observe_pose_actual):
        log("[export_rl_sar] NOTE: this policy observes base linear velocity "
            "and/or base height. rl_sar must be fed a state estimate "
            "(RobotState::base.lin_vel in BODY frame, base.position in WORLD frame).")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=str, required=True, help="RoboDuet run directory")
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--rl_sar_root", type=str, default=None,
                        help="Default: <logdir>/rl_sar/<config_name>/ (config.yaml + policy.pt). "
                             "An explicit root uses policy/<robot>/<config_name>/ and base.yaml.")
    parser.add_argument("--robot", type=str, default=None, choices=sorted(ARM_JOINTS),
                        help="default: inferred from the checkpoint's recorded asset")
    parser.add_argument("--config_name", type=str, default="roboduet_stage1")
    args = parser.parse_args()

    export(args.logdir, args.rl_sar_root, ckpt_id=args.ckptid, robot=args.robot,
           config_name=args.config_name)


if __name__ == "__main__":
    main()
