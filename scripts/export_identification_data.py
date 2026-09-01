"""R9.2 -- export the time series Contribution 2 identifies from.

Records, per control step and per environment: the command, the reference
state, the measured base pose and twist, the gait phase and frequency, the arm
state, the privileged domain parameters (simulation only), and the terrain
label.  Written as ``.npz`` with a sidecar JSON that states the sampling rate,
the frame conventions and the channel index tables -- R9.2 lets the format be
chosen but requires the metadata, and metadata is the part that decides whether
the file is still usable in six months.

    python scripts/export_identification_data.py --policy <ckpt> \
        --seconds 120 --out data/identification_go2_x5.npz

Two things about *what* gets recorded that are decisions, not defaults.

**Both the raw and the detrended measurement are stored.**  Detrending is a
modelling choice -- it removes the gait-phase-conditioned oscillation R3
estimates -- and a data file that has already made that choice cannot be used to
question it.  The phase is stored alongside so the detrending can be redone or
undone.

**The excitation plan is stored per step.**  R6 gives each identification
environment one signal on one channel, re-drawn at reset, and a record spanning
two plans is not one experiment.  Segmenting by ``plan_generation`` is the only
reliable way to cut it -- (signal, channel) collides about one time in fifteen.
"""

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401  must precede torch
import numpy as np
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.response import DECISION_CHANNEL_NAMES, DECISION_CHANNEL_UNITS, DOG_COMMAND_NAMES
from go1_gym.response.excitation import SIGNAL_NAMES
from go1_gym.utils import global_switch


def build_env(num_envs, sim_device, robot):
    args = argparse.Namespace(
        robot=robot, num_envs=num_envs, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0.0, video=False,
    )
    cfg = build_roboduet_config(args)
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    global_switch.pretrained_to_wbc_start = 10 ** 9
    global_switch.pretrained_to_wbc_end = 10 ** 9 + 1
    global_switch.init_sigmoid_lr()
    return HistoryWrapper(WBCEnv(sim_device=sim_device, headless=True, cfg=cfg)), cfg


def load_policy(path, cfg, device):
    from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic

    model = DogActorCritic(
        num_obs=cfg.dog.dog_num_observations,
        num_privileged_obs=cfg.dog.dog_num_privileged_obs,
        num_obs_history=cfg.dog.dog_num_obs_history,
        num_actions=cfg.dog.dog_actions,
        use_adaptation_module=cfg.dog.use_adaptation_module,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def metadata(cfg, base, args, steps):
    """The sidecar.  Without it the arrays are unusable in six months."""
    return {
        "schema": "roboduet-identification-v1",
        "robot": args.robot,
        "policy": args.policy,
        "sample_rate_hz": round(1.0 / base.dt, 6),
        "dt_s": base.dt,
        "steps": steps,
        "num_envs": int(base.num_envs),
        "terrain": cfg.terrain.mesh_type,
        "frames": {
            "base_position": "world",
            "base_quaternion": "world, xyzw (IsaacGym convention)",
            "base_linear_velocity": "BODY frame",
            "base_angular_velocity": "BODY frame",
            "ee_position": "BODY frame, relative to base origin",
            "height": (
                "base z minus the local terrain height, i.e. RELATIVE to terrain "
                "-- absolute world height is not comparable across environments"
            ),
        },
        "channels": {
            "decision_order": list(DECISION_CHANNEL_NAMES),
            "units": dict(DECISION_CHANNEL_UNITS),
            "command_columns": list(DOG_COMMAND_NAMES[: cfg.dog.dog_num_commands]),
            "note": (
                "decision channel order is (vx, vy, wyaw, height, pitch); the "
                "command vector puts pitch at column 3 and height at column 5. "
                "These are different orderings and conflating them is the "
                "easiest silent bug in this dataset."
            ),
        },
        "reference_model": {
            "omega_n": dict(cfg.response.omega_n),
            "rate_limit": dict(cfg.response.rate_limit),
            "form": "critically damped second order with a rate limit, zeta = 1",
        },
        "excitation": {
            "signals": list(SIGNAL_NAMES),
            "note": (
                "segment records by plan_generation, not by (signal, channel): "
                "a redraw lands on the same pair about one time in fifteen"
            ),
        },
        "fields": {
            "command": "(T, E, C) decision-channel commands, physical units",
            "reference_state": "(T, E, C) xi",
            "reference_rate": "(T, E, C) xi_dot",
            "measured": "(T, E, C) raw response, terrain-relative height",
            "measured_detrended": "(T, E, C) with the phase residual removed",
            "gait_phase": "(T, E) in [0, 1)",
            "gait_frequency_hz": "(T, E)",
            "base_position": "(T, E, 3)",
            "base_quaternion": "(T, E, 4)",
            "base_linear_velocity": "(T, E, 3)",
            "base_angular_velocity": "(T, E, 3)",
            "arm_dof_pos": "(T, E, A)",
            "arm_dof_vel": "(T, E, A)",
            "ee_pos_in_base": "(T, E, 3)",
            "excitation_signal": "(T, E) index into excitation.signals, -1 if none",
            "excitation_channel": "(T, E) decision channel index, -1 if none",
            "plan_generation": "(T, E) increments on every excitation re-plan",
            "is_identification": "(E,) environments running designed excitation",
            "is_nominal_twin": "(E,) environments held at nominal domain",
            "group_of": "(E,) R5 group index, -1 if ungrouped",
            "group_valid": "(T, E) 1 when the group is phase-synchronised",
            "reset": "(T, E) 1 on the step the episode ended",
            "domain_*": "(E,) privileged domain parameters, SIMULATION ONLY",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--out", default="data/identification.npz")
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        raise SystemExit(f"checkpoint not found: {args.policy}")

    env, cfg = build_env(args.num_envs, args.sim_device, args.robot)
    base = env.env
    policy = load_policy(args.policy, cfg, base.device)
    steps = int(args.seconds / base.dt)
    arm_actions = torch.zeros(base.num_envs, base.num_actions_arm, device=base.device)
    sampler = base.response_excitation
    arm_slice = slice(base.num_actions_loco, base.num_actions_loco + base.num_actions_arm)

    series = {name: [] for name in (
        "command", "reference_state", "reference_rate", "measured",
        "measured_detrended", "gait_phase", "gait_frequency_hz",
        "base_position", "base_quaternion", "base_linear_velocity",
        "base_angular_velocity", "arm_dof_pos", "arm_dof_vel", "ee_pos_in_base",
        "excitation_signal", "excitation_channel", "plan_generation",
        "group_valid", "reset",
    )}

    def record():
        reference = base.response_ref
        series["command"].append(reference.gather_commands(base.commands_dog).cpu())
        series["reference_state"].append(reference.xi.cpu())
        series["reference_rate"].append(reference.xi_dot.cpu())
        series["measured"].append(base.response_measured.cpu())
        series["measured_detrended"].append(base.response_detrended.cpu())
        series["gait_phase"].append(base.gait_indices.cpu())
        frequency = (base.commands_dog[:, 6] if base.commands_dog.shape[1] > 6
                     else torch.zeros(base.num_envs, device=base.device))
        series["gait_frequency_hz"].append(frequency.cpu())
        series["base_position"].append(base.base_pos.cpu())
        series["base_quaternion"].append(base.base_quat.cpu())
        series["base_linear_velocity"].append(base.base_lin_vel.cpu())
        series["base_angular_velocity"].append(base.base_ang_vel.cpu())
        series["arm_dof_pos"].append(base.dof_pos[:, arm_slice].cpu())
        series["arm_dof_vel"].append(base.dof_vel[:, arm_slice].cpu())
        series["ee_pos_in_base"].append(base.response_ee_pos_in_base.cpu())
        active = sampler.is_identification
        series["excitation_signal"].append(
            torch.where(active, sampler.signal, torch.full_like(sampler.signal, -1)).cpu()
        )
        series["excitation_channel"].append(
            torch.where(active, sampler.channel, torch.full_like(sampler.channel, -1)).cpu()
        )
        series["plan_generation"].append(sampler.plan_generation.cpu())
        series["group_valid"].append(base.grouping.valid.cpu())
        series["reset"].append((base.reset_buf != 0).float().cpu())

    env.reset()
    with torch.no_grad():
        for _ in range(steps):
            observations = env.get_dog_observations()
            actions = policy.act_inference({"obs_history": observations["obs_history"]})
            env.step(actions, arm_actions)
            record()

    payload = {name: torch.stack(values).numpy() for name, values in series.items()}
    payload["is_identification"] = sampler.is_identification.cpu().numpy()
    payload["is_nominal_twin"] = base.is_nominal_twin.cpu().numpy()
    payload["group_of"] = base.grouping.group_of.cpu().numpy()
    # Privileged domain parameters: simulation only, and labelled as such in the
    # metadata so nobody builds a hardware pipeline that expects them.
    for name, tensor in (
        ("domain_friction", base.friction_coeffs[:, 0]),
        ("domain_restitution", base.restitutions[:, 0]),
        ("domain_payload", base.payloads),
        ("domain_com_displacement", base.com_displacements),
        ("domain_motor_strength", base.motor_strengths),
        ("domain_Kp_factor", base.Kp_factors),
        ("domain_Kd_factor", base.Kd_factors),
    ):
        payload[name] = tensor.cpu().numpy()
    if hasattr(base, "stage1_ee_payload_mass"):
        payload["domain_ee_payload"] = base.stage1_ee_payload_mass.cpu().numpy()
    if hasattr(base, "arm_mount_bucket_of_env"):
        payload["domain_mount_bucket"] = base.arm_mount_bucket_of_env.cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    sidecar = os.path.splitext(args.out)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata(cfg, base, args, steps), handle, indent=2)
        handle.write("\n")

    size_mb = os.path.getsize(args.out) / 1e6
    identification = int(sampler.is_identification.sum())
    print(f"  {steps} steps x {base.num_envs} envs at "
          f"{1.0 / base.dt:.0f} Hz -> {args.out} ({size_mb:.1f} MB)")
    print(f"  {identification} identification envs, "
          f"{int(base.is_nominal_twin.sum())} nominal twins")
    print(f"  metadata: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
