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

Four things about *what* gets recorded that are decisions, not defaults.

**The environment is built from the checkpoint's own ``parameters.pkl``.**  Not
from the current source tree.  ``response.omega_n`` and ``response.rate_limit``
are edited by hand in ``wbc.py`` between Run A and Run B -- that is exactly what
the calibration gate does -- so a checkpoint exported after that edit would be
measured against a reference model it was never trained against, and
``reference_state``, ``reference_rate`` and ``measured_detrended`` would all be
wrong in a way that reads as a modelling result rather than as a bug.  The
R8 curriculum iteration is restored for the same reason: it sets the *width* of
the domain randomisation, and at iteration 0 friction spans [0.75, 1.57]
instead of [0.10, 3.00].

**Domain parameters are split by whether they are re-drawn during the run.**
Motor strength, the Kp/Kd factors and the EE payload are sampled fresh on every
reset *and* again every ``domain_rand.rand_interval`` steps mid-episode -- of
292 motor-strength changes in a 25 s / 64 env export, 238 were mid-episode -- so
they are stored per step.  Friction, restitution, base payload and COM offset
are drawn once during ``_create_envs`` and are stored per env, after being
compared before and after the rollout, so the split is a checked property of
this file rather than a belief about the config.  A single end-of-rollout
snapshot of a per-episode parameter mispairs almost every sample in the file,
and nothing downstream can detect that from the data alone.

Within the per-step group the leg and arm blocks are *not* interchangeable: the
legs share one randomised scalar, each arm joint is drawn independently, and the
two come from different ranges.  Hence the ``[leg, arm_0..arm_{A-1}]`` column
layout rather than one number per env.

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
import pickle
import sys
from pathlib import Path

import isaacgym  # noqa: F401  must precede torch
import numpy as np
import torch

from go1_gym.envs.config import (
    apply_config_snapshot,
    build_roboduet_config,
    recompute_observation_dims,
)
from go1_gym.envs.config.wbc import ROBOT_ASSET_FILES
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.response import DECISION_CHANNEL_NAMES, DECISION_CHANNEL_UNITS, DOG_COMMAND_NAMES
from go1_gym.response.excitation import SIGNAL_NAMES
from go1_gym.utils import global_switch


def run_directory(policy_path, logdir=None):
    """``<run>/checkpoints_dog/ac_weights_*.pt`` -> ``<run>``."""
    if logdir:
        return Path(logdir)
    return Path(policy_path).resolve().parent.parent


def load_snapshot(logdir):
    """The config the checkpoint was actually trained with.

    Rebuilding from source instead is the single most expensive mistake this
    script can make, and it is silent.  omega_n and rate_limit are *edited by
    hand in wbc.py* between Run A and Run B (that is what the calibration gate
    does), so a checkpoint exported a week later is measured against a reference
    model it was never trained against -- and every derived quantity here,
    reference_state, reference_rate and measured_detrended alike, is wrong in a
    way that looks like a modelling result.  Every other consumer of a
    checkpoint in this repo reads this file (scripts/load_policy.py,
    scripts/export_rl_sar.py, go1_gym_learn/ppo_cse_automatic).
    """
    path = Path(logdir) / "parameters.pkl"
    if not path.exists():
        return None, path
    with open(path, "rb") as handle:
        return pickle.load(handle), path


def infer_iteration(policy_path, snapshot):
    """Which training iteration this checkpoint is from, and how we know.

    It matters because the R8 curriculum keys the *domain randomisation width*
    off it: ``randomization_intensity`` is 0.30 in stages 1-2 and only reaches
    1.0 during stage 3.  Exporting at iteration 0 -- which is what leaving
    global_switch.count alone does -- samples friction from [0.75, 1.57]
    instead of [0.10, 3.00].  A domain-identification dataset whose domains
    span a third of their trained width is not obviously broken; it just
    produces confidently wrong answers.
    """
    path = Path(policy_path)
    tail = path.stem.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail), "checkpoint filename"
    # "last": read the iteration off the highest numbered sibling instead.
    # RunnerArgs.max_iterations is NOT a usable fallback -- auto_train.py takes
    # --num_learning_iterations from the command line and the snapshot keeps the
    # default, so the 20k run's parameters.pkl claims 1500 while its checkpoints
    # go to 19999.  Off by 13x, and in the direction that silently narrows the
    # randomisation to its stage-1 floor.
    numbered = [
        int(sibling.stem.rsplit("_", 1)[-1])
        for sibling in path.parent.glob("ac_weights_*.pt")
        if sibling.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if numbered:
        return max(numbered), f"highest numbered checkpoint beside {path.name}"
    return None, None


def build_env(num_envs, sim_device, robot, snapshot=None, iteration=0):
    args = argparse.Namespace(
        robot=robot, num_envs=num_envs, dyna_gait=True, goal_reaching=False,
        traj_tracking=False, arm_action_mode=None, no_reach_table=False,
        dyna_gait_min_frequency=0.0, video=False,
    )
    cfg = build_roboduet_config(args)
    if snapshot is not None:
        apply_config_snapshot(cfg, snapshot["Cfg"], drop_unknown=True)
        recompute_observation_dims(cfg)
        expected = ROBOT_ASSET_FILES.get(robot)
        if expected is not None and cfg.asset.file != expected:
            raise SystemExit(
                f"--robot {robot} expects asset {expected!r}, but the checkpoint "
                f"was trained on {cfg.asset.file!r}. Pass the matching --robot."
            )
    # These three describe *this* export, not the training run, so they are
    # restored after the snapshot rather than taken from it.
    cfg.env.num_envs = num_envs
    cfg.env.arm_policy_enabled = False
    cfg.env.record_video = False
    if getattr(cfg.env, "num_eval_envs", 0):
        cfg.env.num_eval_envs = min(int(cfg.env.num_eval_envs), max(num_envs // 8, 0))
    global_switch.pretrained_to_wbc_start = 10 ** 9
    global_switch.pretrained_to_wbc_end = 10 ** 9 + 1
    # Both curricula are driven by these counters: the R8 randomisation width by
    # ``count``, the stage-1 arm disturbance by ``stage1_count``.  Left at 0 the
    # export reproduces neither of the domains the checkpoint was trained in.
    global_switch.count = int(iteration)
    global_switch.stage1_count = int(iteration)
    global_switch.stage1_arm_ramp_iterations = int(
        getattr(cfg.env, "stage1_arm_ramp_iterations", 1)
    )
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


def metadata(cfg, base, args, steps, provenance):
    """The sidecar.  Without it the arrays are unusable in six months."""
    return {
        "schema": "roboduet-identification-v2",
        "robot": args.robot,
        "policy": args.policy,
        "provenance": provenance,
        "sample_rate_hz": round(1.0 / base.dt, 6),
        "dt_s": base.dt,
        "steps": steps,
        "num_envs": int(base.num_envs),
        "terrain": {
            "mesh_type": cfg.terrain.mesh_type,
            "note": (
                "v1 is flat ground, so this global label IS the terrain label. "
                "per-env terrain_level/terrain_type arrays are written instead "
                "whenever the terrain has levels"
            ),
        },
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
            "is_standing": (
                "(T, E) 1 where the gait clock is stopped (commanded speed "
                "below 0.1 forces gait_frequency to 0). Phase-binned estimates "
                "must exclude or separate these -- the phase is not advancing"
            ),
            "is_identification": "(E,) environments running designed excitation",
            "is_nominal_twin": "(E,) environments held at nominal domain",
            "group_of": "(E,) R5 group index, -1 if ungrouped",
            "group_valid": "(T, E) 1 when the group is comparable with its twin",
            "reset": "(T, E) 1 on the step the episode ended",
            "terrain_level": "(E,) terrain curriculum level, absent on flat ground",
            "terrain_type": "(E,) terrain column index, absent on flat ground",
        },
        "domain_parameters": {
            "note": (
                "privileged, SIMULATION ONLY. Split by whether the parameter "
                "is re-drawn during the run: the per_step ones are resampled on "
                "every reset AND every domain_rand.rand_interval steps "
                "mid-episode, so storing them per-env would mispair almost "
                "every sample in the file"
            ),
            "per_step": {
                "domain_motor_strength": "(T, E, 1+A) [leg, arm_0..arm_{A-1}]",
                "domain_Kp_factor": "(T, E, 1+A) [leg, arm_0..arm_{A-1}]",
                "domain_Kd_factor": "(T, E, 1+A) [leg, arm_0..arm_{A-1}]",
                "domain_ee_payload": "(T, E) kg at the end effector",
            },
            "per_env": {
                "domain_friction": "(E,)",
                "domain_restitution": "(E,)",
                "domain_payload": "(E,) kg added to the base",
                "domain_com_displacement": "(E, 3) m",
                "domain_mount_bucket": "(E,) arm mount-TF bucket index",
            },
            "dof_columns": list(base.response_export_dof_columns),
            "dof_columns_note": (
                "the leg DoFs share ONE randomised scalar, so they collapse to "
                "a single column; each arm DoF is drawn INDEPENDENTLY and from a "
                "different range (domain_rand.motor_strength_range vs "
                "domain_rand.stage1_arm.motor_strength_range), so every arm "
                "joint gets its own column. Column 0 is not the arm's value"
            ),
            "per_env_is_checked": (
                "the per_env values are compared before and after the rollout "
                "and the export fails if any moved, so this split is a fact "
                "about the file and not an assumption about the config"
            ),
            "randomization_intensity": float(
                getattr(base, "domain_randomization_intensity", 1.0)
            ),
            "randomization_intensity_note": (
                "R8 opens the randomisation ranges from nominal towards their "
                "configured width; 0.30 is the stage-1/2 floor and 1.0 is the "
                "full width. Ranges below are the CONFIGURED ones, so the "
                "realised spread is this fraction of them, measured around the "
                "nominal value"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--logdir", default=None,
                        help="run directory holding parameters.pkl "
                             "(default: two levels up from --policy)")
    parser.add_argument("--curriculum_iteration", type=int, default=None,
                        help="training iteration to reproduce the R8 domain "
                             "randomisation width at (default: inferred)")
    parser.add_argument("--allow_config_drift", action="store_true",
                        help="export without the checkpoint's parameters.pkl. "
                             "Every derived channel is then computed against "
                             "whatever wbc.py says today")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--robot", type=str, default="go2_x5")
    parser.add_argument("--out", default="data/identification.npz")
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        raise SystemExit(f"checkpoint not found: {args.policy}")

    logdir = run_directory(args.policy, args.logdir)
    snapshot, snapshot_path = load_snapshot(logdir)
    if snapshot is None and not args.allow_config_drift:
        raise SystemExit(
            f"no config snapshot at {snapshot_path}.\n"
            "Exporting against the current source config silently measures the "
            "policy with a reference model it was never trained against -- "
            "omega_n and rate_limit are edited by hand between runs.\n"
            "Pass --logdir, or --allow_config_drift if you really mean to."
        )
    iteration, iteration_source = infer_iteration(args.policy, snapshot)
    if args.curriculum_iteration is not None:
        iteration, iteration_source = args.curriculum_iteration, "--curriculum_iteration"
    if iteration is None:
        iteration, iteration_source = 0, "UNKNOWN -- domain randomisation will be at its floor"
    print(f"  config: {snapshot_path if snapshot else 'CURRENT SOURCE (drift allowed)'}")
    print(f"  curriculum iteration {iteration} (from {iteration_source})")

    provenance = {
        "config_snapshot": str(snapshot_path) if snapshot else None,
        "config_source": "checkpoint parameters.pkl" if snapshot else "current source tree",
        "curriculum_iteration": int(iteration),
        "curriculum_iteration_source": iteration_source,
    }
    env, cfg = build_env(args.num_envs, args.sim_device, args.robot, snapshot, iteration)
    base = env.env
    print(f"  domain randomisation intensity "
          f"{float(getattr(base, 'domain_randomization_intensity', 1.0)):.2f}")
    policy = load_policy(args.policy, cfg, base.device)
    steps = int(args.seconds / base.dt)
    arm_actions = torch.zeros(base.num_envs, base.num_actions_arm, device=base.device)
    sampler = base.response_excitation
    arm_slice = slice(base.num_actions_loco, base.num_actions_loco + base.num_actions_arm)

    series = {name: [] for name in (
        "command", "reference_state", "reference_rate", "measured",
        "measured_detrended", "gait_phase", "gait_frequency_hz", "is_standing",
        "base_position", "base_quaternion", "base_linear_velocity",
        "base_angular_velocity", "arm_dof_pos", "arm_dof_vel", "ee_pos_in_base",
        "excitation_signal", "excitation_channel", "plan_generation",
        "group_valid", "reset",
        # Domain parameters re-drawn on every reset.  These MUST be per-step:
        # a single end-of-rollout snapshot pairs every sample before the last
        # reset with the wrong domain, which is the one failure mode a
        # closed-loop identification cannot detect from the data alone.
        "domain_motor_strength", "domain_Kp_factor", "domain_Kd_factor",
        "domain_ee_payload",
    )}

    # ...and the ones that are drawn once and never redrawn, under this config.
    # Recorded as (E,), and checked at the end rather than assumed: whether they
    # are static is decided by domain_rand.randomize_rigids_after_start, which
    # lives in the very config file this script used to ignore.
    static_domain = {
        "domain_friction": base.friction_coeffs[:, 0],
        "domain_restitution": base.restitutions[:, 0],
        "domain_payload": base.payloads,
        "domain_com_displacement": base.com_displacements,
    }
    if hasattr(base, "arm_mount_bucket_of_env"):
        static_domain["domain_mount_bucket"] = base.arm_mount_bucket_of_env
    static_before = {name: tensor.clone() for name, tensor in static_domain.items()}

    # motor_strength and the Kp/Kd factors are sampled with DIFFERENT structure
    # on the two DoF blocks, and the difference is easy to miss:
    #
    #   legs  _randomize_dof_props   one scalar per env, broadcast to all DoFs
    #   arm   _randomize_arm_dof_props   torch.rand(n, num_actions_arm) -- one
    #                                    INDEPENDENT value per arm joint
    #
    # ...from different ranges too (legs [0.9, 1.1], arm [0.7, 1.3] for motor
    # strength).  So the leg block collapses to one column losslessly and the
    # arm block does not collapse at all.  Column layout is therefore
    # [leg, arm_0, ..., arm_{A-1}].
    num_arm_dofs = int(base.num_actions_arm)
    leg_block = slice(0, int(base.num_actions_loco))
    arm_block = slice(int(base.num_actions_loco), int(base.num_actions_loco) + num_arm_dofs)
    dof_columns = ["leg"] + [f"arm_{i}" for i in range(num_arm_dofs)]

    def dof_blocks(tensor, name):
        """(E, 1 + A) -- the leg scalar, then every arm DoF.

        The leg block's uniformity is checked on every recorded step rather than
        once at startup, because a startup-only check runs before the first
        reset has written anything and so proves nothing -- which is exactly how
        the arm block's per-DoF sampling went unnoticed until this guard fired.
        """
        legs = tensor[:, leg_block]
        if legs.shape[1] > 1 and not torch.allclose(legs, legs[:, :1].expand_as(legs)):
            raise SystemExit(
                f"{name} varies within the leg DoF block; this exporter records "
                "one column for it. Record the full (T, E, D) tensor before "
                "using this dataset."
            )
        return torch.cat((tensor[:, :1], tensor[:, arm_block]), dim=-1)

    def record():
        reference = base.response_ref
        series["command"].append(reference.gather_commands(base.commands_dog).cpu())
        series["reference_state"].append(reference.xi.cpu())
        series["reference_rate"].append(reference.xi_dot.cpu())
        series["measured"].append(base.response_measured.cpu())
        series["measured_detrended"].append(base.response_detrended.cpu())
        series["gait_phase"].append(base.gait_indices.cpu())
        # _gait_frequency_hz(), not commands_dog[:, 6]: the fallback when
        # dynamic gait is off is a fixed 3 Hz clock, not zero, and recording
        # zero there would label the whole run as standing.
        frequency = base._gait_frequency_hz()
        series["gait_frequency_hz"].append(frequency.cpu())
        # R12's standing flag.  gait_frequency is forced to 0 when the commanded
        # velocity norm drops below 0.1, which stops the phase clock: ~17-24% of
        # samples.  Phase-binned residual estimates have to treat those
        # separately or the bins are polluted by a clock that is not running.
        # Recorded rather than reconstructed, because the R6 excitation can move
        # vx/vy/wyaw mid-episode without the frequency being re-evaluated -- so
        # the command norm at read time does not always agree with the clock.
        series["is_standing"].append((frequency == 0).float().cpu())
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
        series["domain_motor_strength"].append(
            dof_blocks(base.motor_strengths, "motor_strengths").cpu())
        series["domain_Kp_factor"].append(
            dof_blocks(base.Kp_factors, "Kp_factors").cpu())
        series["domain_Kd_factor"].append(
            dof_blocks(base.Kd_factors, "Kd_factors").cpu())
        series["domain_ee_payload"].append(
            base.stage1_ee_payload_mass.cpu() if hasattr(base, "stage1_ee_payload_mass")
            else torch.zeros(base.num_envs)
        )

    base.response_export_dof_columns = dof_columns

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
    #
    # Verified static rather than assumed static.  Under the shipped config
    # (randomize_rigids_after_start = False) these are drawn during
    # _create_envs and never redrawn, so one (E,) row is the whole truth -- but
    # flipping that one flag makes them per-episode, and then an (E,) row pairs
    # every pre-reset sample with the wrong domain.  Fail rather than write the
    # file.
    drifted = [name for name, tensor in static_domain.items()
               if not torch.equal(tensor, static_before[name])]
    if drifted:
        raise SystemExit(
            "these domain parameters changed during the rollout and can no "
            f"longer be stored per-env: {', '.join(sorted(drifted))}.\n"
            "domain_rand.randomize_rigids_after_start is on; record them "
            "per-step (as motor_strength already is) before exporting."
        )
    for name, tensor in static_domain.items():
        payload[name] = tensor.cpu().numpy()

    # Terrain.  v1 is flat and the R0 plan says so, which is why the sidecar's
    # global mesh_type is the label -- but the per-env terrain assignment exists
    # whenever the terrain has levels, and writing it costs one array.
    if getattr(cfg.terrain, "curriculum", False) or cfg.terrain.mesh_type != "plane":
        payload["terrain_level"] = base.terrain_levels.cpu().numpy()
        payload["terrain_type"] = base.terrain_types.cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    sidecar = os.path.splitext(args.out)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata(cfg, base, args, steps, provenance), handle, indent=2)
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
