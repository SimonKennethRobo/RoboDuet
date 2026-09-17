"""Diff the rl_sar observation assembly against the real RoboDuet dog observation.

This is the check to run before touching a robot. It builds the live IsaacGym
env, and at every step compares:

  truth   = WBCEnv.get_dog_observations()["obs"]        (RoboDuet's own code)
  replay  = the rl_sar term list from the exported config.yaml, reassembled
            here in Python from the same raw sensor quantities rl_sar would see

Any per-term mismatch means the exported config and rl_sar's ComputeObservation()
disagree with training, and the policy would be fed garbage on hardware. The
Python port below deliberately mirrors rl_sar's C++ term-for-term, so a diff
localises the bug to one observation term instead of "the robot fell over".

Usage::

    python scripts/verify_rl_sar_obs.py \
        --logdir runs/<date>/<run> \
        --config <rl_sar>/policy/go2_x5/roboduet_stage1/config.yaml \
        --steps 200
"""

import argparse
import sys
from pathlib import Path

import isaacgym  # noqa: F401  must precede torch
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_gym.envs.roboduet.wbc_env_wrapper import KeyboardStage1Wrapper  # noqa: E402
from go1_gym.utils.global_switch import global_switch  # noqa: E402
from scripts.load_policy import load_dog_policy, load_env  # noqa: E402
from scripts.rl_sar_obs import RlSarObservation, dog_command_values, effective_gait_frequency  # noqa: E402


# Operator command schedule. rl_sar drives the commands, so the env is driven to
# the same values before every comparison -- otherwise the env's own curriculum
# resampling would show up as a false mismatch. Includes a zero entry so the
# standing branch of the gait clock is exercised too.
#                (vx,    vy,   yaw,  pitch,  roll, height)
COMMAND_SCHEDULE = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),      # standing: clock forced to stand phase
    (0.6, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.4, -0.5, 0.0, 0.0, 0.0),
    (-0.5, -0.3, 0.8, 0.15, -0.1, 0.05),
    (1.0, 0.0, 0.0, -0.2, 0.2, -0.08),
]


def apply_rl_sar_commands(env, params, command):
    """Write rl_sar's command vector into the env, clamped exactly as
    RL::StateController() clamps operator input."""
    limits = ["limit_vel_x", "limit_vel_y", "limit_vel_yaw",
              "limit_body_pitch", "limit_body_roll", "limit_body_height"]
    clamped = [float(np.clip(value, params[key][0], params[key][1]))
               for value, key in zip(command, limits)]
    full = dog_command_values(params, clamped)
    inner = env.env
    width = inner.commands_dog.shape[1]
    if len(full) != width:
        raise ValueError(f"rl_sar sends {len(full)} dog commands, env wants {width}")
    inner.commands_dog[0, :] = torch.tensor(full, device=inner.device, dtype=inner.commands_dog.dtype)
    return clamped


def raw_state_from_env(env, command):
    """The quantities rl_sar reads from MuJoCo sensors / the robot SDK / FAST-LIO.

    `command` is what rl_sar's operator input would be, not read back from the
    env -- so a scale or ordering mistake in the command term actually fails.
    """
    inner = env.env
    return {
        "quat": inner.base_quat[0].cpu().numpy().astype(np.float64),  # xyzw
        "dof_pos": inner.dof_pos[0].cpu().numpy().astype(np.float64),
        "dof_vel": inner.dof_vel[0].cpu().numpy().astype(np.float64),
        "actions": inner.actions[0].cpu().numpy().astype(np.float64),
        "ang_vel": inner.base_ang_vel[0].cpu().numpy().astype(np.float64),
        "lin_vel": inner.base_lin_vel[0].cpu().numpy().astype(np.float64),
        "base_height": float(inner.base_pos[0, 2].item()),
        "cmd_x": command[0], "cmd_y": command[1], "cmd_yaw": command[2],
        "cmd_pitch": command[3], "cmd_roll": command[4], "cmd_height": command[5],
        # rl_sar runs its own gait integrator; seeding from the env isolates the
        # phase-warp math from integrator drift, which is checked separately.
        "gait_indices": float(inner.gait_indices[0].item()),
    }


def env_resampled_commands(env, params, applied):
    """True if the env overwrote our command (episode reset / curriculum resample).

    _step_contact_targets() would then have run against a different command than
    the replay assumes, so that sample is not comparable and gets skipped.
    """
    live = env.env.commands_dog[0].detach().cpu().numpy().astype(np.float64)
    expected = dog_command_values(params, applied)
    return not np.allclose(live, expected, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True,
                        help="config.yaml produced by export_rl_sar.py")
    parser.add_argument("--ckptid", type=str, default="last")
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    raw_yaml = yaml.safe_load(open(args.config))
    if len(raw_yaml) != 1:
        raise ValueError(f"{args.config} should hold exactly one top-level key")
    params = next(iter(raw_yaml.values()))

    # Stage 1: the arm policy never runs and the arm command slot stays zeroed.
    global_switch.switch_flag = False
    global_switch.count = 0
    global_switch.pretrained_to_wbc_start = args.steps + 1
    global_switch.pretrained_to_wbc_end = args.steps + 2

    # load_env builds `wrapper` as the env and applies HistoryWrapper itself.
    # KeyboardStage1Wrapper is the stage-1 env play uses; its key bindings are
    # inert headless, and it keeps this check on the same path as play.
    env, cfg = load_env(args.logdir, wrapper=KeyboardStage1Wrapper, headless=True,
                        device=args.sim_device, robot=args.robot)
    dog_policy = load_dog_policy(args.logdir, args.ckptid, cfg)

    builder = RlSarObservation(params)
    widths = builder.widths()
    if sum(widths[name] for name in params["observations"]) != int(cfg.dog.dog_num_observations):
        raise ValueError("exported term widths do not sum to cfg.dog.dog_num_observations")

    offsets, cursor = {}, 0
    for name in params["observations"]:
        offsets[name] = (cursor, cursor + widths[name])
        cursor += widths[name]

    env.reset()
    worst = {name: 0.0 for name in params["observations"]}
    gait_drift = 0.0
    gait_offset_reference = None
    rl_sar_gait = 0.0
    compared = 0
    skipped = 0
    policy_dt = float(cfg.sim.dt) * int(cfg.control.decimation)
    steps_per_command = max(1, args.steps // len(COMMAND_SCHEDULE))

    # Order matters: the env writes clock_inputs inside step(), from the command
    # active during that step. Comparing before stepping would diff a fresh
    # command against a clock computed from the previous one. So: apply the
    # command, step, then observe -- which is also the real control-loop order.
    dog_obs = env.get_dog_observations()
    for step in range(args.steps):
        command = COMMAND_SCHEDULE[min(step // steps_per_command, len(COMMAND_SCHEDULE) - 1)]
        applied = apply_rl_sar_commands(env, params, command)

        with torch.no_grad():
            actions_dog = dog_policy(dog_obs).to(env.env.device)
        env.step(actions_dog, env.arm_fake_actions)

        dog_obs = env.get_dog_observations()
        if env_resampled_commands(env, params, applied):
            skipped += 1
            continue

        truth = dog_obs["obs"][0].cpu().numpy().astype(np.float64)
        state = raw_state_from_env(env, applied)
        replay = builder.assemble(state)
        compared += 1

        for name, (lo, hi) in offsets.items():
            worst[name] = max(worst[name], float(np.max(np.abs(truth[lo:hi] - replay[lo:hi]))))

        # Independently check rl_sar's own gait integrator against the env's.
        # A constant phase offset is harmless -- the policy only ever sees a
        # phase, and rl_sar zeroes gait_indices when the RL state is entered.
        # What matters is the *rate*, so measure how far the offset moves.
        rl_sar_gait = np.fmod(rl_sar_gait + policy_dt * effective_gait_frequency(params, applied[:3]), 1.0)
        circular = abs(rl_sar_gait - state["gait_indices"])
        offset = min(circular, 1.0 - circular)
        if gait_offset_reference is None:
            gait_offset_reference = offset
        gait_drift = max(gait_drift, abs(offset - gait_offset_reference))

    if not compared:
        print("No comparable samples -- the env resampled commands every step.")
        return 1

    print(f"\nper-term max |truth - rl_sar| over {compared} compared steps "
          f"({skipped} skipped after env resets, tolerance {args.atol}):\n")
    failures = 0
    for name in params["observations"]:
        lo, hi = offsets[name]
        status = "ok  " if worst[name] <= args.atol else "FAIL"
        if worst[name] > args.atol:
            failures += 1
        print(f"  [{status}] {name:<32s} dims {lo:>3d}:{hi:<3d}  max_err {worst[name]:.3e}")

    print(f"\n  gait clock rate drift (rl_sar vs env): {gait_drift:.3e}")
    if gait_drift > 1e-3:
        print("  WARNING: the gait phase is advancing at a different rate than "
              "training. Check dt, decimation and gait_frequency in the exported "
              "config. (A constant phase offset is fine; a growing one is not.)")

    if failures:
        print(f"\n{failures} term(s) disagree -- fix the export/ComputeObservation "
              f"before deploying.")
        return 1
    print("\nAll observation terms match. The exported config reproduces "
          "get_dog_observations() exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
