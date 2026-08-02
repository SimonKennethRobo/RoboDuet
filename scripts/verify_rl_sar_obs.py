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


def quat_rotate_inverse_np(quat_xyzw, vec):
    """IsaacGym's quat_rotate_inverse, matching rl_sar's QuatRotateInverse."""
    x, y, z, w = quat_xyzw
    a = vec * (2.0 * w * w - 1.0)
    b = np.cross(np.array([x, y, z]), vec) * w * 2.0
    c = np.array([x, y, z]) * np.dot(np.array([x, y, z]), vec) * 2.0
    return a - b + c


def quat_to_euler_np(quat_xyzw):
    """[roll, pitch, yaw]; identical formula to rl_sar's QuaternionToEuler."""
    x, y, z, w = quat_xyzw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class RlSarObservation:
    """Python port of rl_sar's RL::ComputeObservation() for the roboduet terms.

    Reads exactly the keys the C++ reads, from the exported config.yaml, so a
    typo in the generated config shows up here rather than on the robot.
    """

    def __init__(self, params):
        self.p = params
        self.num_leg = int(params["num_leg_dofs"])
        self.num_arm = int(params["num_arm_dofs"])
        self.default_dof_pos = np.array(params["default_dof_pos"], dtype=np.float64)

    def widths(self):
        p = self.p
        return {
            "gravity_vec": 3,
            "ang_vel": 3,
            "lin_vel": 3,
            "roboduet/leg_dof_pos": self.num_leg,
            "roboduet/leg_dof_vel": self.num_leg,
            "roboduet/leg_actions": self.num_leg,
            "roboduet/dog_commands": len(p["dog_commands_scale"]),
            "roboduet/arm_commands": int(p["arm_num_commands"]),
            "roboduet/clock_inputs": 4,
            "roboduet/base_lin_vel": 3,
            "roboduet/body_pose_actual": 3,
            "roboduet/body_pose_error": 3,
            "roboduet/velocity_error": 3,
            "roboduet/arm_dof_pos": self.num_arm,
            "roboduet/arm_dof_vel": self.num_arm,
        }

    def term(self, name, s):
        """s: dict of raw quantities rl_sar would have available."""
        p = self.p
        num_leg, num_arm = self.num_leg, self.num_arm

        if name == "gravity_vec":
            return quat_rotate_inverse_np(s["quat"], np.array([0.0, 0.0, -1.0]))
        if name == "ang_vel":
            return s["ang_vel"] * p["ang_vel_scale"]
        if name == "roboduet/leg_dof_pos":
            return (s["dof_pos"][:num_leg] - self.default_dof_pos[:num_leg]) * p["dof_pos_scale"]
        if name == "roboduet/leg_dof_vel":
            return s["dof_vel"][:num_leg] * p["dof_vel_scale"]
        if name == "roboduet/leg_actions":
            return s["actions"][:num_leg]
        if name == "roboduet/arm_dof_pos":
            sl = slice(num_leg, num_leg + num_arm)
            return (s["dof_pos"][sl] - self.default_dof_pos[sl]) * p["dof_pos_scale"]
        if name == "roboduet/arm_dof_vel":
            return s["dof_vel"][num_leg:num_leg + num_arm] * p["dof_vel_scale"]
        if name == "roboduet/dog_commands":
            cmd = np.concatenate([
                [s["cmd_x"], s["cmd_y"], s["cmd_yaw"],
                 s["cmd_pitch"], s["cmd_roll"], s["cmd_height"]],
                np.array(p["dog_commands_extra"], dtype=np.float64),
            ])
            return cmd * np.array(p["dog_commands_scale"], dtype=np.float64)
        if name == "roboduet/arm_commands":
            return np.zeros(int(p["arm_num_commands"]))
        if name == "roboduet/clock_inputs":
            phases, offsets, bounds = p["gait_phases"]
            duration = float(p["gait_duration"])
            gait = s["gait_indices"]
            foot = [gait + phases + offsets + bounds, gait + offsets, gait + bounds, gait + phases]
            standing = np.linalg.norm([s["cmd_x"], s["cmd_y"], s["cmd_yaw"]]) < 0.1
            out = np.zeros(4)
            for i in range(4):
                idx = 0.25 if standing else np.fmod(foot[i], 1.0)
                if idx < 0.0:
                    idx += 1.0
                idx = (idx * (0.5 / duration) if idx < duration
                       else 0.5 + (idx - duration) * (0.5 / (1.0 - duration)))
                out[i] = np.sin(2.0 * np.pi * idx)
            return out
        if name in ("lin_vel", "roboduet/base_lin_vel"):
            if name == "roboduet/base_lin_vel" and not p.get("observe_lin_vel", True):
                return np.zeros(3)
            return s["lin_vel"] * p["lin_vel_scale"]
        if name == "roboduet/body_pose_actual":
            if not p.get("observe_pose_actual", True):
                return np.zeros(3)
            euler = quat_to_euler_np(s["quat"])
            return np.array([s["base_height"] * p["body_height_cmd_scale"],
                             euler[1] * p["body_pitch_cmd_scale"],
                             euler[0] * p["body_roll_cmd_scale"]])
        if name == "roboduet/body_pose_error":
            if not p.get("observe_track_error", True):
                return np.zeros(3)
            euler = quat_to_euler_np(s["quat"])
            height_target = float(p["base_height_target"]) + s["cmd_height"]
            return np.array([(height_target - s["base_height"]) * p["body_height_cmd_scale"],
                             (s["cmd_pitch"] - euler[1]) * p["body_pitch_cmd_scale"],
                             (s["cmd_roll"] - euler[0]) * p["body_roll_cmd_scale"]])
        if name == "roboduet/velocity_error":
            if not p.get("observe_track_error", True):
                return np.zeros(3)
            return np.array([(s["cmd_x"] - s["lin_vel"][0]) * p["lin_vel_scale"],
                             (s["cmd_y"] - s["lin_vel"][1]) * p["lin_vel_scale"],
                             (s["cmd_yaw"] - s["ang_vel"][2]) * p["ang_vel_scale"]])
        raise KeyError(f"rl_sar term '{name}' is not implemented in this verifier")

    def assemble(self, s):
        parts = [np.asarray(self.term(name, s), dtype=np.float64) for name in self.p["observations"]]
        return np.clip(np.concatenate(parts), -self.p["clip_obs"], self.p["clip_obs"])


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
    full = clamped + [float(value) for value in params["dog_commands_extra"]]
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
    expected = np.array(list(applied) + list(params["dog_commands_extra"]), dtype=np.float64)
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
        rl_sar_gait = np.fmod(rl_sar_gait + policy_dt * float(params["gait_frequency"]), 1.0)
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
