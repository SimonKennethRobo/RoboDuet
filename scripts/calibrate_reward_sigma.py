"""Calibrate the R4.1 error scale ``sigma``, per channel.

R4 forbids guessing sigma.  The procedure it specifies: build the reference
trajectory and a candidate with twice the bandwidth, then choose sigma so the
integrated-reward difference over the transient window is 30-50% of the largest
achievable difference.  This script runs that, for every channel, from the
values actually in the config.

    python scripts/calibrate_reward_sigma.py
    python scripts/calibrate_reward_sigma.py --target 0.45 --out configs/reward_sigma.json

No simulator and no policy involved -- it is a closed-form property of the
reference model -- so it takes milliseconds.  **Re-run it whenever omega_n,
rate_limit or the command sampling ranges change**, because sigma scales with
the step amplitude and with the bandwidth gap.
"""

import argparse
import json
import os

from go1_gym.envs.config import build_roboduet_config
from go1_gym.response import (
    DEFAULT_TARGET_DISCRIMINATION,
    build_channels,
    calibrate_channels,
)


def build_cfg(robot):
    args = argparse.Namespace(
        robot=robot, num_envs=1, dyna_gait=True, goal_reaching=False, traj_tracking=False,
        arm_action_mode=None, no_reach_table=False, dyna_gait_min_frequency=0.0, video=False,
    )
    return build_roboduet_config(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="go2_x5")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET_DISCRIMINATION,
                        help="fraction of the maximum integrated-reward gap (R4 asks for 0.3-0.5)")
    parser.add_argument("--faster_multiple", type=float, default=2.0,
                        help='bandwidth ratio of the "too aggressive" candidate')
    parser.add_argument("--window_factor", type=float, default=4.0,
                        help="transient window, in units of 1/omega_n")
    parser.add_argument("--out", default=None, help="write the result to this JSON file")
    args = parser.parse_args()

    if not 0.0 < args.target < 1.0:
        raise SystemExit(f"--target must be in (0, 1), got {args.target}")
    if not 0.3 <= args.target <= 0.5:
        print(f"[warn] --target {args.target} is outside the 0.3-0.5 band R4 specifies")

    cfg = build_cfg(args.robot)
    dt = cfg.sim.dt * cfg.control.decimation
    channels = build_channels(
        cfg.response.channel_order, cfg.response.omega_n, cfg.response.rate_limit
    )
    amplitudes = {name: float(value)
                  for name, value in cfg.response.reward.calibration_amplitudes.items()}

    results = calibrate_channels(
        channels, amplitudes, dt=dt,
        target_discrimination=args.target,
        window_factor=args.window_factor,
        faster_multiple=args.faster_multiple,
    )

    print(f"dt={dt:g}s  target discrimination={args.target}  "
          f"candidate={args.faster_multiple}x bandwidth  window={args.window_factor}/omega_n\n")
    print(f"{'channel':<10}{'omega_n':>9}{'rate lim':>10}{'step':>8}"
          f"{'window s':>10}{'peak gap':>10}{'sigma':>10}{'current':>10}")
    current = dict(cfg.response.reward.sigma)
    changed = []
    for channel in channels:
        result = results[channel.name]
        existing = current.get(channel.name)
        print(f"{channel.name:<10}{channel.omega_n:>9.2f}{channel.rate_limit:>10.3f}"
              f"{result.amplitude:>8.3f}{result.window_s:>10.3f}{result.peak_gap:>10.4f}"
              f"{result.sigma:>10.4f}{existing:>10.4f}")
        if abs(result.sigma - existing) > 1e-3:
            changed.append(channel.name)

    print("\nconfig snippet for go1_gym/envs/config/wbc.py:\n")
    body = ", ".join(f'"{name}": {results[name].sigma:.4f}' for name in results)
    print(f'    "response.reward.sigma": {{{body}}},')

    if changed:
        print(f"\n[warn] differs from the config for: {', '.join(changed)} -- "
              "the config is stale, update it")
    else:
        print("\nconfig matches this calibration")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = {
            "target_discrimination": args.target,
            "faster_multiple": args.faster_multiple,
            "window_factor": args.window_factor,
            "dt": dt,
            "robot": args.robot,
            "channels": {
                name: {
                    "sigma": result.sigma,
                    "omega_n": next(c.omega_n for c in channels if c.name == name),
                    "rate_limit": next(c.rate_limit for c in channels if c.name == name),
                    "amplitude": result.amplitude,
                    "window_s": result.window_s,
                    "peak_gap": result.peak_gap,
                    "discrimination": result.discrimination,
                }
                for name, result in results.items()
            },
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
