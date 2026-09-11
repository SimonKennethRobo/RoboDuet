"""Independent open-loop excitation; neither MPC commands nor hold-push data are fitted."""
import argparse
import json
from pathlib import Path

from benchmark.dog_policy.servo_runtime import ServoPlant, DEFAULT_POLICY, NOMINAL_Q
import numpy as np


def trial_specs():
    # Entire postures and random phases are held out, not individual rows.
    postures = {
        "train": [[0., .6, .6, 0., 0., 0.], [.25, .9, .85, .2, -.2, .1], [-.25, .75, 1., -.2, .25, -.1]],
        "validation": [[.12, .8, .7, -.12, .12, .15]],
        "test": [[-.15, 1., .75, .15, .15, -.2], [.3, .65, .95, -.15, -.15, .2]],
    }
    return [dict(split=split, posture=q, channel=channel, seed=1000 + 100 * si + 17 * pi + ci)
            for si, (split, poses) in enumerate(postures.items()) for pi, q in enumerate(poses)
            for ci, channel in enumerate([f"arm{j}" for j in range(6)] + [f"base{j}" for j in range(5)] + ["combined"])]


def collect(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    specs = trial_specs()
    if args.confirmation:
        # Prospective trajectories collected only AFTER the model bundle is frozen.
        poses = [[-.1, .7, .8, .1, -.1, .12], [.22, .95, .9, -.1, .2, -.15]]
        specs = [dict(split="test", posture=q, channel=c, seed=8000 + 37*pi + ci)
                 for pi, q in enumerate(poses)
                 for ci, c in enumerate([f"arm{j}" for j in range(6)] + [f"base{j}" for j in range(5)] + ["combined"])]
    plant = ServoPlant(len(specs), args.seed, args.seconds, args.logdir, args.device)
    alive = np.ones(len(specs), dtype=bool)
    try:
        centers = np.array([s["posture"] for s in specs])
        amplitude = np.array([.16, .13, .13, .2, .2, .2])
        if np.any(centers - amplitude < plant.q_low) or np.any(centers + amplitude > plant.q_high):
            raise ValueError("Excitation postures do not fit the runtime joint envelope")
        for k in range(round(4. / plant.dt)):
            alpha = min((k + 1) * plant.dt / 2., 1.)
            done, _ = plant.step(np.zeros(5), target=NOMINAL_Q + alpha * (centers - NOMINAL_Q))
            alive &= ~done
        phases = np.array([np.random.default_rng(s["seed"]).uniform(-np.pi, np.pi, (11, 3)) for s in specs])
        arm_mask = np.array([[s["channel"] in (f"arm{j}", "combined") for j in range(6)] for s in specs])
        base_mask = np.array([[s["channel"] in (f"base{j}", "combined") for j in range(5)] for s in specs])
        rows = {k: [] for k in ("q", "dq", "response", "root", "ee", "base_angular_velocity", "q_target", "command", "valid", "done")}
        max_target_error = 0.
        for k in range(round(args.seconds / plant.dt)):
            t = k * plant.dt
            before = plant.state()
            # Independent multisine phases and split-specific frequencies; smooth startup.
            freq = np.array([.17, .43, .83])[None, None, :] * np.array(
                [1. if s["split"] == "train" else (1.13 if s["split"] == "validation" else .91) for s in specs])[:, None, None]
            phase_time = t + .006*t*t if args.confirmation else t
            wave = np.sum(np.sin(2 * np.pi * freq * phase_time + phases) * np.array([.5, .3, .2]), axis=-1)
            wave *= min(t / 1., 1.)
            target = centers + amplitude * wave[:, 5:] * arm_mask
            command = wave[:, :5] * np.array([.25, .18, .3, .025, .12]) * base_mask
            # Exercise exactly the persistent velocity-to-position interface.
            velocity = (target - plant.q_target) / plant.dt
            done, applied = plant.step(command, velocity=velocity)
            max_target_error = max(max_target_error, plant.last_target_error)
            alive &= ~done
            for key in before:
                if key in rows:
                    rows[key].append(before[key])
            rows["q_target"].append(applied)
            rows["command"].append(command)
            rows["valid"].append(alive.copy())
            rows["done"].append(done)
            if k % 250 == 0:
                print(f"ID t={t:.2f}s valid={alive.sum()}/{len(specs)} target_error={max_target_error:.3g}", flush=True)
        # Last measured sample closes the final transition.
        after = plant.state()
        for key in ("q", "dq", "response", "root", "ee", "base_angular_velocity"):
            rows[key].append(after[key])
        np.savez_compressed(output / "excitation.npz", **{k: np.asarray(v) for k, v in rows.items()})
        manifest = dict(plant.manifest, trials=specs, seconds=args.seconds, prospective_confirmation=args.confirmation,
                        max_applied_target_error_rad=max_target_error, surviving_trials=int(alive.sum()),
                        identification_boundary="command interface + fixed RL policy + arm drive + robot; no MPC")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"SAVED {output} ({alive.sum()}/{len(specs)} trials survived)", flush=True)
    finally:
        plant.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=24.)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--logdir", default=DEFAULT_POLICY)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--confirmation", action="store_true")
    args = p.parse_args()
    if args.seconds <= 0:
        p.error("seconds must be positive")
    collect(args)
