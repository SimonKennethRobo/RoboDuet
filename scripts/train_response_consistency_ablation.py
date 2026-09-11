"""Launch a matched Stage-1 ablation without modifying the shared profile file."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


VARIANTS = {
    "reference_only": (0.0, 0.0, 0.0),
    "within_domain": (-20.0, -0.5, 0.0),
    "full_consistency": (-20.0, -0.5, -0.5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_learning_iterations", type=int, default=25000)
    parser.add_argument("--num_envs", type=int, default=4096)
    args = parser.parse_args()

    # IsaacGym must be imported before anything that imports torch.
    import isaacgym  # noqa: F401
    from go1_gym.envs.config import ConfigProfile
    from go1_gym.envs.config import wbc
    from go1_gym_learn.ppo_cse_automatic import RunnerArgs

    phase, steady, domain = VARIANTS[args.variant]
    overrides = {
        "reward_scales.ref_tracking": 2.0,
        "reward_scales.phase_variance": phase,
        "reward_scales.steady_gain": steady,
        "reward_scales.domain_consistency": domain,
    }
    profile = wbc.ROBODUET_PROFILE
    wbc.ROBODUET_PROFILE = ConfigProfile(
        name="response_ablation_" + args.variant,
        overrides={**profile.overrides, **overrides},
        allow_new=profile.allow_new,
    )
    # Preserve all response observations, identification groups, excitation,
    # reference tracking, reward handover and DR/push schedules in every arm.
    # Only the three auxiliary reward scales above differ between arms.
    RunnerArgs.save_interval = 1000
    RunnerArgs.log_video = False
    RunnerArgs.save_video_interval = 0

    root = Path(__file__).resolve().parents[1]
    run_name = "stage1_rc_ablation_{}_s{}".format(args.variant, args.seed)
    manifest = {
        "experiment": "e4-response-consistency-20260911",
        "variant": args.variant,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_learning_iterations": args.num_learning_iterations,
        "num_steps_per_env": 24,
        "train_stage": "stage1",
        "robot": "go2_x5",
        "domain_rand_mode": "benchmark",
        "from_scratch": True,
        "reward_overrides": overrides,
        "source_archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "profile_overrides": dict(wbc.ROBODUET_PROFILE.overrides),
        "interpretation": "Single-seed matched screening, not across-seed evidence.",
    }
    with (root / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("RESPONSE_ABLATION " + json.dumps({k: v for k, v in manifest.items()
                                            if k != "profile_overrides"}), flush=True)

    entrypoint = root / "scripts" / "auto_train.py"
    sys.argv = [
        str(entrypoint), "--headless", "--sim_device", "cuda:0",
        "--graphics_device_id", "0", "--train_stage", "stage1",
        "--robot", "go2_x5", "--dyna_gait", "--domain_rand_mode", "benchmark",
        "--num_envs", str(args.num_envs), "--num_steps_per_env", "24",
        "--num_learning_iterations", str(args.num_learning_iterations),
        "--seed", str(args.seed), "--run_name", run_name,
        "--notes", "Matched auxiliary-consistency ablation; " + args.variant,
        "--tags", "response-consistency-ablation", "e4-4090", args.variant,
    ]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
