#!/usr/bin/env python3
"""Preview six independent runs; --launch starts them with separate logs/PIDs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from go1_gym.envs.config import build_roboduet_config, cfg_to_dict
from go1_gym.envs.config.coordination import DEFAULT_CONFIG, read_experiment


def make_jobs(config, experiments, gpus, python):
    if len(experiments) != len(gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Assign exactly one distinct GPU to each experiment")
    jobs = []
    for name, gpu in zip(experiments, gpus):
        spec = read_experiment(name, config)
        training = spec["training"]
        args = argparse.Namespace(num_envs=training["num_envs"], robot="go2_x5", train_stage="stage1",
                                  dyna_gait=True, experiment=name, experiment_config=str(config))
        cfg = build_roboduet_config(args)
        command = [python, "-u", str(ROOT / "scripts/auto_train.py"), "--experiment", name,
                   "--experiment_config", str(config), "--train_stage", "stage1", "--dyna_gait",
                   "--robot", "go2_x5", "--headless", "--no_video", "--sim_device", "cuda:0",
                   "--run_name", f"stage1_coord6_{name}"]
        jobs.append(dict(experiment=name, gpu=str(gpu), command=command, training=training, cfg=cfg_to_dict(cfg)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiments", nargs="+", choices=list("ABCDEF"), default=list("ABCDEF"))
    parser.add_argument("--gpus", nargs="+", default=list(map(str, range(6))))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--launch", action="store_true", help="Actually start training; omitted means preview only")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/coordination_launches")
    args = parser.parse_args()
    config = args.config.resolve()
    jobs = make_jobs(config, args.experiments, args.gpus, args.python)
    for job in jobs:
        if args.offline:
            job["command"].append("--offline")
        print(f"GPU {job['gpu']} / {job['experiment']}: {job['training']}")
        print(f"CUDA_VISIBLE_DEVICES={shlex.quote(job['gpu'])} " + shlex.join(job["command"]))
    if not args.launch:
        print("Preview only. Add --launch to start these runs.")
        return
    available = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    if any(job["gpu"] not in available for job in jobs):
        raise ValueError(f"Requested GPUs {args.gpus}; available indices {available}")
    output = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    frozen_config = output / "coordination_recipe.json"
    shutil.copyfile(config, frozen_config)
    for job in jobs:
        job["command"][job["command"].index("--experiment_config") + 1] = str(frozen_config)
    manifest = output / "manifest.json"
    # Record intent/config before launching; record each PID immediately after.
    manifest.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    for job in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpu"], OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        job["log"] = str(output / f"{job['experiment']}.log")
        with open(job["log"], "w") as log:
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        job["pid"] = process.pid
        manifest.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    print(f"Started processes; inspect startup/progress in {manifest}")


if __name__ == "__main__":
    main()
