"""Identify multiple RL-SAR policies, then evaluate ideal/selected MPC pairs.

Examples:
  python sysid/run_policy_identification_benchmark.py \
    --policies I_Q NH_D_s11 coord_I_26499 \
    --output tmp/experiments/policy_identification_benchmark

  python sysid/run_policy_identification_benchmark.py \
    --policies-file policies.txt --output tmp/experiments/policy_batch --resume
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sysid.identification_bundle import resolve_bundle

DEFAULT_STACK_ROOT = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
DEFAULT_LIBRARY = ROOT / "benchmark/data/frozen_trajectory_library2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_policy_file(path):
    """Read a JSON array or a text file containing one policy per line."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        values = json.loads(text)
        if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
            raise ValueError("JSON policy file must contain an array of strings")
        return values
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    with log_path.open("a") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {shlex.join(command)}\n")
        log.flush()
        return subprocess.run(command, cwd=ROOT, env=environment,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def prepare(args, policy_arguments):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stack = args.stack_root.resolve()
    robot_default = (args.robot_dir or stack / "rl_sar/policy/go2_x5").resolve()
    policies = []
    seen = set()
    for argument in policy_arguments:
        robot_dir, key = resolve_bundle(argument, robot_default)
        if key in seen:
            raise ValueError(f"duplicate policy key: {key}")
        seen.add(key)
        policies.append({
            "requested_argument": argument,
            "argument": str(robot_dir / key / "policy.pt"),
            "policy": key,
            "source_robot_dir": str(robot_dir),
            "source_files": {
                str(path): sha(path) for path in (
                    robot_dir / "base.yaml", robot_dir / key / "config.yaml",
                    robot_dir / key / "policy.pt")
            },
            "identification": str(output / "identification" / key),
        })
    library = args.library.resolve()
    library_files = {
        str(path): sha(path) for path in (
            library / "manifest.json", library / "trajectories.npz",
            library / "suite/trajectory_suite.json",
            library / "suite/reference.npz")
    }
    plan = {
        "schema": "multi-policy-identification-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policies": policies,
        "stack_root": str(stack),
        "library": str(library),
        "library_files": library_files,
        "sample_seed": args.seed,
        "sample_count": args.count,
        "identification_workers": args.identification_workers,
        "evaluation_workers": args.evaluation_workers,
        "evaluation": str(output / "evaluation"),
    }
    write_json(output / "plan.json", plan)
    write_json(output / "state.json", {
        "status": "prepared", "policies": {item["policy"]: "pending" for item in policies},
        "evaluation": "pending",
    })
    return plan


def verify_plan(plan):
    for path, digest in plan["library_files"].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError(f"trajectory library input changed: {path}")


def identification_command(plan, item):
    experiment = Path(item["identification"])
    snapshot_policy = experiment / "policy_bundle/go2_x5" / item["policy"] / "policy.pt"
    policy = str(snapshot_policy) if snapshot_policy.is_file() else item["argument"]
    command = [sys.executable, "-u", str(ROOT / "sysid/identify_policy.py"),
               "--policy", policy, "--output", str(experiment),
               "--stack-root", plan["stack_root"]]
    if experiment.exists():
        command.append("--resume")
    return command


def run_identification(plan, item, output):
    command = identification_command(plan, item)
    code = run_logged(command, output / "logs" / f"identify_{item['policy']}.log")
    state_path = Path(item["identification"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if code or state.get("status") != "all_complete":
        raise RuntimeError(
            f"identification failed for {item['policy']} (exit {code}); "
            f"see {output / 'logs' / ('identify_' + item['policy'] + '.log')}"
        )
    required = ["policy_bundle/manifest.json", "models_v2/selection.json",
                "mpc/manifest.json", "identification_quality/quality.json"]
    for relative in required:
        if not (Path(item["identification"]) / relative).is_file():
            raise RuntimeError(f"incomplete identification artifact: {item['identification']}/{relative}")
    return item["policy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policies", nargs="+", help="Policy keys, bundle directories, or policy.pt files")
    source.add_argument("--policies-file", type=Path, help="JSON array or one policy per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--robot-dir", type=Path)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--identification-workers", type=int, default=1)
    parser.add_argument("--evaluation-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.identification_workers <= 0 or args.evaluation_workers <= 0:
        parser.error("counts and worker counts must be positive")
    policy_arguments = args.policies or load_policy_file(args.policies_file)
    if not policy_arguments:
        parser.error("policy list is empty")
    output = args.output.resolve()
    plan_path = output / "plan.json"
    if args.resume:
        if not plan_path.is_file():
            raise FileNotFoundError(f"resume plan does not exist: {plan_path}")
        plan = json.loads(plan_path.read_text())
        requested = [str(x) for x in policy_arguments]
        frozen_requested = [item.get("requested_argument", item["argument"])
                            for item in plan["policies"]]
        if requested != frozen_requested:
            raise ValueError("resume policy list/order differs from the frozen plan")
    else:
        plan = prepare(args, [str(x) for x in policy_arguments])
    verify_plan(plan)
    if args.prepare_only:
        print(f"Prepared {len(plan['policies'])} policies: {plan_path}")
        return 0
    state_path = output / "state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "identifying"
    write_json(state_path, state)
    pending = []
    for item in plan["policies"]:
        pipeline_state = Path(item["identification"]) / "pipeline_state.json"
        complete = (pipeline_state.is_file() and
                    json.loads(pipeline_state.read_text()).get("status") == "all_complete")
        if complete:
            state["policies"][item["policy"]] = "complete"
        else:
            pending.append(item)
    write_json(state_path, state)
    with ThreadPoolExecutor(max_workers=plan["identification_workers"]) as executor:
        futures = {executor.submit(run_identification, plan, item, output): item for item in pending}
        try:
            for future in as_completed(futures):
                item = futures[future]
                future.result()
                state["policies"][item["policy"]] = "complete"
                write_json(state_path, state)
        except Exception as error:
            state.update(status="failed", error=str(error))
            write_json(state_path, state)
            raise
    experiments = [item["identification"] for item in plan["policies"]]
    evaluation = Path(plan["evaluation"])
    command = [sys.executable, "-u", str(ROOT / "sysid/run_policy_library_benchmark.py"),
               "--experiments", *experiments, "--library", plan["library"],
               "--output", str(evaluation), "--seed", str(plan["sample_seed"]),
               "--count", str(plan["sample_count"]), "--workers", str(plan["evaluation_workers"])]
    if evaluation.exists():
        command.append("--resume")
    state.update(status="evaluating", evaluation="running")
    write_json(state_path, state)
    code = run_logged(command, output / "logs/evaluation.log")
    completion = evaluation / "completion.json"
    evidence = json.loads(completion.read_text()) if completion.is_file() else {}
    if code or evidence.get("status") != "complete":
        state.update(status="failed", evaluation="failed",
                     error=f"evaluation failed (exit {code}); see {output / 'logs/evaluation.log'}")
        write_json(state_path, state)
        raise RuntimeError(state["error"])
    verify_plan(plan)
    state.update(status="complete", evaluation="complete", error=None,
                 report=str(evaluation / "REPORT.md"), results=str(evaluation / "results.json"))
    write_json(state_path, state)
    print(f"Complete: {evaluation / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Batch stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
