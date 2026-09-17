"""One-command RL-SAR policy collection, response fitting and OCS2 .info export.

Example:
  python sysid/identify_policy.py --policy I_Q --output tmp/experiments/iq_new
  python sysid/identify_policy.py --policy /path/go2_x5/MY_POLICY --output tmp/my_policy

The policy must already be exported for the supported Go2-X5 deployment contract.
Use --stage check or --smoke before starting the complete 72-episode experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sysid.identification_bundle import (
    load_bundle_contract,
    policy_argument_key,
    resolve_bundle,
    snapshot_policy_bundle,
    verify_bundle_snapshot,
)


# User-editable defaults. All relative output paths are relative to your shell.
DEFAULT_STACK_ROOT = Path("/home/simon/Projects/Simon/wbc_rl_mpc")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "tmp/experiments"
DEFAULT_MODEL_DIR = "models_v2"


def read_json(path):
    return json.loads(path.read_text())


def check_frozen_fit(root, sha):
    selection = read_json(root / DEFAULT_MODEL_DIR / "selection.json")
    if selection["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Frozen fit belongs to a different collection protocol")
    if not selection.get("model_hashes") or not selection.get("raw_hashes"):
        raise ValueError("Missing resumable fit provenance; choose a fresh experiment directory")
    for name, digest in selection["model_hashes"].items():
        if sha(root / DEFAULT_MODEL_DIR / f"{name}.json") != digest:
            raise ValueError(f"Frozen model changed: {name}")
    for episode, digest in selection["raw_hashes"].items():
        if sha(root / "raw" / f"{episode}.npz") != digest:
            raise ValueError(f"Frozen raw data changed: {episode}")
    if not (root / f"{DEFAULT_MODEL_DIR}_prediction_validation.json").is_file():
        raise ValueError("Frozen fit has no completed test report; inspect fit.log")


def execute_stage(name, command, root, state, write_json):
    log_path = root / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    print(f"\n[{name}] {shlex.join(command)}\nLog: {log_path}", flush=True)
    state.update(status="running", stage=name)
    write_json(root / "pipeline_state.json", state)
    with log_path.open("a") as log:
        log.write(f"\n[{datetime.now().isoformat()}] {shlex.join(command)}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    if result.returncode:
        raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log_path}")
    state["completed_stages"] = sorted(set(state["completed_stages"]) | {name})
    write_json(root / "pipeline_state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="RL-SAR key, bundle directory, or exported policy.pt")
    parser.add_argument("--output", type=Path, help="Fresh experiment directory; automatically named if omitted")
    parser.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    parser.add_argument("--robot-dir", type=Path, help="Defaults to STACK/rl_sar/policy/go2_x5")
    parser.add_argument("--scene", type=Path, help="Defaults to the stack's Go2-X5 scene.xml")
    parser.add_argument("--stage", choices=["all", "check", "collect", "fit", "report", "export"], default="all")
    parser.add_argument("--resume", action="store_true", help="Continue the same frozen experiment")
    parser.add_argument("--smoke", action="store_true", help="Collect only one 20-second episode; do not fit/export")
    args = parser.parse_args()
    if args.smoke and args.stage not in ("all", "collect"):
        parser.error("--smoke requires --stage all or collect")

    # Import the simulator after argparse, so --help works without MuJoCo/Torch.
    import torch
    from sysid.identify_iq_mujoco import IdentificationPlant, sha, write_json

    stack = args.stack_root.expanduser().resolve()
    output = (args.output or DEFAULT_RESULTS_DIR /
              f"identify_{policy_argument_key(args.policy)}_{datetime.now():%Y%m%d_%H%M%S_%f}").expanduser().resolve()
    snapshot_manifest = output / "policy_bundle/manifest.json"
    legacy_identity = output / "pipeline_inputs.json"
    if snapshot_manifest.is_file():
        if not args.resume:
            raise FileExistsError(f"Experiment exists: {output}; use --resume or a new --output")
        robot_dir, key, bundle_snapshot = verify_bundle_snapshot(
            output, policy_argument_key(args.policy))
    elif legacy_identity.is_file():
        if not args.resume:
            raise FileExistsError(f"Experiment exists: {output}; use --resume or a new --output")
        # Existing experiments predate bundle snapshots. Preserve their exact
        # identity instead of inserting a current bundle into old evidence.
        robot_dir, key = resolve_bundle(
            args.policy, args.robot_dir or stack / "rl_sar/policy/go2_x5")
        bundle_snapshot = None
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not an identification pipeline directory: {output}")
        source_robot_dir, key = resolve_bundle(
            args.policy, args.robot_dir or stack / "rl_sar/policy/go2_x5")
        robot_dir, key, bundle_snapshot = snapshot_policy_bundle(
            source_robot_dir, key, output)
    contract = load_bundle_contract(robot_dir, key)
    scene = (args.scene or stack / "rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml").expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    source_files = [ROOT / "sysid" / name for name in [
        "__init__.py", "identify_policy.py", "identification_bundle.py", "identify_iq_mujoco.py",
        "identification_models.py", "iq_response_model.py", "report_identification_quality.py",
        "run_iq_mpc.py"]]
    source_files += [ROOT / "scripts" / name for name in [
        "sim2sim_mujoco.py", "rl_sar_obs.py"]]
    inputs = source_files + [robot_dir / "base.yaml", robot_dir / key / "config.yaml",
                            robot_dir / key / "policy.pt", stack / "go2_x5_ocs2/config/task_floating.info"]
    inputs += sorted(scene.parent.glob("*.xml"))
    identity = dict(contract=contract, scene=str(scene), stack_root=str(stack),
                    files={str(p): sha(p) for p in inputs})
    if bundle_snapshot is not None:
        identity["policy_bundle_snapshot"] = bundle_snapshot
    identity_path = output / "pipeline_inputs.json"
    if identity_path.exists():
        if not args.resume:
            raise FileExistsError(f"Experiment exists: {output}; use --resume or a new --output")
        if read_json(identity_path) != identity:
            raise ValueError("Policy/config/scene/source changed; use a new experiment directory")
    elif output.exists() and any(path.name != "policy_bundle" for path in output.iterdir()):
        raise FileExistsError(f"Output is not an identification pipeline directory: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)
    state_path = output / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else dict(policy=key, output=str(output), completed_stages=[])
    print(f"Policy snapshot: {robot_dir/key}\nOutput: {output}\n"
          f"Observation: {contract['num_observations']} x {len(contract['history_indices'])}; "
          f"gait: {contract['gait_frequency_hz']:g} Hz", flush=True)
    torch.set_num_threads(1)
    try:
        if args.stage in ("check", "all", "collect"):
            state.update(status="running", stage="check")
            write_json(state_path, state)
            plant = IdentificationPlant(robot_dir, scene, 6101, key)
            failure = plant.failure()
            if failure:
                raise RuntimeError(f"Policy failed the standing preflight: {failure}")
            del plant
            print("Policy input/output and standing preflight passed.", flush=True)
        if args.stage in ("all", "collect"):
            command = [sys.executable, "-u", str(ROOT/"sysid/identify_iq_mujoco.py"), "collect",
                       "--output", str(output), "--robot-dir", str(robot_dir), "--policy-key", key, "--scene", str(scene)]
            if args.smoke:
                command += ["--split", "train", "--limit", "1"]
            execute_stage("collect", command, output, state, write_json)
            if args.smoke:
                receipt = read_json(output/"raw/train_000.json")
                if not receipt["success"]:
                    raise RuntimeError(f"Smoke episode failed: {receipt['failure']}")
        if not args.smoke and args.stage in ("all", "fit"):
            if (output / DEFAULT_MODEL_DIR / "selection.json").exists():
                check_frozen_fit(output, sha)
                print("Frozen fit verified; skipping refit.", flush=True)
            else:
                execute_stage("fit", [sys.executable, "-u", str(ROOT/"sysid/iq_response_model.py"),
                              "--root", str(output), "--model-dir", DEFAULT_MODEL_DIR], output, state, write_json)
        if not args.smoke and args.stage in ("all", "report"):
            check_frozen_fit(output, sha)
            report_path = output / "identification_quality/quality.json"
            if report_path.exists():
                report = read_json(report_path)
                if report["selection_sha256"] != sha(output / DEFAULT_MODEL_DIR / "selection.json"):
                    raise ValueError("Quality report belongs to a different frozen fit")
                print("Frozen prediction-quality report verified; skipping plotting.", flush=True)
            else:
                execute_stage("report", [sys.executable, "-u", str(ROOT/"sysid/report_identification_quality.py"),
                              "--root", str(output), "--model-dir", DEFAULT_MODEL_DIR], output, state, write_json)
        if not args.smoke and args.stage in ("all", "export"):
            check_frozen_fit(output, sha)
            manifest_path = output / "mpc/manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                for label, digest in {"ideal": manifest["ideal_task_sha256"], **{
                    name: item["task_sha256"] for name, item in manifest["models"].items()}}.items():
                    if sha(output / "mpc" / f"task_{key}_{label}.info") != digest:
                        raise ValueError(f"Exported .info was modified: {label}")
                if manifest["selection_sha256"] != sha(output / DEFAULT_MODEL_DIR / "selection.json"):
                    raise ValueError("Exported .info belongs to a different fit")
                print("Exported .info files verified; skipping export.", flush=True)
            else:
                execute_stage("export", [sys.executable, "-u", str(ROOT/"sysid/run_iq_mpc.py"), "export",
                              "--root", str(output), "--stack-root", str(stack)], output, state, write_json)
            print(f"\nMPC files: {output}/mpc/task_{key}_{{ideal,first_order,first_order_delay,gait,second_order,selected}}.info", flush=True)
        state.update(status="smoke_complete" if args.smoke else f"{args.stage}_complete")
        state.pop("error", None)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        write_json(state_path, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Identification stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
