"""Paired ideal/identified MPC evaluation on a seeded frozen-library sample."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.wbc.mujoco import FrozenReference
from benchmark.wbc.trace import score_trace_archive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checked_copy(source, target, digest):
    source, target = Path(source), Path(target)
    if sha(source) != digest:
        raise ValueError(f"input hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def prepare(args):
    out = args.output.resolve()
    if out.exists():
        if (out / "plan.json").exists():
            raise FileExistsError(f"use --resume with existing output: {out}")
        unexpected = {path.name for path in out.iterdir()} - {"suite", "inputs", "source"}
        if unexpected:
            raise FileExistsError(
                f"output contains files unrelated to an interrupted preparation: {out}"
            )
    library = args.library.resolve()
    suite_path = library / "suite/trajectory_suite.json"
    suite = json.loads(suite_path.read_text())
    for filename, key in [("manifest.json", "manifest_sha256"), ("trajectories.npz", "archive_sha256")]:
        if sha(library / filename) != suite["source_library"][key]:
            raise ValueError(f"suite does not match source library: {filename}")
    selected = random.Random(args.seed).sample(suite["trajectories"], args.count)
    for task in selected:
        FrozenReference(suite_path, task["task_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite").mkdir(exist_ok=True)
    checked_copy(suite_path, out / "suite/trajectory_suite.json", sha(suite_path))
    checked_copy(suite_path.parent / suite["reference_archive"]["path"],
                 out / "suite" / suite["reference_archive"]["path"], suite["reference_archive"]["sha256"])
    policies = []
    monitored = {}
    for source_root in args.experiments:
        source_root = source_root.resolve()
        manifest = json.loads((source_root / "mpc/manifest.json").read_text())
        spec = json.loads((source_root / "protocol.json").read_text())
        inputs = json.loads((source_root / "input_manifest.json").read_text())["files"]
        key = manifest["policy"]
        manifest_bundle = Path(manifest["robot_dir"])
        local_bundle = source_root / "policy_bundle/go2_x5"
        bundle = local_bundle if local_bundle.is_dir() else manifest_bundle
        frozen = out / "inputs" / key
        robot = frozen / "rl_sar/policy/go2_x5"
        base = bundle / "base.yaml"
        expected_base = inputs[str(manifest_bundle / "base.yaml")]
        base_bytes = base.read_bytes()
        base_origin = "current_file"
        if hashlib.sha256(base_bytes).hexdigest() != expected_base:
            if bundle == local_bundle:
                raise ValueError(f"experiment-local policy snapshot changed: {base}")
            # Recover the exact collected config into this experiment only.
            repo = Path(manifest["stack_root"]) / "rl_sar"
            relative = base.relative_to(repo).as_posix()
            commits = subprocess.check_output(
                ["git", "-C", str(repo), "log", "-30", "--format=%H", "--", relative], text=True).split()
            for commit in commits:
                candidate = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{relative}"])
                if hashlib.sha256(candidate).hexdigest() == expected_base:
                    base_bytes, base_origin = candidate, f"git:{commit}:{relative}"
                    break
            else:
                raise ValueError(f"cannot recover collected base config: {source_root}")
        robot.mkdir(parents=True, exist_ok=True)
        (robot / "base.yaml").write_bytes(base_bytes)
        for name in ["policy.pt", "config.yaml"]:
            source = bundle / key / name
            checked_copy(source, robot / key / name,
                         inputs[str(manifest_bundle / key / name)])
        selected_model = manifest["models"]["selected"]
        tasks = {}
        for mode in ["ideal", "selected"]:
            source = source_root / "mpc" / f"task_{key}_{mode}.info"
            digest = manifest["ideal_task_sha256"] if mode == "ideal" else selected_model["task_sha256"]
            tasks[mode] = checked_copy(source, frozen / source.name, digest)
        # Manifests retain collection-time absolute paths. Resolve models from
        # the experiment root so a complete identification directory remains
        # relocatable as one unit.
        selected_model_path = source_root / "models_v2" / f"{selected_model['model_name']}.json"
        checked_copy(selected_model_path, frozen / "selected_model.json",
                     selected_model["model_sha256"])
        for name in ["mpc/manifest.json", "protocol.json", "models_v2/selection.json", "input_manifest.json"]:
            checked_copy(source_root / name, frozen / name, sha(source_root / name))
        if sha(source_root / "protocol.json") != manifest["protocol_sha256"]:
            raise ValueError("protocol changed after export")
        if sha(source_root / "models_v2/selection.json") != manifest["selection_sha256"]:
            raise ValueError("selection changed after export")
        scene = Path(spec["scene"])
        for source in scene.parent.glob("*.xml"):
            if sha(source) != inputs[str(source)]:
                raise ValueError(f"identified scene changed: {source}")
        for source in scene.parent.rglob("*"):
            if source.is_file():
                monitored[str(source)] = sha(source)
        policies.append(dict(policy=key, source=str(source_root), selected_model=selected_model["model_name"],
            rl_sar_root=str(frozen / "rl_sar"), stack_root=manifest["stack_root"], scene=str(scene),
            tasks=tasks, command_limits=manifest["command_limits"], base_config_origin=base_origin,
            source_task_sha256=manifest["source_task_sha256"]))
    source_origins = {}
    for source in [Path(__file__), ROOT / "benchmark/__init__.py", ROOT / "benchmark/wbc/__init__.py",
                   ROOT / "benchmark/wbc/mujoco.py", ROOT / "benchmark/wbc/controllers.py",
                   ROOT / "benchmark/wbc/trace.py", ROOT / "benchmark/wbc/scoring.py", ROOT / "benchmark/wbc/suite.py",
                   ROOT / "scripts/sim2sim_mujoco.py", ROOT / "scripts/rl_sar_obs.py"]:
        target = out / "source" / source.relative_to(ROOT)
        checked_copy(source, target, sha(source))
        monitored[str(target)] = sha(target)
        source_origins[str(source)] = sha(target)
    for policy in policies:
        stack = Path(policy["stack_root"])
        for relative in ["ros2_ws/install/go2_x5_ocs2_bridge/lib/go2_x5_ocs2_bridge/wbc_benchmark_sync",
                         "ros2_ws/install/ocs2_mobile_manipulator/lib/libocs2_mobile_manipulator.a",
                         "go2_x5_description/urdf/arx5_ac1_floating.urdf"]:
            monitored[str(stack / relative)] = sha(stack / relative)
    for source in (out / "inputs").rglob("*"):
        if source.is_file():
            monitored[str(source)] = sha(source)
    for source in (out / "suite").iterdir():
        monitored[str(source)] = sha(source)
    plan = dict(schema="policy-library-paired-v1", created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        seed=args.seed, sampling="random.Random(seed).sample, without replacement, source suite order",
        library=str(library), suite=str(out / "suite/trajectory_suite.json"), suite_sha256=suite["suite_sha256"],
        reference_sha256=suite["reference_archive"]["sha256"], evaluation_protocol=suite["evaluation_protocol"],
        initial_state="same existing I_Q-settled TaskSpec state for every policy; histories and gait clock reset",
        tasks=selected, policies=policies, requested_trials=2 * len(policies) * len(selected),
        reference_window_s=1., reference_window_dt_s=.02, runtime_hashes=monitored,
        runtime_root=str(out / "source"), source_original_hashes=source_origins,
        scope="paired exported configurations; coord_I_26499 retains its different arm cost/posture weights; no retuning")
    write_json(out / "plan.json", plan)
    return plan


def verify_inputs(plan):
    for path, digest in plan["runtime_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"frozen runtime input changed: {path}")


def trial(args, plan, policy, task, mode, smoke=False):
    directory = args.output.resolve() / ("smoke" if smoke else "runs") / policy["policy"] / mode / task["source_trajectory_id"]
    record_path = directory / "execution.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        for name, digest in record.get("artifacts", {}).items():
            if sha(directory / name) != digest:
                raise ValueError(f"existing trial artifact changed: {directory / name}")
        return record
    if directory.exists():
        raise FileExistsError(f"unfinished trial retained; choose a new output or inspect: {directory}")
    directory.mkdir(parents=True)
    command = [sys.executable, "-m", "benchmark.wbc.mujoco", "--suite", plan["suite"],
        "--task-id", task["task_id"], "--rl-sar-root", policy["rl_sar_root"], "--policy-key", policy["policy"],
        "--scene", policy["scene"], "--upper-controller", "floating_base_ocs2_mpc", "--ocs2-root", policy["stack_root"],
        "--ocs2-task-profile", "native_ideal", "--ocs2-task-file", policy["tasks"][mode],
        "--ocs2-command-mode", "full", "--ocs2-transport", "synchronous", "--ocs2-timeout-s", "300",
        "--ocs2-command-limits", *map(str, policy["command_limits"]), "--seed", str(plan["seed"]),
        "--output", str(directory)]
    if smoke:
        command += ["--max-steps", "2"]
    print(f"START {'smoke ' if smoke else ''}{policy['policy']} {mode} {task['source_trajectory_id']}", flush=True)
    begin = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    timeout = False
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=plan["runtime_root"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.wall_timeout_s)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    result = receipt.get("result") or {}
    trace_path = directory / "trace.npz"
    if trace_path.is_file():
        rescored = score_trace_archive(trace_path)[0]
        if rescored != result:
            raise ValueError(f"offline scorer differs from receipt: {directory}")
        if receipt["suite_sha256"] != plan["suite_sha256"] or receipt["task_id"] != task["task_id"]:
            raise ValueError("trial TaskSpec identity mismatch")
    record = dict(policy=policy["policy"], mode=mode, model=policy["selected_model"] if mode == "selected" else "ideal",
        task_id=task["task_id"], trajectory=task["source_trajectory_id"], directory=str(directory),
        returncode=code, wall_timeout=timeout, wall_seconds=time.monotonic()-begin, command=command,
        operational_failure=code != 0 or receipt.get("status") != "complete" or not trace_path.is_file(),
        error=receipt.get("error"), recorded_steps=receipt.get("recorded_steps", 0),
        deadline_steps=receipt.get("deadline_steps"), result=result,
        artifacts={p.name: sha(p) for p in [receipt_path, trace_path, directory / "trace.partial.npz", directory / "run.log"] if p.is_file()})
    write_json(record_path, record)
    print(f"END {policy['policy']} {mode} {task['source_trajectory_id']} exit={code} "
          f"reason={result.get('end_reason', record['error'])} pos={result.get('ee_pos_rmse_m')}", flush=True)
    return record


def mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def report(out, plan):
    records = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/*/*/execution.json"))]
    groups, pairs = [], []
    for policy in plan["policies"]:
        key = policy["policy"]
        for mode in ["ideal", "selected"]:
            rows = [r for r in records if r["policy"] == key and r["mode"] == mode]
            groups.append(dict(policy=key, mode=mode, model=policy["selected_model"] if mode == "selected" else "ideal",
                requested=len(plan["tasks"]), finished=len(rows), successes=sum(r["result"].get("success", False) for r in rows),
                falls=sum(r["result"].get("fall", False) for r in rows),
                numerical_faults=sum(r["result"].get("numerical_fault", False) for r in rows),
                operational_failures=sum(r["operational_failure"] for r in rows),
                timeouts=sum(r["result"].get("end_reason") == "timeout" for r in rows),
                metrics_available=sum(bool(r["result"]) for r in rows),
                ee_pos_rmse_m=mean(r["result"].get("ee_pos_rmse_m") for r in rows),
                ee_rot_rmse_rad=mean(r["result"].get("ee_rot_rmse_rad") for r in rows),
                tracking_tube_fraction=mean(r["result"].get("tracking_tube_fraction") for r in rows),
                whole_body_abs_mechanical_power_mean_w=mean(
                    r["result"].get("whole_body_abs_mechanical_power_mean_w") for r in rows),
                leg_torque_rms_nm=mean(r["result"].get("leg_torque_rms_nm") for r in rows),
                ee_jerk_mean=mean((r["result"].get("smoothness") or {}).get("ee_jerk", {}).get("mean")
                                  for r in rows),
                base_jerk_mean=mean((r["result"].get("smoothness") or {}).get("base_jerk", {}).get("mean")
                                    for r in rows),
                arm_joint_jerk_mean=mean(
                    (r["result"].get("smoothness") or {}).get("arm_joint_jerk", {}).get("mean")
                    for r in rows)))
        for task in plan["tasks"]:
            rows = {r["mode"]: r for r in records if r["policy"] == key and r["task_id"] == task["task_id"]}
            if len(rows) != 2:
                continue
            pair = dict(policy=key, trajectory=task["source_trajectory_id"],
                        ideal_success=rows["ideal"]["result"].get("success", False),
                        selected_success=rows["selected"]["result"].get("success", False))
            paths = {k: Path(v["directory"]) / "trace.npz" for k, v in rows.items()}
            if all(p.is_file() for p in paths.values()):
                traces = {k: np.load(p, allow_pickle=False) for k, p in paths.items()}
                n = min(len(t["sample_present"]) for t in traces.values())
                pair["common_steps"] = n
                for mode, trace in traces.items():
                    error = trace["actual_ee_state"][:n, 0, :3] - trace["reference_ee_position_m"][:n, 0]
                    pair[mode + "_common_pos_rmse_m"] = float(np.sqrt(np.mean(np.sum(error.astype(float)**2, axis=-1))))
                    trace.close()
                a, b = pair["ideal_common_pos_rmse_m"], pair["selected_common_pos_rmse_m"]
                pair["common_pos_rmse_delta_m"] = b-a
                pair["common_pos_improvement_pct"] = 100*(a-b)/a if a else None
            pairs.append(pair)
    complete = len(records) == plan["requested_trials"]
    data = dict(status="complete" if complete else "running", requested=plan["requested_trials"], finished=len(records),
                groups=groups, pairs=pairs, records=records)
    write_json(out / "results.json", data)
    with (out / "per_trial.csv").open("w") as stream:
        fields = ["policy", "mode", "model", "trajectory", "returncode", "operational_failure", "recorded_steps", "deadline_steps",
                  "success", "end_reason", "fall", "numerical_fault", "ee_pos_rmse_m", "ee_rot_rmse_rad", "tracking_tube_fraction",
                  "ee_jerk_mean", "base_jerk_mean", "arm_joint_jerk_mean",
                  "whole_body_abs_mechanical_power_mean_w", "leg_torque_rms_nm"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for r in records:
            row = {**r, **r["result"]}
            smoothness = r["result"].get("smoothness") or {}
            for name in ("ee_jerk", "base_jerk", "arm_joint_jerk"):
                row[name + "_mean"] = (smoothness.get(name) or {}).get("mean")
            writer.writerow({k: row.get(k) for k in fields})
    lines = ["# 多策略：辨识模型开关成对闭环测试", "", f"完成 {len(records)}/{plan['requested_trials']}；抽样种子 {plan['seed']}，无放回随机抽取。", "",
        "共用原库 TaskSpec、初始物理状态、参考轨迹和时限；每次重置策略历史和步态时钟。50 Hz 同步原生 OCS2，1 秒参考预览，统一部署命令限幅。",
        "", "启用辨识使用各实验 development 选出的 selected 模型，关闭使用同一实验导出的 ideal。保留各实验原有 MPC 导出参数；跨 policy 的基础代价可能不同，因此重点比较各策略内的开关效果。",
        "", f"成功按现有开发阈值：位置 3 cm、姿态 5°、80% 时间处于跟踪管道、终点进度至少 0.99、保持 0.5 秒。跌倒和运行失败均保留在每组 {len(plan['tasks'])} 条分母。",
        "", "RMSE 为逐轨迹等权平均，包含有 trace 的失败轨迹；早停轨迹只覆盖实际运行区间，不能单独据此评价整体优劣。共同前缀的成对位置 RMSE 位于 results.json/pairs。",
        "", "| Policy | 模式 | 模型 | 完成 | 成功 | 跌倒 | 数值故障 | 运行失败 | 位置 RMSE (m) | 姿态 RMSE (rad) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        fmt = lambda x: f"{x:.4f}" if x is not None else "—"
        lines.append(f"| {g['policy']} | {g['mode']} | {g['model']} | {g['finished']}/{g['requested']} | {g['successes']}/{g['requested']} | {g['falls']} | {g['numerical_faults']} | {g['operational_failures']} | {fmt(g['ee_pos_rmse_m'])} | {fmt(g['ee_rot_rmse_rad'])} |")
    lines += ["", "## 平滑度与能耗", "",
              "各指标为逐轨迹等权平均；jerk 越低越平滑。末端／机身单位为 m/s³，机械臂关节为 rad/s³。", "",
              "| Policy | 模式 | 末端 jerk | 机身 jerk | 机械臂 jerk | 整机功率 (W) | 腿扭矩 RMS (Nm) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        fmt = lambda x: f"{x:.3f}" if x is not None else "—"
        lines.append(f"| {g['policy']} | {g['mode']} | {fmt(g['ee_jerk_mean'])} | {fmt(g['base_jerk_mean'])} | {fmt(g['arm_joint_jerk_mean'])} | {fmt(g['whole_body_abs_mechanical_power_mean_w'])} | {fmt(g['leg_torque_rms_nm'])} |")
    lines += ["", "## 抽中的轨迹", ""] + [f"- {t['source_trajectory_id']}：参考 {t['duration_s']:.3f} s，截止 {t['deadline_s']:.3f} s；`{t['task_id']}`" for t in plan["tasks"]]
    lines += ["", "## 证据", "", "- `plan.json`：输入散列、随机抽样、完整 TaskSpec、策略和模型来源。",
              "- `inputs/`：冻结部署包及原始 MPC 导出；旧 base.yaml 从匹配散列的 Git 提交恢复到本实验目录。",
              "- `runs/<policy>/<ideal|selected>/<trajectory>/`：进程退出码、日志、receipt、trace 及散列。",
              f"- `results.json`、`per_trial.csv`：离线重新评分后汇总；`smoke/` 不计入正式 {plan['requested_trials']} 次。",
              "- 当前证据仅为这一随机样本上的平地 MuJoCo 闭环结果。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "benchmark/data/frozen_trajectory_library2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wall-timeout-s", type=float, default=900.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    plan = json.loads((out / "plan.json").read_text()) if args.resume else prepare(args)
    verify_inputs(plan)
    if args.report_only:
        report(out, plan)
        return
    if args.prepare_only:
        print(f"Prepared {plan['requested_trials']} trials in {out}")
        return
    # Sequential first use avoids races while CppAD compiles shared caches.
    for policy in plan["policies"]:
        for mode in ["ideal", "selected"]:
            record = trial(args, plan, policy, plan["tasks"][0], mode, smoke=True)
            if record["operational_failure"] or record["recorded_steps"] != 2:
                raise RuntimeError(f"preflight failed: {record['directory']}")
    if args.smoke_only:
        return
    def run_policy(policy):
        for task in plan["tasks"]:
            for mode in ["ideal", "selected"]:
                yield trial(args, plan, policy, task, mode)
    # One lane per policy; every lane executes the same task order.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(lambda p=p: list(run_policy(p))) for p in plan["policies"]]
        for future in as_completed(futures):
            future.result()
            report(out, plan)
    verify_inputs(plan)
    final = report(out, plan)
    write_json(out / "completion.json", dict(status=final["status"], finished=final["finished"],
        requested=final["requested"], runtime_inputs_verified=True, results_sha256=sha(out / "results.json")))
    print(f"COMPLETE {final['finished']}/{final['requested']} {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
