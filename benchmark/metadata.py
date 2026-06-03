"""Metadata helpers for benchmark artifacts."""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit


def _git_output(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=Path.cwd(), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _redact_url_credentials(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"<redacted>@{host}", parts.path, parts.query, parts.fragment))


def _redact_remote_line(line: str) -> str:
    parts = line.split()
    return " ".join(_redact_url_credentials(part) if "://" in part else part for part in parts)


def git_snapshot() -> Dict[str, Any]:
    status = _git_output(["status", "--short"])
    remotes = _git_output(["remote", "-v"])
    return {
        "commit": _git_output(["rev-parse", "HEAD"]) or "unknown",
        "short_commit": _git_output(["rev-parse", "--short", "HEAD"]) or "unknown",
        "branch": _git_output(["branch", "--show-current"]) or "unknown",
        "dirty": bool(status),
        "status_short": status or "",
        "remotes": [_redact_remote_line(line) for line in remotes.splitlines()] if remotes else [],
    }


def runtime_snapshot() -> Dict[str, Any]:
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    try:
        import torch

        runtime.update(
            {
                "torch": getattr(torch, "__version__", "unknown"),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            runtime["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        runtime["torch_error"] = str(exc)
    return runtime


def build_benchmark_metadata(
    *,
    mode: str,
    protocol: str,
    profile: str,
    candidate_dir: str,
    names: List[str],
    logdirs: List[str],
    ckptids: List[str],
    args: Any,
    total_envs: int,
    control_dt_s: Any,
    layout: Any,
    command_argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    git = git_snapshot()
    command_argv = list(sys.argv if command_argv is None else command_argv)
    candidates = [
        {
            "name": name,
            "logdir": logdir,
            "ckptid": ckptid,
        }
        for name, logdir, ckptid in zip(names, logdirs, ckptids)
    ]

    metadata = {
        "benchmark_mode": mode,
        "benchmark_protocol": protocol,
        "benchmark_version": protocol,
        "profile": profile,
        "candidate_dir": candidate_dir,
        "generated_at": generated_at,
        "git_commit": git["short_commit"],
        "runs": names,
        "logdirs": logdirs,
        "ckptids": ckptids,
        "num_envs_per_policy": args.num_envs_per_policy,
        "total_envs": total_envs,
        "num_eval_steps": args.num_eval_steps,
        "seed": args.seed,
        "control_dt_s": control_dt_s,
        "headless": args.headless,
        "robot": args.robot,
        "sim_device": args.sim_device,
        "arm_intensity": args.arm_intensity,
        "scenario_config": getattr(args, "scenario_config", {}),
        "profile_description": getattr(args, "profile_data", {}).get("description") if hasattr(args, "profile_data") else None,
        "dog_num_commands": getattr(layout, "n_dims", "unknown"),
        "use_dynamic_gait": bool(getattr(layout, "has_dynamic_gait", False)),
        "scenario_d_enabled": bool(getattr(layout, "has_dynamic_gait", False)) and not args.skip_d,
        "benchmark": {
            "mode": mode,
            "protocol": protocol,
            "profile": profile,
            "entrypoint": "python -m benchmark.cli",
            "command": shlex.join(command_argv),
            "generated_at": generated_at,
            "output_dir": os.path.abspath(args.output_dir),
        },
        "git": git,
        "runtime": runtime_snapshot(),
        "candidates": candidates,
        "scenarios": {
            "vel_grid": not args.skip_a,
            "arm_sweep": not args.skip_b,
            "body_pose": not args.skip_c,
            "gait": bool(getattr(layout, "has_dynamic_gait", False)) and not args.skip_d,
        },
    }
    return metadata
