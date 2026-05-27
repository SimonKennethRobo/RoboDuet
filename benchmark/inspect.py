"""Inspect RoboDuet run-like policy bundles without creating IsaacGym envs."""

from __future__ import annotations

import argparse
import json
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from benchmark.candidates import discover_run_logdirs


def _load_cfg(logdir: Path) -> Dict[str, Any]:
    path = logdir / "parameters.pkl"
    if not path.is_file():
        raise ValueError(f"{logdir}: missing parameters.pkl")
    with path.open("rb") as f:
        data = pkl.load(f)
    cfg = data.get("Cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a dict-like Cfg entry")
    return cfg


def _checkpoint_path(logdir: Path, policy: str, ckpt_id: str) -> Path:
    if policy == "dog":
        ckpt_name = "last_dog" if ckpt_id == "last" else ckpt_id.zfill(6)
        return logdir / "checkpoints_dog" / f"ac_weights_{ckpt_name}.pt"
    if policy == "arm":
        ckpt_name = "last_arm" if ckpt_id == "last" else ckpt_id.zfill(6)
        return logdir / "checkpoints_arm" / f"ac_weights_{ckpt_name}.pt"
    raise ValueError(f"unknown policy type: {policy}")


def _shape(ckpt: Dict[str, Any], key: str) -> Optional[List[int]]:
    value = ckpt.get(key)
    if not hasattr(value, "shape"):
        return None
    return list(value.shape)


def _load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu")


def _cfg_section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = cfg.get(name, {})
    return section if isinstance(section, dict) else {}


def _dog_cfg_dims(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dog = _cfg_section(cfg, "dog")
    return {
        "observations": dog.get("dog_num_observations"),
        "privileged_obs": dog.get("dog_num_privileged_obs"),
        "history_len": dog.get("dog_num_observation_history"),
        "obs_history": dog.get("dog_num_obs_history"),
        "actions": dog.get("dog_actions"),
        "commands": dog.get("dog_num_commands"),
        "adaptation_module": dog.get("use_adaptation_module", True),
    }


def _arm_cfg_dims(cfg: Dict[str, Any]) -> Dict[str, Any]:
    arm = _cfg_section(cfg, "arm")
    return {
        "observations": arm.get("arm_num_observations"),
        "privileged_obs": arm.get("arm_num_privileged_obs"),
        "history_len": arm.get("arm_num_observation_history"),
        "obs_history": arm.get("arm_num_obs_history"),
        "actions": arm.get("num_actions_arm_cd"),
        "commands": arm.get("arm_num_commands"),
        "adaptation_module": arm.get("use_adaptation_module", False),
    }


def _dog_ckpt_dims(ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if ckpt is None:
        return {"present": False}
    adapt = _shape(ckpt, "adaptation_module.0.weight")
    actor = _shape(ckpt, "actor_body.0.weight")
    critic = _shape(ckpt, "critic_body.0.weight")
    out = _shape(ckpt, "actor_body.6.weight")
    return {
        "present": True,
        "keys": len(ckpt),
        "adaptation_module": adapt is not None,
        "adaptation_input": adapt[1] if adapt else None,
        "adaptation_latent": _shape(ckpt, "adaptation_module.4.weight")[0] if adapt and _shape(ckpt, "adaptation_module.4.weight") else None,
        "actor_input": actor[1] if actor else None,
        "critic_input": critic[1] if critic else None,
        "actions": out[0] if out else None,
    }


def _arm_ckpt_dims(ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if ckpt is None:
        return {"present": False}
    adapt = _shape(ckpt, "adaptation_module.0.weight")
    adapt_out = _shape(ckpt, "adaptation_module.4.weight")
    hist = _shape(ckpt, "actor_history_encoder.0.weight")
    hist_out = _shape(ckpt, "actor_history_encoder.4.weight")
    actor = _shape(ckpt, "actor_body.0.weight")
    critic = _shape(ckpt, "critic_body.0.weight")
    out = _shape(ckpt, "actor_body.6.weight")
    return {
        "present": True,
        "keys": len(ckpt),
        "adaptation_module": adapt is not None,
        "adaptation_input": adapt[1] if adapt else None,
        "adaptation_latent": adapt_out[0] if adapt_out else None,
        "history_encoder_input": hist[1] if hist else None,
        "history_encoder_latent": hist_out[0] if hist_out else None,
        "actor_input": actor[1] if actor else None,
        "critic_input": critic[1] if critic else None,
        "actions": out[0] if out else None,
    }


def _dog_internal_checks(cfg_dims: Dict[str, Any], ckpt_dims: Dict[str, Any]) -> List[str]:
    if not ckpt_dims.get("present"):
        return ["dog checkpoint missing"]
    issues = []
    obs_history = cfg_dims.get("obs_history")
    privileged = cfg_dims.get("privileged_obs")
    actions = cfg_dims.get("actions")
    uses_adapt = ckpt_dims.get("adaptation_module")
    if cfg_dims.get("adaptation_module") != uses_adapt:
        issues.append(
            f"dog adaptation flag cfg={cfg_dims.get('adaptation_module')} checkpoint={uses_adapt}"
        )
    expected_actor = obs_history + privileged if uses_adapt and obs_history is not None and privileged is not None else obs_history
    expected_critic = obs_history + privileged if obs_history is not None and privileged is not None else None

    if obs_history is not None and ckpt_dims.get("adaptation_input") not in (None, obs_history):
        issues.append(f"dog adaptation input {ckpt_dims.get('adaptation_input')} != cfg obs_history {obs_history}")
    if expected_actor is not None and ckpt_dims.get("actor_input") != expected_actor:
        issues.append(f"dog actor input {ckpt_dims.get('actor_input')} != expected {expected_actor}")
    if expected_critic is not None and ckpt_dims.get("critic_input") != expected_critic:
        issues.append(f"dog critic input {ckpt_dims.get('critic_input')} != expected {expected_critic}")
    if actions is not None and ckpt_dims.get("actions") != actions:
        issues.append(f"dog actions {ckpt_dims.get('actions')} != cfg actions {actions}")
    return issues


def _arm_internal_checks(cfg_dims: Dict[str, Any], ckpt_dims: Dict[str, Any]) -> List[str]:
    if not ckpt_dims.get("present"):
        return ["arm checkpoint missing"]
    issues = []
    obs = cfg_dims.get("observations")
    obs_history = cfg_dims.get("obs_history")
    privileged = cfg_dims.get("privileged_obs")
    actions = cfg_dims.get("actions")
    hist_input = obs_history - obs if obs_history is not None and obs is not None else None
    hist_latent = ckpt_dims.get("history_encoder_latent")
    uses_adapt = ckpt_dims.get("adaptation_module")
    if cfg_dims.get("adaptation_module") != uses_adapt:
        issues.append(
            f"arm adaptation flag cfg={cfg_dims.get('adaptation_module')} checkpoint={uses_adapt}"
        )
    expected_actor = obs + hist_latent if obs is not None and hist_latent is not None else None
    if uses_adapt and expected_actor is not None and privileged is not None:
        expected_actor += privileged
    expected_critic = obs + privileged + hist_latent if obs is not None and privileged is not None and hist_latent is not None else None

    if obs_history is not None and ckpt_dims.get("adaptation_input") not in (None, obs_history):
        issues.append(f"arm adaptation input {ckpt_dims.get('adaptation_input')} != cfg obs_history {obs_history}")
    if hist_input is not None and ckpt_dims.get("history_encoder_input") != hist_input:
        issues.append(f"arm history input {ckpt_dims.get('history_encoder_input')} != expected {hist_input}")
    if expected_actor is not None and ckpt_dims.get("actor_input") != expected_actor:
        issues.append(f"arm actor input {ckpt_dims.get('actor_input')} != expected {expected_actor}")
    if expected_critic is not None and ckpt_dims.get("critic_input") != expected_critic:
        issues.append(f"arm critic input {ckpt_dims.get('critic_input')} != expected {expected_critic}")
    if actions is not None and ckpt_dims.get("actions") != actions:
        issues.append(f"arm actions {ckpt_dims.get('actions')} != cfg actions {actions}")
    return issues


def inspect_logdir(logdir: Path, ckpt_id: str) -> Dict[str, Any]:
    cfg = _load_cfg(logdir)
    dog_ckpt = _load_checkpoint(_checkpoint_path(logdir, "dog", ckpt_id))
    arm_ckpt = _load_checkpoint(_checkpoint_path(logdir, "arm", ckpt_id))
    dog_cfg = _dog_cfg_dims(cfg)
    arm_cfg = _arm_cfg_dims(cfg)
    dog_ckpt_dims = _dog_ckpt_dims(dog_ckpt)
    arm_ckpt_dims = _arm_ckpt_dims(arm_ckpt)
    dog_issues = _dog_internal_checks(dog_cfg, dog_ckpt_dims)
    arm_issues = _arm_internal_checks(arm_cfg, arm_ckpt_dims)
    return {
        "logdir": str(logdir),
        "ckpt_id": ckpt_id,
        "dog": {
            "cfg": dog_cfg,
            "checkpoint": dog_ckpt_dims,
            "internally_compatible": not dog_issues,
            "issues": dog_issues,
        },
        "arm": {
            "cfg": arm_cfg,
            "checkpoint": arm_ckpt_dims,
            "internally_compatible": not arm_issues,
            "issues": arm_issues,
        },
    }


def _discover_logdirs(candidate_dir: Path) -> List[Path]:
    return discover_run_logdirs(candidate_dir)


def _print_text(reports: List[Dict[str, Any]]):
    for report in reports:
        print(f"\n{report['logdir']}")
        print(f"  ckpt: {report['ckpt_id']}")
        for policy in ("dog", "arm"):
            section = report[policy]
            cfg = section["cfg"]
            ckpt = section["checkpoint"]
            print(f"  {policy}:")
            print(
                "    cfg: "
                f"obs={cfg.get('observations')} hist={cfg.get('obs_history')} "
                f"priv={cfg.get('privileged_obs')} actions={cfg.get('actions')} "
                f"commands={cfg.get('commands')} adapt={cfg.get('adaptation_module')}"
            )
            if ckpt.get("present"):
                extras = []
                for key in ("adaptation_module", "adaptation_input", "history_encoder_input", "actor_input", "critic_input", "actions"):
                    if key in ckpt and ckpt[key] is not None:
                        extras.append(f"{key}={ckpt[key]}")
                print("    ckpt: " + " ".join(extras))
            else:
                print("    ckpt: missing")
            verdict = "ok" if section["internally_compatible"] else "mismatch"
            print(f"    internal check: {verdict}")
            for issue in section["issues"]:
                print(f"      - {issue}")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Inspect run-like RoboDuet policy bundles")
    parser.add_argument("--logdirs", nargs="+", default=None, help="Run-like logdirs to inspect")
    parser.add_argument("--candidate_dir", default=None, help="Scan every run-like logdir under this candidate root")
    parser.add_argument("--ckptid", default="last", help="Checkpoint id to inspect, default: last")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text summary")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    logdirs: List[Path] = []
    if args.logdirs:
        logdirs.extend(Path(path) for path in args.logdirs)
    if args.candidate_dir:
        logdirs.extend(_discover_logdirs(Path(args.candidate_dir)))
    if not logdirs:
        raise ValueError("Provide --logdirs or --candidate_dir")

    reports = [inspect_logdir(path, args.ckptid) for path in logdirs]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        _print_text(reports)


if __name__ == "__main__":
    main()
