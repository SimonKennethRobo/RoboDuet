import argparse
import dataclasses
import inspect


def _to_wandb_value(value):
    if isinstance(value, argparse.Namespace):
        return _to_wandb_value(vars(value))

    if inspect.isclass(value):
        return _class_to_dict(value)

    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_wandb_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }

    if isinstance(value, dict):
        return {str(k): _to_wandb_value(v) for k, v in value.items() if _include_config_key(str(k), v)}

    if isinstance(value, (list, tuple)):
        return [_to_wandb_value(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return str(value)


def _include_config_key(key, value):
    if key.startswith("__"):
        return False
    if key in {"cli", "cli_prefix", "proto", "subgroups"}:
        return False
    if inspect.ismodule(value) or inspect.isfunction(value) or inspect.ismethod(value):
        return False
    return True


def _class_to_dict(cls):
    values = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        for key, value in vars(base).items():
            if _include_config_key(key, value):
                values[key] = _to_wandb_value(value)
    return values


def build_wandb_config(**sections):
    return {key: _to_wandb_value(value) for key, value in sections.items()}
