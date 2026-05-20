"""Public RoboDuet env API.

The heavy IsaacGym environment is imported lazily so config-only imports such
as `go1_gym.envs.roboduet.legged_robot_config` do not initialize simulator
dependencies.
"""

__all__ = [
    "EvaluationWrapper",
    "HistoryWrapper",
    "JoyWrapper",
    "KeyboardWrapper",
    "WBCEnv",
]


def __getattr__(name):
    if name in __all__:
        from . import wbc_env

        return getattr(wbc_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
