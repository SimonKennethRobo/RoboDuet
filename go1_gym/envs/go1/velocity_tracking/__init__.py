"""Legacy Go1 velocity-tracking env entrypoint.

The old locomotion-only implementation is no longer present in this codebase.
Keep this module importable so legacy discovery fails with an explicit message
instead of a missing-module traceback.
"""


class VelocityTrackingEasyEnv:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "go1_gym.envs.go1.velocity_tracking.VelocityTrackingEasyEnv is a legacy entrypoint. "
            "Use go1_gym.envs.roboduet.WBCEnv for current RoboDuet training."
        )


__all__ = ["VelocityTrackingEasyEnv"]
