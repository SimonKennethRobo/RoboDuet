"""Shared planar waypoint PID for policies accepting vx, vy and yaw-rate.

World XY is the ground plane. The EE projection is offset backwards along its
tangent so it lies beyond the front of the desired robot footprint. This is a
world waypoint, independent of the measured base position. Frozen EE references
are never modified. The locomotion policy, not this kinematic model, drives legs.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def wrap_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def limit_norm(vector, limit):
    return vector * min(1., limit / max(float(np.linalg.norm(vector)), 1e-12))


@dataclass(frozen=True)
class OmniFollowerConfig:
    # Conservative Go2 standing footprint including feet, in the desired-yaw
    # body frame. Fixed dimensions avoid target jitter from swinging feet.
    footprint_half_length_m: float = .30
    footprint_half_width_m: float = .30
    footprint_clearance_m: float = .06
    position_kp: float = 1.5
    position_ki: float = .10
    position_kd: float = .80
    position_integral_limit_ms: float = .30
    yaw_kp: float = 2.
    yaw_ki: float = .10
    yaw_kd: float = .25
    yaw_integral_limit_rads: float = .50
    max_speed_mps: float = .35
    max_lateral_speed_mps: float = .35
    max_backward_speed_mps: float = .35
    max_acceleration_mps2: float = .70
    max_yaw_rate_rps: float = .60
    max_yaw_acceleration_rps2: float = 1.5
    position_tolerance_m: float = .025
    yaw_tolerance_rad: float = .03
    tangent_epsilon_mps: float = 1e-5


class OmniWaypointFollower:
    algorithm = "ee-ground-footprint-tangent-omni-pid-v1"

    def __init__(self, dt, cfg=None):
        self.dt = float(dt)
        self.cfg = cfg or OmniFollowerConfig()
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control dt must be positive and finite")
        if any(not np.isfinite(v) or v < 0 for v in asdict(self.cfg).values()):
            raise ValueError("follower configuration must be finite and nonnegative")
        self.reset()

    def reset(self):
        self.integral_xy = np.zeros(2)
        self.integral_yaw = 0.
        self.last_tangent = None
        self.last_goal = None
        self.last_yaw_target = None
        self.last_base_yaw = None
        self.base_yaw_unwrapped = None
        self.previous_velocity_world = np.zeros(2)
        self.previous_yaw_rate = 0.
        self.diagnostics = None

    def _sample(self, reference, time_s, base_yaw):
        duration = getattr(reference, "duration", np.inf)
        t = float(np.clip(time_s, 0., duration))
        _, position, _ = reference.at(t)
        lo, hi = max(0., t - self.dt / 2), min(duration, t + self.dt / 2)
        velocity = np.zeros(2)
        if hi > lo:
            velocity = (np.asarray(reference.at(hi)[1])[:2] - np.asarray(reference.at(lo)[1])[:2]) / (hi - lo)
        moving = np.linalg.norm(velocity) > self.cfg.tangent_epsilon_mps
        if moving:
            tangent = velocity / np.linalg.norm(velocity)
        elif self.last_tangent is not None:
            tangent = self.last_tangent.copy()
        else:
            # A time law may start at zero speed. Find the first nonzero planar
            # displacement without deriving yaw from the EE orientation.
            tangent = np.array([np.cos(base_yaw), np.sin(base_yaw)])
            for horizon in (.05, .1, .25, .5, 1.):
                a, b = max(0., t-horizon), min(duration, t+horizon)
                delta = np.asarray(reference.at(b)[1])[:2] - np.asarray(reference.at(a)[1])[:2]
                if np.linalg.norm(delta) > 1e-7:
                    tangent = delta / np.linalg.norm(delta)
                    break
        self.last_tangent = tangent.copy()
        if t >= duration:
            velocity[:] = 0.
            moving = False
        return np.asarray(position, dtype=float).copy(), velocity, tangent, moving

    def compute(self, reference, time_s, base_position_world, base_yaw,
                base_velocity_world, base_yaw_rate):
        cfg = self.cfg
        position = np.asarray(base_position_world, dtype=float)[:2]
        measured_velocity = np.asarray(base_velocity_world, dtype=float)[:2]
        ee, reference_velocity, tangent, moving = self._sample(reference, time_s, base_yaw)
        offset = cfg.footprint_half_length_m + cfg.footprint_clearance_m
        goal = ee[:2] - offset * tangent
        yaw_target_wrapped = float(np.arctan2(tangent[1], tangent[0]))
        yaw_target = (float(base_yaw + wrap_pi(yaw_target_wrapped - base_yaw))
                      if self.last_yaw_target is None else
                      self.last_yaw_target + float(wrap_pi(yaw_target_wrapped - self.last_yaw_target)))
        goal_velocity = (reference_velocity if self.last_goal is None else
                         (goal - self.last_goal) / self.dt)
        goal_velocity = limit_norm(goal_velocity, cfg.max_speed_mps)
        tangent_yaw_rate = (0. if self.last_yaw_target is None else
                            (yaw_target - self.last_yaw_target) / self.dt)
        tangent_yaw_rate = float(np.clip(tangent_yaw_rate, -cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps))
        self.last_goal, self.last_yaw_target = goal.copy(), yaw_target

        error_xy = goal - position
        if self.last_base_yaw is None:
            self.base_yaw_unwrapped = float(base_yaw)
        else:
            self.base_yaw_unwrapped += float(wrap_pi(base_yaw - self.last_base_yaw))
        self.last_base_yaw = float(base_yaw)
        error_yaw = float(yaw_target - self.base_yaw_unwrapped)
        candidate_xy = np.clip(self.integral_xy + self.dt * error_xy,
                               -cfg.position_integral_limit_ms, cfg.position_integral_limit_ms)
        candidate_yaw = float(np.clip(self.integral_yaw + self.dt * error_yaw,
                                     -cfg.yaw_integral_limit_rads, cfg.yaw_integral_limit_rads))
        # User-specified vector sum, in one world frame. Drop the measured-speed
        # feedforward at a stationary/terminal waypoint so momentum cannot keep
        # regenerating a nonzero target speed after the trajectory stops.
        velocity_sum = measured_velocity + reference_velocity if moving else np.zeros(2)
        derivative_xy = goal_velocity - measured_velocity
        correction_xy = cfg.position_kp * error_xy + cfg.position_ki * candidate_xy + cfg.position_kd * derivative_xy
        raw_velocity = velocity_sum + correction_xy
        raw_yaw_rate = (tangent_yaw_rate + cfg.yaw_kp * error_yaw + cfg.yaw_ki * candidate_yaw
                        + cfg.yaw_kd * (tangent_yaw_rate - base_yaw_rate))
        # Conditional integration: do not accumulate error that drives further
        # into speed saturation; integral clamps also bound prolonged lag.
        if np.linalg.norm(raw_velocity) <= cfg.max_speed_mps or np.dot(error_xy, raw_velocity) < 0:
            self.integral_xy = candidate_xy
        if abs(raw_yaw_rate) <= cfg.max_yaw_rate_rps or error_yaw * raw_yaw_rate < 0:
            self.integral_yaw = candidate_yaw
        if not moving and np.linalg.norm(error_xy) <= cfg.position_tolerance_m:
            raw_velocity[:] = 0.
            self.integral_xy[:] = 0.
        if not moving and abs(error_yaw) <= cfg.yaw_tolerance_rad:
            raw_yaw_rate = 0.
            self.integral_yaw = 0.

        desired_velocity = limit_norm(raw_velocity, cfg.max_speed_mps)
        c, s = np.cos(base_yaw), np.sin(base_yaw)
        world_to_body = np.array([[c, s], [-s, c]])
        desired_body = world_to_body @ desired_velocity
        desired_body[0] = max(desired_body[0], -cfg.max_backward_speed_mps)
        desired_body[1] = np.clip(desired_body[1], -cfg.max_lateral_speed_mps, cfg.max_lateral_speed_mps)
        desired_velocity = world_to_body.T @ desired_body
        velocity_world = self.previous_velocity_world + limit_norm(
            desired_velocity - self.previous_velocity_world, cfg.max_acceleration_mps2 * self.dt)
        yaw_rate = self.previous_yaw_rate + float(np.clip(
            np.clip(raw_yaw_rate, -cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps) - self.previous_yaw_rate,
            -cfg.max_yaw_acceleration_rps2 * self.dt, cfg.max_yaw_acceleration_rps2 * self.dt))
        self.previous_velocity_world, self.previous_yaw_rate = velocity_world.copy(), yaw_rate
        body_velocity = world_to_body @ velocity_world
        # Rotating the base can rotate a previous world command outside an
        # axis limit. Enforce policy limits again after the world slew limiter.
        body_velocity[0] = max(body_velocity[0], -cfg.max_backward_speed_mps)
        body_velocity[1] = np.clip(body_velocity[1], -cfg.max_lateral_speed_mps, cfg.max_lateral_speed_mps)
        velocity_world = world_to_body.T @ body_velocity
        self.previous_velocity_world = velocity_world.copy()
        command = np.r_[body_velocity, yaw_rate]
        if not np.isfinite(command).all():
            raise ValueError("nonfinite omni follower command")
        self.diagnostics = dict(
            base_waypoint_world_m=np.r_[goal, 0.],
            ee_ground_projection_world_m=np.r_[ee[:2], 0.],
            trajectory_tangent_world=tangent.copy(),
            base_yaw_target_rad=yaw_target,
            base_yaw_target_wrapped_rad=yaw_target_wrapped,
            base_yaw_measured_unwrapped_rad=self.base_yaw_unwrapped,
            reference_velocity_world_mps=reference_velocity.copy(),
            measured_velocity_world_mps=measured_velocity.copy(),
            velocity_sum_world_mps=velocity_sum.copy(),
            pid_correction_world_mps=correction_xy.copy(),
            command_velocity_world_mps=velocity_world.copy(),
            position_error_world_m=error_xy.copy(),
            yaw_error_rad=error_yaw,
            trajectory_moving=moving,
        )
        return command.copy(), self.diagnostics


def configure_follower(sim, cfg=None):
    sim.follower = OmniWaypointFollower(sim.policy_dt, cfg)
    sim.follower_source = Path(__file__).resolve()
    sim.follower_config = asdict(sim.follower.cfg)


def follow_reference(sim, reference, time_s):
    _, _, _, gyro, _, linear_velocity_body = sim.read_state()
    rotation = sim.data.xmat[sim.base].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    command, _ = sim.follower.compute(
        reference, time_s, sim.data.xpos[sim.base], yaw,
        rotation @ linear_velocity_body, float((rotation @ gyro)[2]))
    sim.command[:3] = command.tolist()
