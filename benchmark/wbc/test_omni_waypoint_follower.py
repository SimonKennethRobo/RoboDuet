"""Geometry, coordinate frames, PID saturation, stopping and task reset."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.wbc.omni_waypoint_follower import OmniFollowerConfig, OmniWaypointFollower, wrap_pi


def line_reference(velocity=(.1, 0.), start=(1., .2), duration=3.):
    def at(t):
        xy = np.asarray(start) + min(max(t, 0.), duration) * np.asarray(velocity)
        return t, np.r_[xy, .6], np.array([0., 0., 0., 1.])
    return SimpleNamespace(at=at, duration=duration)


def test_footprint_offset_and_vector_sum_are_world_quantities():
    follower = OmniWaypointFollower(.02)
    reference = line_reference((0., .1))
    original = reference.at(1.)[1].copy()
    command, d = follower.compute(reference, 1., [0., 0., .3], np.pi/2, [.12, -.03, 0.], 0.)
    np.testing.assert_allclose(d['trajectory_tangent_world'], [0., 1.], atol=1e-12)
    np.testing.assert_allclose(d['velocity_sum_world_mps'], [.12, .07], atol=1e-12)
    np.testing.assert_allclose(d['base_waypoint_world_m'], [1., -.06, 0.], atol=1e-12)
    assert d['base_yaw_target_rad'] == pytest.approx(np.pi/2)
    delta = original[:2] - d['base_waypoint_world_m'][:2]
    assert np.dot(delta, d['trajectory_tangent_world']) > follower.cfg.footprint_half_length_m
    np.testing.assert_allclose(command[:2], [d['command_velocity_world_mps'][1], -d['command_velocity_world_mps'][0]])
    np.testing.assert_array_equal(reference.at(1.)[1], original)


def test_waypoint_does_not_move_with_measured_robot():
    reference = line_reference()
    a, b = OmniWaypointFollower(.02), OmniWaypointFollower(.02)
    _, da = a.compute(reference, 1., [0., 0., .3], 0., [0., 0., 0.], 0.)
    _, db = b.compute(reference, 1., [3., -2., .3], 2., [.2, .1, 0.], 0.)
    np.testing.assert_array_equal(da['base_waypoint_world_m'], db['base_waypoint_world_m'])
    assert da['base_yaw_target_rad'] == db['base_yaw_target_rad']


def test_lateral_command_does_not_require_turning_towards_waypoint():
    reference = line_reference((.1, 0.), start=(.36, 1.))
    follower = OmniWaypointFollower(.02)
    command, d = follower.compute(reference, 0., [0., 0., .3], 0., np.zeros(3), 0.)
    assert command[1] > 0.
    assert command[2] == pytest.approx(0.)
    assert d['base_yaw_target_rad'] == pytest.approx(0.)


def test_saturation_antiwindup_and_reset():
    follower = OmniWaypointFollower(.02)
    ref = line_reference((0., .1), (100., 100.), duration=100.)
    previous = np.zeros(3)
    for i in range(150):
        command, _ = follower.compute(ref, i*.02, [0., 0., .3], 0., np.zeros(3), 0.)
        assert np.linalg.norm(command[:2]) <= follower.cfg.max_speed_mps + 1e-12
        assert abs(command[2]) <= follower.cfg.max_yaw_rate_rps + 1e-12
        assert np.linalg.norm(command[:2] - previous[:2]) <= follower.cfg.max_acceleration_mps2 * .02 + 1e-12
        assert abs(command[2] - previous[2]) <= follower.cfg.max_yaw_acceleration_rps2 * .02 + 1e-12
        previous = command
    np.testing.assert_array_equal(follower.integral_xy, 0.)
    assert follower.integral_yaw == 0.
    follower.reset()
    assert follower.diagnostics is None and follower.last_tangent is None
    np.testing.assert_array_equal(follower.previous_velocity_world, 0.)


def test_yaw_wrap_and_stationary_endpoint():
    angle = np.deg2rad(-179.)
    ref = line_reference(.1 * np.array([np.cos(angle), np.sin(angle)]))
    follower = OmniWaypointFollower(.02)
    cmd, d = follower.compute(ref, 1., [0., 0., .3], np.deg2rad(179.), np.zeros(3), 0.)
    assert d['yaw_error_rad'] == pytest.approx(np.deg2rad(2.))
    assert cmd[2] > 0.
    endpoint = ref.at(ref.duration)[1][:2] - .36 * np.array([np.cos(angle), np.sin(angle)])
    for _ in range(80):
        cmd, d = follower.compute(ref, ref.duration + 1., np.r_[endpoint, .3], angle, [.2, .1, 0.], .2)
    np.testing.assert_allclose(cmd, 0., atol=1e-12)
    np.testing.assert_array_equal(d['velocity_sum_world_mps'], 0.)
    assert wrap_pi(d['base_yaw_target_rad'] - angle) == pytest.approx(0.)
    assert not d['trajectory_moving']


def test_pid_integral_removes_static_offset_on_omni_kinematic_plant():
    cfg = replace(OmniFollowerConfig(), position_tolerance_m=.001)
    follower = OmniWaypointFollower(.02, cfg)
    ref = line_reference((0., 0.), start=(.36, .4))
    position, velocity, yaw, yaw_rate = np.zeros(3), np.zeros(3), 0., 0.
    max_integral = 0.
    for i in range(1000):
        command, d = follower.compute(ref, i*.02, position, yaw, velocity, yaw_rate)
        c, s = np.cos(yaw), np.sin(yaw)
        velocity[:2] = np.array([[c, -s], [s, c]]) @ command[:2]
        position += velocity * .02
        yaw_rate = command[2]
        yaw = float(wrap_pi(yaw + yaw_rate*.02))
        max_integral = max(max_integral, np.linalg.norm(follower.integral_xy))
    assert max_integral > .01
    assert np.linalg.norm(position[:2] - d['base_waypoint_world_m'][:2]) < .02


def test_point_reference_is_finite_and_keeps_initial_heading():
    follower = OmniWaypointFollower(.02)
    for t in (0., 1., 3., 4.):
        cmd, d = follower.compute(line_reference((0., 0.)), t, [0., 0., .3], .7, np.zeros(3), 0.)
        assert np.isfinite(cmd).all()
        assert d['base_yaw_target_rad'] == pytest.approx(.7)


def test_circle_tangent_follows_path_across_angle_wrap():
    def at(t):
        angle = .4 * t
        return t, np.array([np.cos(angle), np.sin(angle), .6]), np.array([0., 0., 0., 1.])
    reference = SimpleNamespace(at=at, duration=20.)
    follower = OmniWaypointFollower(.02)
    for t in np.linspace(.02, 19.9, 120):
        _, d = follower.compute(reference, t, [0., 0., .3], 0., np.zeros(3), 0.)
        expected = np.array([-np.sin(.4*t), np.cos(.4*t)])
        np.testing.assert_allclose(d['trajectory_tangent_world'], expected, atol=1e-10)
        np.testing.assert_allclose(d['base_waypoint_world_m'][:2], at(t)[1][:2] - .36*expected, atol=1e-10)


def test_policy_axis_limits_hold_even_when_body_rotates():
    cfg = replace(OmniFollowerConfig(), max_lateral_speed_mps=.02, max_backward_speed_mps=.03)
    follower = OmniWaypointFollower(.02, cfg)
    for i, yaw in enumerate(np.linspace(0., 2*np.pi, 150)):
        cmd, _ = follower.compute(line_reference(), i*.02, [0., 0., .3], yaw, [.2, .1, 0.], 0.)
        assert abs(cmd[1]) <= .02 + 1e-12
        assert cmd[0] >= -.03 - 1e-12


def test_yaw_command_keeps_winding_direction_when_more_than_pi_behind():
    def at(t):
        angle = t
        return t, np.array([np.cos(angle), np.sin(angle), .6]), np.array([0., 0., 0., 1.])
    follower = OmniWaypointFollower(.02, replace(
        OmniFollowerConfig(), max_yaw_rate_rps=1., max_yaw_acceleration_rps2=10.))
    reference = SimpleNamespace(at=at, duration=10.)
    commands, errors, targets = [], [], []
    for t in np.arange(.02, 7., .02):
        command, d = follower.compute(reference, t, [0., 0., .3], 0., np.zeros(3), 0.)
        commands.append(command[2])
        errors.append(d['yaw_error_rad'])
        targets.append(d['base_yaw_target_rad'])
    assert targets[-1] > 2*np.pi
    assert errors[-1] > 2*np.pi
    assert min(commands[100:]) > 0.
