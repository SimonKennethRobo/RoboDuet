import pytest
import torch

from modules.curriculum import CurriculumManager
from modules.trajectory_generator import TimingGenerator, TrajectoryFactory


@pytest.mark.parametrize("seed", [7, 12345, 54321])
def test_hardest_geometry_has_five_metre_xy_displacement(seed):
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    geom, timing = curriculum.params_for_cell(5, 5)
    gamma, _ = TrajectoryFactory().generate(
        dict(geom, seed=seed), dict(timing, seed=seed + 100000)
    )

    xy_delta = gamma.p[-1, :2] - gamma.p[0, :2]
    assert torch.linalg.vector_norm(xy_delta).item() == pytest.approx(5.0, abs=2e-3)


def test_xy_travel_increases_monotonically_with_geometry_level():
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    travels = [curriculum.params_for_cell(a, 0)[0]["xy_displacement"] for a in range(6)]

    assert travels == sorted(travels)
    assert travels[0] == pytest.approx(0.25)
    assert travels[-1] == pytest.approx(5.0)


def test_hardest_geometry_samples_multiple_xy_directions():
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    geom, timing = curriculum.params_for_cell(5, 5)
    directions = []
    for seed in range(12):
        gamma, _ = TrajectoryFactory().generate(
            dict(geom, seed=seed), dict(timing, seed=seed + 100000)
        )
        directions.append(gamma.p[-1, :2] - gamma.p[0, :2])
    directions = torch.stack(directions)

    assert (directions[:, 0] > 0).any() and (directions[:, 0] < 0).any()
    assert (directions[:, 1] > 0).any() and (directions[:, 1] < 0).any()


@pytest.mark.parametrize("seed", [7, 12345, 54321])
def test_hardest_geometry_covers_ground_to_one_point_five_metres(seed):
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    geom, timing = curriculum.params_for_cell(5, 5)
    gamma, _ = TrajectoryFactory().generate(
        dict(geom, seed=seed), dict(timing, seed=seed + 100000)
    )

    assert gamma.p[:, 2].min().item() == pytest.approx(0.0, abs=2e-3)
    assert gamma.p[:, 2].max().item() == pytest.approx(1.5, abs=2e-3)
    assert gamma.p[0, 2].item() == pytest.approx(0.75, abs=2e-3)
    assert gamma.p[-1, 2].item() == pytest.approx(0.75, abs=2e-3)


def test_hardest_geometry_contains_high_curvature_segments():
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    geom, timing = curriculum.params_for_cell(5, 5)
    gamma, _ = TrajectoryFactory().generate(
        dict(geom, seed=12345), dict(timing, seed=112345)
    )
    ds = (gamma.s_grid[1:] - gamma.s_grid[:-1]).clamp_min(1e-6)
    dots = (gamma.tangent[:-1] * gamma.tangent[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    curvature = torch.acos(dots) / ds

    assert torch.quantile(curvature, 0.99).item() > 2.0


def _peak_abs_acceleration(time_law):
    return torch.gradient(time_law.sdot_of_t, spacing=(time_law.t_grid,))[0].abs().max()


def test_timing_limits_are_physical_bounds_and_endpoint_is_exact():
    time_law = TimingGenerator().generate(
        L=12.0,
        f_max=0.4,
        v_max=0.75,
        a_max=0.30,
        T=8.0,
        dt=0.02,
        seed=12345,
    )

    assert time_law.T >= 8.0
    assert time_law.s_of_t[0].item() == pytest.approx(0.0, abs=1e-7)
    assert time_law.s_of_t[-1].item() == pytest.approx(12.0, abs=1e-5)
    assert torch.all(time_law.s_of_t[1:] >= time_law.s_of_t[:-1])
    assert time_law.sdot_of_t.max().item() <= 0.75 + 1e-5
    assert _peak_abs_acceleration(time_law).item() <= 0.30 + 1e-4


def test_v_max_changes_duration_instead_of_being_normalized_away():
    generator = TimingGenerator()
    common = dict(L=10.0, f_max=0.25, a_max=10.0, T=8.0, dt=0.02, seed=7)
    slow = generator.generate(v_max=0.25, **common)
    fast = generator.generate(v_max=0.75, **common)

    assert slow.T > fast.T
    assert slow.sdot_of_t.max().item() == pytest.approx(0.25, rel=1e-5)
    assert fast.sdot_of_t.max().item() == pytest.approx(0.75, rel=1e-5)


def test_long_path_extends_duration_instead_of_exceeding_speed_limit():
    generator = TimingGenerator()
    common = dict(f_max=0.2, v_max=0.5, a_max=1.0, T=8.0, dt=0.02, seed=11)
    short = generator.generate(L=1.0, **common)
    long = generator.generate(L=15.0, **common)

    assert short.T == pytest.approx(8.0)
    assert long.T > short.T
    assert long.sdot_of_t.max().item() <= 0.5 + 1e-5


def test_factory_stretches_high_curvature_timing_to_bound_linear_acceleration():
    curriculum = CurriculumManager(1, "cpu", n_levels_A=6, n_levels_B=6)
    geom, timing = curriculum.params_for_cell(5, 5)
    gamma, time_law = TrajectoryFactory().generate(
        dict(geom, seed=12345), dict(timing, seed=112345)
    )
    position = gamma.p_at(time_law.s_of_t)
    dt = time_law.t_grid[1:] - time_law.t_grid[:-1]
    velocity = (position[1:] - position[:-1]) / dt.unsqueeze(-1)
    velocity_t = 0.5 * (time_law.t_grid[1:] + time_law.t_grid[:-1])
    acceleration = (velocity[1:] - velocity[:-1]) / (
        velocity_t[1:] - velocity_t[:-1]
    ).unsqueeze(-1)

    peak = torch.linalg.vector_norm(acceleration, dim=-1).max().item()
    assert peak <= timing["linear_a_max"] + 1e-4
