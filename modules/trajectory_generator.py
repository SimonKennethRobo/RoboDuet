"""M5: procedural trajectory generation.

Standalone module built on top of M1 (``modules/trajectory.py``). Implements
``GeometryGenerator`` (multi-frequency-sine SE(3) geometric path) and
``TimingGenerator`` (multi-frequency-sine time law, softplus-clamped to
nonnegative arc-length speed and normalized to traverse exactly L). Composed
by ``TrajectoryFactory``.

Deliberately NOT implemented here (depends on modules not in scope for this
pass -- M2's ReachabilityTable and M3's OnlineBaseNom): the design doc's
``TrajectoryFactory.generate_and_label`` (post-hoc difficulty labeling via
base_nom implied base velocity/acceleration) and ``generate_for_level``
(curriculum-level rejection sampling). ``TrajectoryFactory.generate`` covers
the geometry+timing composition; difficulty labeling and curriculum belong to
a later pass once M2/M3/M10 exist.
"""

import math

import torch

from .trajectory import TimeLaw, mat_to_quat, quat_slerp, quat_to_mat, so3_exp, to_canonical


class GeometryGenerator:
    """Forward-generates an EE geometric path (position + SO(3) orientation)."""

    def generate(self, params):
        """
        params: dict with keys
            f_max:        float, frequency ceiling (Hz)
            amplitude:    float, spatial amplitude (m)
            f_rot_max:    float, rotation frequency ceiling (Hz)
            tangent_align_ratio: float [0,1], fraction of rotation that
                          tracks the path tangent vs. an independent
                          multi-freq rotation-vector wobble
            drift_speed:  float, drift speed (m/s), keeps min difficulty > 0
            drift_dir:    (2,) drift direction (xy), will be normalized
            center:       (3,) workspace center the path is generated around
            duration:     float, duration (s)
            dt:           float, sample interval (s)
            lam:          float, SE(3) arc-length rotation-translation factor
            n_freqs:      int, number of superposed sine components per axis
            seed:         optional int

        Returns: Gamma
        """
        gen = torch.Generator().manual_seed(params["seed"]) if "seed" in params else None
        T = params["duration"]
        dt = params["dt"]
        n_freqs = int(params.get("n_freqs", 8))
        t = torch.arange(0, T, dt)

        def rand(*shape):
            return torch.rand(*shape, generator=gen)

        # position: multi-frequency sine superposition per axis
        p = torch.zeros(len(t), 3)
        for axis in range(3):
            f_k = rand(n_freqs) * params["f_max"]
            A_k = (rand(n_freqs) - 0.5) * 2 * params["amplitude"]
            phi_k = rand(n_freqs) * 2 * math.pi
            p[:, axis] = (A_k * torch.sin(2 * math.pi * f_k * t.unsqueeze(1) + phi_k)).sum(dim=1)

        drift_dir = torch.as_tensor(params.get("drift_dir", [1.0, 0.0]), dtype=torch.float32)
        drift_dir = drift_dir / drift_dir.norm().clamp_min(1e-8)
        p[:, 0] += drift_dir[0] * params.get("drift_speed", 0.0) * t
        p[:, 1] += drift_dir[1] * params.get("drift_speed", 0.0) * t

        center = torch.as_tensor(params.get("center", [0.3, 0.0, 0.5]), dtype=torch.float32)
        p += center

        R = self._generate_rotation(t, p, params, rand)

        gamma, _ = to_canonical(
            p, R, dt, lam=params.get("lam", 0.15), ds_grid=params.get("ds_grid", 0.005)
        )
        return gamma

    def _generate_rotation(self, t, p, params, rand):
        """Mix tangent-aligned and independent rotation modes."""
        n_freqs = int(params.get("n_freqs", 8))
        f_rot_max = params.get("f_rot_max", 0.3)
        ratio = params.get("tangent_align_ratio", 0.5)

        # independent multi-freq rotation-vector wobble
        rotvec = torch.zeros(len(t), 3)
        for axis in range(3):
            f_k = rand(n_freqs) * f_rot_max
            A_k = (rand(n_freqs) - 0.5) * 2 * params.get("f_rot_amplitude", 0.5)
            phi_k = rand(n_freqs) * 2 * math.pi
            rotvec[:, axis] = (A_k * torch.sin(2 * math.pi * f_k * t.unsqueeze(1) + phi_k)).sum(dim=1)
        R_indep = so3_exp(rotvec)

        # tangent-aligned: x-axis of the gripper points along the path's
        # forward finite-difference direction
        fwd = torch.zeros_like(p)
        fwd[:-1] = p[1:] - p[:-1]
        fwd[-1] = fwd[-2]
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        world_up = torch.tensor([0.0, 0.0, 1.0]).expand_as(fwd)
        right = torch.cross(fwd, world_up, dim=-1)
        right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        up = torch.cross(right, fwd, dim=-1)
        R_tangent = torch.stack([fwd, right, up], dim=-1)  # columns = local frame axes

        q_indep = mat_to_quat(R_indep)
        q_tangent = mat_to_quat(R_tangent)
        frac = torch.full((len(t),), ratio)
        q_mixed = quat_slerp(q_indep, q_tangent, frac)
        return quat_to_mat(q_mixed)


class TimingGenerator:
    """Generates the time law s_ref(t)."""

    def generate(self, L, f_max, v_max, T, dt, n_freqs=6, seed=None):
        """
        1. raw = sum A_k sin(2*pi*f_k*t + phi_k)
        2. sdot = v_max * softplus(raw)          # keeps sdot >= 0
        3. sdot *= L / trapz(sdot, dt)            # normalize to traverse exactly L
        4. s_ref = cumsum(sdot) * dt

        Returns: TimeLaw
        """
        n_freqs = int(n_freqs)
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        t = torch.arange(0, T, dt)
        f_k = torch.rand(n_freqs, generator=gen) * f_max
        A_k = torch.rand(n_freqs, generator=gen) * 2 - 1
        phi_k = torch.rand(n_freqs, generator=gen) * 2 * math.pi
        raw = (A_k * torch.sin(2 * math.pi * f_k * t.unsqueeze(1) + phi_k)).sum(dim=1)

        sdot = v_max * torch.nn.functional.softplus(raw)
        area = torch.trapz(sdot, dx=dt).clamp_min(1e-6)
        sdot = sdot * (L / area)
        s_ref = torch.cumsum(sdot, dim=0) * dt
        s_ref = s_ref - s_ref[0]  # start at 0
        # last point should land near L; clamp for numerical safety
        s_ref = s_ref.clamp(max=L)

        return TimeLaw(t, s_ref, sdot)


class TrajectoryFactory:
    """Integrates geometry + timing generation."""

    def __init__(self, geometry_generator=None, timing_generator=None):
        self.geom_gen = geometry_generator or GeometryGenerator()
        self.timing_gen = timing_generator or TimingGenerator()

    def generate(self, geom_params, timing_params):
        """
        Returns: (gamma, time_law)
        """
        gamma = self.geom_gen.generate(geom_params)
        time_law = self.timing_gen.generate(L=gamma.L, **timing_params)
        return gamma, time_law


class DifficultySchedule:
    """Deterministic step->params curriculum ramp (a simplified stand-in for
    the design doc's M10 ``CurriculumManager``).

    This is NOT the full M10: there is no per-env success-rate tracking, no
    promote/demote logic, no history mixing across unlocked levels -- all of
    that requires live success/fail feedback from real RL rollouts, which a
    standalone script generating trajectories in isolation doesn't have.
    What this gives you instead is the simpler, non-adaptive half of a
    curriculum: a monotonic linear ramp from an "easy" generation-parameter
    preset to a "hard" one over a fixed number of training steps, so you can
    at least verify the generator actually produces harder trajectories as
    "training" progresses. Wiring this up to real success/fail signal later
    (promote faster if the policy is doing well, hold back if it's failing)
    is exactly the M10 gap this leaves open.
    """

    def __init__(self, easy_geom, hard_geom, easy_timing, hard_timing, ramp_steps):
        self.easy_geom = easy_geom
        self.hard_geom = hard_geom
        self.easy_timing = easy_timing
        self.hard_timing = hard_timing
        self.ramp_steps = max(1, ramp_steps)

    def progress(self, step):
        """Fraction of the ramp completed, clamped to [0, 1]."""
        return min(1.0, max(0.0, step / self.ramp_steps))

    def params_at(self, step):
        """Returns (geom_params, timing_params, alpha) for the given step."""
        alpha = self.progress(step)
        geom = {k: self._lerp(v, self.hard_geom[k], alpha) for k, v in self.easy_geom.items()}
        timing = {k: self._lerp(v, self.hard_timing[k], alpha) for k, v in self.easy_timing.items()}
        return geom, timing, alpha

    @staticmethod
    def _lerp(a, b, alpha):
        if isinstance(a, (list, tuple)):
            return [DifficultySchedule._lerp(ai, bi, alpha) for ai, bi in zip(a, b)]
        if isinstance(a, (int, float)):
            return a + (b - a) * alpha
        return a if alpha < 0.5 else b  # non-numeric fields (e.g. strings): hard switch at the midpoint
