"""M5: procedural trajectory generation.

Standalone module built on top of M1 (``modules/trajectory.py``). Implements
``GeometryGenerator`` (multi-frequency-sine SE(3) geometric path) and
``TimingGenerator`` (multi-frequency-sine time law with explicit arc-speed,
arc-acceleration, and minimum-duration bounds). Composed
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
            drift_direction_spread: optional half-width (rad) of a seeded
                          uniform heading offset around drift_dir
            xy_displacement: optional exact start-to-end XY travel (m). When
                          present, it supersedes drift_speed and removes the
                          incidental endpoint drift of the sine components.
            planar_curve_amplitude: deterministic cross-track wave amplitude
                          (m), used to retain high-curvature segments on long
                          trajectories
            planar_curve_cycles: number of cross-track wave cycles
            z_min/z_max: optional exact ground-relative height range (m)
            vertical_cycles: smooth height sweeps over the trajectory duration
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
        direction_spread = float(params.get("drift_direction_spread", 0.0))
        if direction_spread > 0.0:
            base_angle = torch.atan2(drift_dir[1], drift_dir[0])
            angle = base_angle + (rand(1)[0] * 2.0 - 1.0) * direction_spread
            drift_dir = torch.stack((torch.cos(angle), torch.sin(angle)))
        frac = (t - t[0]) / (t[-1] - t[0]).clamp_min(1e-8)
        curve_amplitude = float(params.get("planar_curve_amplitude", 0.0))
        curve_cycles = float(params.get("planar_curve_cycles", 0.0))
        if curve_amplitude > 0.0 and curve_cycles > 0.0:
            cross_track = torch.stack((-drift_dir[1], drift_dir[0]))
            wave = curve_amplitude * torch.sin(2.0 * math.pi * curve_cycles * frac)
            p[:, :2] += wave.unsqueeze(1) * cross_track
        if "xy_displacement" in params:
            # Make planar travel an explicit generator contract. The raw sine
            # sum generally has a random endpoint offset of its own; cancelling
            # that linear component before adding the requested travel makes
            # the hardest curriculum level reproducibly exercise a 5 m
            # continuous base translation rather than merely having a large
            # bounding box for one lucky seed.
            raw_delta_xy = p[-1, :2] - p[0, :2]
            requested_delta_xy = drift_dir * float(params["xy_displacement"])
            p[:, :2] += frac.unsqueeze(1) * (requested_delta_xy - raw_delta_xy)
        else:
            p[:, 0] += drift_dir[0] * params.get("drift_speed", 0.0) * t
            p[:, 1] += drift_dir[1] * params.get("drift_speed", 0.0) * t

        center = torch.as_tensor(params.get("center", [0.3, 0.0, 0.5]), dtype=torch.float32)
        p += center
        if "z_min" in params and "z_max" in params:
            z_min = float(params["z_min"])
            z_max = float(params["z_max"])
            if z_max < z_min:
                raise ValueError("z_max must be greater than or equal to z_min")
            # Use a smooth full-range sweep instead of stretching the random
            # high-frequency Z signal to 1.5 m. It starts and ends at the
            # reachable mid-height, visits the top and ground once, and avoids
            # turning height coverage into an unrealistically long vertical
            # jitter path. XY supplies the explicit high-curvature segments.
            vertical_cycles = float(params.get("vertical_cycles", 1.0))
            z_unit = 0.5 + 0.5 * torch.sin(2.0 * math.pi * vertical_cycles * frac)
            z_unit = (z_unit - z_unit.min()) / (z_unit.max() - z_unit.min()).clamp_min(1e-8)
            p[:, 2] = z_min + (z_max - z_min) * z_unit

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
    """Generates a bounded time law ``s_ref(t)``.

    Arc length uses the same metre-equivalent SE(3) metric as ``Gamma``. Thus
    ``v_max`` and ``a_max`` bound derivatives of that metric; Cartesian linear
    speed is no greater than the arc speed, but can be lower when orientation
    contributes to the path length.
    """

    def generate(
        self,
        L,
        f_max,
        v_max,
        T,
        dt,
        n_freqs=6,
        seed=None,
        a_max=None,
        min_speed_fraction=0.25,
    ):
        """
        ``T`` is a minimum duration, not a forced duration. A dimensionless
        positive speed profile is generated over that horizon, then the actual
        duration is enlarged until both ``v_max`` and optional ``a_max`` are
        satisfied. The profile is integrated exactly from 0 to ``L``.

        ``f_max`` is the modulation-frequency ceiling on the minimum-duration
        horizon. Duration stretching can only lower its physical frequency.

        Returns: TimeLaw
        """
        length = float(L)
        minimum_duration = float(T)
        sample_dt = float(dt)
        speed_limit = float(v_max)
        acceleration_limit = None if a_max is None else float(a_max)
        minimum_fraction = float(min_speed_fraction)
        if length <= 0.0:
            raise ValueError("L must be positive")
        if minimum_duration <= 0.0 or sample_dt <= 0.0:
            raise ValueError("T and dt must be positive")
        if speed_limit <= 0.0:
            raise ValueError("v_max must be positive")
        if acceleration_limit is not None and acceleration_limit <= 0.0:
            raise ValueError("a_max must be positive when provided")
        if not 0.0 <= minimum_fraction < 1.0:
            raise ValueError("min_speed_fraction must be in [0, 1)")

        n_freqs = int(n_freqs)
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        n_points = max(2, int(round(minimum_duration / sample_dt)) + 1)
        nominal_t = torch.linspace(0.0, minimum_duration, n_points)
        phase = nominal_t / minimum_duration
        f_k = torch.rand(n_freqs, generator=gen) * f_max
        A_k = torch.rand(n_freqs, generator=gen) * 2 - 1
        phi_k = torch.rand(n_freqs, generator=gen) * 2 * math.pi
        raw = (
            A_k
            * torch.sin(2 * math.pi * f_k * nominal_t.unsqueeze(1) + phi_k)
        ).sum(dim=1)

        positive = torch.nn.functional.softplus(raw)
        positive = positive / positive.max().clamp_min(1e-8)
        profile = minimum_fraction + (1.0 - minimum_fraction) * positive
        profile_area = torch.trapz(profile, phase).clamp_min(1e-8)

        # For sdot = L/T_actual * profile/integral(profile), derive the
        # minimum duration needed by the requested derivative limits.
        peak_speed_ratio = float((profile.max() / profile_area).item())
        duration = max(minimum_duration, length * peak_speed_ratio / speed_limit)
        if acceleration_limit is not None:
            dprofile_dphase = torch.gradient(profile, spacing=(phase,))[0]
            peak_acceleration_ratio = float(
                (dprofile_dphase.abs().max() / profile_area).item()
            )
            duration = max(
                duration,
                math.sqrt(length * peak_acceleration_ratio / acceleration_limit),
            )

        t = phase * duration
        sdot = length / duration * profile / profile_area
        segment_area = 0.5 * (profile[1:] + profile[:-1]) * (phase[1:] - phase[:-1])
        s_ref = torch.cat(
            (torch.zeros(1, dtype=profile.dtype), torch.cumsum(segment_area, dim=0))
        )
        s_ref = (length * s_ref / profile_area).clamp(min=0.0, max=length)
        s_ref[-1] = length

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
        timing = dict(timing_params)
        linear_acceleration_limit = timing.pop("linear_a_max", None)
        time_law = self.timing_gen.generate(L=gamma.L, **timing)
        if linear_acceleration_limit is not None:
            time_law = self._limit_cartesian_acceleration(
                gamma, time_law, float(linear_acceleration_limit)
            )
        return gamma, time_law

    @staticmethod
    def _limit_cartesian_acceleration(gamma, time_law, limit):
        """Globally stretch a time law to bound physical reference accel.

        The arc-tangential ``a_max`` does not cover centripetal acceleration
        on a curved path. A uniform time stretch preserves the frozen geometry
        and profile while linear velocity and acceleration scale as 1/k and
        1/k^2 respectively.
        """
        if limit <= 0.0:
            raise ValueError("linear_a_max must be positive")
        positions = gamma.p_at(time_law.s_of_t)
        dt = (time_law.t_grid[1:] - time_law.t_grid[:-1]).clamp_min(1e-8)
        velocity = (positions[1:] - positions[:-1]) / dt.unsqueeze(-1)
        if velocity.shape[0] < 2:
            return time_law
        velocity_t = 0.5 * (time_law.t_grid[1:] + time_law.t_grid[:-1])
        acceleration = (velocity[1:] - velocity[:-1]) / (
            velocity_t[1:] - velocity_t[:-1]
        ).clamp_min(1e-8).unsqueeze(-1)
        peak = float(torch.linalg.vector_norm(acceleration, dim=-1).max().item())
        if peak <= limit:
            return time_law
        # Small margin absorbs float32 interpolation/differencing differences
        # between generation and manifest reconstruction.
        stretch = math.sqrt(peak / limit) * (1.0 + 1e-4)
        return TimeLaw(
            time_law.t_grid * stretch,
            time_law.s_of_t.clone(),
            time_law.sdot_of_t / stretch,
        )


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
