#!/usr/bin/env python3
"""
trajectory_curriculum.py
────────────────────────
Velocity-Sampled Full-Pose Trajectory Curriculum — library form.

This is the IsaacGym-friendly wrapper around the trajectory generator that the
test_velocity.py script demonstrates.  It exposes:

    VelocityTrajectorySimulator   — the core position generator (the same one
                                    used by test_velocity.py).
    TrajectoryCurriculum          — a per-env runtime buffer:
                                    construct once at env init, call
                                    `resample(env_ids, D)` on every env reset,
                                    call `advance()` once per control step,
                                    read `get_target(...)` for the current
                                    reference pose.

The simulator produces trajectories in the arm-base (body-fixed) frame used by
the langevin field generator (body box at origin, ground at −base_z).  For
whole-body RL, call `resample(..., base_pos, base_quat, measured_heights)` at
reset and `get_target_lpy_world(...)` every step.  That anchors the trajectory
in world coordinates and recomputes the current base-relative (l, p, y), so
base motion can reduce the target error.  `get_target_lpy(...)` remains
available for deliberately body-fixed references.

See trajectory.md for the math, curriculum stages, API reference, and an
end-to-end IsaacGym integration example.
"""

from __future__ import annotations

import os
import sys
import time
import contextlib
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

# The trajectory library imports older generator modules that import matplotlib at
# module load time.  Keep matplotlib's cache in a writable location when this file
# is imported from headless training jobs.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

# Shared machinery: fields + SDF queries (ltg) and the feasible-attitude
# simulator + quaternion helpers (ptg).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import langevin_traj_gen as ltg                       # noqa: E402
import pose_traj_gen as ptg                           # noqa: E402

ConfigLike = Union[dict, str, Path]
TensorOrFloat = Union[float, torch.Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Speed profile
# ─────────────────────────────────────────────────────────────────────────────

def speed_profile(k_steps: int, k_ramp: int) -> torch.Tensor:
    """
    Normalised trapezoidal speed shape (values in [0,1], length k_steps).
    Smoothstep accel 0→1 over k_ramp, flat cruise, smoothstep decel 1→0.
    Starts and ends at zero so the trajectory starts and ends at rest.
    """
    if k_steps <= 0:
        raise ValueError("speed_profile: k_steps must be positive")
    k_ramp = min(max(int(k_ramp), 0), k_steps // 2)
    s = torch.ones(k_steps, dtype=torch.float32)
    if k_ramp > 0:
        t = torch.arange(k_ramp, dtype=torch.float32) / k_ramp
        ramp = 3.0 * t ** 2 - 2.0 * t ** 3                       # smoothstep, C¹
        s[:k_ramp]  = ramp
        s[-k_ramp:] = ramp.flip(0)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# VelocityTrajectorySimulator  (position, single difficulty)
# ─────────────────────────────────────────────────────────────────────────────

class VelocityTrajectorySimulator:
    """
    Position trajectories at a SAMPLED, difficulty-scaled, length-controlled
    speed.  Stages:

        'point'    (D ≤ point_max)         single held point.
        'velocity' (point < D ≤ outreach_min)
            length L = lerp(length_min, length_max, t) ± jitter;
            wander turn_sigma = D**wander_exp · disturb;
            stays INSIDE the arm workspace (boundary + body + ground avoidance).
        'outreach' (D > outreach_min)
            roams FREELY — only an outer bound (workspace SDF < outreach_max)
            keeps it within base-translation range; body avoidance OFF; each
            trajectory STARTS in a hard region (out of arm reach OR inside the
            body) so part of it is unfollowable by the arm alone.

    Drift-anchored wander
    ─────────────────────
    Every trajectory is assigned a fixed XY-only drift direction `dir_drift`.
    The OU wander is anchored to `dir_drift` (not to the current direction),
    so the heading oscillates around `dir_drift` instead of accumulating
    away from it.  Consequence:

        net XY displacement ≈ L · cos(typical wander angle)  →  grows with D.

    Without the drift anchor a long path folds in on itself and the net XY
    span stays tiny — which means the dog's base can stay still and the arm
    alone tracks the EE.  With the anchor, longer paths must sweep the EE
    across the floor → the base has to translate to keep the arm in reach.

    The integration always uses an exact trapezoidal speed profile
    (v_cruise = L / (Σ shape · dt)), so ONLY the direction is steered — the
    path length stays exactly L irrespective of how much the avoidance turns.
    """

    def __init__(self, fg, D, n, k_steps, dt, cfg, device):
        self.fg      = fg
        self.D       = float(D)
        self.N       = int(n)
        self.K       = int(k_steps)
        self.dt      = float(dt)
        self.device  = device
        self.command_limits = cfg.get('command_limits', None)
        self.command_margin_l = float(cfg.get('command_margin_l', 0.08))
        self.command_margin_angle = float(cfg.get('command_margin_angle', 0.15))
        self.grid_margin = float(cfg.get('grid_margin', 0.12))

        fg.set_workspace(cfg.get('workspace', 'dynamic'))
        self.point_max = float(cfg['point_max'])
        outreach_min   = float(cfg.get('outreach_min', 0.7))
        outreach_max   = float(cfg.get('outreach_max', 0.40))

        reach_t = (self.D - outreach_min) / max(1.0 - outreach_min, 1e-6)
        reach_t = float(min(max(reach_t, 0.0), 1.0))

        if self.D <= self.point_max:
            self.stage = 'point'
        elif reach_t > 0.0:
            self.stage = 'outreach'
        else:
            self.stage = 'velocity'

        # ── workspace shell + start position ─────────────────────────────────
        N = self.N
        if self.stage == 'outreach':
            self.shell_hi   = outreach_max
            self.body_avoid = 0.0
            n_body  = int(round(N * float(cfg.get('body_fraction', 0.4))))
            n_body = min(max(n_body, 0), N)
            n_reach = N - n_body
            x_reach = self._sample_safe(n_reach, 0.10, outreach_max - 0.05, use_body=False)
            x_body  = self._sample_body(n_body)
            self.x  = torch.cat([x_reach, x_body], dim=0)
        else:
            self.shell_hi   = 0.0
            self.body_avoid = 1.0
            self.x = self._sample_safe(N, -10.0, -0.15, use_body=True)
        self.target_length = torch.zeros(self.N, device=device)
        if self.stage == 'point':
            return

        # ── curriculum: length scales with D (short → long) ──────────────────
        t = (self.D - self.point_max) / max(1.0 - self.point_max, 1e-6)
        t = float(min(max(t, 0.0), 1.0))
        Lmin, Lmax = float(cfg['length_min']), float(cfg['length_max'])
        L_center = Lmin + t * (Lmax - Lmin)
        jit = float(cfg.get('length_jitter', 0.15))
        L = L_center * (1.0 + (torch.rand(self.N, device=device) * 2 - 1) * jit)
        self.target_length = L.clamp(Lmin, Lmax)

        # ── exact speed profile: v_cruise so Σ speed·dt = L ──────────────────
        k_ramp = min(int(round(float(cfg['ramp_frac']) * self.K)), self.K // 2)
        shape  = speed_profile(self.K, k_ramp).to(device)            # [K]
        v_cruise = self.target_length / (shape.sum() * self.dt)      # [N]
        self.speed = shape.unsqueeze(0) * v_cruise.unsqueeze(1)      # [N, K]

        # ── curriculum: curviness scales steeply with D ──────────────────────
        self.turn_decay = float(cfg['turn_decay'])
        self.turn_sigma = (self.D ** float(cfg.get('wander_exp', 2.0))) \
                          * float(cfg['disturb'])
        self.k_avoid       = float(cfg['k_avoid'])
        self.max_turn      = float(np.radians(cfg.get('max_turn_deg', 10.0)))
        self.bound_margin  = float(cfg['bound_margin'])
        self.body_margin   = float(cfg['body_margin'])
        self.ground_margin = float(cfg['ground_margin'])

        # ── XY drift direction ────────────────────────────────────────────────
        # Per-trajectory unit XY heading.  The wander is anchored to this
        # direction so the path travels (rather than folds), so net XY span
        # scales with L → the dog's base must translate to follow long paths.
        # The Z component is left at 0 so vertical excursion comes only from
        # the wander (≈ ±0.3 m at D=1), not from a constant climb/descent.
        d_xy = torch.randn(self.N, 2, device=device)
        d_xy = d_xy / d_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.dir_drift = torch.cat(
            [d_xy, torch.zeros(self.N, 1, device=device)], dim=-1)
        self.dir    = self.dir_drift.clone()
        self.wnoise = torch.zeros(self.N, 3, device=device)
        self.length = torch.zeros(self.N, device=device)
        self._last_sdf_w = None
        self._last_grad_w = None
        self._last_sdf_b = None
        self._last_grad_b = None

    # ── sampling ──────────────────────────────────────────────────────────────

    def _sample_safe(self, n: int, sdf_lo: float, sdf_hi: float,
                     use_body: bool) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, 3, device=self.device)
        fg = self.fg
        parts, total, tries = [], 0, 0
        while total < n:
            c = (torch.rand(max(n * 8, 4096), 3, device=self.device) * 2 - 1) * fg.gh
            sv, _ = fg.query_field(fg._field_curr, c)
            ok = (sv > sdf_lo) & (sv < sdf_hi) & (c[:, 2] > fg.ground_z + 0.10)
            if use_body:
                bv, _ = fg.query_field(fg.field_body, c)
                ok = ok & (bv > 0.13)
            if self.stage != 'outreach':
                ok = ok & self._command_limit_mask(c)
            parts.append(c[ok])
            total += int(ok.sum())
            tries += 1
            if tries > 400:
                raise RuntimeError("_sample_safe: safe region too small")
        return torch.cat(parts)[:n]

    def _sample_body(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, 3, device=self.device)
        fg = self.fg
        bc = torch.tensor(fg.body_box_center, device=self.device, dtype=torch.float32)
        bs = torch.tensor(fg.body_box_size,   device=self.device, dtype=torch.float32)
        parts, total, tries = [], 0, 0
        while total < n:
            u = torch.rand(max(n * 4, 2048), 3, device=self.device) * 2 - 1
            c = bc + u * (bs * 0.45)
            bv, _ = fg.query_field(fg.field_body, c)
            ok = (bv < -0.02) & (c[:, 2] > fg.ground_z + 0.05)
            parts.append(c[ok])
            total += int(ok.sum())
            tries += 1
            if tries > 400:
                raise RuntimeError("_sample_body: body region too small")
        return torch.cat(parts)[:n]

    def _command_limit_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.command_limits:
            return torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        lpy = xyz_to_lpy(x)
        ok = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        keys = ('l', 'p', 'y')
        for i, key in enumerate(keys):
            if key in self.command_limits:
                lo, hi = self.command_limits[key]
                ok = ok & (lpy[:, i] >= float(lo)) & (lpy[:, i] <= float(hi))
        return ok

    def _command_limit_avoidance(self, x: torch.Tensor) -> torch.Tensor:
        if not self.command_limits or self.stage == 'outreach':
            return torch.zeros_like(x)

        a = torch.zeros_like(x)
        eps = 1e-6
        x0, y0, z0 = x.unbind(-1)
        rxy = torch.sqrt(x0 * x0 + y0 * y0).clamp(min=eps)
        l = torch.sqrt(x0 * x0 + y0 * y0 + z0 * z0).clamp(min=eps)
        yaw = torch.atan2(y0, x0)
        polar = torch.atan2(z0, rxy)
        unit_l = x / l.unsqueeze(-1)

        def add_limit_barrier(value, grad, lo, hi, margin):
            nonlocal a
            grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=eps)
            low = ((float(lo) + margin - value) / margin).clamp(min=0.0)
            high = ((value - (float(hi) - margin)) / margin).clamp(min=0.0)
            a = a + low.unsqueeze(-1) * grad - high.unsqueeze(-1) * grad

        if 'l' in self.command_limits:
            lo, hi = self.command_limits['l']
            add_limit_barrier(l, unit_l, lo, hi, self.command_margin_l)
        if 'y' in self.command_limits:
            lo, hi = self.command_limits['y']
            grad_yaw = torch.stack([-y0 / (rxy * rxy),
                                    x0 / (rxy * rxy),
                                    torch.zeros_like(x0)], dim=-1)
            add_limit_barrier(yaw, grad_yaw, lo, hi, self.command_margin_angle)
        if 'p' in self.command_limits:
            lo, hi = self.command_limits['p']
            l2 = (l * l).clamp(min=eps)
            grad_p = torch.stack([
                -z0 * x0 / (l2 * rxy),
                -z0 * y0 / (l2 * rxy),
                rxy / l2,
            ], dim=-1)
            add_limit_barrier(polar, grad_p, lo, hi, self.command_margin_angle)
        return a

    # ── barrier avoidance ─────────────────────────────────────────────────────

    def _avoidance(self, x: torch.Tensor) -> torch.Tensor:
        fg = self.fg
        m = self.bound_margin
        a = torch.zeros_like(x)

        sdf_w, grad_w = fg.query_field(fg._field_curr, x)
        self._last_sdf_w = sdf_w
        self._last_grad_w = grad_w
        gw = grad_w / grad_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ramp_in = ((sdf_w - self.shell_hi + m) / m).clamp(min=0.0)
        a = a - ramp_in.unsqueeze(-1) * gw

        if self.body_avoid > 0.0:
            sdf_b, grad_b = fg.query_field(fg.field_body, x)
            self._last_sdf_b = sdf_b
            self._last_grad_b = grad_b
            gb = grad_b / grad_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            ramp_bd = self.body_avoid * ((self.body_margin - sdf_b)
                                         / self.body_margin).clamp(min=0.0)
            a = a + ramp_bd.unsqueeze(-1) * gb
        else:
            self._last_sdf_b = None
            self._last_grad_b = None

        h = x[:, 2] - fg.ground_z
        ramp_g = ((self.ground_margin - h) / self.ground_margin).clamp(min=0.0)
        a[:, 2] = a[:, 2] + ramp_g
        cage_hi = ((x - (fg.gh - self.grid_margin)) / self.grid_margin).clamp(min=0.0)
        cage_lo = (((-fg.gh + self.grid_margin) - x) / self.grid_margin).clamp(min=0.0)
        a = a - cage_hi + cage_lo
        a = a + self._command_limit_avoidance(x)
        return a

    def _renorm_direction(self, d: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = d.norm(dim=-1, keepdim=True)
        fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.where(norm > 1e-6, d / norm.clamp(min=1e-6), fallback)

    def _damp_forbidden_direction(self, d: torch.Tensor) -> torch.Tensor:
        if self._last_sdf_w is not None and self._last_grad_w is not None:
            gw = self._last_grad_w / self._last_grad_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            ramp = ((self._last_sdf_w - self.shell_hi + self.bound_margin)
                    / self.bound_margin).clamp(0.0, 1.0).unsqueeze(-1)
            outward = (d * gw).sum(dim=-1, keepdim=True).clamp(min=0.0)
            d = d - ramp * outward * gw
            d = self._renorm_direction(d, -gw)

        if self.body_avoid > 0.0 and self._last_sdf_b is not None and self._last_grad_b is not None:
            gb = self._last_grad_b / self._last_grad_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            ramp = ((self.body_margin - self._last_sdf_b)
                    / self.body_margin).clamp(0.0, 1.0).unsqueeze(-1)
            inward = (d * gb).sum(dim=-1, keepdim=True).clamp(max=0.0)
            d = d - ramp * inward * gb
            d = self._renorm_direction(d, gb)

        h = self.x[:, 2] - self.fg.ground_z
        ramp_g = ((self.ground_margin - h) / self.ground_margin).clamp(0.0, 1.0)
        downward = d[:, 2].clamp(max=0.0)
        d[:, 2] = d[:, 2] - ramp_g * downward

        ramp_hi = ((self.x - (self.fg.gh - self.grid_margin))
                   / self.grid_margin).clamp(0.0, 1.0)
        ramp_lo = (((-self.fg.gh + self.grid_margin) - self.x)
                   / self.grid_margin).clamp(0.0, 1.0)
        d = d - ramp_hi * d.clamp(min=0.0)
        d = d - ramp_lo * d.clamp(max=0.0)
        d = self._damp_command_direction(d)
        fallback_up = torch.zeros_like(d)
        fallback_up[:, 2] = 1.0
        return self._renorm_direction(d, fallback_up)

    def _damp_command_direction(self, d: torch.Tensor) -> torch.Tensor:
        if not self.command_limits or self.stage == 'outreach':
            return d

        eps = 1e-6
        x = self.x
        x0, y0, z0 = x.unbind(-1)
        rxy = torch.sqrt(x0 * x0 + y0 * y0).clamp(min=eps)
        l = torch.sqrt(x0 * x0 + y0 * y0 + z0 * z0).clamp(min=eps)
        yaw = torch.atan2(y0, x0)
        polar = torch.atan2(z0, rxy)
        unit_l = x / l.unsqueeze(-1)

        def damp_scalar(value, grad, lo, hi, margin):
            nonlocal d
            grad = grad / grad.norm(dim=-1, keepdim=True).clamp(min=eps)
            comp = (d * grad).sum(dim=-1, keepdim=True)
            high = ((value - (float(hi) - margin)) / margin).clamp(0.0, 1.0).unsqueeze(-1)
            low = (((float(lo) + margin) - value) / margin).clamp(0.0, 1.0).unsqueeze(-1)
            d = d - high * comp.clamp(min=0.0) * grad
            d = d - low * comp.clamp(max=0.0) * grad

        if 'l' in self.command_limits:
            lo, hi = self.command_limits['l']
            damp_scalar(l, unit_l, lo, hi, self.command_margin_l)
        if 'y' in self.command_limits:
            lo, hi = self.command_limits['y']
            grad_yaw = torch.stack([-y0 / (rxy * rxy),
                                    x0 / (rxy * rxy),
                                    torch.zeros_like(x0)], dim=-1)
            damp_scalar(yaw, grad_yaw, lo, hi, self.command_margin_angle)
        if 'p' in self.command_limits:
            lo, hi = self.command_limits['p']
            l2 = (l * l).clamp(min=eps)
            grad_p = torch.stack([
                -z0 * x0 / (l2 * rxy),
                -z0 * y0 / (l2 * rxy),
                rxy / l2,
            ], dim=-1)
            damp_scalar(polar, grad_p, lo, hi, self.command_margin_angle)
        # Fall back to the previous frame's direction (always [N, 3]) if the
        # damping cancelled everything out — gives a smooth recovery rather
        # than snapping to a coordinate axis.
        return self._renorm_direction(d, self.dir)

    # ── integration ───────────────────────────────────────────────────────────

    def step(self, k: int):
        s_k = self.speed[:, k]

        avoid    = self._avoidance(self.x)
        strength = avoid.norm(dim=-1, keepdim=True)

        # 1. OU wander (∝ D), faded smoothly to 0 near barriers so the
        #    disturbance never fights the avoidance.
        self.wnoise = ((1.0 - self.turn_decay) * self.wnoise
                       + self.turn_sigma * torch.randn(self.N, 3, device=self.device))
        wander_scale = (1.0 - strength).clamp(0.0, 1.0)

        # 2. Direction anchor.  Far from any barrier we anchor on `dir_drift`
        #    (a fixed per-trajectory XY heading) so the wander does NOT
        #    accumulate and the path keeps progressing in one XY direction.
        #    Near a barrier we anchor on the previous direction (which the
        #    avoidance had already rotated to safety), so the avoidance work
        #    is not immediately undone.  Both factors fade smoothly via
        #    `wander_scale`, preserving C¹ heading continuity.
        anchor = wander_scale * self.dir_drift + (1.0 - wander_scale) * self.dir
        anchor = anchor / anchor.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 3. Project wander onto the plane PERPENDICULAR to the anchor.  This
        #    is critical: an unprojected wander whose magnitude exceeds 1 and
        #    happens to point anti-parallel to the anchor would flip the
        #    direction (anchor + (-1.5)·anchor → −0.5·anchor → normalised flip).
        #    The perp-only component can only rotate, never flip — the angle
        #    from `anchor` is bounded by atan(|wander_scale · wn_perp|) < 90°.
        wn_para = (self.wnoise * anchor).sum(-1, keepdim=True) * anchor
        wn_perp = self.wnoise - wn_para
        d_target = anchor + wander_scale * wn_perp
        d_target = d_target / d_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 4. Fold the avoidance into the target direction: rotate `d_target`
        #    toward `safe` by a bounded amount before steering self.dir there.
        safe   = avoid / strength.clamp(min=1e-6)
        s_alg  = (d_target * safe).sum(-1, keepdim=True).clamp(-1.0, 1.0)
        s_turn = (self.k_avoid * strength * (1.0 - s_alg)).clamp(max=self.max_turn)
        s_prp  = safe - s_alg * d_target
        s_prp  = s_prp / s_prp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        d_target = torch.cos(s_turn) * d_target + torch.sin(s_turn) * s_prp

        # 5. Damping safety net — projects forbidden velocity components out of
        #    `d_target` (workspace shell, body, ground, grid cage, command
        #    cone).  Applied to the TARGET, not to self.dir, so the single
        #    bounded rotation in step 6 still caps the per-step heading change.
        d_target = self._damp_forbidden_direction(d_target)

        # 6. Single bounded rotation from self.dir to d_target, capped at
        #    max_turn per step.  Because this is the ONLY place self.dir
        #    changes, the per-step heading change is GUARANTEED ≤ max_turn —
        #    no large wander tail or aggressive damping can violate it.
        align = (self.dir * d_target).sum(-1, keepdim=True).clamp(-1.0, 1.0)
        turn  = torch.acos(align).clamp(max=self.max_turn)
        perp  = d_target - align * self.dir
        perp  = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.dir = torch.cos(turn) * self.dir + torch.sin(turn) * perp

        v = s_k.unsqueeze(-1) * self.dir
        self.x = self.x + v * self.dt
        self.length = self.length + s_k * self.dt
        return self.x.clone(), v

    def simulate(self):
        """Returns history_x [N,K,3], history_v [N,K,3]."""
        if self.stage == 'point':
            hx = self.x.unsqueeze(1).expand(self.N, self.K, 3).contiguous()
            hv = torch.zeros(self.N, self.K, 3, device=self.device)
            return hx, hv
        hx = torch.empty(self.N, self.K, 3, device=self.device)
        hv = torch.empty(self.N, self.K, 3, device=self.device)
        for k in range(self.K):
            hx[:, k], hv[:, k] = self.step(k)
        return hx, hv


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers — matches env's (l, p, y) ↔ XYZ convention
#   (see LeggedRobot._lpy_to_world_xyz / get_lpy_in_base_coord)
# ─────────────────────────────────────────────────────────────────────────────

def xyz_to_lpy(xyz: torch.Tensor) -> torch.Tensor:
    """
    Cartesian (base frame) → cylindrical command (length, polar, yaw).
        l = ||xyz||,  p = atan2(z, sqrt(x²+y²)),  y = atan2(y, x)
    Inverse of LeggedRobot._lpy_to_world_xyz's base-frame portion.
    """
    x, y, z = xyz.unbind(-1)
    l = torch.sqrt(x * x + y * y + z * z)
    p = torch.atan2(z, torch.sqrt(x * x + y * y))
    yaw = torch.atan2(y, x)
    return torch.stack([l, p, yaw], dim=-1)


def lpy_to_xyz(lpy: torch.Tensor) -> torch.Tensor:
    """(l, p, y) → Cartesian (base frame).  Matches LeggedRobot._lpy_to_world_xyz."""
    l, p, y = lpy.unbind(-1)
    x = l * torch.cos(p) * torch.cos(y)
    yv = l * torch.cos(p) * torch.sin(y)
    z = l * torch.sin(p)
    return torch.stack([x, yv, z], dim=-1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Base yaw from an IsaacGym/RoboDuet quaternion in (x, y, z, w) order."""
    qx, qy, qz, qw = q.unbind(-1)
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    return torch.atan2(fy, fx)


def yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.sin(0.5 * yaw)
    w = torch.cos(0.5 * yaw)
    return torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), z, w], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion product q ⊗ r, both in (x, y, z, w) order."""
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    w = qw * rw - qx * rx - qy * ry - qz * rz
    out = torch.stack([x, y, z, w], dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _mean_terrain_height(measured_heights: Optional[torch.Tensor],
                         ref: torch.Tensor) -> torch.Tensor:
    if measured_heights is None:
        return torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
    h = measured_heights.to(device=ref.device, dtype=ref.dtype)
    if h.dim() == 0:
        return h.reshape(1).expand(ref.shape[0])
    if h.dim() == 1:
        return h
    return h.reshape(h.shape[0], -1).mean(dim=1)


def _unsqueeze_to_points(x: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while x.dim() < points.dim() - 1:
        x = x.unsqueeze(-1)
    return x


def base_xyz_to_world(
    xyz: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    measured_heights: Optional[torch.Tensor] = None,
    base_height: float = 0.38,
) -> torch.Tensor:
    """
    Convert arm-base-frame XYZ to world XYZ using the same yaw + terrain-height
    convention as LeggedRobot._lpy_to_world_xyz.
    """
    base_pos = base_pos.to(device=xyz.device, dtype=xyz.dtype)
    base_quat = base_quat.to(device=xyz.device, dtype=xyz.dtype)
    yaw = _unsqueeze_to_points(yaw_from_quat_xyzw(base_quat), xyz)
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = xyz.unbind(-1)
    wx = x * c - y * s + _unsqueeze_to_points(base_pos[:, 0], xyz)
    wy = x * s + y * c + _unsqueeze_to_points(base_pos[:, 1], xyz)
    terrain = _unsqueeze_to_points(_mean_terrain_height(measured_heights, base_pos), xyz)
    wz = z + terrain + float(base_height)
    return torch.stack([wx, wy, wz], dim=-1)


def world_xyz_to_base(
    xyz_world: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    measured_heights: Optional[torch.Tensor] = None,
    base_height: float = 0.38,
) -> torch.Tensor:
    """
    Convert world XYZ into the arm-base frame used by commands_arm.
    This is the inverse of base_xyz_to_world for the env's yaw-only command frame.
    """
    base_pos = base_pos.to(device=xyz_world.device, dtype=xyz_world.dtype)
    base_quat = base_quat.to(device=xyz_world.device, dtype=xyz_world.dtype)
    yaw = _unsqueeze_to_points(yaw_from_quat_xyzw(base_quat), xyz_world)
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = xyz_world[..., 0] - _unsqueeze_to_points(base_pos[:, 0], xyz_world)
    dy = xyz_world[..., 1] - _unsqueeze_to_points(base_pos[:, 1], xyz_world)
    bx = dx * c + dy * s
    by = -dx * s + dy * c
    terrain = _unsqueeze_to_points(_mean_terrain_height(measured_heights, base_pos), xyz_world)
    bz = xyz_world[..., 2] - terrain - float(base_height)
    return torch.stack([bx, by, bz], dim=-1)


def base_quat_to_world(q_base: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
    yaw_q = yaw_quat_xyzw(yaw_from_quat_xyzw(base_quat.to(q_base.device)))
    return quat_mul(yaw_q.to(dtype=q_base.dtype), q_base)


def world_quat_to_base(q_world: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
    yaw_q = yaw_quat_xyzw(yaw_from_quat_xyzw(base_quat.to(q_world.device)))
    return quat_mul(quat_conjugate(yaw_q.to(dtype=q_world.dtype)), q_world)


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryCurriculum — per-env runtime buffer for IsaacGym envs
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryCurriculum:
    """
    Per-env trajectory buffer for IsaacGym training.

    Construct once at env init.  At every env reset call `resample(env_ids, D)`;
    pass the current base pose to world-anchor the new target.  At every env
    control step call `advance()` and read `get_target_lpy_world(...)`.

    Internal buffers (GPU-resident):
        positions          [num_envs, K, 3]  EE position, arm-base frame
        world_positions    [num_envs, K, 3]  world target after anchoring
        quaternions        [num_envs, K, 4]  EE orientation, (x, y, z, w)
        world_quaternions  [num_envs, K, 4]  world orientation after anchoring
        velocities         [num_envs, K, 3]  EE linear velocity
        eulers             [num_envs, K, 3]  EE attitude as (roll, pitch, yaw)
        abg                [num_envs, K, 3]  env-reward 3-angle readout
        target_len         [num_envs]        commanded path length (m)
        stages             [num_envs]        per-env stage tag (0=point,1=vel,2=outreach)
        cursor             [num_envs] long   current sample index in [0, K)

    `K` is the number of trajectory samples generated per env; the simulator
    advances at `dt` seconds per sample, so a trajectory spans `K · dt`
    seconds of env time when you advance the cursor once per env step.
    """

    STAGE_ID = {'point': 0, 'velocity': 1, 'outreach': 2}

    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device] = 'cuda',
        fields_path: Optional[Union[str, Path]] = None,
        config: ConfigLike = 'configs/trajectory_generation.yaml',
        K: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        """
        num_envs    : number of per-env trajectory buffers to allocate.
        device      : torch device for the buffers and the simulator.
        fields_path : pre-computed fields file (.pt).  Overrides
                      `generator.fields_path` in the config.  When both are
                      None, defaults to `runs/langevin/precomputed_fields.pt`.
        config      : dict, or path to a self-contained YAML.  No `base_config`
                      indirection: K, dt, fields_path, velocity, attitude live
                      at the top level — see configs/trajectory_generation.yaml.
        K, dt       : explicit overrides for the trajectory length / step.
        """
        self.num_envs = int(num_envs)
        self.device   = str(device)

        # ── resolve config (dict / yaml path) — fully self-contained ─────────
        cfg = self._load_config(config)
        self.vel_cfg = cfg['velocity']
        self.att_cfg = cfg.get('attitude', {})
        gen_cfg = cfg.get('generator', {})

        self.K  = int(K  if K  is not None else gen_cfg.get('K',  300))
        self.dt = float(dt if dt is not None else gen_cfg.get('dt', 0.01))
        self.base_height = float(gen_cfg.get('base_height', 0.38))
        self.verbose = bool(gen_cfg.get('verbose', False))

        # ── load pre-computed fields and build FieldGenerator ────────────────
        fields_path = Path(fields_path or gen_cfg.get('fields_path',
                                                      'runs/langevin/precomputed_fields.pt'))
        if not fields_path.exists():
            raise FileNotFoundError(
                f"Pre-computed fields not found at {fields_path}.\n"
                f"Run first:  python scripts/langevin_field_gen.py")
        ckpt = torch.load(fields_path, map_location='cpu', weights_only=False)
        for key in ('field_static', 'field_dynamic', 'field_body', 'field_orient'):
            if key not in ckpt:
                raise KeyError(f"{key} missing from {fields_path} — re-run "
                               f"langevin_field_gen.py")
        self._fg = ltg.FieldGenerator(
            field_static    = ckpt['field_static'],
            field_dynamic   = ckpt['field_dynamic'],
            field_body      = ckpt['field_body'],
            grid_half       = float(ckpt['grid_half']),
            ground_z        = float(ckpt['ground_z']),
            device          = self.device,
            body_box_center = ckpt.get('body_box_center', (0.0, 0.0, 0.0)),
            body_box_size   = ckpt.get('body_box_size', (0.3762, 0.13, 0.114)),
        )
        self._grid_half     = float(ckpt['grid_half'])
        self._field_orient  = ckpt['field_orient']
        self._orient_limits = list(ckpt.get('orient_limits', (1.4137, 1.0472, 1.3090)))

        # ── per-env GPU buffers ──────────────────────────────────────────────
        N, K = self.num_envs, self.K
        d = self.device
        self.positions   = torch.zeros(N, K, 3, device=d)
        self.world_positions = torch.zeros(N, K, 3, device=d)
        self.quaternions = torch.zeros(N, K, 4, device=d)
        self.world_quaternions = torch.zeros(N, K, 4, device=d)
        self.quaternions[..., 3] = 1.0                            # identity (w=1)
        self.world_quaternions[..., 3] = 1.0
        self.velocities  = torch.zeros(N, K, 3, device=d)
        self.eulers      = torch.zeros(N, K, 3, device=d)
        self.abg         = torch.zeros(N, K, 3, device=d)
        self.target_len  = torch.zeros(N,        device=d)
        self.stages      = torch.zeros(N, dtype=torch.long, device=d)
        self.cursor      = torch.zeros(N, dtype=torch.long, device=d)
        self.has_world_anchor = torch.zeros(N, dtype=torch.bool, device=d)

    # ── config helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(cfg: ConfigLike) -> dict:
        if isinstance(cfg, dict):
            return cfg
        with open(cfg) as f:
            return yaml.safe_load(f)

    def difficulty_stage(self, D: float) -> str:
        """'point' | 'velocity' | 'outreach' for a given scalar D."""
        vc = self.vel_cfg
        if D <= float(vc['point_max']):
            return 'point'
        if D > float(vc.get('outreach_min', 0.7)):
            return 'outreach'
        return 'velocity'

    # ── core: resample (the per-reset entry point) ────────────────────────────

    @torch.no_grad()
    def resample(
        self,
        env_ids: Optional[torch.Tensor],
        D: float,
        base_pos: Optional[torch.Tensor] = None,
        base_quat: Optional[torch.Tensor] = None,
        measured_heights: Optional[torch.Tensor] = None,
        world_anchor: Optional[bool] = None,
        base_height: Optional[float] = None,
    ) -> None:
        """
        Generate fresh trajectories at difficulty `D` for the given env_ids
        and write them into the per-env buffers (cursor reset to 0).

        env_ids : long tensor of env indices, or None for "all envs".
        D       : scalar difficulty in [0, 1].
        base_pos/base_quat/measured_heights : pass these at reset to anchor the
                  trajectory in world coordinates.  Then use
                  get_target_lpy_world(...) every step.
        """
        env_ids = self._resolve_env_ids(env_ids)
        n = int(env_ids.numel())
        if n == 0:
            return

        # ── position ──────────────────────────────────────────────────────────
        sim = VelocityTrajectorySimulator(
            self._fg, float(D), n, self.K, self.dt, self.vel_cfg, self.device)
        pos, vel = sim.simulate()                                  # [n,K,3]

        # ── attitude (position-conditioned feasible orientation) ──────────────
        att = ptg.AttitudeSimulator(
            float(D), sim.stage, n, self.K, self.dt, self.att_cfg,
            self._field_orient, self._grid_half, self._orient_limits, self.device)
        if self.verbose:
            eul = att.simulate(pos)                                # [n,K,3]
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                eul = att.simulate(pos)
        quat = ptg.euler_to_quat(eul)                              # [n,K,4]
        abg  = ptg.quat_to_abg(quat)                               # [n,K,3]

        # ── scatter into per-env buffers ──────────────────────────────────────
        self.positions[env_ids]   = pos
        self.world_positions[env_ids] = pos
        self.velocities[env_ids]  = vel
        self.quaternions[env_ids] = quat
        self.world_quaternions[env_ids] = quat
        self.eulers[env_ids]      = eul
        self.abg[env_ids]         = abg
        self.target_len[env_ids]  = sim.target_length
        self.stages[env_ids]      = self.STAGE_ID[sim.stage]
        self.cursor[env_ids]      = 0
        self.has_world_anchor[env_ids] = False

        should_anchor = (base_pos is not None) or (base_quat is not None)
        if world_anchor is not None:
            should_anchor = bool(world_anchor)
        if should_anchor:
            self.anchor_world(env_ids, base_pos, base_quat, measured_heights, base_height)

    @torch.no_grad()
    def anchor_world(
        self,
        env_ids: Optional[torch.Tensor],
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        measured_heights: Optional[torch.Tensor] = None,
        base_height: Optional[float] = None,
    ) -> None:
        """Pin current base-frame trajectories to world coordinates at reset."""
        if base_pos is None or base_quat is None:
            raise ValueError("anchor_world requires base_pos and base_quat")
        env_ids = self._resolve_env_ids(env_ids)
        pos = self._select_state(base_pos, env_ids, 'base_pos')
        quat = self._select_state(base_quat, env_ids, 'base_quat')
        heights = self._select_optional_state(measured_heights, env_ids, 'measured_heights')
        bh = self.base_height if base_height is None else float(base_height)
        self.world_positions[env_ids] = base_xyz_to_world(
            self.positions[env_ids], pos, quat, heights, bh)
        self.world_quaternions[env_ids] = base_quat_to_world(
            self.quaternions[env_ids], quat.unsqueeze(1).expand(-1, self.K, -1))
        self.has_world_anchor[env_ids] = True

    # ── cursor management ─────────────────────────────────────────────────────

    @torch.no_grad()
    def advance(self, env_ids: Optional[torch.Tensor] = None, by: int = 1) -> None:
        """Increment each env's cursor by `by`, clamped to K-1."""
        env_ids = self._resolve_env_ids(env_ids)
        self.cursor[env_ids] = (self.cursor[env_ids] + by).clamp(max=self.K - 1)

    @torch.no_grad()
    def set_cursor(self, env_ids: Optional[torch.Tensor], k: TensorOrFloat) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        if torch.is_tensor(k):
            self.cursor[env_ids] = k.to(self.cursor.dtype).to(self.cursor.device)
        else:
            self.cursor[env_ids] = int(k)

    @torch.no_grad()
    def is_done(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Bool tensor — True for envs whose cursor is at the final sample."""
        env_ids = self._resolve_env_ids(env_ids)
        return self.cursor[env_ids] >= (self.K - 1)

    # ── reads (per-env, current cursor) ───────────────────────────────────────

    @torch.no_grad()
    def get_target(self, env_ids: Optional[torch.Tensor] = None):
        """
        Returns the EE reference pose at each env's current cursor.

            pos  : [B, 3]   arm-base-frame Cartesian (m)
            quat : [B, 4]   (x, y, z, w)
        """
        env_ids = self._resolve_env_ids(env_ids)
        k = self.cursor[env_ids]
        pos  = self.positions[env_ids,  k]
        quat = self.quaternions[env_ids, k]
        return pos, quat

    @torch.no_grad()
    def get_target_lpy(self, env_ids: Optional[torch.Tensor] = None):
        """
        Same as `get_target` but the position is in the env's cylindrical
        (length, polar, yaw) form.  This is body-fixed; whole-body training
        should usually use get_target_lpy_world(...) instead.
        """
        pos, quat = self.get_target(env_ids)
        return xyz_to_lpy(pos), quat

    @torch.no_grad()
    def get_target_world(self, env_ids: Optional[torch.Tensor] = None):
        """Returns the world-anchored target pose at the current cursor."""
        env_ids = self._resolve_env_ids(env_ids)
        if not self.has_world_anchor[env_ids].all():
            raise RuntimeError("get_target_world called before anchor_world/resample(..., base_pos, base_quat)")
        k = self.cursor[env_ids]
        return self.world_positions[env_ids, k], self.world_quaternions[env_ids, k]

    @torch.no_grad()
    def get_target_lpy_world(
        self,
        env_ids: Optional[torch.Tensor] = None,
        base_pos: Optional[torch.Tensor] = None,
        base_quat: Optional[torch.Tensor] = None,
        measured_heights: Optional[torch.Tensor] = None,
        base_height: Optional[float] = None,
    ):
        """
        World-anchored target converted back to current base-frame commands.
        Use this for whole-body training so base motion changes the L/P/Y error.
        """
        if base_pos is None or base_quat is None:
            raise ValueError("get_target_lpy_world requires current base_pos and base_quat")
        env_ids = self._resolve_env_ids(env_ids)
        pos_w, quat_w = self.get_target_world(env_ids)
        pos = self._select_state(base_pos, env_ids, 'base_pos')
        quat = self._select_state(base_quat, env_ids, 'base_quat')
        heights = self._select_optional_state(measured_heights, env_ids, 'measured_heights')
        bh = self.base_height if base_height is None else float(base_height)
        pos_base = world_xyz_to_base(pos_w, pos, quat, heights, bh)
        quat_base = world_quat_to_base(quat_w, quat)
        return xyz_to_lpy(pos_base), quat_base

    @torch.no_grad()
    def get_target_abg_world(
        self,
        env_ids: Optional[torch.Tensor] = None,
        base_pos: Optional[torch.Tensor] = None,
        base_quat: Optional[torch.Tensor] = None,
        measured_heights: Optional[torch.Tensor] = None,
        base_height: Optional[float] = None,
    ):
        """World-anchored target as current base-frame Cartesian position + ABG."""
        lpy, quat = self.get_target_lpy_world(
            env_ids, base_pos, base_quat, measured_heights, base_height)
        return lpy_to_xyz(lpy), ptg.quat_to_abg(quat)

    @torch.no_grad()
    def get_target_abg(self, env_ids: Optional[torch.Tensor] = None):
        """Returns (pos [B,3], abg [B,3]) — abg is the env reward's 3-angle readout."""
        env_ids = self._resolve_env_ids(env_ids)
        k = self.cursor[env_ids]
        return self.positions[env_ids, k], self.abg[env_ids, k]

    @torch.no_grad()
    def get_velocity(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        env_ids = self._resolve_env_ids(env_ids)
        return self.velocities[env_ids, self.cursor[env_ids]]

    # ── reads (per-env, full trajectory) ──────────────────────────────────────

    @torch.no_grad()
    def get_full(self, env_ids: Optional[torch.Tensor] = None):
        """Returns the full trajectories (pos [B,K,3], quat [B,K,4], target_length [B])."""
        env_ids = self._resolve_env_ids(env_ids)
        return (self.positions[env_ids],
                self.quaternions[env_ids],
                self.target_len[env_ids])

    # ── utilities ─────────────────────────────────────────────────────────────

    def _resolve_env_ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return env_ids.to(self.device).long()

    def _select_state(self, value, env_ids, name: str):
        if value is None:
            raise ValueError(f"{name} is required")
        t = value if torch.is_tensor(value) else torch.as_tensor(value)
        t = t.to(self.device)
        if t.shape[0] == self.num_envs:
            return t[env_ids]
        if t.shape[0] == env_ids.numel():
            return t
        raise ValueError(
            f"{name} first dimension must be num_envs ({self.num_envs}) "
            f"or len(env_ids) ({env_ids.numel()}), got {t.shape[0]}")

    def _select_optional_state(self, value, env_ids, name: str):
        if value is None:
            return None
        return self._select_state(value, env_ids, name)

    def __repr__(self) -> str:
        return (f"TrajectoryCurriculum(num_envs={self.num_envs}, K={self.K}, "
                f"dt={self.dt}, device='{self.device}')")


# ─────────────────────────────────────────────────────────────────────────────
# CLI: a short sanity check (not a visualisation tool — see test_velocity.py)
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_check():
    import argparse
    ap = argparse.ArgumentParser(
        description='Sanity-check the TrajectoryCurriculum library: build it, '
                    'resample a few envs at several D values, print stats.')
    ap.add_argument('--config',      default='configs/trajectory_generation.yaml')
    ap.add_argument('--fields',      default='runs/langevin/precomputed_fields.pt')
    ap.add_argument('--num_envs',    type=int, default=64)
    ap.add_argument('--K',           type=int, default=None)
    ap.add_argument('--dt',          type=float, default=None)
    ap.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--difficulties', type=float, nargs='+',
                    default=[0.0, 0.3, 0.6, 0.9])
    args = ap.parse_args()

    curr = TrajectoryCurriculum(
        num_envs    = args.num_envs,
        device      = args.device,
        fields_path = args.fields,
        config      = args.config,
        K           = args.K,
        dt          = args.dt,
    )
    print(curr)

    for D in args.difficulties:
        t0 = time.perf_counter()
        curr.resample(None, D)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3

        stage = curr.difficulty_stage(D)
        L = curr.target_len
        pos, quat = curr.get_target()                              # cursor still 0
        ach = (curr.positions[:, 1:] - curr.positions[:, :-1]).norm(dim=-1).sum(dim=1)
        err_mm = (ach - L).abs().max().item() * 1e3
        print(f"D={D:.2f}  stage={stage:8s}  resample={dt_ms:6.1f} ms  "
              f"length mean={L.mean():.2f} m  [{L.min():.2f}, {L.max():.2f}]  "
              f"|err|≤{err_mm:.2f} mm")

        # walk every env through its full trajectory
        for _ in range(curr.K - 1):
            curr.advance()
        assert curr.is_done().all(), 'all envs should be done at the last sample'

    print('OK.')


if __name__ == '__main__':
    _sanity_check()
