"""M1: SE(3) trajectory representation.

Standalone module -- pure PyTorch (+ scipy for one-off offline filtering /
resampling in ``to_canonical``), no IsaacGym dependency. Implements the
``docs/coding-agent-guide.md`` M1 spec: an arc-length-parameterized geometric
path ``Gamma``, a time law ``s_ref(t)``, a batched GPU container
``TrajectoryBatch``, and the two entry points that turn a raw pose sequence
into that representation (``to_canonical``) and track progress along it
(``update_s``).

Rotations are represented as (..., 3, 3) matrices at the public API boundary
(matching the design doc), with quaternions used internally only where SLERP
needs them.

Not implemented here (out of scope for this pass, see M2/M3 in the design
doc): direction-dependent reachability (``ReachabilityTable``) and the
preview-averaged online base feedforward (``OnlineBaseNom``) -- both are
separate modules that would consume ``Gamma``/``TrajectoryBatch`` from here.

Perf note: ``TrajectoryBatch``'s per-step methods (``p_at``/``R_at``/
``sample_preview``) are vectorized across the batch dimension (no Python loop
over envs) via ``torch.searchsorted`` + ``gather``, but this module has not
been profiled against the design doc's N=4096/<0.1ms target -- it's written
for correctness and to be a faithful, reusable M1 implementation, not yet
perf-validated for production RL rollout use.
"""

import math

import numpy as np
import scipy.interpolate
import scipy.signal
import torch

# ============================================================
# Rotation helpers (batched, arbitrary leading dims)
# ============================================================


def mat_to_quat(R):
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion, xyzw order."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    s1 = torch.sqrt((trace + 1.0).clamp_min(1e-12)) * 2
    x1, y1, z1, w1 = (m21 - m12) / s1, (m02 - m20) / s1, (m10 - m01) / s1, 0.25 * s1

    s2 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    x2, y2, z2, w2 = 0.25 * s2, (m01 + m10) / s2, (m02 + m20) / s2, (m21 - m12) / s2

    s3 = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-12)) * 2
    x3, y3, z3, w3 = (m01 + m10) / s3, 0.25 * s3, (m12 + m21) / s3, (m02 - m20) / s3

    s4 = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-12)) * 2
    x4, y4, z4, w4 = (m02 + m20) / s4, (m12 + m21) / s4, 0.25 * s4, (m10 - m01) / s4

    c1 = trace > 0
    c2 = (~c1) & (m00 > m11) & (m00 > m22)
    c3 = (~c1) & (~c2) & (m11 > m22)

    x = torch.where(c1, x1, torch.where(c2, x2, torch.where(c3, x3, x4)))
    y = torch.where(c1, y1, torch.where(c2, y2, torch.where(c3, y3, y4)))
    z = torch.where(c1, z1, torch.where(c2, z2, torch.where(c3, z3, z4)))
    w = torch.where(c1, w1, torch.where(c2, w2, torch.where(c3, w3, w4)))

    q = torch.stack([x, y, z, w], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_to_mat(q):
    """(..., 4) quaternion xyzw -> (..., 3, 3) rotation matrix."""
    n = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = (q / n).unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_slerp(q0, q1, frac):
    """Batched SLERP. q0, q1: (..., 4). frac: (...,) in [0, 1]."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-6)
    frac_ = frac.unsqueeze(-1)
    w0 = torch.sin((1.0 - frac_) * theta) / sin_theta
    w1 = torch.sin(frac_ * theta) / sin_theta
    slerp = w0 * q0 + w1 * q1
    lerp = q0 + frac_ * (q1 - q0)
    out = torch.where(theta.abs() < 1e-4, lerp, slerp)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def so3_log(R):
    """(..., 3, 3) -> (..., 3) rotation vector (axis * angle), angle in [0, pi]."""
    q = mat_to_quat(R)
    xyz = q[..., :3]
    w = q[..., 3]
    xyz_norm = xyz.norm(dim=-1)
    angle = 2 * torch.atan2(xyz_norm, w)
    axis = xyz / xyz_norm.clamp_min(1e-12).unsqueeze(-1)
    small = xyz_norm < 1e-8
    axis = torch.where(small.unsqueeze(-1), torch.zeros_like(axis), axis)
    return axis * angle.unsqueeze(-1)


def so3_exp(rotvec):
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1e-12)
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    small = angle.squeeze(-1) < 1e-8
    xyz = torch.where(small.unsqueeze(-1), torch.zeros_like(xyz), xyz)
    w = torch.where(small.unsqueeze(-1), torch.ones_like(w), w)
    q = torch.cat([xyz, w], dim=-1)
    return quat_to_mat(q)


def _make_quat_continuous(q):
    """Flip signs along the leading (time) axis so consecutive quaternions
    take the shorter path -- required before filtering quaternion components
    independently, otherwise a sign flip looks like a huge discontinuity."""
    q = q.clone()
    for t in range(1, q.shape[0]):
        if (q[t] * q[t - 1]).sum() < 0:
            q[t] = -q[t]
    return q


# ============================================================
# Gamma: arc-length-parameterized SE(3) geometric path
# ============================================================


class Gamma:
    """Arc-length-parameterized SE(3) geometric path (single trajectory)."""

    def __init__(self, s_grid, p, R, tangent, lam=0.15):
        """
        s_grid: (M,)     ascending arc-length samples
        p:      (M, 3)   position
        R:      (M, 3, 3) rotation
        tangent:(M, 3)   unit tangent direction
        lam:    float    rotation-translation equivalence factor (m/rad)
        """
        self.s_grid = s_grid
        self.p = p
        self.R = R
        self.tangent = tangent
        self.lam = lam
        self.L = float(s_grid[-1].item())

    def _locate(self, s):
        s_grid = self.s_grid
        s_clamped = s.clamp(s_grid[0], s_grid[-1])
        idx = torch.searchsorted(s_grid, s_clamped, right=True).clamp(1, s_grid.shape[0] - 1) - 1
        s0 = s_grid[idx]
        s1 = s_grid[idx + 1]
        frac = ((s_clamped - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0.0, 1.0)
        return idx, frac

    def p_at(self, s):
        """s: (batch,) -> (batch, 3), linear interpolation."""
        idx, frac = self._locate(s)
        p0, p1 = self.p[idx], self.p[idx + 1]
        return p0 + frac.unsqueeze(-1) * (p1 - p0)

    def R_at(self, s):
        """s: (batch,) -> (batch, 3, 3), SLERP interpolation."""
        idx, frac = self._locate(s)
        q0, q1 = mat_to_quat(self.R[idx]), mat_to_quat(self.R[idx + 1])
        return quat_to_mat(quat_slerp(q0, q1, frac))

    def tangent_at(self, s):
        """s: (batch,) -> (batch, 3), unit tangent."""
        idx, frac = self._locate(s)
        t0, t1 = self.tangent[idx], self.tangent[idx + 1]
        v = t0 + frac.unsqueeze(-1) * (t1 - t0)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def rot_6d_at(self, s):
        """s: (batch,) -> (batch, 6). First two columns of R, concatenated."""
        R = self.R_at(s)
        return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


class TimeLaw:
    """Time law s_ref(t): maps wall-clock time to reference arc length."""

    def __init__(self, t_grid, s_of_t, sdot_of_t):
        self.t_grid = t_grid
        self.s_of_t = s_of_t
        self.sdot_of_t = sdot_of_t
        self.T = float(t_grid[-1].item())

    def _interp(self, values, t):
        t_grid = self.t_grid
        t_clamped = t.clamp(t_grid[0], t_grid[-1])
        idx = torch.searchsorted(t_grid, t_clamped, right=True).clamp(1, t_grid.shape[0] - 1) - 1
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        frac = ((t_clamped - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0.0, 1.0)
        v0, v1 = values[idx], values[idx + 1]
        return v0 + frac * (v1 - v0)

    def s_ref(self, t):
        return self._interp(self.s_of_t, t)

    def sdot_ref(self, t):
        return self._interp(self.sdot_of_t, t)


# ============================================================
# TrajectoryBatch: batched GPU container, N envs each holding (gamma, time_law)
# ============================================================

_PREVIEW_FRACS = torch.tensor([0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.85, 1.0])


def preview_fractions(K, device=None):
    """The first K look-ahead fractions of the preview horizon (near-dense,
    far-sparse). Consumers that need to weight the preview points -- e.g. the
    base feedforward's exp(-2*Delta/L_h) blend -- must use these same
    fractions as ``sample_preview``, so they live in one place."""
    fracs = _PREVIEW_FRACS[:K]
    return fracs.to(device) if device is not None else fracs


class TrajectoryBatch:
    """GPU-resident container for N envs, each holding one (gamma, time_law)."""

    def __init__(self, N, max_gamma_points, max_tl_points=None, device="cpu"):
        """
        max_gamma_points: capacity of each env's arc-length-resampled Gamma grid
        max_tl_points:    capacity of each env's time-law grid (raw sample
                          count, typically much larger than max_gamma_points
                          since it's not arc-length-resampled) -- defaults to
                          max_gamma_points if omitted.
        """
        max_tl_points = max_tl_points or max_gamma_points
        self.N, self.max_gamma_points, self.max_tl_points, self.device = N, max_gamma_points, max_tl_points, device
        self.gamma_s = torch.zeros(N, max_gamma_points, device=device)
        self.gamma_p = torch.zeros(N, max_gamma_points, 3, device=device)
        # Orientations are stored as xyzw QUATERNIONS, not 3x3 matrices: 4
        # floats per point instead of 9 (at N=4096, M=1536 that is 96 MB
        # instead of 216 MB, and the pre-generated bank saves the same
        # fraction again), and every consumer -- SLERP here, the observation
        # builder, the viewer overlay -- wants a quaternion anyway, so this
        # also removes a matrix->quaternion conversion from the per-step path.
        # ``R_at`` still returns matrices for callers that want them.
        self.gamma_quat = torch.zeros(N, max_gamma_points, 4, device=device)
        self.gamma_quat[..., 3] = 1.0
        self.gamma_tangent = torch.zeros(N, max_gamma_points, 3, device=device)
        self.L = torch.zeros(N, device=device)

        self.tl_t = torch.zeros(N, max_tl_points, device=device)
        self.tl_s = torch.zeros(N, max_tl_points, device=device)
        self.tl_sdot = torch.zeros(N, max_tl_points, device=device)
        self.T = torch.zeros(N, device=device)
        self._row_cache = None

    def load(self, env_ids, gammas, time_laws):
        """Write newly generated (gamma, time_law) pairs into the given env slots."""
        for local_i, env_id in enumerate(env_ids):
            gamma, tl = gammas[local_i], time_laws[local_i]
            # Truncate rather than assert: a rare over-long procedural
            # trajectory just gets its tail clipped (path ends earlier), which
            # is harmless for training and never crashes a long run.
            n = min(gamma.s_grid.shape[0], self.max_gamma_points)
            self.gamma_s[env_id, :n] = gamma.s_grid[:n].to(self.device)
            self.gamma_s[env_id, n:] = gamma.s_grid[n - 1].to(self.device)
            self.gamma_p[env_id, :n] = gamma.p[:n].to(self.device)
            self.gamma_p[env_id, n:] = gamma.p[n - 1].to(self.device)
            q = mat_to_quat(gamma.R[:n].to(self.device))
            self.gamma_quat[env_id, :n] = q
            self.gamma_quat[env_id, n:] = q[n - 1]
            self.gamma_tangent[env_id, :n] = gamma.tangent[:n].to(self.device)
            self.gamma_tangent[env_id, n:] = gamma.tangent[n - 1].to(self.device)
            self.L[env_id] = gamma.s_grid[n - 1].to(self.device)

            m = min(tl.t_grid.shape[0], self.max_tl_points)
            self.tl_t[env_id, :m] = tl.t_grid[:m].to(self.device)
            self.tl_t[env_id, m:] = tl.t_grid[m - 1].to(self.device)
            self.tl_s[env_id, :m] = tl.s_of_t[:m].to(self.device)
            self.tl_s[env_id, m:] = tl.s_of_t[m - 1].to(self.device)
            self.tl_sdot[env_id, :m] = tl.sdot_of_t[:m].to(self.device)
            self.tl_sdot[env_id, m:] = tl.sdot_of_t[m - 1].to(self.device)
            self.T[env_id] = tl.t_grid[m - 1].to(self.device)

    def load_from_stacked(self, env_ids, src, src_idx):
        """Copy pre-stacked trajectory rows from another TrajectoryBatch (a
        bank) into the given env slots -- a pure GPU gather, no per-trajectory
        CPU->tensor conversion. ``src`` must share this batch's field layout
        (same max_gamma_points / max_tl_points). ``src_idx`` (len(env_ids),)
        selects which bank row each env receives."""
        assert src.max_gamma_points == self.max_gamma_points
        assert src.max_tl_points == self.max_tl_points
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        src_idx = torch.as_tensor(src_idx, device=self.device, dtype=torch.long)
        self.gamma_s[env_ids] = src.gamma_s[src_idx]
        self.gamma_p[env_ids] = src.gamma_p[src_idx]
        self.gamma_quat[env_ids] = src.gamma_quat[src_idx]
        self.gamma_tangent[env_ids] = src.gamma_tangent[src_idx]
        self.L[env_ids] = src.L[src_idx]
        self.tl_t[env_ids] = src.tl_t[src_idx]
        self.tl_s[env_ids] = src.tl_s[src_idx]
        self.tl_sdot[env_ids] = src.tl_sdot[src_idx]
        self.T[env_ids] = src.T[src_idx]

    @staticmethod
    def _to_2d(s):
        return (s.unsqueeze(-1), True) if s.dim() == 1 else (s, False)

    def _rows(self, N):
        """(N, 1) row index for advanced indexing, cached per device."""
        if self._row_cache is None or self._row_cache.shape[0] != N:
            self._row_cache = torch.arange(N, device=self.device).unsqueeze(1)
        return self._row_cache

    # NOTE on the indexing style used by every lookup below. The obvious
    # formulation -- expand a field to (N, K, M, ...) and torch.gather along
    # the M axis -- allocates that whole expanded tensor: at N=4096, K=9,
    # M=1536 a single call peaked at 217 MB, and a trajectory-mode step makes
    # several of them. Two changes remove it entirely:
    #   * torch.searchsorted accepts a batched (N, M) sorted_sequence against
    #     an (N, K) query directly, so the (N, K, M) `.contiguous()` copy that
    #     dominated that 217 MB is unnecessary;
    #   * advanced indexing field[rows, idx] with rows (N,1) and idx (N,K)
    #     broadcasts to (N, K) and allocates only the (N, K, ...) result.
    # Same values, ~1000x less transient memory. Keep it this way.

    def _locate_gamma(self, s2d):
        # s2d: (N, K)
        s_clamped = s2d.clamp(self.gamma_s[:, :1], self.L.unsqueeze(-1))
        idx = torch.searchsorted(self.gamma_s, s_clamped, right=True)
        idx = idx.clamp(1, self.max_gamma_points - 1) - 1
        s0 = torch.gather(self.gamma_s, 1, idx)
        s1 = torch.gather(self.gamma_s, 1, idx + 1)
        frac = ((s_clamped - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0.0, 1.0)
        return idx, frac  # (N, K) each

    def p_at(self, s):
        """s: (N,) or (N,K) -> matching-shape (...,3)."""
        s2d, was_1d = self._to_2d(s)
        idx, frac = self._locate_gamma(s2d)
        rows = self._rows(idx.shape[0])
        p0, p1 = self.gamma_p[rows, idx], self.gamma_p[rows, idx + 1]
        out = p0 + frac.unsqueeze(-1) * (p1 - p0)
        return out.squeeze(1) if was_1d else out

    def quat_at(self, s):
        """s: (N,) or (N,K) -> matching-shape (...,4) xyzw, SLERP interpolated.
        Prefer this over ``R_at`` -- it is what the stored data already is."""
        s2d, was_1d = self._to_2d(s)
        idx, frac = self._locate_gamma(s2d)
        rows = self._rows(idx.shape[0])
        q0, q1 = self.gamma_quat[rows, idx], self.gamma_quat[rows, idx + 1]
        out = quat_slerp(q0, q1, frac)
        return out.squeeze(1) if was_1d else out

    def R_at(self, s):
        """s: (N,) or (N,K) -> matching-shape (...,3,3)."""
        return quat_to_mat(self.quat_at(s))

    def tangent_at(self, s):
        s2d, was_1d = self._to_2d(s)
        idx, frac = self._locate_gamma(s2d)
        rows = self._rows(idx.shape[0])
        t0, t1 = self.gamma_tangent[rows, idx], self.gamma_tangent[rows, idx + 1]
        v = t0 + frac.unsqueeze(-1) * (t1 - t0)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return v.squeeze(1) if was_1d else v

    def _locate_time(self, t2d):
        t_clamped = t2d.clamp(self.tl_t[:, :1], self.T.unsqueeze(-1))
        idx = torch.searchsorted(self.tl_t, t_clamped, right=True)
        idx = idx.clamp(1, self.max_tl_points - 1) - 1
        return idx, t_clamped

    def s_ref(self, t):
        t2d, was_1d = self._to_2d(t)
        idx, t_clamped = self._locate_time(t2d)
        s0 = torch.gather(self.tl_s, 1, idx)
        s1 = torch.gather(self.tl_s, 1, idx + 1)
        t0 = torch.gather(self.tl_t, 1, idx)
        t1 = torch.gather(self.tl_t, 1, idx + 1)
        frac = ((t_clamped - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0.0, 1.0)
        out = s0 + frac * (s1 - s0)
        return out.squeeze(1) if was_1d else out

    def sdot_ref(self, t):
        """Reference arc-length speed at time t. t: (N,) or (N,K)."""
        t2d, was_1d = self._to_2d(t)
        idx, _ = self._locate_time(t2d)
        out = torch.gather(self.tl_sdot, 1, idx)
        return out.squeeze(1) if was_1d else out

    def sample_preview(self, s_current, L_h, K=9):
        """s_current: (N,) -> s_k (N,K), p_k (N,K,3), q_k (N,K,4) xyzw, sdot_k (N,K)."""
        delta = L_h * preview_fractions(K, device=self.device)  # (K,)
        s_k = (s_current.unsqueeze(1) + delta.unsqueeze(0)).clamp(max=self.L.unsqueeze(1))
        p_k = self.p_at(s_k)
        q_k = self.quat_at(s_k)
        # sdot at s_k: approximate via the time law by inverse-locating s in tl_s
        sdot_k = self._sdot_at_s(s_k)
        return s_k, p_k, q_k, sdot_k

    def _sdot_at_s(self, s2d):
        s_clamped = s2d.clamp(self.tl_s[:, :1], self.L.unsqueeze(-1))
        idx = torch.searchsorted(self.tl_s, s_clamped, right=True)
        idx = idx.clamp(1, self.max_tl_points - 1) - 1
        return torch.gather(self.tl_sdot, 1, idx)


# ============================================================
# to_canonical / update_s
# ============================================================


def to_canonical(p_raw, R_raw, dt, lam=0.15, ds_grid=0.005, butter_fc=10.0):
    """Turn a raw pose sequence into (Gamma, TimeLaw).

    p_raw: (T, 3) array-like
    R_raw: (T, 3, 3) array-like
    dt:    float, sample interval of the raw sequence (s)
    """
    p_raw = torch.as_tensor(np.asarray(p_raw), dtype=torch.float32)
    R_raw = torch.as_tensor(np.asarray(R_raw), dtype=torch.float32)
    T = p_raw.shape[0]

    # 1. low-pass filter (Butterworth, offline/zero-phase). Quaternion
    # components are filtered independently then renormalized -- an
    # approximation that's fine for a smooth, continuous input sequence.
    fs = 1.0 / dt
    if T > 15 and butter_fc < fs / 2:
        b, a = scipy.signal.butter(4, butter_fc / (fs / 2))
        p_np = p_raw.numpy()
        p_filt = np.stack([scipy.signal.filtfilt(b, a, p_np[:, i]) for i in range(3)], axis=1)
        p_raw = torch.as_tensor(p_filt, dtype=torch.float32)

        q = _make_quat_continuous(mat_to_quat(R_raw))
        q_np = q.numpy()
        q_filt = np.stack([scipy.signal.filtfilt(b, a, q_np[:, i]) for i in range(4)], axis=1)
        q_filt = torch.as_tensor(q_filt, dtype=torch.float32)
        q_filt = q_filt / q_filt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        R_raw = quat_to_mat(q_filt)

    # 2. SE(3) arc length: ds^2 = ||dp||^2 + (lam*dtheta)^2
    dp = p_raw[1:] - p_raw[:-1]
    dR = torch.matmul(R_raw[1:], R_raw[:-1].transpose(-1, -2))
    dtheta = so3_log(dR).norm(dim=-1)
    ds = torch.sqrt((dp * dp).sum(-1) + (lam * dtheta) ** 2)
    s_of_t = torch.cat([torch.zeros(1), torch.cumsum(ds, dim=0)])
    L = s_of_t[-1].item()

    t_grid = torch.arange(T, dtype=torch.float32) * dt
    sdot_of_t = torch.gradient(s_of_t, spacing=(dt,))[0]
    time_law = TimeLaw(t_grid, s_of_t, sdot_of_t)

    # 3. uniform resample in s: PCHIP for position, piecewise SLERP for rotation
    n_grid = max(2, int(L / ds_grid) + 1)
    s_grid = torch.linspace(0.0, L, n_grid)

    s_of_t_np = s_of_t.numpy()
    # PCHIP needs a strictly increasing x; s_of_t is monotonic non-decreasing
    # by construction (ds >= 0), nudge flat runs by a tiny epsilon.
    s_of_t_np = s_of_t_np + np.arange(T) * 1e-9
    p_np = p_raw.numpy()
    pchip = [scipy.interpolate.PchipInterpolator(s_of_t_np, p_np[:, i]) for i in range(3)]
    p_grid = torch.as_tensor(np.stack([f(s_grid.numpy()) for f in pchip], axis=1), dtype=torch.float32)

    q_of_t = mat_to_quat(R_raw)
    s_of_t_grid = torch.as_tensor(s_of_t_np, dtype=torch.float32)
    idx = torch.searchsorted(s_of_t_grid, s_grid, right=True).clamp(1, T - 1) - 1
    s0, s1 = s_of_t_grid[idx], s_of_t_grid[idx + 1]
    frac = ((s_grid - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0.0, 1.0)
    q_grid = quat_slerp(q_of_t[idx], q_of_t[idx + 1], frac)
    R_grid = quat_to_mat(q_grid)

    tangent = torch.zeros_like(p_grid)
    tangent[1:-1] = p_grid[2:] - p_grid[:-2]
    tangent[0] = p_grid[1] - p_grid[0]
    tangent[-1] = p_grid[-1] - p_grid[-2]
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    gamma = Gamma(s_grid, p_grid, R_grid, tangent, lam=lam)
    return gamma, time_law


def update_s(s_prev, ee_p, ee_R, gamma, window=0.15, M=16, lam=0.15):
    """Forward window projection. Fully batched: s_prev (N,), ee_p (N,3), ee_R (N,3,3).

    Only searches forward from s_prev (s never regresses), scored by SE(3)
    distance using the same lam as the trajectory's own arc length metric.
    Returns (s_new, d_lat) where d_lat is the pure positional distance to the
    chosen point (used for termination / reward).
    """
    device = s_prev.device
    N = s_prev.shape[0]
    frac = torch.linspace(0.0, 1.0, M, device=device)  # (M,)
    L = gamma.L if not torch.is_tensor(gamma.L) else gamma.L
    s_max = (s_prev + window).clamp(max=L)
    s_cand = s_prev.unsqueeze(-1) + frac.unsqueeze(0) * (s_max - s_prev).clamp_min(0.0).unsqueeze(-1)  # (N,M)

    p_cand = gamma.p_at(s_cand.reshape(-1)).reshape(N, M, 3)
    R_cand = gamma.R_at(s_cand.reshape(-1)).reshape(N, M, 3, 3)

    dp = p_cand - ee_p.unsqueeze(1)
    dR = torch.matmul(R_cand, ee_R.unsqueeze(1).transpose(-1, -2))
    dtheta = so3_log(dR.reshape(-1, 3, 3)).reshape(N, M, 3).norm(dim=-1)
    dist = torch.sqrt((dp * dp).sum(-1) + (lam * dtheta) ** 2)  # (N,M)

    best = dist.argmin(dim=-1)  # (N,)
    s_new = torch.gather(s_cand, 1, best.unsqueeze(-1)).squeeze(-1)
    s_new = torch.maximum(s_new, s_prev)

    p_best = torch.gather(p_cand, 1, best.view(N, 1, 1).expand(-1, 1, 3)).squeeze(1)
    d_lat = (ee_p - p_best).norm(dim=-1)
    return s_new, d_lat


def update_s_batch(s_prev, ee_p, ee_R, traj_batch, window=0.15, M=16, lam=0.15):
    """Batched twin of ``update_s`` for a ``TrajectoryBatch`` (N distinct
    trajectories) instead of a single shared ``Gamma``.

    s_prev (N,), ee_p (N,3), ee_R (N,3,3). Forward-only window projection
    scored by SE(3) distance; returns (s_new, d_lat) with s_new monotonic.
    """
    device = s_prev.device
    N = s_prev.shape[0]
    frac = torch.linspace(0.0, 1.0, M, device=device)  # (M,)
    s_max = (s_prev + window).clamp(max=traj_batch.L)
    s_cand = s_prev.unsqueeze(-1) + frac.unsqueeze(0) * (s_max - s_prev).clamp_min(0.0).unsqueeze(-1)  # (N,M)

    p_cand = traj_batch.p_at(s_cand)  # (N,M,3)
    R_cand = traj_batch.R_at(s_cand)  # (N,M,3,3)

    dp = p_cand - ee_p.unsqueeze(1)
    dR = torch.matmul(R_cand, ee_R.unsqueeze(1).transpose(-1, -2))
    dtheta = so3_log(dR.reshape(-1, 3, 3)).reshape(N, M, 3).norm(dim=-1)
    dist = torch.sqrt((dp * dp).sum(-1) + (lam * dtheta) ** 2)  # (N,M)

    best = dist.argmin(dim=-1)  # (N,)
    s_new = torch.gather(s_cand, 1, best.unsqueeze(-1)).squeeze(-1)
    s_new = torch.maximum(s_new, s_prev)

    p_best = torch.gather(p_cand, 1, best.view(N, 1, 1).expand(-1, 1, 3)).squeeze(1)
    d_lat = (ee_p - p_best).norm(dim=-1)
    return s_new, d_lat
