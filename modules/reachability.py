"""M2: direction-dependent reachability table R_max(u) and the rho it defines.

Standalone -- pure PyTorch, no IsaacGym. The table itself is built offline by
``scripts/build_reach_table.py`` (which is where the IsaacGym FK lives) and
loaded here at env init; this module only owns the data structure, the
interpolated query, and the rho / rho-profile math built on top of it.

Why a table at all (project-design-v3.md §4.2): rho = r / R_max(u) is meant to
answer "how much of the arm's reach *in this direction* is the target using
up". Approximating R_max by a single scalar (a sphere) makes rho blind to the
fact that an arm reaches much further forward than backward or straight up,
and -- more importantly for §4.5 -- makes rho nearly invariant to base pitch,
which is exactly the channel that is supposed to *control* rho.

Two table modes:

- ``2d``: R_max indexed by the target's bearing from the shoulder
  (azimuth x elevation). The default; start here.
- ``4d``: additionally indexed by the tool's pointing direction, for when
  "rho is in the comfort band but tracking still fails" turns out to be
  common (§4.6). Same build path, two more axes.

Angle conventions, shared by build and query:
    az = atan2(u_y, u_x)      in [-pi, pi),  wraps around
    el = asin(u_z)            in [-pi/2, pi/2], clamped
both in the SHOULDER frame (the arm mount link's frame), never the world's.
"""

import math

import torch

# Bucket centers sit at (i + 0.5) of each bin, so a query exactly at a bucket
# center interpolates to that bucket's value and nothing else.
_TWO_PI = 2.0 * math.pi


def dir_to_angles(u):
    """(..., 3) unit vectors in the shoulder frame -> (az, el), each (...,)."""
    az = torch.atan2(u[..., 1], u[..., 0])
    el = torch.asin(u[..., 2].clamp(-1.0, 1.0))
    return az, el


def _frac_index(value, lo, span, n, wrap):
    """Map a value in [lo, lo+span] onto continuous bin coordinates, returning
    the two neighbouring bin indices and the blend weight between them."""
    x = (value - lo) / span * n - 0.5  # bucket centers at integer x
    i0 = torch.floor(x)
    w = (x - i0).clamp(0.0, 1.0)
    i0 = i0.long()
    i1 = i0 + 1
    if wrap:
        i0 = i0 % n
        i1 = i1 % n
    else:
        i0 = i0.clamp(0, n - 1)
        i1 = i1.clamp(0, n - 1)
    return i0, i1, w


class ReachabilityTable:
    """Direction-dependent maximum reach R_max(u), resident on the GPU."""

    def __init__(self, table, mode="2d", occupancy=None, meta=None):
        """
        table: (n_az, n_el)                                    for mode='2d'
               (n_az, n_el, n_az_tool, n_el_tool)              for mode='4d'
        occupancy: same shape, sample count per bucket (diagnostics; buckets
               with 0 samples were filled -- see build_* 's empty_fill).
        meta:  free-form dict recorded at build time (robot, n_samples, ...).
        """
        if mode not in ("2d", "4d"):
            raise ValueError(f"mode must be '2d' or '4d', got {mode!r}")
        if table.dim() != (2 if mode == "2d" else 4):
            raise ValueError(f"table.dim()={table.dim()} does not match mode={mode}")
        self.table = table
        self.mode = mode
        self.occupancy = occupancy
        self.meta = dict(meta or {})

    @property
    def device(self):
        return self.table.device

    def to(self, device):
        self.table = self.table.to(device)
        if self.occupancy is not None:
            self.occupancy = self.occupancy.to(device)
        return self

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(self, u, tool_dir=None):
        """R_max for each direction.

        u:        (..., 3) unit vectors in the SHOULDER frame
        tool_dir: (..., 3) tool pointing direction in the shoulder frame;
                  required for mode='4d', ignored for '2d'
        returns:  (...) same leading shape as u
        """
        az, el = dir_to_angles(u)
        n_az, n_el = self.table.shape[0], self.table.shape[1]
        a0, a1, wa = _frac_index(az, -math.pi, _TWO_PI, n_az, wrap=True)
        e0, e1, we = _frac_index(el, -math.pi / 2, math.pi, n_el, wrap=False)

        if self.mode == "2d":
            t = self.table
            v = (
                t[a0, e0] * (1 - wa) * (1 - we)
                + t[a1, e0] * wa * (1 - we)
                + t[a0, e1] * (1 - wa) * we
                + t[a1, e1] * wa * we
            )
            return v

        if tool_dir is None:
            raise ValueError("mode='4d' requires tool_dir")
        taz, tel = dir_to_angles(tool_dir)
        n_taz, n_tel = self.table.shape[2], self.table.shape[3]
        b0, b1, wb = _frac_index(taz, -math.pi, _TWO_PI, n_taz, wrap=True)
        f0, f1, wf = _frac_index(tel, -math.pi / 2, math.pi, n_tel, wrap=False)

        t = self.table
        v = 0.0
        for ia, wia in ((a0, 1 - wa), (a1, wa)):
            for ie, wie in ((e0, 1 - we), (e1, we)):
                for ib, wib in ((b0, 1 - wb), (b1, wb)):
                    for jf, wjf in ((f0, 1 - wf), (f1, wf)):
                        v = v + t[ia, ie, ib, jf] * (wia * wie * wib * wjf)
        return v

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(
            {
                "table": self.table.detach().cpu(),
                "mode": self.mode,
                "occupancy": None if self.occupancy is None else self.occupancy.detach().cpu(),
                "meta": self.meta,
            },
            path,
        )

    @staticmethod
    def load(path, device="cpu"):
        blob = torch.load(path, map_location="cpu")
        occ = blob.get("occupancy")
        return ReachabilityTable(
            blob["table"].to(device),
            mode=blob.get("mode", "2d"),
            occupancy=None if occ is None else occ.to(device),
            meta=blob.get("meta", {}),
        )

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    @staticmethod
    def build_2d(
        fk_fn,
        q_lo,
        q_hi,
        jac_fn=None,
        n_samples=2_000_000,
        n_az=72,
        n_el=36,
        sigma_rot_min=0.05,
        quantile=0.98,
        smooth_sigma=1.0,
        chunk=4096,
        seed=0,
        meta=None,
        progress=None,
    ):
        """Offline build (project-design-v3.md §4.3).

        fk_fn(q) -> (p, R) or (p, R, keep)
            Batched forward kinematics, with p (M,3) / R (M,3,3) already
            expressed in the SHOULDER frame -- only the caller knows the mount
            transform, so the transform is its job, not this module's. The
            optional third return is a bool (M,) mask of samples to keep
            (e.g. self-collision-free); absent means keep everything.
        jac_fn(q) -> J (M, 6, n_joints)
            Optional. When given, configurations whose ROTATIONAL sub-Jacobian
            has a minimum singular value below sigma_rot_min are dropped: the
            wrist there has lost a rotational DoF, so although the point is
            positionally reachable it is not usable for full-SE(3) tracking.
        quantile
            0.98 rather than the max on purpose: the farthest configuration in
            any direction is the fully-extended singular one, which we do not
            want to advertise as usable reach.
        smooth_sigma
            Gaussian blur in bucket units. rho feeds a reward, so a
            discontinuous table would make the reward gradient discontinuous.
        """
        return _build(
            fk_fn, q_lo, q_hi, jac_fn, n_samples, (n_az, n_el), None,
            sigma_rot_min, quantile, smooth_sigma, chunk, seed, meta, progress,
        )

    @staticmethod
    def build_4d(
        fk_fn,
        q_lo,
        q_hi,
        jac_fn=None,
        n_samples=40_000_000,
        n_az_pos=72,
        n_el_pos=36,
        n_az_tool=24,
        n_el_tool=12,
        tool_axis=0,
        sigma_rot_min=0.05,
        quantile=0.98,
        smooth_sigma=1.0,
        chunk=4096,
        seed=0,
        meta=None,
        progress=None,
    ):
        """Same pipeline as build_2d plus two tool-orientation axes (§4.6).

        The tool direction is column ``tool_axis`` of the end-effector's
        rotation matrix in the shoulder frame (default 0 = the link's local x,
        which is the axis the grasp-point offset runs along on this robot).

        Needs far more samples than 2D simply because there are ~290x more
        buckets to fill; with the default grid, 40M keeps the median bucket
        population comparable to build_2d's.
        """
        return _build(
            fk_fn, q_lo, q_hi, jac_fn, n_samples,
            (n_az_pos, n_el_pos, n_az_tool, n_el_tool), tool_axis,
            sigma_rot_min, quantile, smooth_sigma, chunk, seed, meta, progress,
        )


def _build(fk_fn, q_lo, q_hi, jac_fn, n_samples, shape, tool_axis,
           sigma_rot_min, quantile, smooth_sigma, chunk, seed, meta, progress):
    mode = "2d" if len(shape) == 2 else "4d"
    device = q_lo.device
    n_joints = q_lo.numel()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    n_bins = int(torch.tensor(shape).prod())
    flat_bins, radii = [], []
    kept = 0
    done = 0
    while done < n_samples:
        # Always a full chunk, even if that overshoots n_samples: a sim-backed
        # fk_fn batches over a fixed number of envs and cannot take a short
        # final batch.
        m = chunk
        q = torch.rand(m, n_joints, generator=gen).to(device) * (q_hi - q_lo) + q_lo

        out = fk_fn(q)
        p, R = out[0], out[1]
        keep = out[2] if len(out) > 2 else torch.ones(m, dtype=torch.bool, device=device)

        if jac_fn is not None and sigma_rot_min > 0:
            J = jac_fn(q)
            sigma_min = torch.linalg.svdvals(J[:, 3:6, :]).min(dim=-1).values
            keep = keep & (sigma_min > sigma_rot_min)

        r = p.norm(dim=-1)
        keep = keep & (r > 1e-4)
        if keep.any():
            p, R, r = p[keep], R[keep], r[keep]
            u = p / r.unsqueeze(-1)
            idx = _bucket_index(u, R, shape, tool_axis, mode)
            flat_bins.append(idx)
            radii.append(r)
            kept += int(keep.sum())

        done += m
        if progress is not None:
            progress(done, n_samples, kept)

    if not flat_bins:
        raise RuntimeError("no samples survived the filters -- check fk_fn / sigma_rot_min")
    flat_bins = torch.cat(flat_bins)
    radii = torch.cat(radii)

    values, counts = _bucket_quantile(flat_bins, radii, n_bins, quantile)

    # Buckets nothing reached are genuinely unreachable directions. Fill them
    # with the smallest reach that WAS observed rather than 0: rho = r/R_max
    # then comes out large (the barrier reads "way past the limit") instead of
    # dividing by zero, and the table stays finite everywhere for smoothing.
    empty = counts == 0
    if empty.all():
        raise RuntimeError("every bucket empty")
    values[empty] = values[~empty].min()

    table = values.view(*shape)
    occupancy = counts.view(*shape)
    if smooth_sigma > 0:
        table = _smooth(table, smooth_sigma, mode)

    info = dict(meta or {})
    info.update(
        mode=mode, n_samples=int(n_samples), n_kept=int(kept), quantile=float(quantile),
        smooth_sigma=float(smooth_sigma), sigma_rot_min=float(sigma_rot_min),
        empty_buckets=int(empty.sum()), shape=list(shape),
    )
    if tool_axis is not None:
        info["tool_axis"] = int(tool_axis)
    return ReachabilityTable(table, mode=mode, occupancy=occupancy, meta=info)


def _bucket_index(u, R, shape, tool_axis, mode):
    """Flat bucket index for each sample."""
    az, el = dir_to_angles(u)
    n_az, n_el = shape[0], shape[1]
    ia = (((az + math.pi) / _TWO_PI) * n_az).long().clamp(0, n_az - 1)
    ie = (((el + math.pi / 2) / math.pi) * n_el).long().clamp(0, n_el - 1)
    if mode == "2d":
        return ia * n_el + ie

    n_taz, n_tel = shape[2], shape[3]
    tool = R[:, :, tool_axis]
    taz, tel = dir_to_angles(tool)
    ib = (((taz + math.pi) / _TWO_PI) * n_taz).long().clamp(0, n_taz - 1)
    jf = (((tel + math.pi / 2) / math.pi) * n_tel).long().clamp(0, n_tel - 1)
    return ((ia * n_el + ie) * n_taz + ib) * n_tel + jf


def _bucket_quantile(bins, values, n_bins, q):
    """Per-bucket quantile of `values`, fully vectorized.

    Sorting by value first and then stably by bucket leaves every bucket's
    slice internally ascending, so the quantile is one gather off each
    bucket's start offset.
    """
    order = torch.argsort(values)
    bins_s = bins[order]
    order2 = torch.argsort(bins_s, stable=True)
    bins_s = bins_s[order2]
    values_s = values[order][order2]

    counts = torch.bincount(bins_s, minlength=n_bins)
    starts = torch.cumsum(counts, 0) - counts
    offset = ((counts - 1).clamp_min(0).float() * q).round().long()
    pick = (starts + offset).clamp(0, max(values_s.numel() - 1, 0))
    out = values_s[pick]
    out[counts == 0] = 0.0
    return out, counts


def _gaussian_kernel1d(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur_axis(t, axis, k, wrap):
    """1-D convolution of tensor `t` along `axis`, circular or replicate."""
    radius = (k.numel() - 1) // 2
    t = t.movedim(axis, -1)
    shape = t.shape
    flat = t.reshape(-1, 1, shape[-1])
    mode = "circular" if wrap else "replicate"
    flat = torch.nn.functional.pad(flat, (radius, radius), mode=mode)
    flat = torch.nn.functional.conv1d(flat, k.view(1, 1, -1))
    return flat.reshape(shape).movedim(-1, axis)


def _smooth(table, sigma, mode):
    k = _gaussian_kernel1d(sigma, table.device, table.dtype)
    # azimuth axes wrap; elevation axes do not (poles are boundaries)
    wrap_by_axis = (True, False) if mode == "2d" else (True, False, True, False)
    for axis, wrap in enumerate(wrap_by_axis):
        table = _blur_axis(table, axis, k, wrap)
    return table


# ======================================================================
# rho
# ======================================================================


def compute_rho(p_tgt_world, shoulder_pos_world, shoulder_quat_world, table,
                tool_dir_world=None, quat_rotate_inverse=None):
    """rho = r / R_max(u), batched (project-design-v3.md §4.2).

    Signature deviates from the design doc's 4x4-matrix form to match this
    codebase, which carries poses as position + xyzw quaternion:

    p_tgt_world:        (N, K, 3) or (N, 3) target positions, world frame
    shoulder_pos_world: (N, 3)
    shoulder_quat_world:(N, 4) xyzw, the arm MOUNT link's orientation (not the
                        trunk's -- they differ by the mount's fixed rpy)
    tool_dir_world:     (N, K, 3) or (N, 3), only for a 4d table
    quat_rotate_inverse: callable(quat, vec) rotating world -> local; injected
                        so this module stays free of any isaacgym import.

    returns rho, r, u -- u in the shoulder frame, all with the target's
    leading shape.
    """
    squeeze = p_tgt_world.dim() == 2
    if squeeze:
        p_tgt_world = p_tgt_world.unsqueeze(1)
        if tool_dir_world is not None:
            tool_dir_world = tool_dir_world.unsqueeze(1)
    N, K = p_tgt_world.shape[0], p_tgt_world.shape[1]

    d_world = p_tgt_world - shoulder_pos_world.unsqueeze(1)  # (N, K, 3)
    q = shoulder_quat_world.unsqueeze(1).expand(N, K, 4).reshape(N * K, 4)
    d = quat_rotate_inverse(q, d_world.reshape(N * K, 3)).reshape(N, K, 3)

    r = d.norm(dim=-1).clamp_min(1e-6)
    u = d / r.unsqueeze(-1)

    if table.mode == "4d":
        if tool_dir_world is None:
            raise ValueError("a 4d reach table needs tool_dir_world")
        tool = quat_rotate_inverse(q, tool_dir_world.reshape(N * K, 3)).reshape(N, K, 3)
        tool = tool / tool.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r_max = table.query(u, tool)
    else:
        r_max = table.query(u)

    rho = r / r_max.clamp_min(1e-3)
    if squeeze:
        return rho[:, 0], r[:, 0], u[:, 0]
    return rho, r, u


def rho_profile_summary(rho_k, rho_hi):
    """Collapse a (N, K) look-ahead rho profile into the two scalars the
    policy actually needs (project-design-v3.md §6.2): how far past the comfort
    band the worst look-ahead point is, and how soon the first violation comes.

    Returns (urgency, s_to_viol); s_to_viol is normalized to [0, 1] and is 1
    when no point in the horizon violates.
    """
    K = rho_k.shape[-1]
    urgency = (rho_k - rho_hi).clamp_min(0.0).max(dim=-1).values
    violated = rho_k > rho_hi
    first = violated.float().argmax(dim=-1).float() / max(K, 1)
    return urgency, torch.where(violated.any(dim=-1), first, torch.ones_like(first))
