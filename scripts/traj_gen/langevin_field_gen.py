#!/usr/bin/env python3
"""
Offline Workspace SDF Field Pre-computation for the Langevin Trajectory Generator.

Uses the ARX5 arm URDF forward kinematics (identical chain to workspace_analysis.py)
to compute genuine workspace SDFs in arm-base frame coordinates:

  SDF_static  : arm workspace with fixed quadruped base (ground at Z = -base_z)
  SDF_dynamic : arm workspace with base pitching/rolling/height variation
  field_delta : field_dynamic − field_static  (base-motion delta field)
  SDF_body    : robot trunk collision box (so the EE cannot pass through the dog)

Each field has its spatial gradient pre-baked into 3 extra channels:
  combined tensor = [1, 4, G, G, G]  channels: (SDF, ∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z)

Speed benefit to langevin_traj_gen.py
  Blended field at difficulty D is pre-cached once per D value:
    field_curr = field_static + D * field_delta
  boundary_force() then requires only 1 grid_sample call (down from 4).

Pipeline
────────
  FK sampling → arm-base frame → 3D occupancy → Gaussian fill → EDT → SDF
  → central-difference gradient field → save [1,4,G,G,G] tensors

Outputs  (under field_gen.save_dir from config):
  precomputed_fields.pt   — field tensors + metadata
  workspace_sdf.png       — SDF cross-sections (2 workspaces × 3 slices + 3D)
  force_fields.png        — boundary force and ground repulsion maps
  base_delta_field.png    — delta field cross-sections (workspace expansion by base motion)
  body_collision_field.png — robot trunk collision SDF cross-sections

Usage:
  conda run -n roboduet python scripts/langevin_field_gen.py
  conda run -n roboduet python scripts/langevin_field_gen.py --show
  conda run -n roboduet python scripts/langevin_field_gen.py --n-samples 500000
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kinematics — identical chain to workspace_analysis.py
# ─────────────────────────────────────────────────────────────────────────────

_URDF_TRANS = [
    [0.,       0.,       0.057  ],   # zarx5p2_mount fixed
    [0.,       0.,       0.0605 ],   # zarx_j1  xyz
    [0.02,     0.,       0.04   ],   # zarx_j2  xyz
    [-0.264,   0.,       0.     ],   # zarx_j3  xyz  (elbow position)
    [0.245,    0.,      -0.056  ],   # zarx_j4  xyz
    [0.06675,  0.,      -0.084  ],   # zarx_j5  xyz
    [0.03045,  0.,       0.084  ],   # zarx_j6  xyz
    [0.073574, 0.,       0.     ],   # EE tip
]

def _rotX(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([o,z,z, z,c,-s, z,s,c], -1).reshape(-1, 3, 3)

def _rotY(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c,z,s, z,o,z, -s,z,c], -1).reshape(-1, 3, 3)

def _rotZ(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c,-s,z, s,c,z, z,z,o], -1).reshape(-1, 3, 3)


def matrix_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix → (roll, pitch, yaw) for the intrinsic Z-Y-X convention
    R = Rz(yaw)·Ry(pitch)·Rx(roll) — the same composition the env uses to build
    EE-orientation commands.  R : [..., 3, 3]  →  [..., 3].
    """
    pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1.0, 1.0))
    roll  = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw   = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def fk_batch(q: torch.Tensor, bp: torch.Tensor, base_z: float):
    """
    q  : (B,6) joint angles
    bp : (B,5) = [roll, pitch, x, y, dz]
    Returns ee_pos (B,3), ee_rot (B,3,3), elbow_pos (B,3) — all in world frame.
    """
    B, dev, dt = q.shape[0], q.device, q.dtype
    ts = [torch.tensor(t, device=dev, dtype=dt).unsqueeze(0).expand(B, -1)
          for t in _URDF_TRANS]

    def Ht(t):
        T = q.new_zeros(B, 4, 4)
        T[:, 0, 0] = T[:, 1, 1] = T[:, 2, 2] = T[:, 3, 3] = 1.
        T[:, :3, 3] = t
        return T

    def Hr(R):
        T = q.new_zeros(B, 4, 4)
        T[:, :3, :3] = R; T[:, 3, 3] = 1.
        return T

    mm  = torch.bmm
    Rpi = q.new_zeros(B, 3, 3)
    Rpi[:, 0, 0] = 1.; Rpi[:, 1, 1] = -1.; Rpi[:, 2, 2] = -1.

    R_base = mm(_rotY(bp[:, 1]), _rotX(bp[:, 0]))
    t_base = torch.stack([bp[:, 2], bp[:, 3],
                          q.new_full((B,), base_z) + bp[:, 4]], -1)
    T = Hr(R_base); T[:, :3, 3] = t_base

    T = mm(T, Ht(ts[0]))
    T = mm(T, Ht(ts[1])); T = mm(T, Hr(_rotZ(q[:, 0])))
    T = mm(T, Ht(ts[2])); T = mm(T, Hr(_rotY(q[:, 1])))
    T = mm(T, Ht(ts[3])); elbow = T[:, :3, 3].clone()
    T = mm(T, Hr(Rpi));   T = mm(T, Hr(_rotY(q[:, 2])))
    T = mm(T, Ht(ts[4])); T = mm(T, Hr(_rotY(q[:, 3])))
    T = mm(T, Ht(ts[5])); T = mm(T, Hr(_rotZ(q[:, 4])))
    T = mm(T, Ht(ts[6])); T = mm(T, Hr(Rpi))
    T = mm(T, Hr(_rotX(q[:, 5])))
    ee_rot = T[:, :3, :3].clone()          # EE orientation (before the tip translation)
    T = mm(T, Ht(ts[7]))
    return T[:, :3, 3], ee_rot, elbow


# ─────────────────────────────────────────────────────────────────────────────
# FK Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_ee_armframe(
    n:          int,
    q_limits:   np.ndarray,   # [6, 2]  lower/upper per joint
    base_z:     float,
    base_cfg:   dict,
    is_dynamic: bool,
    device:     str,
    chunk:      int = 250_000,
):
    """
    Sample n valid EE poses in the arm-base frame.

    Returns (ee_pos [M,3], ee_euler [M,3]) — positions in the arm-base frame and
    the EE orientation as (roll,pitch,yaw) Z-Y-X Euler.

    Coordinate convention:
      arm-base frame Z = world Z − base_z_nominal
      Ground in arm-base frame = Z = −base_z

    Valid = EE and elbow both above ground (world z ≥ 0).

    Static  (is_dynamic=False): base pose = zeros
    Dynamic (is_dynamic=True) : random roll/pitch within limits + height dz
    """
    q_lo = torch.tensor(q_limits[:, 0], device=device, dtype=torch.float32)
    q_hi = torch.tensor(q_limits[:, 1], device=device, dtype=torch.float32)

    max_roll  = base_cfg.get('max_roll',   0.08)
    max_pitch = base_cfg.get('max_pitch',  0.18)
    dz_lo     = base_cfg.get('height_min', 0.28) - base_z   # e.g. −0.06 m
    dz_hi     = base_cfg.get('height_max', 0.44) - base_z   # e.g. +0.10 m

    parts, parts_e, done = [], [], 0
    while done < n:
        B  = min(chunk, n - done)
        q  = q_lo + torch.rand(B, 6, device=device) * (q_hi - q_lo)

        if is_dynamic:
            roll  = (torch.rand(B, device=device) * 2 - 1) * max_roll
            pitch = (torch.rand(B, device=device) * 2 - 1) * max_pitch
            dz    = dz_lo + torch.rand(B, device=device) * (dz_hi - dz_lo)
            bp    = torch.stack([roll, pitch,
                                 torch.zeros(B, device=device),
                                 torch.zeros(B, device=device), dz], dim=-1)
        else:
            bp = torch.zeros(B, 5, device=device)

        ee_w, ee_rot, elbow_w = fk_batch(q, bp, base_z)   # world frame

        valid = (ee_w[:, 2] >= 0.) & (elbow_w[:, 2] >= 0.)
        ee_valid    = ee_w[valid].cpu().float().numpy()
        euler_valid = matrix_to_euler_zyx(ee_rot[valid]).cpu().float().numpy()

        # Convert from world frame to arm-base frame: subtract nominal base height
        ee_valid[:, 2] -= base_z
        parts.append(ee_valid)
        parts_e.append(euler_valid)
        done += B

    return (np.concatenate(parts, axis=0),      # [M, 3]  arm-base positions
            np.concatenate(parts_e, axis=0))    # [M, 3]  (roll,pitch,yaw) ZYX


# ─────────────────────────────────────────────────────────────────────────────
# SDF Construction from FK Point Cloud
# ─────────────────────────────────────────────────────────────────────────────

def build_sdf(
    ee_arm:       np.ndarray,   # [M, 3]  EE positions in arm-base frame
    grid_half:    float,
    grid_res:     int,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """
    Build a signed distance field from an EE point cloud.

    Steps:
      1. Rasterise into 3D occupancy grid (axes: z, y, x — matches D-H-W convention)
      2. Gaussian-fill sampling gaps (non-uniform FK density)
      3. Threshold at 1% of peak → binary workspace mask
      4. EDT on inside + outside separately
      5. SDF = dist_outside − dist_inside  (−ve inside, +ve outside)

    Returns [grid_res, grid_res, grid_res] float32 SDF in metres.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    G     = grid_res
    voxel = 2.0 * grid_half / (G - 1)

    def _idx(coord: np.ndarray) -> np.ndarray:
        return np.clip(
            np.round((coord / grid_half + 1.0) / 2.0 * (G - 1)).astype(np.int32),
            0, G - 1,
        )

    # axes: z→0, y→1, x→2
    iz = _idx(ee_arm[:, 2])
    iy = _idx(ee_arm[:, 1])
    ix = _idx(ee_arm[:, 0])

    lin = iz.astype(np.int64) * G * G + iy.astype(np.int64) * G + ix.astype(np.int64)
    occ = np.bincount(lin, minlength=G**3).reshape(G, G, G).astype(np.float32)

    occ  = gaussian_filter(occ, sigma=smooth_sigma)
    mask = occ > occ.max() * 0.01   # 1% threshold → workspace mask

    d_out = distance_transform_edt(~mask) * voxel
    d_in  = distance_transform_edt( mask) * voxel
    return (d_out - d_in).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EE Reachable-Orientation Field
# ─────────────────────────────────────────────────────────────────────────────

def build_orientation_field(
    ee_arm:       np.ndarray,   # [M, 3]  EE positions, arm-base frame
    ee_euler:     np.ndarray,   # [M, 3]  EE (roll,pitch,yaw) ZYX Euler
    cmd_box,                    # (roll, pitch, yaw) command-box half-extents (rad)
    grid_half:    float,
    grid_res:     int,
    smooth_sigma: float = 2.0,
):
    """
    Per-voxel statistics of the EE orientations that are actually REACHABLE at
    each workspace position.

    Every FK sample gives (EE position, EE orientation).  Per voxel we accumulate
    the mean and std of the EE Euler angles, so the trajectory generator can pick
    an orientation that is feasible *at that position* — a position may be
    reachable yet not with an arbitrary orientation.

    Only samples whose orientation lies inside the EE command box are kept: those
    are the task-relevant, gimbal-safe orientations (pitch < 90°).  Empty voxels
    are filled by count-weighted Gaussian smoothing.

    Returns ([1, 6, G, G, G] field, n_inbox):
      channels 0-2 = mean (roll,pitch,yaw),  3-5 = std (roll,pitch,yaw)
      axes (z, y, x), matching build_sdf().
    """
    from scipy.ndimage import gaussian_filter

    G   = grid_res
    box = np.asarray(cmd_box, dtype=np.float32)

    # keep only command-box orientations (gimbal-safe, task-relevant)
    inbox = np.all(np.abs(ee_euler) <= box[None, :], axis=1)
    pos   = ee_arm[inbox]
    eul   = ee_euler[inbox].astype(np.float64)

    def _idx(coord):
        return np.clip(
            np.round((coord / grid_half + 1.0) / 2.0 * (G - 1)).astype(np.int64),
            0, G - 1)

    iz, iy, ix = _idx(pos[:, 2]), _idx(pos[:, 1]), _idx(pos[:, 0])
    lin = iz * G * G + iy * G + ix

    cnt   = np.bincount(lin, minlength=G ** 3).astype(np.float64).reshape(G, G, G)
    cnt_f = gaussian_filter(cnt, sigma=smooth_sigma)          # count-weighted fill
    eps   = 1e-6

    mean = np.zeros((3, G, G, G), dtype=np.float32)
    std  = np.zeros((3, G, G, G), dtype=np.float32)
    for a in range(3):
        s1 = np.bincount(lin, weights=eul[:, a],      minlength=G ** 3).reshape(G, G, G)
        s2 = np.bincount(lin, weights=eul[:, a] ** 2, minlength=G ** 3).reshape(G, G, G)
        m  = gaussian_filter(s1, sigma=smooth_sigma) / np.maximum(cnt_f, eps)
        v  = gaussian_filter(s2, sigma=smooth_sigma) / np.maximum(cnt_f, eps) - m ** 2
        mean[a] = m.astype(np.float32)
        std[a]  = np.sqrt(np.maximum(v, 0.0)).astype(np.float32)

    field = np.concatenate([mean, std], axis=0)              # [6, G, G, G]
    field_t = torch.from_numpy(field).unsqueeze(0).contiguous().float()
    return field_t, int(inbox.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Field & Combined Tensor
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_field(
    sdf:       np.ndarray,   # [G, G, G]  axes: z, y, x
    grid_half: float,
    grid_res:  int,
) -> torch.Tensor:
    """
    Compute ∇SDF on the full voxel grid via central differences.

    Axis mapping (D-H-W = Z-Y-X):
      ∂/∂x → diff along axis 2  (W)
      ∂/∂y → diff along axis 1  (H)
      ∂/∂z → diff along axis 0  (D)

    Returns [G, G, G, 3] tensor of (gx, gy, gz) in m⁻¹.
    """
    voxel = 2.0 * grid_half / (grid_res - 1)
    sdf_t = torch.from_numpy(sdf).float()

    p = F.pad(sdf_t.unsqueeze(0).unsqueeze(0),
               [1, 1, 1, 1, 1, 1], mode='replicate').squeeze()   # [G+2, G+2, G+2]

    gx = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * voxel)
    gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * voxel)
    gz = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * voxel)
    return torch.stack([gx, gy, gz], dim=-1)   # [G, G, G, 3]


def build_combined_field(
    sdf:  np.ndarray,     # [G, G, G]
    grad: torch.Tensor,   # [G, G, G, 3]
) -> torch.Tensor:
    """
    Stack SDF value and gradient into one [1, 4, G, G, G] tensor.
    Channel layout: (SDF, ∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z)

    A single grid_sample call on this tensor yields value AND gradient together,
    halving the number of GPU memory transactions vs separate queries.
    """
    sdf_t    = torch.from_numpy(sdf).float()
    combined = torch.cat([sdf_t.unsqueeze(-1), grad.cpu()], dim=-1)   # [G,G,G,4]
    return combined.permute(3, 0, 1, 2).unsqueeze(0).contiguous()      # [1,4,G,G,G]


# ─────────────────────────────────────────────────────────────────────────────
# Robot Body (Trunk) Collision SDF
# ─────────────────────────────────────────────────────────────────────────────
#
# Why a separate body field is needed
# ───────────────────────────────────
# SDF_static / SDF_dynamic are built from FK-sampled *EE positions* — they mark
# where the end-effector CAN reach (the outer reachability envelope).  The robot
# trunk sits *inside* that envelope, so the workspace SDF labels the trunk region
# as "reachable / safe".  Nothing modelled the trunk as an obstacle, so EE
# particles could pass straight through the robot's own body.
#
# Coordinate frame
# ────────────────
# The Langevin frame origin coincides with the URDF `base`/`trunk` link:
#   base --floating_base(xyz=0)--> trunk ;  fk_batch() places `base` at world
#   (0,0,base_z) and the workspace pipeline subtracts base_z, so the trunk centre
#   sits exactly at the field origin (0,0,0).  All URDF trunk geometry therefore
#   drops into this frame with no transform.
#
# Geometry  (URDF arx5p2Go1.urdf, link "trunk")
# ─────────────────────────────────────────────
# The trunk <collision> uses mesh trunk.dae with a commented-out primitive box
#   <box size="0.3762 0.0935 0.114"/>  at the trunk origin.
# We use that abstract box as the trunk proxy.  Its 0.0935 m width is only the
# hip-mount span (hip joints at Y=±0.04675); widened to 0.13 m to cover the real
# body shell.  Legs are intentionally excluded — kept simple per design, and the
# trunk is the dominant self-collision hazard for the EE workspace.
#
# These are only the DEFAULTS — the box is configurable per run via
# field_gen.body_box_center / field_gen.body_box_size in traj_gen.yaml.

BODY_BOX_CENTER = (0.0, 0.0, 0.0)        # trunk centre, arm-base frame (m)
BODY_BOX_SIZE   = (0.3762, 0.13, 0.114)  # (X length, Y width, Z height) (m)


def _sdf_box(pts: torch.Tensor, center, size) -> torch.Tensor:
    """
    Exact signed distance from points to an axis-aligned box.
      pts : [M, 3]      center, size : 3-tuples (size = full extents)
    Returns [M] — negative inside the box, positive (true distance) outside.
    """
    c = torch.tensor(center, device=pts.device, dtype=pts.dtype)
    h = torch.tensor(size,   device=pts.device, dtype=pts.dtype) * 0.5
    q = (pts - c).abs() - h                          # [M, 3]
    outside = q.clamp(min=0.0).norm(dim=-1)          # [M]
    inside  = q.max(dim=-1).values.clamp(max=0.0)    # [M]
    return outside + inside


def generate_body_sdf(grid_half: float, grid_res: int, device: str,
                      box_center, box_size) -> np.ndarray:
    """
    Signed distance field of the robot body (trunk box) on the voxel grid.

    box_center, box_size : 3-tuples (m) — the trunk collision box, configurable
    via field_gen.body_box_center / body_box_size in traj_gen.yaml.

    The body is approximated by the trunk box only (legs excluded by design).
    The returned array has axes (z, y, x) — matching build_sdf() — so it shares
    the gradient / combined-field pipeline.

    Returns [G, G, G] float32 SDF in metres  (< 0 inside body, > 0 outside).
    """
    G   = grid_res
    lin = torch.linspace(-grid_half, grid_half, G, device=device)
    # voxel (iz,iy,ix) → position (lin[ix], lin[iy], lin[iz])  — axes (z,y,x)
    xx  = lin.view(1, 1, G).expand(G, G, G)
    yy  = lin.view(1, G, 1).expand(G, G, G)
    zz  = lin.view(G, 1, 1).expand(G, G, G)
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)     # [G³, 3]
    sdf = _sdf_box(pts, box_center, box_size)                  # [G³]
    return sdf.reshape(G, G, G).cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — SDF Cross-sections
# ─────────────────────────────────────────────────────────────────────────────

def _sdf_panel(ax, data2d, hx, hy, xlabel, ylabel, title,
               hline_z=None, hline_label=None):
    """Plot one 2D SDF cross-section. data2d axes: (H, W) for imshow."""
    vmax = max(float(np.percentile(np.abs(data2d), 95)), 0.01)
    im = ax.imshow(
        data2d, origin='lower', aspect='equal',
        extent=[hx[0], hx[-1], hy[0], hy[-1]],
        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
    )
    ax.contour(hx, hy, data2d, levels=[0.0],  colors='k',    linewidths=1.2)
    ax.contour(hx, hy, data2d, levels=[-0.05], colors='gray', linewidths=0.7,
               linestyles='--')
    if hline_z is not None:
        ax.axhline(hline_z, color='saddlebrown', linewidth=1.0, linestyle='-.',
                   label=hline_label or f'Z={hline_z:.2f}')
    ax.scatter(0, 0, c='red', s=40, zorder=6)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, pad=2)
    ax.grid(alpha=0.2)
    return im


def plot_sdf_fields(
    field_s:   torch.Tensor,
    field_d:   torch.Tensor,
    ground_z:  float,
    grid_half: float,
    grid_res:  int,
    save_path: Optional[str] = None,
    show:      bool = False,
):
    """
    2×4 subplot grid:
      Rows: static / dynamic
      Cols: XZ@Y=0 (front) / YZ@X=0 (side) / XY@Z=0 (top) / 3D scatter
    """
    G   = grid_res
    lin = np.linspace(-grid_half, grid_half, G)
    iy0 = G // 2   # Y=0 slice
    ix0 = G // 2   # X=0 slice
    iz0 = int(round((0.0 / grid_half + 1.0) / 2.0 * (G - 1)))   # Z=0 slice

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(22, 9))
    fig.suptitle(
        'ARX5 Arm Workspace SDF  ·  Row 0: Static base  ·  Row 1: Dynamic base\n'
        'Red = outside workspace, Blue = inside  ·  Black contour = boundary (SDF=0)  '
        '·  Grey dashed = −5 cm safety margin',
        fontsize=10, fontweight='bold', y=1.02,
    )

    for row, (field, label) in enumerate([(field_s, 'Static'), (field_d, 'Dynamic')]):
        sdf = field[0, 0].numpy()   # [G, G, G] axes (z, y, x)

        col_offset = row * 4

        # XZ @ Y=0: axes (z→rows, x→cols)  → imshow rows=z, cols=x
        ax = fig.add_subplot(2, 4, col_offset + 1)
        slc = sdf[:, iy0, :]   # [G_z, G_x]
        im = _sdf_panel(ax, slc, lin, lin,
                        'X — forward (m)', 'Z — arm-base (m)',
                        f'{label}: XZ  (Y = 0)',
                        hline_z=ground_z, hline_label='Ground')
        fig.colorbar(im, ax=ax, shrink=0.75, label='SDF (m)')

        # YZ @ X=0: axes (z→rows, y→cols)
        ax = fig.add_subplot(2, 4, col_offset + 2)
        slc = sdf[:, :, ix0]   # [G_z, G_y]
        im = _sdf_panel(ax, slc, lin, lin,
                        'Y — lateral (m)', 'Z — arm-base (m)',
                        f'{label}: YZ  (X = 0)',
                        hline_z=ground_z)
        fig.colorbar(im, ax=ax, shrink=0.75, label='SDF (m)')

        # XY @ Z=0: axes (y→rows, x→cols)
        ax = fig.add_subplot(2, 4, col_offset + 3)
        slc = sdf[iz0, :, :]   # [G_y, G_x]
        im = _sdf_panel(ax, slc, lin, lin,
                        'X — forward (m)', 'Y — lateral (m)',
                        f'{label}: XY  (Z = 0, arm-mount level)')
        fig.colorbar(im, ax=ax, shrink=0.75, label='SDF (m)')

        # 3D: scatter voxels near SDF=0 surface
        ax3 = fig.add_subplot(2, 4, col_offset + 4, projection='3d')
        near = np.abs(sdf) < 0.05
        iz_n, iy_n, ix_n = np.where(near)
        if len(iz_n) > 3000:
            idx_s = rng.choice(len(iz_n), 3000, replace=False)
            iz_n, iy_n, ix_n = iz_n[idx_s], iy_n[idx_s], ix_n[idx_s]
        col_v = sdf[iz_n, iy_n, ix_n]
        sc = ax3.scatter(lin[ix_n], lin[iy_n], lin[iz_n],
                         c=col_v, cmap='RdBu_r', s=1.0, alpha=0.3,
                         vmin=-0.05, vmax=0.05)
        gx_ = np.linspace(-grid_half, grid_half, 4)
        gX, gY = np.meshgrid(gx_, gx_)
        ax3.plot_surface(gX, gY, np.full_like(gX, ground_z),
                         alpha=0.08, color='saddlebrown')
        ax3.scatter(0, 0, 0, c='red', s=60, zorder=10)
        ax3.set_xlabel('X', fontsize=7); ax3.set_ylabel('Y', fontsize=7)
        ax3.set_zlabel('Z', fontsize=7); ax3.tick_params(labelsize=6)
        ax3.set_title(f'{label}: 3D surface (|SDF| < 5 cm)', fontsize=8, pad=2)
        fig.colorbar(sc, ax=ax3, shrink=0.55, label='SDF (m)', pad=0.1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — Force Fields
# ─────────────────────────────────────────────────────────────────────────────

def plot_force_fields(
    field_s:    torch.Tensor,
    field_d:    torch.Tensor,
    ground_z:   float,
    grid_half:  float,
    grid_res:   int,
    k_bound:    float,
    k_ground:   float,
    decay_rate: float,
    save_path:  Optional[str] = None,
    show:       bool = False,
):
    """
    1×4 subplot figure:
      Col 0 — Ground repulsion magnitude: Z profile (analytical, log scale)
      Col 1 — Boundary force magnitude, static SDF  (XZ cross-section)
      Col 2 — Boundary force magnitude, dynamic SDF  (XZ cross-section)
      Col 3 — SDF gradient magnitude, dynamic        (XZ cross-section)
    """
    G   = grid_res
    lin = np.linspace(-grid_half, grid_half, G)
    iy0 = G // 2   # Y=0 cross-section

    def _bforce_xz(field: torch.Tensor) -> np.ndarray:
        """Boundary force magnitude [G, G] at Y=0 cross-section."""
        sdf_np = field[0, 0].numpy()         # [G,G,G]
        gx_np  = field[0, 1].numpy()
        gy_np  = field[0, 2].numpy()
        gz_np  = field[0, 3].numpy()
        active = (sdf_np > -0.05).astype(float)
        mag    = k_bound / np.maximum(0.051 + sdf_np, 1e-4)
        gn     = np.sqrt(gx_np**2 + gy_np**2 + gz_np**2)
        return (active * mag * gn)[:, iy0, :]   # [G_z, G_x]

    def _grad_mag_xz(field: torch.Tensor) -> np.ndarray:
        gx = field[0, 1].numpy()
        gy = field[0, 2].numpy()
        gz = field[0, 3].numpy()
        return np.sqrt(gx**2 + gy**2 + gz**2)[:, iy0, :]

    bf_s   = _bforce_xz(field_s)
    bf_d   = _bforce_xz(field_d)
    gm_d   = _grad_mag_xz(field_d)
    sdf_d_xz = field_d[0, 0].numpy()[:, iy0, :]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'APF Force Fields  (cross-sections at Y=0 / arm-base frame)\n'
        f'k_bound={k_bound}  k_ground={k_ground}  decay_rate={decay_rate} m⁻¹  '
        f'Black = SDF=0 boundary  ·  Ground: Z={ground_z:.2f} m',
        fontsize=10, fontweight='bold', y=1.03,
    )

    # ── Col 0: ground repulsion profile ──────────────────────────────────────
    ax = axes[0]
    z_vals = np.linspace(ground_z, ground_z + 0.35, 400)
    fz     = k_ground * np.exp(-decay_rate * np.maximum(0.0, z_vals - ground_z))
    ax.plot(fz, z_vals, 'royalblue', linewidth=2.0)
    ax.axhline(ground_z, color='saddlebrown', linestyle='-.', linewidth=1.1,
               label=f'Ground Z={ground_z:.2f} m')
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8,
               label='Arm-mount Z=0')
    ax.set_xlabel('Ground repulsion Fz (N)', fontsize=9)
    ax.set_ylabel('Z in arm-base frame (m)', fontsize=9)
    ax.set_title('Ground Repulsion Profile', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xscale('log')

    # ── Col 1-2: boundary force magnitude (XZ) ───────────────────────────────
    vmax_b = max(float(np.percentile(bf_d[bf_d > 0], 97)) if (bf_d > 0).any() else 1.0,
                 0.01)
    for col, (bf_xz, field, title) in enumerate([
        (bf_s, field_s, 'Boundary |F|: Static SDF'),
        (bf_d, field_d, 'Boundary |F|: Dynamic SDF'),
    ], start=1):
        ax = axes[col]
        sdf_xz = field[0, 0].numpy()[:, iy0, :]
        im = ax.imshow(bf_xz, origin='lower', aspect='equal',
                       extent=[lin[0], lin[-1], lin[0], lin[-1]],
                       cmap='hot_r', vmin=0, vmax=vmax_b)
        ax.contour(lin, lin, sdf_xz, levels=[0.0],   colors='k',  linewidths=1.0)
        ax.contour(lin, lin, sdf_xz, levels=[-0.05], colors='w',  linewidths=0.6,
                   linestyles='--')
        ax.axhline(ground_z, color='cyan', linewidth=1.0, linestyle='-.', label='Ground')
        ax.scatter(0, 0, c='lime', s=50, zorder=6, label='Origin')
        ax.set_xlabel('X (m)', fontsize=9); ax.set_ylabel('Z (m)', fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, label='|F_boundary| (N)')

    # ── Col 3: gradient magnitude (XZ, dynamic) ──────────────────────────────
    ax = axes[3]
    im3 = ax.imshow(gm_d, origin='lower', aspect='equal',
                    extent=[lin[0], lin[-1], lin[0], lin[-1]],
                    cmap='viridis', vmin=0)
    ax.contour(lin, lin, sdf_d_xz, levels=[0.0], colors='red', linewidths=1.2)
    ax.axhline(ground_z, color='cyan', linewidth=1.0, linestyle='-.')
    ax.scatter(0, 0, c='red', s=50, zorder=6)
    ax.set_xlabel('X (m)', fontsize=9); ax.set_ylabel('Z (m)', fontsize=9)
    ax.set_title('|∇SDF| Dynamic (Red = boundary)\n'
                 'Well-defined gradient inside & outside', fontsize=9)
    fig.colorbar(im3, ax=ax, shrink=0.85, label='|∇SDF| (m⁻¹)')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — Delta Field (base-motion workspace expansion)
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_field(
    field_delta: torch.Tensor,   # [1, 4, G, G, G]  = field_dynamic − field_static
    field_s:     torch.Tensor,   # [1, 4, G, G, G]  for SDF=0 boundary reference
    ground_z:    float,
    grid_half:   float,
    grid_res:    int,
    save_path:   Optional[str] = None,
    show:        bool = False,
):
    """
    Visualise the delta (base-motion) field: field_dynamic − field_static.

    Negative delta → region newly inside workspace when base moves (expansion).
    Positive delta → region less accessible when base moves (contraction).

    Layout: 1 row × 4 cols
      Col 0 — XZ @ Y=0  with static SDF=0 boundary overlay
      Col 1 — YZ @ X=0
      Col 2 — XY @ Z=0
      Col 3 — 3D scatter of strongest expansion voxels (delta < −0.02 m)
    """
    G   = grid_res
    lin = np.linspace(-grid_half, grid_half, G)
    iy0 = G // 2
    ix0 = G // 2
    iz0 = int(round((0.0 / grid_half + 1.0) / 2.0 * (G - 1)))

    delta_np = field_delta[0, 0].numpy()   # [G,G,G]  SDF channel only
    sdf_s_np = field_s[0, 0].numpy()

    rng  = np.random.default_rng(7)
    fig  = plt.figure(figsize=(22, 5))
    fig.suptitle(
        'Base-Motion Delta Field  ·  field_dynamic − field_static  (SDF channel)\n'
        'Blue = workspace expansion (base motion opens new EE space)  '
        '·  Red = contraction  ·  Black contour = static SDF=0 boundary',
        fontsize=10, fontweight='bold', y=1.04,
    )

    slices = [
        (delta_np[:, iy0, :],   sdf_s_np[:, iy0, :],   lin, lin, 'X (m)', 'Z (m)', 'XZ  (Y=0)'),
        (delta_np[:, :, ix0],   sdf_s_np[:, :, ix0],   lin, lin, 'Y (m)', 'Z (m)', 'YZ  (X=0)'),
        (delta_np[iz0, :, :],   sdf_s_np[iz0, :, :],   lin, lin, 'X (m)', 'Y (m)', 'XY  (Z=0, arm-mount level)'),
    ]

    vmax = max(float(np.percentile(np.abs(delta_np), 97)), 0.01)

    for col, (d2d, s2d, hx, hy, xlabel, ylabel, title) in enumerate(slices):
        ax = fig.add_subplot(1, 4, col + 1)
        im = ax.imshow(d2d, origin='lower', aspect='equal',
                       extent=[hx[0], hx[-1], hy[0], hy[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.contour(hx, hy, s2d, levels=[0.0], colors='k', linewidths=1.2)
        ax.contour(hx, hy, d2d, levels=[0.0], colors='purple', linewidths=0.8,
                   linestyles='--')
        if ylabel == 'Z (m)':
            ax.axhline(ground_z, color='saddlebrown', linewidth=0.9, linestyle='-.',
                       label='Ground')
        ax.scatter(0, 0, c='red', s=40, zorder=6)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'ΔSDF  {title}', fontsize=8, pad=2)
        ax.grid(alpha=0.2)
        fig.colorbar(im, ax=ax, shrink=0.75, label='ΔSDF (m)')

    # 3D: strongest expansion voxels (delta < −0.02 m)
    ax3 = fig.add_subplot(1, 4, 4, projection='3d')
    expand = delta_np < -0.02
    iz_e, iy_e, ix_e = np.where(expand)
    if len(iz_e) > 3000:
        idx_s  = rng.choice(len(iz_e), 3000, replace=False)
        iz_e, iy_e, ix_e = iz_e[idx_s], iy_e[idx_s], ix_e[idx_s]
    vals_e = delta_np[iz_e, iy_e, ix_e]
    sc = ax3.scatter(lin[ix_e], lin[iy_e], lin[iz_e],
                     c=vals_e, cmap='RdBu_r', s=1.5, alpha=0.4,
                     vmin=-vmax, vmax=vmax)
    gx_ = np.linspace(-grid_half, grid_half, 4)
    gX, gY = np.meshgrid(gx_, gx_)
    ax3.plot_surface(gX, gY, np.full_like(gX, ground_z),
                     alpha=0.08, color='saddlebrown')
    ax3.scatter(0, 0, 0, c='red', s=60, zorder=10)
    ax3.set_xlabel('X', fontsize=7); ax3.set_ylabel('Y', fontsize=7)
    ax3.set_zlabel('Z', fontsize=7); ax3.tick_params(labelsize=6)
    ax3.set_title('3D expansion voxels\n(ΔSDF < −0.02 m)', fontsize=8, pad=2)
    fig.colorbar(sc, ax=ax3, shrink=0.55, label='ΔSDF (m)', pad=0.1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — Robot Body Collision Field
# ─────────────────────────────────────────────────────────────────────────────

def plot_body_field(
    field_body:  torch.Tensor,   # [1, 4, G, G, G]
    margin_body: float,
    ground_z:    float,
    grid_half:   float,
    grid_res:    int,
    box_center,
    box_size,
    save_path:   Optional[str] = None,
    show:        bool = False,
):
    """
    Robot-body collision SDF cross-sections.
      Black contour  = body surface (SDF = 0)
      Orange dashed  = force activation edge (SDF = margin_body)
    Layout 1×4: XZ@Y=0 / YZ@X=0 / XY@Z=0 / 3D trunk surface.
    """
    G   = grid_res
    lin = np.linspace(-grid_half, grid_half, G)
    iy0 = G // 2
    ix0 = G // 2
    iz0 = int(round((0.0 / grid_half + 1.0) / 2.0 * (G - 1)))
    sdf = field_body[0, 0].numpy()   # [G, G, G]  axes (z, y, x)

    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(22, 5))
    fig.suptitle(
        'Robot-Body Collision Field  ·  trunk box '
        f'{box_size[0]}×{box_size[1]}×{box_size[2]} m @ centre {tuple(box_center)}\n'
        'Blue = inside body (SDF<0)  ·  Black = body surface  '
        f'·  Orange dashed = force activation edge (SDF = {margin_body} m)',
        fontsize=10, fontweight='bold', y=1.04,
    )

    slices = [
        (sdf[:, iy0, :], 'X (m)', 'Z (m)', 'XZ  (Y=0)', True),
        (sdf[:, :, ix0], 'Y (m)', 'Z (m)', 'YZ  (X=0)', True),
        (sdf[iz0, :, :], 'X (m)', 'Y (m)', 'XY  (Z=0)', False),
    ]
    # Tight colour range — the trunk box spans <0.1% of the grid, so a
    # percentile-based vmax washes out all near-body detail.
    vmax = 0.30

    for col, (s2d, xl, yl, title, show_ground) in enumerate(slices):
        ax = fig.add_subplot(1, 4, col + 1)
        im = ax.imshow(s2d, origin='lower', aspect='equal',
                       extent=[lin[0], lin[-1], lin[0], lin[-1]],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.contour(lin, lin, s2d, levels=[0.0], colors='k', linewidths=1.4)
        ax.contour(lin, lin, s2d, levels=[margin_body], colors='darkorange',
                   linewidths=0.9, linestyles='--')
        if show_ground:
            ax.axhline(ground_z, color='saddlebrown', linewidth=0.9,
                       linestyle='-.', label='Ground')
            ax.legend(fontsize=7)
        ax.scatter(0, 0, c='red', s=40, zorder=6)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(f'Body SDF  {title}', fontsize=8, pad=2)
        ax.grid(alpha=0.2)
        fig.colorbar(im, ax=ax, shrink=0.75, label='SDF (m)')

    # 3D: trunk surface voxels
    ax3 = fig.add_subplot(1, 4, 4, projection='3d')
    surf = np.abs(sdf) < 0.02
    iz_s, iy_s, ix_s = np.where(surf)
    if len(iz_s) > 4000:
        sel = rng.choice(len(iz_s), 4000, replace=False)
        iz_s, iy_s, ix_s = iz_s[sel], iy_s[sel], ix_s[sel]
    ax3.scatter(lin[ix_s], lin[iy_s], lin[iz_s], c='dimgray', s=2.0, alpha=0.5)
    gx_ = np.linspace(-grid_half, grid_half, 4)
    gX, gY = np.meshgrid(gx_, gx_)
    ax3.plot_surface(gX, gY, np.full_like(gX, ground_z),
                     alpha=0.08, color='saddlebrown')
    ax3.scatter(0, 0, 0, c='red', s=60, zorder=10)
    ax3.set_xlabel('X', fontsize=7); ax3.set_ylabel('Y', fontsize=7)
    ax3.set_zlabel('Z', fontsize=7); ax3.tick_params(labelsize=6)
    ax3.set_title('3D trunk surface (|SDF| < 2 cm)', fontsize=8, pad=2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — EE Reachable-Orientation Field
# ─────────────────────────────────────────────────────────────────────────────

def plot_orientation_field(field_orient, limits, ground_z, grid_half, grid_res,
                           save_path=None, show=False):
    """
    XZ@Y=0 cross-sections of the mean reachable EE orientation (roll/pitch/yaw)
    and the orientation spread |σ| — how free the EE orientation is per position.
    """
    G   = grid_res
    lin = np.linspace(-grid_half, grid_half, G)
    iy0 = G // 2
    mean = field_orient[0, :3].numpy()        # [3, G, G, G]
    std  = field_orient[0, 3:].numpy()
    deg  = 180.0 / np.pi
    names = ['roll', 'pitch', 'yaw']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('EE Reachable-Orientation Field  ·  mean feasible orientation per '
                 'workspace voxel  ·  XZ slice @ Y=0',
                 fontsize=10, fontweight='bold', y=1.03)
    for a in range(3):
        ax = axes[a]
        sl = mean[a][:, iy0, :] * deg
        vm = float(limits[a]) * deg
        im = ax.imshow(sl, origin='lower', aspect='equal',
                       extent=[lin[0], lin[-1], lin[0], lin[-1]],
                       cmap='Spectral', vmin=-vm, vmax=vm)
        ax.axhline(ground_z, color='saddlebrown', lw=0.9, ls='-.')
        ax.scatter(0, 0, c='k', s=28)
        ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Z (m)', fontsize=8)
        ax.set_title(f'mean {names[a]} (deg)', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.78, label='deg')

    ax = axes[3]
    spread = np.linalg.norm(std, axis=0)[:, iy0, :] * deg
    im = ax.imshow(spread, origin='lower', aspect='equal',
                   extent=[lin[0], lin[-1], lin[0], lin[-1]], cmap='viridis', vmin=0)
    ax.axhline(ground_z, color='cyan', lw=0.9, ls='-.')
    ax.scatter(0, 0, c='red', s=28)
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Z (m)', fontsize=8)
    ax.set_title('orientation spread |σ| (deg)\n(larger = freer EE orientation)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.78, label='deg')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute workspace SDF fields for Langevin trajectory generator.')
    parser.add_argument('--config',    default='configs/traj_gen.yaml',
                        help='Path to traj_gen.yaml')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Override FK sample count')
    parser.add_argument('--show',      action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    robot_cfg = cfg['robot']
    base_cfg  = cfg['base_traj']
    fg_cfg    = cfg['field_gen']
    lc_cfg    = cfg['langevin']
    device    = cfg['generator'].get('device', 'cuda')
    if not torch.cuda.is_available():
        device = 'cpu'

    base_z     = float(robot_cfg['base_z'])
    q_limits   = np.array(robot_cfg['q_limits'], dtype=np.float32)
    ground_z   = -base_z

    grid_half   = float(fg_cfg['grid_half'])
    grid_res    = int(fg_cfg['grid_res'])
    n_samples   = args.n_samples or int(fg_cfg['n_samples'])
    smooth_sig  = float(fg_cfg.get('smooth_sigma', 1.5))
    save_dir    = Path(fg_cfg['save_dir'])
    fields_fn   = fg_cfg['fields_filename']

    # Robot body collision box — configurable via field_gen.body_box_* in YAML
    body_center = tuple(float(v) for v in
                        fg_cfg.get('body_box_center', BODY_BOX_CENTER))
    body_size   = tuple(float(v) for v in
                        fg_cfg.get('body_box_size', BODY_BOX_SIZE))
    if any(s <= 0.0 for s in body_size):
        raise ValueError(f"field_gen.body_box_size must be all-positive, got {body_size}")

    k_bound     = float(lc_cfg['k_bound'])
    k_ground    = float(lc_cfg['k_ground'])
    decay_rate  = float(lc_cfg['decay_rate'])
    margin_body = float(lc_cfg.get('margin_body', 0.10))

    # EE reachable-orientation field — command box + fill sigma
    eo_cfg = cfg.get('ee_orientation', {})
    orient_limits = (float(eo_cfg.get('roll_limit',  1.4137)),
                     float(eo_cfg.get('pitch_limit', 1.0472)),
                     float(eo_cfg.get('yaw_limit',   1.3090)))
    orient_sigma  = float(eo_cfg.get('smooth_sigma', 2.0))

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device.upper()}")
    print(f"base_z    : {base_z} m  (ground at Z_arm = {ground_z} m)")
    print(f"grid_half : {grid_half} m  |  grid_res : {grid_res}³  |  n_samples : {n_samples:,}\n")

    fields = {}
    ee_static = eul_static = None
    for label, is_dyn in [('static', False), ('dynamic', True)]:
        print(f"── {label.capitalize()} workspace {'─'*42}")
        t0 = time.perf_counter()
        ee_arm, ee_euler = sample_ee_armframe(
            n_samples, q_limits, base_z, base_cfg, is_dyn, device)
        elapsed = time.perf_counter() - t0
        pct_valid = 100.0 * len(ee_arm) / n_samples
        print(f"  {len(ee_arm):,} valid EE positions  "
              f"({pct_valid:.1f}% of {n_samples:,})  [{elapsed:.1f} s]")
        print(f"  Z arm-base: [{ee_arm[:,2].min():.3f}, {ee_arm[:,2].max():.3f}] m")
        if not is_dyn:
            ee_static, eul_static = ee_arm, ee_euler   # keep for the orientation field

        t0 = time.perf_counter()
        sdf = build_sdf(ee_arm, grid_half, grid_res, smooth_sig)
        print(f"  SDF built   [{time.perf_counter()-t0:.1f} s]")

        grad  = compute_gradient_field(sdf, grid_half, grid_res)
        field = build_combined_field(sdf, grad)   # [1, 4, G, G, G]  CPU
        fields[label] = field
        inside = (field[0, 0] < 0).float().mean().item() * 100
        print(f"  Workspace volume: {inside:.1f}% of grid  |  "
              f"SDF range: [{field[0,0].min():.3f}, {field[0,0].max():.3f}] m\n")

    # Delta field: encodes how base motion expands/contracts the workspace.
    # Cached at D-change time by langevin_traj_gen.py → 1 grid_sample call/step.
    field_delta = fields['dynamic'] - fields['static']   # [1, 4, G, G, G]
    delta_range = (field_delta[0, 0].min().item(), field_delta[0, 0].max().item())
    print(f"Delta field (SDF channel): [{delta_range[0]:.3f}, {delta_range[1]:.3f}] m")
    print(f"  Expansion voxels (Δ<−0.02 m): "
          f"{(field_delta[0,0] < -0.02).float().mean().item()*100:.1f}% of grid\n")

    # Robot-body collision field: trunk box as a static obstacle in arm-base
    # frame.  The EE workspace SDFs label the trunk region as "reachable", so
    # without this field particles pass straight through the robot's own body.
    print(f"── Robot body collision field {'─'*34}")
    body_sdf   = generate_body_sdf(grid_half, grid_res, device,
                                   body_center, body_size)
    body_grad  = compute_gradient_field(body_sdf, grid_half, grid_res)
    field_body = build_combined_field(body_sdf, body_grad)   # [1, 4, G, G, G]
    inside_b   = (field_body[0, 0] < 0).float().mean().item() * 100
    print(f"  trunk box  L×W×H = {body_size} m  @ centre {body_center}")
    print(f"  body volume: {inside_b:.2f}% of grid  |  "
          f"SDF range: [{field_body[0,0].min():.3f}, {field_body[0,0].max():.3f}] m\n")

    # EE reachable-orientation field: per-voxel mean/std of the EE orientations
    # that are actually achievable there.  pose_traj_gen.py samples the attitude
    # around this position-conditioned mean → generated poses stay jointly
    # feasible (a position may be reachable but not with an arbitrary attitude).
    print(f"── EE reachable-orientation field {'─'*30}")
    field_orient, n_inbox = build_orientation_field(
        ee_static, eul_static, orient_limits, grid_half, grid_res, orient_sigma)
    pct_in = 100.0 * n_inbox / max(len(ee_static), 1)
    print(f"  command box: roll±{orient_limits[0]:.3f}  pitch±{orient_limits[1]:.3f}"
          f"  yaw±{orient_limits[2]:.3f} rad")
    print(f"  {n_inbox:,} samples inside the command box ({pct_in:.1f}% of static)")
    deg = 180.0 / np.pi
    print(f"  mean reachable-orientation spread |σ| ≈ "
          f"{field_orient[0,3:].norm(dim=0).mean().item()*deg:.1f}°\n")

    # Save
    out_pt = save_dir / fields_fn
    torch.save({
        'field_static':    fields['static'],
        'field_dynamic':   fields['dynamic'],
        'field_delta':     field_delta,
        'field_body':      field_body,
        'field_orient':    field_orient,
        'orient_limits':   orient_limits,
        'body_box_center': body_center,
        'body_box_size':   body_size,
        'grid_half':       grid_half,
        'grid_res':        grid_res,
        'base_z':          base_z,
        'ground_z':        ground_z,
    }, out_pt)
    print(f"Saved → {out_pt}\n")

    # Visualise
    show = args.show
    print("Generating visualisations ...")
    plot_sdf_fields(
        fields['static'], fields['dynamic'],
        ground_z, grid_half, grid_res,
        save_path=str(save_dir / 'workspace_sdf.png'), show=show,
    )
    plot_force_fields(
        fields['static'], fields['dynamic'],
        ground_z, grid_half, grid_res,
        k_bound, k_ground, decay_rate,
        save_path=str(save_dir / 'force_fields.png'), show=show,
    )
    plot_delta_field(
        field_delta, fields['static'],
        ground_z, grid_half, grid_res,
        save_path=str(save_dir / 'base_delta_field.png'), show=show,
    )
    plot_body_field(
        field_body, margin_body, ground_z, grid_half, grid_res,
        body_center, body_size,
        save_path=str(save_dir / 'body_collision_field.png'), show=show,
    )
    plot_orientation_field(
        field_orient, orient_limits, ground_z, grid_half, grid_res,
        save_path=str(save_dir / 'orientation_field.png'), show=show,
    )
    print("Done.")


if __name__ == '__main__':
    main()
