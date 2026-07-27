"""M10: curriculum manager + trajectory bank for trajectory tracking.

Standalone (no IsaacGym). Two pieces:

- ``CurriculumManager``: a 2D grid curriculum (level_A = geometry difficulty,
  level_B = timing difficulty). Tracks a per-cell EMA success rate, expands an
  unlocked "frontier" as cells are mastered (promote), and samples each env's
  next cell with a 60/40 frontier/history mix (anti-forgetting). Per-env
  current cell is stored as two ``(N,)`` long tensors, mirroring the codebase's
  ``env_command_bins`` convention (go1_gym/envs/base/curriculum.py).

- ``TrajectoryBank``: builds, once at init, a pool of pre-generated
  ``(Gamma, TimeLaw)`` per grid cell (via the M5 ``TrajectoryFactory``) packed
  into a ``TrajectoryBatch`` so a reset only GPU-gathers a row instead of
  running CPU/scipy generation on the hot path.
"""

import numpy as np
import torch

from .trajectory import TrajectoryBatch, to_canonical
from .trajectory_generator import TrajectoryFactory


def _lerp(a, b, alpha):
    if isinstance(a, (list, tuple)):
        return [_lerp(ai, bi, alpha) for ai, bi in zip(a, b)]
    return a + (b - a) * alpha


# Difficulty presets. level_A interpolates geometry (path shape) difficulty;
# level_B interpolates timing (traversal speed) difficulty. Kept reach-feasible
# (small amplitude, workspace-centered) so rho stays mostly within the comfort
# band -- see the plan's rho risk note.
# center=[0,0,0]: paths are generated around the origin and translated to a
# reachable anchor in front of the shoulder at env reset (see wbc_env
# _arm_post_reset_refresh_hook). Kept reach-feasible (small amplitude) so rho
# stays mostly within the comfort band -- see the plan's rho risk note.
EASY_GEOM = dict(
    f_max=0.15, amplitude=0.04, f_rot_max=0.10, f_rot_amplitude=0.20,
    tangent_align_ratio=0.6, drift_speed=0.0, drift_dir=[1.0, 0.0],
    center=[0.0, 0.0, 0.0], duration=8.0, dt=0.02, n_freqs=8, lam=0.15, ds_grid=0.01,
)
HARD_GEOM = dict(
    f_max=0.6, amplitude=0.16, f_rot_max=0.5, f_rot_amplitude=0.8,
    tangent_align_ratio=0.6, drift_speed=0.05, drift_dir=[1.0, 0.2],
    center=[0.0, 0.0, 0.0], duration=8.0, dt=0.02, n_freqs=8, lam=0.15, ds_grid=0.01,
)
EASY_TIMING = dict(f_max=0.10, v_max=0.06, T=8.0, dt=0.02, n_freqs=6)
HARD_TIMING = dict(f_max=0.40, v_max=0.30, T=8.0, dt=0.02, n_freqs=6)


class CurriculumManager:
    def __init__(self, num_envs, device, n_levels_A=6, n_levels_B=6,
                 success_threshold=0.7, fail_threshold=0.3, ema_alpha=0.05, seed=0):
        self.num_envs = num_envs
        self.device = device
        self.nA, self.nB = n_levels_A, n_levels_B
        self.success_threshold = success_threshold
        self.fail_threshold = fail_threshold
        self.ema_alpha = ema_alpha
        self.rng = np.random.RandomState(seed)

        self.success_ema = np.zeros((self.nA, self.nB), dtype=np.float64)
        self.trials = np.zeros((self.nA, self.nB), dtype=np.int64)
        self.unlocked = np.zeros((self.nA, self.nB), dtype=bool)
        self.unlocked[0, 0] = True  # start at the easiest cell

        # per-env current cell (mirrors env_command_bins convention)
        self.cell_A = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cell_B = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ---- cell -> generator params ----
    def params_for_cell(self, a, b):
        alpha_A = a / max(1, self.nA - 1)
        alpha_B = b / max(1, self.nB - 1)
        geom = {k: _lerp(EASY_GEOM[k], HARD_GEOM[k], alpha_A) for k in EASY_GEOM}
        timing = {k: _lerp(EASY_TIMING[k], HARD_TIMING[k], alpha_B) for k in EASY_TIMING}
        return geom, timing

    # ---- sampling ----
    def _unlocked_cells(self):
        return list(zip(*np.nonzero(self.unlocked)))

    def _frontier_cells(self):
        # unlocked but not yet mastered
        return [(a, b) for (a, b) in self._unlocked_cells()
                if self.success_ema[a, b] < self.success_threshold]

    def sample_cells(self, env_ids):
        """Assign each env in env_ids a new (A,B) cell: 60% frontier, 40% history."""
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        unlocked = self._unlocked_cells()
        frontier = self._frontier_cells()
        for i in range(len(env_ids)):
            use_frontier = frontier and self.rng.rand() < 0.6
            pool = frontier if use_frontier else unlocked
            a, b = pool[self.rng.randint(len(pool))]
            self.cell_A[env_ids[i]] = int(a)
            self.cell_B[env_ids[i]] = int(b)

    # ---- success reporting / promotion ----
    def report_result(self, env_ids, success):
        """success: (len(env_ids),) bool tensor. Updates per-cell EMA and
        promotes the frontier when a cell is mastered."""
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        a = self.cell_A[env_ids].cpu().numpy()
        b = self.cell_B[env_ids].cpu().numpy()
        succ = success.detach().float().cpu().numpy()
        for i in range(len(env_ids)):
            ai, bi = int(a[i]), int(b[i])
            self.trials[ai, bi] += 1
            self.success_ema[ai, bi] = (
                (1 - self.ema_alpha) * self.success_ema[ai, bi] + self.ema_alpha * succ[i]
            )
            if self.success_ema[ai, bi] > self.success_threshold:
                self._promote(ai, bi)

    def _promote(self, a, b):
        for na, nb in ((a + 1, b), (a, b + 1)):
            if na < self.nA and nb < self.nB:
                self.unlocked[na, nb] = True

    # ---- logging ----
    def stats(self):
        return dict(
            mean_level_A=float(self.cell_A.float().mean().item()),
            mean_level_B=float(self.cell_B.float().mean().item()),
            frontier_success_rate=float(
                np.mean([self.success_ema[a, b] for a, b in self._unlocked_cells()])
            ),
            unlocked_cells=int(self.unlocked.sum()),
        )


class TrajectoryBank:
    """Pre-generated pool of trajectories, ``per_cell`` per grid cell, packed
    into a single ``TrajectoryBatch`` for GPU-gather at reset."""

    def __init__(self, curriculum, per_cell, max_gamma_points, max_tl_points,
                 device="cpu", seed=0):
        self.curriculum = curriculum
        self.per_cell = per_cell
        self.nA, self.nB = curriculum.nA, curriculum.nB
        total = self.nA * self.nB * per_cell
        self.batch = TrajectoryBatch(total, max_gamma_points, max_tl_points, device=device)

        factory = TrajectoryFactory()
        gammas, tls, row = [], [], 0
        # cell_offset[a,b] = first bank row for that cell
        self.cell_offset = np.zeros((self.nA, self.nB), dtype=np.int64)
        for a in range(self.nA):
            for b in range(self.nB):
                self.cell_offset[a, b] = row
                geom, timing = curriculum.params_for_cell(a, b)
                for k in range(per_cell):
                    g = dict(geom, seed=seed + row)
                    t = dict(timing, seed=seed + 100000 + row)
                    gamma, tl = factory.generate(g, t)
                    gammas.append(gamma)
                    tls.append(tl)
                    row += 1
        self.batch.load(list(range(total)), gammas, tls)

    def sample_rows(self, cell_A, cell_B, rng):
        """cell_A, cell_B: (M,) long tensors -> (M,) long bank row indices."""
        a = cell_A.cpu().numpy()
        b = cell_B.cpu().numpy()
        k = rng.randint(0, self.per_cell, size=len(a))
        rows = self.cell_offset[a, b] + k
        return torch.as_tensor(rows, dtype=torch.long)
