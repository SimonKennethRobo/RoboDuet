"""WBC benchmark scenarios.

Each scenario returns ``{run_name: [result_dict]}``, where ``result_dict`` is
the flat dict from ``WBCAccumulator.wbc_summary()`` with extra metadata
(cell_A, cell_B, label). These go directly into the benchmark's ``results.json``
and are consumed by the HTML report and comparison tool.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import isaacgym  # noqa: F401 - must precede torch
import torch

from benchmark.wbc.evaluation import (
    WBCAccumulator,
    WBCPolicyHandle,
    wbc_acc_to_result,
    wbc_eval_loop,
)


def run_wbc_aggregate(
    env,
    handles: List[WBCPolicyHandle],
    cells: Optional[List[tuple]] = None,
    n_steps: int = 500,
    settle_steps: int = 30,
    device: str = "cuda:0",
) -> Dict[str, List[dict]]:
    """Run every enabled curriculum cell; accumulate WBC metrics.

    ``cells`` is a list of (A, B) pairs. The default is every cell in the
    6x6 grid, which is 36 points -- enough to fill the profile from the easiest
    to the hardest difficulty the curriculum spans.

    Each cell runs exactly one policy step loop. The env's ``_load_trajectory_for``
    is called with the cell pinned, so every env in the group tracks a fresh
    held-out bank trajectory at that difficulty.
    """
    base = env.env
    if cells is None:
        cells = [(a, b) for a in range(base.traj_curriculum.nA) for b in range(base.traj_curriculum.nB)]

    out: Dict[str, List[dict]] = {h.name: [] for h in handles}
    dt = float(base.dt)
    all_ids = torch.arange(base.num_envs, device=base.device)

    total = len(cells)
    for i, (a, b) in enumerate(cells):
        label = f"cell_A={a}_B={b}"
        print(f"  [{i + 1:2d}/{total}] {label}", end="  ", flush=True)

        # Pin the cell and reload trajectories.
        base.traj_curriculum.cell_A[:] = a
        base.traj_curriculum.cell_B[:] = b
        base._load_trajectory_for(all_ids)

        accs = wbc_eval_loop(env, handles, n_steps, device, settle_steps=settle_steps)
        for h, acc in zip(handles, accs):
            result = wbc_acc_to_result(acc, h.name, "wbc_aggregate", label, dt, a, b)
            out[h.name].append(result)
            print(f"{h.name}: pos_err={result['ee_pos_rmse_m']:.3f}m  "
                  f"rho={result['rho_mean']:.2f}  "
                  f"util={result['base_util_mean']:.2f}  "
                  f"power={result['motor_power_mean_w']:.1f}W",
                  end="  " if len(handles) > 1 else "", flush=True)
        print()

    return out
