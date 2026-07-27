"""Standalone demo/test for M1 (trajectory representation) + M5 (trajectory
generation) from docs/coding-agent-guide.md.

Pure PyTorch/NumPy/SciPy/Matplotlib -- no IsaacGym, no RoboDuet env. Run with
the plain system python3 (not the isaacgym conda env):

    python3 scripts/demo_trajectory_generation.py [--out DIR]

What it does:
  1. Self-tests matching the design doc's stated M1 test criteria:
       - circular arc: to_canonical's L should match 2*pi*r
       - figure-8 path: update_s should never jump backward or skip a lobe
       - sample_preview: should clamp correctly once s approaches L
  2. M5 demo: procedurally generates a geometric path (GeometryGenerator) and
     a time law (TimingGenerator), composed by TrajectoryFactory.
  3. Simulates "stepping" through the trajectory at a fixed control dt using
     the time law (s_ref(t) -> Gamma.p_at/R_at), and separately simulates a
     lagging tracker being kept on-path by update_s, to test both directions
     of M1 (reference generation forward, and re-projection backward).
  4. Curriculum ramp demo: uses DifficultySchedule to generate trajectories
     at several simulated training-step checkpoints and checks that measured
     difficulty (top speed required, workspace excursion) rises monotonically
     as the simulated step count increases. Path length is reported alongside
     these as descriptive geometry info only -- per the design doc, path
     length by itself is NOT a difficulty axis (a long, slow, gentle path can
     be easy; a short, violent one can be hard). The doc's real difficulty
     metric is v_base_max/a_base_max/v_base_mean, the base velocity/
     acceleration implied by OnlineBaseNom (M3) tracking the EE path -- not
     implemented here, so sdot_max/excursion_p95 are proxies standing in for
     it, not the literal spec quantities.
  5. Saves multi-panel matplotlib figures to --out (default: scratch dir).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.trajectory import Gamma, TrajectoryBatch, so3_exp, to_canonical, update_s  # noqa: E402
from modules.trajectory_generator import (  # noqa: E402
    DifficultySchedule,
    GeometryGenerator,
    TimingGenerator,
    TrajectoryFactory,
)


# ============================================================
# Self-tests (design doc M1 test criteria)
# ============================================================


def test_circular_arc_length():
    r = 0.3
    n = 400
    dt = 0.01
    theta = torch.linspace(0, 2 * math.pi, n)
    p = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.zeros(n)], dim=-1)
    R = torch.eye(3).expand(n, 3, 3).contiguous()  # constant orientation, rotation shouldn't add arc length
    gamma, _ = to_canonical(p, R, dt, lam=0.15)
    expected = 2 * math.pi * r
    err = abs(gamma.L - expected) / expected
    ok = err < 0.01
    print(f"[test] circular arc length: L={gamma.L:.4f}, expected={expected:.4f}, rel_err={err:.4%} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_figure_eight_update_s_no_jump():
    n = 800
    dt = 0.01
    t = torch.linspace(0, 4 * math.pi, n)
    a = 0.3
    p = torch.stack([a * torch.sin(t), a * torch.sin(t) * torch.cos(t), torch.zeros(n)], dim=-1)
    R = torch.eye(3).expand(n, 3, 3).contiguous()
    gamma, _ = to_canonical(p, R, dt, lam=0.15)

    # simulate a tracker walking along the ground-truth path with small noise
    N = 1
    s = torch.zeros(N)
    s_hist = []
    torch.manual_seed(0)
    n_steps = 300
    for i in range(n_steps):
        s_query = torch.tensor([min(gamma.L, i / n_steps * gamma.L)])
        ee_p = gamma.p_at(s_query) + torch.randn(N, 3) * 0.005
        ee_R = gamma.R_at(s_query)
        s, d_lat = update_s(s, ee_p, ee_R, gamma, window=0.15, M=16, lam=0.15)
        s_hist.append(s.item())

    s_hist_t = torch.tensor(s_hist)
    backward = (s_hist_t[1:] - s_hist_t[:-1] < -1e-6).any().item()
    # a "jump" = advancing by much more than one window in a single step
    big_jump = ((s_hist_t[1:] - s_hist_t[:-1]).abs() > 0.16).any().item()
    ok = (not backward) and (not big_jump)
    print(
        f"[test] figure-8 update_s monotonic/no-jump: backward={backward}, "
        f"big_jump={big_jump}, final s/L={s_hist[-1] / gamma.L:.3f} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, gamma, s_hist_t


def test_preview_clamp():
    n = 200
    dt = 0.01
    t = torch.linspace(0, 1, n)
    p = torch.stack([0.3 * t, torch.zeros(n), torch.zeros(n)], dim=-1)
    R = torch.eye(3).expand(n, 3, 3).contiguous()
    gamma, time_law = to_canonical(p, R, dt, lam=0.15)

    batch = TrajectoryBatch(
        N=1, max_gamma_points=gamma.s_grid.shape[0] + 1, max_tl_points=time_law.t_grid.shape[0] + 1, device="cpu"
    )
    batch.load([0], [gamma], [time_law])

    s_current = torch.tensor([gamma.L - 0.02])  # near the end
    s_k, p_k, R_k, sdot_k = batch.sample_preview(s_current, L_h=0.5, K=9)
    ok = bool((s_k <= gamma.L + 1e-4).all())
    print(f"[test] preview clamp near s=L: max(s_k)={s_k.max().item():.4f}, L={gamma.L:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# M5 demo: procedural generation + step-by-step walk-through
# ============================================================


def run_m5_demo(out_dir, seed=0):
    torch.manual_seed(seed)
    factory = TrajectoryFactory(GeometryGenerator(), TimingGenerator())

    geom_params = dict(
        f_max=0.4,
        amplitude=0.12,
        f_rot_max=0.25,
        f_rot_amplitude=0.6,
        tangent_align_ratio=0.6,
        drift_speed=0.03,
        drift_dir=[1.0, 0.3],
        center=[0.3, 0.0, 0.55],
        duration=8.0,
        dt=0.01,
        n_freqs=8,
        lam=0.15,
        seed=seed,
    )
    timing_params = dict(f_max=0.3, v_max=0.25, T=6.0, dt=0.02, n_freqs=6, seed=seed + 1)

    gamma, time_law = factory.generate(geom_params, timing_params)
    print(f"[m5] generated gamma: L={gamma.L:.3f} m, {gamma.s_grid.shape[0]} arc-length samples")
    print(f"[m5] generated time_law: T={time_law.T:.2f} s, s_ref(T)={time_law.s_ref(torch.tensor([time_law.T])).item():.3f} (should be ~= L)")

    # --- step through the trajectory at a fixed control dt, reading the
    # reference point at each step (this is "轨迹随 step 的生成效果") ---
    control_dt = 0.02
    steps = int(time_law.T / control_dt)
    t_steps = torch.arange(steps) * control_dt
    s_ref_steps = time_law.s_ref(t_steps)
    sdot_ref_steps = time_law.sdot_ref(t_steps)
    p_ref_steps = gamma.p_at(s_ref_steps)
    R_ref_steps = gamma.R_at(s_ref_steps)
    tangent_steps = gamma.tangent_at(s_ref_steps)

    # --- simulate a base-mounted tracker that lags the reference (first-order
    # lag, like a real controller would), and drive it back onto the path
    # with update_s each step (this exercises the "backward" M1 direction) ---
    T_lag = 0.15
    ee_p = p_ref_steps[0].clone()
    ee_R = R_ref_steps[0].clone()
    s_tracked = torch.zeros(1)
    s_tracked_hist, d_lat_hist, ee_p_hist = [], [], []
    for i in range(steps):
        target_p = p_ref_steps[i]
        ee_p = ee_p + (target_p - ee_p) * (control_dt / T_lag) + torch.randn(3) * 0.003
        ee_R = R_ref_steps[i]  # orientation tracked exactly for simplicity
        s_tracked, d_lat = update_s(s_tracked, ee_p.unsqueeze(0), ee_R.unsqueeze(0), gamma, window=0.2, M=16, lam=0.15)
        s_tracked_hist.append(s_tracked.item())
        d_lat_hist.append(d_lat.item())
        ee_p_hist.append(ee_p.clone())
    s_tracked_hist = torch.tensor(s_tracked_hist)
    d_lat_hist = torch.tensor(d_lat_hist)
    ee_p_hist = torch.stack(ee_p_hist)

    # --- preview points at one representative step, for visualization ---
    mid_step = steps // 3
    batch = TrajectoryBatch(
        N=1, max_gamma_points=gamma.s_grid.shape[0] + 1, max_tl_points=time_law.t_grid.shape[0] + 1, device="cpu"
    )
    batch.load([0], [gamma], [time_law])
    s_k, p_k, R_k, sdot_k = batch.sample_preview(s_ref_steps[mid_step : mid_step + 1], L_h=0.5, K=9)

    return dict(
        gamma=gamma,
        time_law=time_law,
        t_steps=t_steps,
        s_ref_steps=s_ref_steps,
        sdot_ref_steps=sdot_ref_steps,
        p_ref_steps=p_ref_steps,
        tangent_steps=tangent_steps,
        s_tracked_hist=s_tracked_hist,
        d_lat_hist=d_lat_hist,
        ee_p_hist=ee_p_hist,
        mid_step=mid_step,
        p_k=p_k[0],
    )


def visualize(result, out_dir, fig8_gamma, fig8_s_hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 11))

    gamma = result["gamma"]
    p_ref = result["p_ref_steps"].numpy()
    ee_p = result["ee_p_hist"].numpy()
    t_steps = result["t_steps"].numpy()
    s_ref = result["s_ref_steps"].numpy()
    sdot_ref = result["sdot_ref_steps"].numpy()
    tangent = result["tangent_steps"].numpy()
    s_tracked = result["s_tracked_hist"].numpy()
    d_lat = result["d_lat_hist"].numpy()
    p_k = result["p_k"].numpy()
    mid_step = result["mid_step"]

    # 1) 3D geometric path + stepped reference points + preview window
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    gp = gamma.p.numpy()
    ax.plot(gp[:, 0], gp[:, 1], gp[:, 2], "-", color="gray", lw=1, label="Gamma (full geometry)")
    sc = ax.scatter(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], c=t_steps, cmap="viridis", s=8, label="s_ref(t) steps")
    ax.scatter(*p_ref[mid_step], color="red", s=60, marker="*", label=f"step {mid_step}")
    ax.scatter(p_k[:, 0], p_k[:, 1], p_k[:, 2], color="orange", s=25, marker="^", label="preview (K=9)")
    ax.set_title("Gamma geometry + time-law-driven stepping")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="t (s)")
    ax.legend(fontsize=7, loc="upper left")

    # 2) s_ref(t) and sdot_ref(t)
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t_steps, s_ref, label="s_ref(t)")
    ax.axhline(gamma.L, color="gray", ls="--", lw=1, label="L")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("s (m)")
    ax.set_title("Time law: s_ref(t)")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t_steps, sdot_ref, color="tab:orange", alpha=0.6, label="sdot_ref(t)")
    ax2.set_ylabel("sdot (m/s)", color="tab:orange")

    # 3) tangent direction sanity (should stay unit-norm, vary smoothly)
    ax = fig.add_subplot(2, 3, 3)
    tnorm = np.linalg.norm(tangent, axis=-1)
    ax.plot(t_steps, tnorm, label="||tangent||")
    ax.plot(t_steps, tangent[:, 0], alpha=0.6, label="tangent_x")
    ax.plot(t_steps, tangent[:, 1], alpha=0.6, label="tangent_y")
    ax.plot(t_steps, tangent[:, 2], alpha=0.6, label="tangent_z")
    ax.set_xlabel("t (s)")
    ax.set_title("tangent_at(s_ref(t))")
    ax.legend(fontsize=7)

    # 4) update_s tracking: s_tracked vs s_ref(t) (open-loop reference used as
    # the *target* the lagging simulated tracker chases, not literally equal)
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t_steps, s_ref, label="s_ref(t) (reference)")
    ax.plot(t_steps, s_tracked, label="s from update_s(lagging tracker)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("s (m)")
    ax.set_title("update_s re-projection vs. time-law reference")
    ax.legend(fontsize=8)

    # 5) lateral deviation d_lat over steps
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t_steps, d_lat)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("d_lat (m)")
    ax.set_title("update_s lateral deviation")

    # 6) figure-8 self-test trajectory (monotonic-s sanity check)
    ax = fig.add_subplot(2, 3, 6)
    gp8 = fig8_gamma.p.numpy()
    ax.plot(gp8[:, 0], gp8[:, 1], "-", color="gray", lw=1, label="figure-8 Gamma")
    s_np = fig8_s_hist.numpy()
    frac = s_np / fig8_gamma.L
    pts = fig8_gamma.p_at(fig8_s_hist).numpy()
    sc2 = ax.scatter(pts[:, 0], pts[:, 1], c=frac, cmap="plasma", s=10)
    ax.set_title("M1 self-test: update_s on figure-8", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(sc2, ax=ax, shrink=0.7, label="s / L")

    fig.suptitle("M1 (trajectory representation) + M5 (procedural generation) demo", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "m1_m5_demo.png")
    fig.savefig(out_path, dpi=130)
    print(f"[viz] saved {out_path}")
    return out_path


def run_curriculum_demo(seed=0):
    """Sweep simulated training steps through a DifficultySchedule ramp and
    check that the generated trajectories actually get harder."""
    schedule = DifficultySchedule(
        easy_geom=dict(
            f_max=0.15,
            amplitude=0.04,
            f_rot_max=0.10,
            f_rot_amplitude=0.20,
            tangent_align_ratio=0.6,
            drift_speed=0.0,
            drift_dir=[1.0, 0.0],
            center=[0.3, 0.0, 0.55],
            duration=8.0,
            dt=0.01,
            n_freqs=8,
            lam=0.15,
        ),
        hard_geom=dict(
            f_max=0.9,
            amplitude=0.22,
            f_rot_max=0.6,
            f_rot_amplitude=1.0,
            tangent_align_ratio=0.6,
            drift_speed=0.08,
            drift_dir=[1.0, 0.3],
            center=[0.3, 0.0, 0.55],
            duration=8.0,
            dt=0.01,
            n_freqs=8,
            lam=0.15,
        ),
        easy_timing=dict(f_max=0.10, v_max=0.08, T=6.0, dt=0.02, n_freqs=6),
        hard_timing=dict(f_max=0.50, v_max=0.50, T=6.0, dt=0.02, n_freqs=6),
        ramp_steps=20000,
    )

    checkpoints = [0, 5000, 10000, 15000, 20000]
    factory = TrajectoryFactory(GeometryGenerator(), TimingGenerator())
    records = []
    for step in checkpoints:
        geom_params, timing_params, alpha = schedule.params_at(step)
        geom_params = dict(geom_params, seed=seed)
        timing_params = dict(timing_params, seed=seed + 1)
        gamma, time_law = factory.generate(geom_params, timing_params)

        t = torch.arange(0, time_law.T, timing_params["dt"])
        s_ref = time_law.s_ref(t)
        sdot_ref = time_law.sdot_ref(t)
        p_ref = gamma.p_at(s_ref)
        center = torch.as_tensor(geom_params["center"])
        dist_from_center = (p_ref - center).norm(dim=-1)

        records.append(
            dict(
                step=step,
                alpha=alpha,
                gamma=gamma,
                p_ref=p_ref,
                sdot_max=sdot_ref.max().item(),
                sdot_mean=sdot_ref.mean().item(),
                excursion_p95=torch.quantile(dist_from_center, 0.95).item(),
                path_length=gamma.L,
            )
        )
        print(
            f"[curriculum] step={step:>6d} alpha={alpha:.2f}  "
            f"sdot_max={records[-1]['sdot_max']:.3f} m/s  "
            f"excursion_p95={records[-1]['excursion_p95']:.3f} m  "
            f"(descriptive only, not a difficulty axis: L={records[-1]['path_length']:.3f} m)"
        )
    return records


def visualize_curriculum(records, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(records)
    fig = plt.figure(figsize=(4 * n, 8))
    # shared grid: n*3 columns so a top-row cell (3 cols wide) and a
    # bottom-row cell (n cols wide) both tile evenly, no layout warning
    gs = fig.add_gridspec(2, n * 3)

    # top row: top-down (x,y) view of the generated path at each checkpoint,
    # all on the same axis limits so "getting bigger/wilder" is visually honest
    all_p = torch.cat([r["p_ref"] for r in records], dim=0)
    pad = 0.02
    xlim = (all_p[:, 0].min().item() - pad, all_p[:, 0].max().item() + pad)
    ylim = (all_p[:, 1].min().item() - pad, all_p[:, 1].max().item() + pad)

    for i, r in enumerate(records):
        ax = fig.add_subplot(gs[0, i * 3 : i * 3 + 3])
        p = r["p_ref"].numpy()
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, color=plt.cm.viridis(r["alpha"]))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"step={r['step']}\n(alpha={r['alpha']:.2f})", fontsize=9)
        if i == 0:
            ax.set_ylabel("y (m)")
        ax.set_xlabel("x (m)")

    # bottom row: left two panels are difficulty proxies (stand-ins for the
    # design doc's v_base_max/a_base_max, which need OnlineBaseNom / M3, not
    # implemented here); path length is descriptive geometry info only, per
    # the doc it is explicitly NOT a difficulty axis on its own.
    steps = [r["step"] for r in records]
    w = n  # each of the 3 bottom cells spans n columns of the n*3 grid
    ax = fig.add_subplot(gs[1, 0 * w : 1 * w])
    ax.plot(steps, [r["sdot_max"] for r in records], "o-")
    ax.set_xlabel("training step")
    ax.set_ylabel("sdot_max (m/s)")
    ax.set_title("required top speed (difficulty proxy)")

    ax = fig.add_subplot(gs[1, 1 * w : 2 * w])
    ax.plot(steps, [r["excursion_p95"] for r in records], "o-", color="tab:orange")
    ax.set_xlabel("training step")
    ax.set_ylabel("p95 dist. from center (m)")
    ax.set_title("workspace excursion (difficulty proxy)")

    ax = fig.add_subplot(gs[1, 2 * w : 3 * w])
    ax.plot(steps, [r["path_length"] for r in records], "o-", color="tab:green")
    ax.set_xlabel("training step")
    ax.set_ylabel("path length L (m)")
    ax.set_title("total path length (descriptive only, NOT difficulty)")

    fig.suptitle("DifficultySchedule ramp: trajectories generated at increasing simulated training steps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(out_dir, "m10_curriculum_demo.png")
    fig.savefig(out_path, dpi=130)
    print(f"[viz] saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="output directory for the figure")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "trajectory_demo")
    out_dir = os.path.abspath(out_dir)

    print("=" * 60)
    print("M1 self-tests")
    print("=" * 60)
    ok1 = test_circular_arc_length()
    ok2, fig8_gamma, fig8_s_hist = test_figure_eight_update_s_no_jump()
    ok3 = test_preview_clamp()
    all_ok = ok1 and ok2 and ok3
    print(f"\nself-tests: {'ALL PASS' if all_ok else 'SOME FAILED'}\n")

    print("=" * 60)
    print("M5 procedural generation demo")
    print("=" * 60)
    result = run_m5_demo(out_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    visualize(result, out_dir, fig8_gamma, fig8_s_hist)

    print("\n" + "=" * 60)
    print("Curriculum ramp demo (DifficultySchedule)")
    print("=" * 60)
    records = run_curriculum_demo(seed=args.seed)
    monotonic = all(
        records[i]["sdot_max"] <= records[i + 1]["sdot_max"] + 1e-6
        and records[i]["excursion_p95"] <= records[i + 1]["excursion_p95"] + 1e-6
        for i in range(len(records) - 1)
    )
    print(f"[curriculum] difficulty monotonically non-decreasing across steps: {'PASS' if monotonic else 'FAIL'}")
    visualize_curriculum(records, out_dir)
    all_ok = all_ok and monotonic

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
