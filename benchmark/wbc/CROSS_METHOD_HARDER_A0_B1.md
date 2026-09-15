# Cross-method MuJoCo A0/B1 benchmark

This run evaluates all eight handoff methods on one frozen trajectory that is
moderately harder than the earlier A0/B0 integration task. The A0/B1 reference
has a 3.325 m SE(3) path and 0.205 m/s peak equivalent speed, compared with
2.567 m and 0.163 m/s for A0/B0. Spatial span is unchanged. The full reference
duration is 30.437 s and the TaskSpec deadline is 31.0 s.

- Task ID: `timed-trajectory-4a663eb29353ed21`
- Suite SHA-256: `ea8a721d6bb101041a4ce26794744199f20f37b6a0dc069a93a1863a1a343f97`
- Controller/render implementation commit: `aee26fda8ea1ee56782dd12146df188a62f8db0e`
- Conditions: nominal and deterministic 80 N base force for 0.1 s (`8 N s`)
- Result root: `benchmark/results/cross_method_mujoco_harder_release`
- Summary: `benchmark_summary.json`, `benchmark_summary.csv`, and
  `trajectory_comparison.png` under the result root

Every scenario has a 25 fps, 960 x 540 H.264 `mujoco_tracking.mp4` and an
individual `trajectory_tracking.png`. The video replays the recorded MuJoCo
root and 18 joint positions in the hashed common scene. Yellow marks the full
reference and cyan marks executed EE history. Rendering runs after control, in
an isolated MuJoCo 3.3.6 environment, so it cannot affect controller timing.

| method (run) | nominal: samples/end, position/rotation RMSE, progress | push: samples/end, position/rotation RMSE, progress |
| --- | --- | --- |
| `roboduet` (`20260915_095023`) | 1550/timeout, 0.4536 m / 1.7691 rad, 0.0000 | 1550/timeout, 0.4480 m / 1.7609 rad, 0.0000 |
| `roboduet_raw` (`20260915_095046`) | 1550/timeout, 0.4506 m / 1.6635 rad, 0.0150 | 1550/timeout, 0.4210 m / 1.6840 rad, 0.4361 |
| `ma2022` (`20260915_095108`) | 1550/timeout, 0.2931 m / 1.6271 rad, 0.0060 | 1550/timeout, 0.2615 m / 1.5244 rad, 0.0150 |
| `deep_whole_body_control` (`20260915_095129`) | 235/fall, 0.6128 m / 2.0019 rad, 0.5534 | 228/fall, 0.6026 m / 2.1587 rad, 0.5564 |
| `visual_wholebody` (`20260915_095137`) | 16/fall, 0.2680 m / 2.1117 rad, 0.1865 | 16/fall, 0.2513 m / 2.1144 rad, 0.1925 |
| `umi` (`20260915_095144`) | 22/fall, 0.8354 m / 2.4983 rad, 0.1594 | 21/fall, 0.7893 m / 2.4740 rad, 0.1594 |
| `wb_locoman` (`20260915_095150`) | 24/fall, 0.2914 m / 1.9231 rad, 0.0000 | 26/fall, 0.2766 m / 1.9241 rad, 0.0000 |
| `qm_control` (`20260915_095347`) | 1551/timeout, 0.1901 m / 0.8574 rad, 0.6580 | 1551/timeout, 0.1895 m / 0.8599 rad, 0.6581 |

All 16 scenario receipts report `success=false` and
`numerical_fault=false`. Common physical metric coverage is complete. The
early falls and timeouts remain in the result set and denominators. The run is
closed-loop common-plant evidence, but it does not support a task-success or
method-promotion claim.

Two superseded launch roots, `cross_method_mujoco_harder_v2` and
`cross_method_mujoco_harder_final`, retain the EGL and framebuffer failure
attempts. They are not included in the table or aggregate summary.
