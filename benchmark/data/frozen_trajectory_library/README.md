# Frozen trajectory library v4

Simulator-free trajectory data for cross-method inspection and later TaskSpec
materialization. This library was built with grid mode `generated_from_roboduet_source` and contains
36 curriculum trajectories, 16 seeded random lines and
16 seeded random circles.

The Rerun gallery places a full-scale static Go2+X5 model beside the start of
every trajectory. The model is compiled from the common benchmark MJCF and
instanced at scale 1.0, so it provides a direct visual reference for distance,
height and workspace scale. A solid ground slab has its top surface at z=0,
with a one-metre grid drawn just above it. The lowest model visual vertex is
placed at z=0 so the robot feet meet the displayed ground.

For the random primitives, the XY projection is exactly a line or a circle.
Height and SO(3) orientation vary smoothly between seeded random control poses
at every path sample. The longest line is 3 m and the
largest circle radius is 3 m. These are geometry references only: a benchmark run must still bind a
canonical physical initial state, anchor, deadline and disturbance schedule to
create a TaskSpec.

Open the complete gallery with the matching Rerun 0.19 viewer:

```bash
/opt/miniconda3/envs/isaacgym/bin/rerun \
  /home/simon/Projects/WBC/RoboDuet/benchmark/data/frozen_trajectory_library/trajectory_library.rrd
```

`trajectories.npz` is the unified padded archive; use `gamma_points` and
`time_law_points` to slice valid samples. `manifest.json` records all seeds,
bounds, generator parameters, per-trajectory hashes and artifact hashes.

Generate the complete library again from RoboDuet source into a new empty directory:

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/isaacgym/bin/python \
  benchmark/data/build_frozen_trajectory_library.py \
  --output benchmark/data/frozen_trajectory_library_v4_new
```

Materialize all 68 geometry references as executable TaskSpecs using one I_Q
get-up/settle state and a common initial EE anchor:

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/isaacgym/bin/python \
  benchmark/data/materialize_trajectory_library.py \
  --library benchmark/data/frozen_trajectory_library \
  --policy-key I_Q \
  --output benchmark/results/iq_frozen_library_suite_$(date +%Y%m%d_%H%M%S)
```

The command refuses to overwrite an existing output. It writes
`trajectory_suite.json` and `reference.npz`; every generated task records the
same physical initial state, maps its first XY and orientation to the measured
canonical EE pose, and preserves the library's absolute ground-relative Z.
The default deadline is trajectory duration plus one second; change it with
`--completion-timeout-s`.

To materialize and run the entire library with I_Q plus synchronous
`native_ideal/full` MPC in one command:

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/isaacgym/bin/python \
  benchmark/data/run_iq_native_ideal_library.py \
  --output benchmark/results/iq_native_ideal_all_$(date +%Y%m%d_%H%M%S)
```

Each task has its own directory and `run.log`. Progress is saved after every
task in `batch_state.json`. To continue an interrupted batch, run the same
command with the same output path and add `--resume`. Use
`--max-tasks 1 --max-steps 2` for a short execution-path smoke test.

Add `--viewer` to open a real-time MuJoCo window while each trajectory runs.
Closing the current window ends that trajectory and the batch proceeds to the
next one; use Ctrl+C in the launching terminal to stop the batch.

The synchronous native MPC receives a forward-moving reference window on every
20 ms control step. The default window is 1.0 s sampled every 0.02 s (51
knots), matching the native_ideal MPC horizon. Configure it with
`--reference-window-s` and `--reference-window-dt-s`. Once the window extends
past the trajectory duration, the remaining knots hold the final EE pose.
This applies to both synchronous and async OCS2 transport. Select async with
`--ocs2-transport async`; its latest-value channel may skip intermediate
state/window messages, and the receipt records command reuse and lag.

Use `--cell A B` to run a specific curriculum trajectory, for example
`--cell 0 1`. Repeat the option to run several cells, such as
`--cell 0 1 --cell 2 3`. Without `--cell`, all curriculum and random primitive
trajectories are run.

Select an OCS2 task configuration with `--mpc-info /absolute/path/task.info`.
When omitted, the bridge uses `go2_x5_ocs2/config/task_floating.info`. The
`native_ideal` profile copies the selected file unchanged into the run output;
the source and runtime hashes are recorded in every receipt.
