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
