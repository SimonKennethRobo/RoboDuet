# Frozen trajectory library v1

Simulator-free trajectory data for cross-method inspection and later TaskSpec
materialization. The library contains the exact 36-trajectory curriculum grid
previously stored under `benchmark/results/trajectory_samples`, plus 16 seeded
random lines and 16 seeded random circles.

For the random primitives, the XY projection is exactly a line or a circle.
Height and SO(3) orientation vary smoothly between seeded random control poses
at every path sample. The longest line is 5 m and the largest circle radius is
5 m. These are geometry references only: a benchmark run must still bind a
canonical physical initial state, anchor, deadline and disturbance schedule to
create a TaskSpec.

Open the complete gallery with the matching Rerun 0.19 viewer:

```bash
/opt/miniconda3/envs/isaacgym/bin/rerun \
  /home/simon/Projects/WBC/RoboDuet/benchmark/data/frozen_trajectory_library_v1/trajectory_library.rrd
```

`trajectories.npz` is the unified padded archive; use `gamma_points` and
`time_law_points` to slice valid samples. `source_grid/` preserves the original
grid manifest and NPZ byte-for-byte. `manifest.json` records all seeds, bounds,
per-trajectory hashes and artifact hashes.

Rebuild into a new empty directory:

```bash
cd /home/simon/Projects/WBC/RoboDuet
/opt/miniconda3/envs/isaacgym/bin/python scripts/build_frozen_trajectory_library.py \
  --output /tmp/frozen_trajectory_library_v1
```
