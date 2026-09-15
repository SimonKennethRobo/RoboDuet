# Benchmark data

Frozen, simulator-independent inputs belong here. Generated evaluation output,
traces, videos and logs remain under `benchmark/results/`.

- `frozen_trajectory_library/`: current from-source library with deterministic
  line/circle families, full-scale Go2+X5 references, a z=0 ground slab and a
  one-metre grid.
- `frozen_trajectory_library_v1/`: compatibility snapshot that imported the
  earlier 6x6 gallery.

Build and validation are provided by
`benchmark/data/build_frozen_trajectory_library.py`. Existing files are not
required unless the optional `--import-grid` compatibility mode is used.

Generate a new library in a timestamped directory:

```bash
./benchmark/data/run.sh
```

Or choose the output directory and seeds explicitly:

```bash
/opt/miniconda3/envs/isaacgym/bin/python \
  benchmark/data/build_frozen_trajectory_library.py \
  --output benchmark/data/my_frozen_library \
  --grid-seed 12345 --seed 20260915 \
  --line-count 16 --circle-count 16
```

可视化：

```
./benchmark/data/run.sh view
```

The generator compiles the common Go2+X5 MJCF into five material-group meshes,
logs each mesh once, and uses Rerun instance poses to place the full robot at
all trajectory starts. This keeps the recorded geometry at physical scale 1.0
without duplicating the mesh payload for every trajectory.

Validate the current frozen library without opening the viewer:

```bash
./benchmark/data/run.sh verify
```
