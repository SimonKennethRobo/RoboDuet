# Formal parallel MuJoCo benchmark

The formal entry freezes one deterministic environment for the complete campaign:

- common Go2-X5 MJCF and mesh-resource hashes;
- MuJoCo 2.5 ms physics with `implicitfast` integration;
- a 16 m by 16 m, seed-fixed rolling heightfield (default range -100 to +100 mm),
  with a zero-height centre sample and no discrete rocks;
- a matte sage-stone surface without checker texture, with a restrained blue-grey
  sky and soft low-angle key/fill lighting for publication-ready slope shading;
- explicit ground friction `0.8 0.02 0.01`;
- the existing frozen TaskSpec-v3 initial states, references and scorer;
- nominal and deterministic 80 N, 0.1 s base-COM push scenarios;
- the common rough scene for every method, with one declared exception: UMI keeps
  its `training_nominal` in-memory dynamics profile.

Start all eight methods on all 68 frozen trajectories with four isolated workers:

```bash
./benchmark/data/run_formal_mujoco.sh --workers 4
```

Each worker owns a distinct ROS domain and each method-task pair runs in its own
process group and output directory. `formal_state.json` is updated atomically;
failed jobs remain in the campaign denominator.

The UMI exception changes only its adapter-local leg damping, joint friction loss
and foot slide friction. The generated heightfield, TaskSpec, initial state,
reference, timing, push schedule and scorer remain shared. The exception and its
numeric values are recorded in `environment/environment.json` and each UMI command.

Prepare and inspect the exact environment and 544-job plan without running:

```bash
./benchmark/data/run_formal_mujoco.sh --workers 4 --dry-run
```

Run a bounded parallel end-to-end check:

```bash
./benchmark/data/run_formal_mujoco.sh \
  --methods roboduet ma2022 qm_control \
  --max-tasks 1 --scenarios nominal --workers 3
```

Resume a campaign without repeating completed or failed jobs:

```bash
./benchmark/data/run_formal_mujoco.sh --resume /absolute/path/to/campaign
```

Add `--rerun-failed` to retry retained failed jobs. Use `--record-video` only
when the storage cost of video for every selected scenario is intentional.
Recorded MP4 files show the reference and actual EE paths, but deliberately do
not render a target-base pose. The interactive viewer follows the same default;
pass `--viewer-base-target` directly to `benchmark.wbc.mujoco` only for follower
diagnostics.

## Distributed NFS campaign

The cluster wrapper uses task-based sharding (`task_index % shard_count`) and
keeps all eight methods for one task on the same node. It materializes the
environment once, writes node state only below `nodes/<hostname>/`, and starts
one worker per node:

```bash
campaign=/home/simon-nfs/Projects/WBC/RoboDuet/benchmark/results/\
formal_mujoco_paper_rolling_v3_distributed/<campaign_id>
bash benchmark/data/run_formal_mujoco_cluster.sh prepare "$campaign"
bash benchmark/data/run_formal_mujoco_cluster.sh preflight "$campaign/environment/scene.xml"
bash benchmark/data/run_formal_mujoco_cluster.sh smoke "$campaign"
bash benchmark/data/run_formal_mujoco_cluster.sh launch "$campaign"
bash benchmark/data/run_formal_mujoco_cluster.sh status "$campaign"
bash benchmark/data/run_formal_mujoco_cluster.sh aggregate "$campaign"
```

`aggregate_formal_mujoco` is read-only until all seven shards are present. It
rejects mismatched suite/scene/heightfield/task identities and duplicate,
missing, or unexpected scenario keys before writing the campaign-level
`formal_results.json`. Failed rows remain in the output and in every metric
denominator.
