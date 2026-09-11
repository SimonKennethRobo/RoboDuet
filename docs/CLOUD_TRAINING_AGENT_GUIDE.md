# RoboDuet cloud training guide for coding agents

This is the common operating guide for assigning and submitting training jobs to the two configured cloud servers. It describes the existing launch scripts and a cooperative GPU reservation procedure. Read the assigned task and the repository's AGENTS.md before acting.

## 1. Scope and host inventory

| Host alias used in this document | SSH command | GPUs | Approximate CPU allocation | Data disk capacity |
|---|---|---|---|---|
| cloud-1 | `ssh -p 30071 root@183.147.142.40` | 1 × RTX 4090, device 0 | 14 cores | 49 GB |
| cloud-3 | `ssh -p 30322 root@183.147.142.40` | 3 × RTX 4090, devices 0–2 | 42 cores | 98 GB |

Inventory was checked on 2026-09-11. Ports and allocations can change when cloud instances are replaced; verify them on connection. Hostnames may be identical. Identify a server by its SSH endpoint, not its hostname.

Both hosts have Ubuntu 24.04, CUDA Toolkit 12.8, driver 580.126.09, Python 3.8.20, pip Torch 2.3.1+cu121, NumPy 1.23.5, IsaacGym Preview 4, and PyTorch3D 0.7.7 with transforms available and without its native extensions. Torch's CUDA 12.1 runtime and the system CUDA 12.8 toolkit are separate components.

Stage 1 was validated on both hosts with 4096 environments and 20 PPO iterations. Checkpoint updates, finite parameters, and TorchScript loading passed. These short runs establish that training works; they do not establish convergence. Stage 2 has not been validated. Default to Stage 1 unless the assigned task explicitly specifies otherwise.

One training process uses one GPU. This deployment does not distribute one experiment across all three GPUs. Separate experiments may use separate GPUs.

## 2. Assignment contract

Before submission, record the following in an experiment note. Use the user's existing instructions; do not ask again for details already specified.

```text
Task ID / unique job name:
Assigning user or coordinator:
Experiment objective and acceptance criteria:
Source commit or source snapshot checksum:
Configuration changes and checkpoint inputs, if any:
Stage / robot / seed / environment count / iteration budget:
Server SSH port / physical GPU index:
Remote experiment source directory:
W&B mode:
Requested action: submit and hand off / monitor to completion
Owner responsible for monitoring and releasing the GPU reservation:
```

A request to configure an environment does not imply an unbounded production training run. A request to submit a specified experiment authorizes preparing its source, submitting it, and checking startup. Continue to completion when that is part of the assignment. Do not silently change rewards, curriculum, robot, seed, iteration budget, or environment count to make a failing experiment pass.

Use unique names, for example `s1_baseline_seed42_20260911_120000`. Allowed characters are letters, digits, underscores, and hyphens. Do not reuse an old name: the script rejects a currently active duplicate, but does not prohibit reuse after completion.

## 3. Shared paths and execution identity

| Purpose | Path on either server |
|---|---|
| Launch scripts and environment bundle | `/root/gpufree-data/roboduet-conda` |
| Default external source | `/root/gpufree-data/RoboDuet` |
| Default source alias | `/home/simon/Projects/WBC/RoboDuet` |
| Conda root | `/home/simon/miniconda3` |
| Conda environment | `/home/simon/miniconda3/envs/isaacgym` |
| IsaacGym SDK | `/home/simon/Apps/Simulator/Isaac/isaacgym` |
| Terminal logs and exit codes | `/root/gpufree-data/roboduet-conda/logs` |
| Generated launch commands | `/root/gpufree-data/roboduet-conda/jobs` |
| Portable environment and SDK archives | `/root/gpufree-data/roboduet-conda/packages` |
| Recommended experiment copies | `/root/gpufree-data/experiments/<job-name>/RoboDuet` |
| Cooperative reservation directories | `/root/gpufree-data/roboduet-conda/allocations/gpu-<index>` |

SSH as root for administration and submission. `run.sh` automatically switches to `simon` (currently UID/GID 1001), activates `isaacgym`, sets library/import/cache paths, and changes into the selected repository. Therefore `~` inside a training shell means `/home/simon`, while `~` in the root SSH shell means `/root`.

```bash
cd /root/gpufree-data/roboduet-conda
./run.sh                         # Interactive zsh as simon; exit returns to root
./run.sh python -m pip check     # Execute a command in the configured environment
```

Run `train.sh` from the root SSH shell so all agents see the same root-owned tmux server. Do not submit it from the interactive simon shell. Environment scripts and SDK are shared by all experiments: routine task submission should not reinstall dependencies or change these shared files.

`run.sh` changes the working directory. Invoke helper scripts outside the repository with absolute paths.

## 4. Inspect resources before allocating

Run on the candidate host:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
ps -eo user,pid,etimes,args | rg '[a]uto_train.py'
tmux list-sessions 2>/dev/null || true
df -h /root/gpufree-data
cat /sys/fs/cgroup/cpu.max
ls -la /root/gpufree-data/roboduet-conda/allocations 2>/dev/null || true
```

GPU availability is live state, not a property of this document. At guide preparation time, GPU 0 on both hosts was occupied; cloud-1 had `rd-ma2022-teacher`. Do not assume those jobs have finished. Some processes can be invisible to container-local process listing; GPU memory/utilization still matters. An absent tmux session does not prove a GPU is free.

Assign at most one training job per GPU by default. The validated 4096-environment workload used about 8.7–8.8 GiB on one 4090, but other configurations can use more. Keep the configured CPU thread default of 4 unless the experiment requires tuning. Check storage for the full run, not just startup: repeated checkpoints can consume substantial space.

Do not interrupt, overwrite, or reclaim another task because you cannot identify its owner. Resolve ownership with the assigning coordinator when needed.

## 5. Prepare a reproducible, isolated source copy

The environment package contains no project source. The initially deployed source on both hosts was commit `0f02d9d476fc2e3e08b4b6cc9e60e19259bf0775`; cloud-3 was populated as a source snapshot without `.git`. Neither host automatically follows the local working tree. Inspect the actual source before selecting it for a new task.

Use a separate source directory per experiment when changing code/configuration. Never edit a source directory while another training process is using it: configuration imports, assets, and later file reads can otherwise mix revisions.

For a committed local revision, the following creates a source snapshot. Run locally from the repository. Set PORT, JOB, and REV to the actual assignment first. This example deliberately exports committed files; uncommitted changes must be captured separately and recorded if the task needs them.

```bash
PORT=30322
JOB=s1_baseline_seed42_20260911_120000
REV=HEAD
ARTIFACT_DIR="$HOME/Downloads/roboduet-jobs/$JOB"
mkdir -p "$ARTIFACT_DIR"
git rev-parse "$REV" > "$ARTIFACT_DIR/source-commit.txt"
git status --short > "$ARTIFACT_DIR/local-status.txt"
# pipefail prevents a failed git archive from being hidden by gzip.
(set -o pipefail; git archive "$REV" | gzip > "$ARTIFACT_DIR/source.tar.gz")
(cd "$ARTIFACT_DIR" && sha256sum source.tar.gz > SHA256SUMS)
ssh -p "$PORT" root@183.147.142.40 "mkdir -p /root/gpufree-data/experiments/$JOB"
scp -P "$PORT" "$ARTIFACT_DIR/"{source.tar.gz,source-commit.txt,local-status.txt,SHA256SUMS} \
  "root@183.147.142.40:/root/gpufree-data/experiments/$JOB/"
```

Then on the selected remote host, set the same JOB:

```bash
JOB=s1_baseline_seed42_20260911_120000
EXPERIMENT=/root/gpufree-data/experiments/$JOB
cd "$EXPERIMENT"
sha256sum -c SHA256SUMS
# Refuse to merge into an existing working copy.
mkdir RoboDuet
tar -xzf source.tar.gz -C RoboDuet
chown -R simon:simon "$EXPERIMENT"
```

Check for required untracked assets, Git LFS objects, submodules, and checkpoint inputs: `git archive` does not automatically materialize all of these. Copy required files explicitly, check their integrity, and record their origin. Keep checkpoint inputs together with their corresponding `parameters.pkl`. After making assigned configuration edits, save the patch or an updated source archive and checksum in the experiment directory.

Training writes `runs/` and W&B data under the selected source directory. Keep this directory on `/root/gpufree-data`, not the shared transfer filesystem. `/root/gpufree-share` can transfer artifacts between these hosts, but should not hold active training outputs.

## 6. Reserve a GPU and submit

The existing `train.sh` is a launcher, not a scheduler: it does not check GPU occupancy, queue jobs, or reserve devices. The procedure below adds a cooperative reservation using atomic `mkdir`. All agents using this guide must follow it; jobs launched outside this procedure can still occupy a GPU.

Run the following in one root Bash session on the chosen host after preparing the source and checking resources. Replace the sample values with the assigned experiment.

```bash
set -euo pipefail
BUNDLE=/root/gpufree-data/roboduet-conda
JOB=s1_baseline_seed42_20260911_120000
GPU=1                         # cloud-1 only has GPU=0
EXPERIMENT=/root/gpufree-data/experiments/$JOB
REPO=$EXPERIMENT/RoboDuet
OWNER=agent-task-identifier

[[ "$JOB" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$GPU" =~ ^[0-9]+$ ]]
test -d "$REPO"
# Use the server's original device numbering. Do not combine this with masking.
unset CUDA_VISIBLE_DEVICES
mkdir -p "$BUNDLE/allocations"
CLAIM="$BUNDLE/allocations/gpu-$GPU"
if ! mkdir "$CLAIM"; then
    echo "GPU reservation exists: $CLAIM; inspect its owner and job."
    exit 1
fi
printf 'job=%s\nowner=%s\ngpu=%s\nrepo=%s\ncreated=%s\n' \
  "$JOB" "$OWNER" "$GPU" "$REPO" "$(date -Is)" > "$CLAIM/owner.txt"

# Recheck after claiming. If an existing workload is present, do not submit.
nvidia-smi -i "$GPU"
```

Inspect that output, then continue in the same session only when the GPU is available:

```bash
cd "$BUNDLE"
ROBODUET_REPO="$REPO" ROBODUET_CPU_THREADS=4 \
  ./train.sh "$JOB" \
  --train_stage stage1 --robot go2_x5 --dyna_gait \
  --sim_device "cuda:$GPU" --seed 42 \
  --num_envs 4096 --num_learning_iterations 100000 \
  | tee "$CLAIM/submission.txt"
```

This is a template, not a prescribed experiment budget. `--train_stage stage1` is explicit because the underlying entrypoint defaults to `two_stage`. Do not add `--debug` to a production experiment: it changes environment count, video/logging behavior, and schedule. Do not assume `--resume` resumes the latest checkpoint; inspect the selected code's checkpoint-loading contract for any assigned resume task.

Use `--sim_device cuda:0`, `cuda:1`, or `cuda:2` with an unmasked CUDA device list. Avoid mixing these physical indices with `CUDA_VISIBLE_DEVICES`, which changes logical numbering. On cloud-1 only device 0 exists. Multi-GPU selection is an independent-job mechanism, not distributed PPO.

The reservation persists after the submitting shell exits. If submission errors or SSH disconnects, inspect tmux, the generated job, and the log before retrying; the task may already have launched. Do not immediately remove the reservation or submit a duplicate. There is no automatic stale-claim timeout. If you abandon a claim before submitting, release only your own claim after confirming no job was launched.

### What the launcher actually does

- Creates root tmux session `rd-<JOB>` and returns immediately.
- Runs training as simon, headless, with W&B offline by default.
- Writes `jobs/<JOB>-<timestamp>.sh`, the exact shell-quoted launch command.
- Writes `logs/<JOB>-<timestamp>.log` and, on normal wrapper exit, the matching `.exit` file.
- Captures `ROBODUET_REPO`, `ROBODUET_CPU_THREADS`, and any explicitly set `CUDA_VISIBLE_DEVICES` in the generated job script.
- Passes supported W&B environment variables to tmux without writing API keys into the generated script.

A successful `train.sh` return means the background session was submitted. It does not mean imports, simulation, or training succeeded.

## 7. Verify startup, monitor, and complete

Use the exact paths printed in `submission.txt`. Replace the example timestamp below.

```bash
LOG=/root/gpufree-data/roboduet-conda/logs/JOB-YYYYMMDD-HHMMSS.log
STATUS=${LOG%.log}.exit
tail -n 80 "$LOG"
rg 'Logging to|Physics Device|GPU Pipeline|Learning iteration|Total timesteps|Traceback|Error' "$LOG"
nvidia-smi
if test -f "$STATUS"; then cat "$STATUS"; fi
```

Confirm the selected device, expected stage/configuration, and advancing PPO iterations. First execution can compile the IsaacGym Torch extension. Allow startup to finish; use log activity and process state rather than a fixed short timeout. Record the actual `Logging to ...` path. It may be `runs/dummy-...` with `--no_wandb`; never derive the run path from the job name alone.

For interactive monitoring:

```bash
tmux attach -t rd-JOB
# Detach: Ctrl+B, then D. Training continues.
```

Exit-state interpretation:

| Observation | Meaning / action |
|---|---|
| No `.exit`, session/process active, iterations advance | Running |
| `.exit` contains 0 | Wrapper completed successfully; verify requested iteration count and artifacts |
| `.exit` contains a nonzero value | Failed or interrupted; inspect the log before choosing a fix or retry |
| No `.exit`, no session/process | Unknown/abnormal termination; check host restart, logs, disk and GPU before reporting failure or resubmitting |
| GPU memory remains allocated without your session | Investigate ownership; do not assume the device is free |

On completion, verify the actual run contains `parameters.pkl`, the dog checkpoint directory, and the expected final checkpoint. For this deployed dual-policy runner the usual final checkpoint is `checkpoints_dog/ac_weights_last_dog.pt`, with export `deploy_model/body_latest_dog.jit`. Confirm these against the selected source if changing runners. When validating training correctness, load trusted checkpoints in the configured environment, check finite floating parameters and actual updates, and check the requested iteration count. A checkpoint file existing by itself is not evidence of convergence.

Keep checkpoint and configuration snapshots together when copying results. Copy a completed checkpoint or wait for the run to finish; do not treat a partially written file as a valid backup.

Release only your own reservation after the process has stopped and its final state has been recorded:

```bash
# CLAIM and JOB must still refer to this task's reservation.
grep -Fx "job=$JOB" "$CLAIM/owner.txt"
# First copy the owner/submission records into the experiment directory.
cp "$CLAIM/owner.txt" "$EXPERIMENT/allocation.txt"
if test -f "$CLAIM/submission.txt"; then
    cp "$CLAIM/submission.txt" "$EXPERIMENT/submission.txt"
fi
# After verifying ownership and no remaining task process:
rm -- "$CLAIM/owner.txt"
if test -f "$CLAIM/submission.txt"; then rm -- "$CLAIM/submission.txt"; fi
rmdir "$CLAIM"
```

For a handoff while training is still running, keep the reservation and transfer monitoring responsibility to the next agent. Ensure that agent has the endpoint, job name, paths, and release procedure.

## 8. Stop or troubleshoot an assigned job

To stop your assigned job, attach to its exact tmux session and press Ctrl+C. For noninteractive operation, `tmux send-keys -t "rd-$JOB" C-c` sends the same interrupt; verify the target first. Confirm termination and inspect the exit file afterward. Interruption does not guarantee an additional checkpoint save. Avoid `kill -9`, killing the tmux server, or broad process-name kills.

| Symptom | First checks |
|---|---|
| Import or library error | Invoke through `run.sh`; check actual Python path, SDK path, and `pip check` |
| Permission denied | Experiment source/output must be accessible to simon; check ownership and parent traversal |
| CUDA out of memory | Verify selected GPU, existing workloads, and the assigned environment count; do not change the experiment silently |
| No new log lines | Check process state, first-run extension compilation, GPU activity, and disk before retrying |
| W&B login/network failure | Default offline mode needs no account; inspect whether the task explicitly enabled online mode |
| Checkpoint/config mismatch | Check the selected source revision and checkpoint's `parameters.pkl`; do not bypass shape checks |
| I/O error or full disk | Inspect the actual run filesystem and free space; preserve checkpoints and diagnose before deleting anything |

W&B online mode is opt-in through `ROBODUET_WANDB_MODE=online` with an authorized account. The deployed baseline uses entity `simon00715`; inspect the selected source when changing accounts. Do not put credentials in experiment notes, shell history examples, or source archives. `--no_wandb` disables W&B entirely. Do not rebuild the environment to fix an ordinary source/configuration error.

## 9. Required handoff report

Report concrete state, not just the submitted command:

```text
Job / owner:
Server: root@183.147.142.40, port ...
GPU index / UUID:
Source path / revision or checksum / configuration changes:
Exact command or generated jobs/*.sh path:
State: submitted / startup verified / running / completed / failed / interrupted
Latest observed iteration and observation time:
Tmux session:
Terminal log / exit-status file and value:
Actual run directory:
Checkpoint + parameters.pkl paths, if available:
Validation performed and limitations:
Reservation path; released or monitoring owner:
```

Submission-only assignments should end after startup is verified and a monitoring owner is identified. Completion assignments require checking the exit state, requested budget, and result artifacts. Do not report a running task as complete or claim a short test demonstrates a trained policy.
