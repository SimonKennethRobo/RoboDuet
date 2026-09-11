#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=${1:?Supply a new output directory}
test ! -e "$root"
mkdir -p "$root/logs"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
failed=0
for task in hold_push reach; do
  for seed in 29 43; do
    for mode in base_only separate coupled; do
      name=${task}_${seed}_${mode}
      printf 'running %s\n' "$name" > "$root/status.txt"
      if ! python -u -m benchmark.dog_policy.ocs2_servo_mpc --mode "$mode" --task "$task" \
          --seed "$seed" --seconds 24 --output "$root/$name" > "$root/logs/$name.log" 2>&1; then
        failed=$((failed+1))
      fi
    done
  done
done
python -m benchmark.dog_policy.report_servo_experiments --closed-loop "$root"
printf 'completed; failed trials=%s\n' "$failed" > "$root/status.txt"
test "$failed" -eq 0
