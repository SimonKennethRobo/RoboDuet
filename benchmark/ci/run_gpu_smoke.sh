#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CANDIDATE_DIR="${CANDIDATE_DIR:-benchmark/candidates}"
PROFILE="${PROFILE:-benchmark/profiles/smoke.json}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/results}"
CKPTIDS="${CKPTIDS:-last}"
SEED="${SEED:-}"
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cmd=(
  python -m benchmark.cli
  --dog_only
  --candidate_dir "$CANDIDATE_DIR"
  --profile "$PROFILE"
  --sim_device "$SIM_DEVICE"
  --output_dir "$OUTPUT_DIR"
  --headless
)

read -r -a ckptid_args <<< "$CKPTIDS"
cmd+=(--ckptids "${ckptid_args[@]}")

if [[ -n "$SEED" ]]; then
  cmd+=(--seed "$SEED")
fi

if [[ -n "$NUM_EVAL_STEPS" ]]; then
  cmd+=(--num_eval_steps "$NUM_EVAL_STEPS")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $EXTRA_ARGS )
  cmd+=("${extra[@]}")
fi

echo "[Benchmark CI] candidate_dir=$CANDIDATE_DIR"
echo "[Benchmark CI] profile=$PROFILE"
echo "[Benchmark CI] sim_device=$SIM_DEVICE"
echo "[Benchmark CI] output_dir=$OUTPUT_DIR"
echo "[Benchmark CI] command: ${cmd[*]}"

"${cmd[@]}"
