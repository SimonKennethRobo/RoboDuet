#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python=/opt/miniconda3/envs/isaacgym/bin/python
run_python=${FORMAL_RUN_PYTHON:-$python}
video_python=${FORMAL_VIDEO_PYTHON:-/opt/miniconda3/envs/base312/bin/python}
export PATH=/opt/miniconda3/envs/isaacgym/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/miniconda3/envs/isaacgym/lib
export PYTHONNOUSERSITE=1
cd "$repo"
exec "$python" -m benchmark.wbc.formal_mujoco \
  --run-python "$run_python" \
  --video-python "$video_python" \
  "$@"
