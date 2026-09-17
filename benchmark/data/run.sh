#!/usr/bin/env bash
set -euo pipefail

data_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
current="${data_dir}/frozen_trajectory_library"
action="${1:-view}"

case "${action}" in
view)
	exec /opt/miniconda3/envs/isaacgym/bin/rerun RoboDuet/benchmark/data/frozen_trajectory_library2/trajectory_library.rrd
	;;
verify)
	exec /opt/miniconda3/envs/isaacgym/bin/python \
		"${data_dir}/build_frozen_trajectory_library.py" \
		--output "${current}" --verify-only
	;;
build)
	stamp="$(date +%Y%m%d_%H%M%S)"
	output="${2:-${data_dir}/frozen_trajectory_library_${stamp}}"

	/opt/miniconda3/envs/isaacgym/bin/python \
		"${data_dir}/build_frozen_trajectory_library.py" --output "${output}"
	/opt/miniconda3/envs/isaacgym/bin/python benchmark/data/materialize_trajectory_library.py \
		--library benchmark/data/$output \
		--policy-key I_Q \
		--completion-timeout-s 1.0 \
		--output benchmark/data/$output/suite

	;;
rollout)

	/opt/miniconda3/envs/isaacgym/bin/python \
		benchmark/data/run_iq_native_ideal_library.py \
		--output benchmark/results/iq_native_ideal_all
	# --max-tasks 1 \
	# --max-steps 2
	;;
*)
	echo "usage: $0 {view|verify|build [output]|materialize}" >&2
	exit 2
	;;
esac
