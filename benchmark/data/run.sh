#!/usr/bin/env bash
set -euo pipefail

data_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
current="${data_dir}/frozen_trajectory_library"
action="${1:-view}"

case "${action}" in
	view)
		exec /opt/miniconda3/envs/isaacgym/bin/rerun "${current}/trajectory_library.rrd"
		;;
	verify)
		exec /opt/miniconda3/envs/isaacgym/bin/python \
			"${data_dir}/build_frozen_trajectory_library.py" \
			--output "${current}" --verify-only
		;;
	build)
		stamp="$(date +%Y%m%d_%H%M%S)"
		output="${2:-${data_dir}/frozen_trajectory_library_${stamp}}"
		exec /opt/miniconda3/envs/isaacgym/bin/python \
			"${data_dir}/build_frozen_trajectory_library.py" --output "${output}"
		;;
	*)
		echo "usage: $0 {view|verify|build [output]}" >&2
		exit 2
		;;
esac
