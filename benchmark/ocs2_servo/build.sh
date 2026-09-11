#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
reference=${OCS2_WORKSPACE:-/home/simon/Projects/Simon/wbc_rl_mpc/ros2_ws}
# Keep ROS/OCS2 C++ dependencies out of the IsaacGym Python environment.
set +u
source /opt/ros/jazzy/setup.bash
source "$reference/install/setup.bash"
set -u
cmake -S "$here" -B "$here/build" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build "$here/build" -j2
