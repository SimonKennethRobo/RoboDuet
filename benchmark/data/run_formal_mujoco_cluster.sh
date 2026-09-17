#!/usr/bin/env bash
set -euo pipefail

repo=${ROBODUET_ROOT:-/home/simon-nfs/Projects/WBC/RoboDuet}
wbc_root=${WBC_RL_MPC_ROOT:-/home/simon-nfs/Projects/Simon/wbc_rl_mpc}
python=${FORMAL_PYTHON:-/home/simon-nfs/miniconda3/envs/isaacgym22/bin/python}
ros_install=${WBC_ROS_INSTALL:-$wbc_root/ros2_ws/install_nfs}
qm_install=${QM_CONTROL_ROS_INSTALL:-$wbc_root/baselines/mpc_baseline/qm_control_baseline/install_nfs}
gateway=${FORMAL_GATEWAY:-simon-nfs@corelab-amd-server.lan}
port=${FORMAL_SSH_PORT:-65522}
action=${1:-status}
campaign=${2:-}
hosts=(simon-5090.lan corelab-4080s.lan corelab-5080-1.lan corelab-5080-2.lan actlab-a4.lan actlab-a5.lan actlab-a6.lan)
suite=$repo/benchmark/data/frozen_trajectory_library2/suite/trajectory_suite.json
task=$wbc_root/go2_x5_ocs2/config/robot_lab_rear_r30o_s42_11497/task_sota.info
raw_bundle=/home/simon-nfs/Projects/WBC/RoboDuetRaw/runs/default_go2x5v3_noselfcollision/0905/default_go2x5v3_noselfcollision_151454_seed6444/rl_sar
common_env="WBC_RL_MPC_ROOT='$wbc_root' WBC_ROS_INSTALL='$ros_install' QM_CONTROL_ROS_INSTALL='$qm_install' ROBODUET_RAW_BUNDLE_ROOT='$raw_bundle'"

remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$port" "$gateway" "$1"
}

case "$action" in
  preflight)
    scene=${campaign:-$wbc_root/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml}
    for host in "${hosts[@]}"; do
      remote "ssh -o BatchMode=yes -o ConnectTimeout=10 -p '$port' '$host' \"cd '$repo' && $common_env MUJOCO_GL=egl '$python' -m benchmark.wbc.formal_mujoco_preflight --scene '$scene' --suite '$suite' --ocs2-task '$task' --output-root '$repo/benchmark/results'\"" \
        >"/tmp/formal_preflight_${host}.json" &
    done
    wait
    ;;
  prepare)
    test -n "$campaign"
    remote "ssh -o BatchMode=yes -o ConnectTimeout=10 -p '$port' '${hosts[0]}' \"cd '$repo' && $common_env '$python' -m benchmark.wbc.formal_mujoco --campaign-root '$campaign' --prepare-campaign --run-python '$python' --video-python '$python' --roboduet-ocs2-task-file '$task'\""
    ;;
  launch)
    test -n "$campaign"
    for index in "${!hosts[@]}"; do
      host=${hosts[$index]}
      domain=$((120 + index * 4))
      remote "ssh -o BatchMode=yes -o ConnectTimeout=10 -p '$port' '$host' \"cd '$repo' || exit; mkdir -p '$campaign/launch' || exit; $common_env MUJOCO_GL=egl nohup '$python' -m benchmark.wbc.formal_mujoco --campaign-root '$campaign' --node-name '$host' --shard-count 7 --shard-index '$index' --workers 1 --ros-domain-base '$domain' --record-video --run-python '$python' --video-python '$python' --roboduet-ocs2-task-file '$task' >'$campaign/launch/$host.log' 2>&1 < /dev/null & echo \\\$! >'$campaign/launch/$host.pid'\""
    done
    ;;
  smoke)
    test -n "$campaign"
    task_id=timed-trajectory-34fad3c2c10aaf11
    scene=$campaign/environment/scene.xml
    smoke_root=$campaign/smoke
    smoke_hosts=(simon-5090.lan corelab-5080-1.lan)
    smoke_methods=(roboduet deep_whole_body_control qm_control)
    for index in "${!smoke_hosts[@]}"; do
      host=${smoke_hosts[$index]}
      methods=(roboduet deep_whole_body_control)
      test "$index" -eq 0 || methods=(qm_control)
      commands=""
      method_index=0
      for method in "${methods[@]}"; do
        domain=$((180 + index * 8 + method_index))
        extra=""
        test "$method" != roboduet || extra="--roboduet-ocs2-task-file '$task'"
        commands+="'$python' -m benchmark.wbc.cross_method_cli --method '$method' --suite '$suite' --task-id '$task_id' --scenarios nominal push --scene '$scene' --output '$smoke_root/nodes/$host/$method' --ros-domain-id '$domain' --timeout-s 1800 --ocs2-transport synchronous --qm-mpc-coupling-mode sync --umi-mujoco-profile training_nominal --python '$python' --video-python '$python' --record-video $extra && "
        method_index=$((method_index + 1))
      done
      commands+="true"
      remote "ssh -o BatchMode=yes -o ConnectTimeout=10 -p '$port' '$host' \"cd '$repo' || exit; mkdir -p '$smoke_root/logs' || exit; $common_env MUJOCO_GL=egl nohup bash -lc \\\"$commands\\\" >'$smoke_root/logs/$host.log' 2>&1 < /dev/null & echo \\\$! >'$smoke_root/logs/$host.pid'\""
    done
    ;;
  status)
    test -n "$campaign"
    remote "for state in '$campaign'/nodes/*/formal_state.json; do test -f \"\$state\" || continue; '$python' -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s.get(\"node_name\"),s.get(\"status\"),s.get(\"summary\"))' \"\$state\"; done"
    ;;
  aggregate)
    test -n "$campaign"
    remote "cd '$repo' && $common_env '$python' -m benchmark.wbc.aggregate_formal_mujoco '$campaign'"
    ;;
  *)
    echo "usage: $0 {preflight [scene]|prepare campaign|smoke campaign|launch campaign|status campaign|aggregate campaign}" >&2
    exit 2
    ;;
esac
