#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export PYTHONPATH=/home/simon/Projects/Simon/wbc_rl_mpc/RoboDuet:$PYTHONPATH

cd /home/simon/Projects/Simon/wbc_rl_mpc/RoboDuet

/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
	--policy /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/policy/go2_x5/NH_D_s11 \
	--output tmp/experiments/identify_NH_D_s11

# 运行辨识后的 MPC：

/opt/miniconda3/envs/base312/bin/python sysid/run_iq_mpc.py run \
	--root tmp/experiments/identify_robotlab_v2 \
	--controller selected \
	--scenario walking_curve \
	--seconds 12 \
	--viewer \
	--output tmp/experiments/identify_robotlab_v2/mpc_run_01

/opt/miniconda3/envs/isaacgym/bin/python -m benchmark.wbc.mujoco \
	--suite benchmark/data/frozen_trajectory_library/suite/trajectory_suite.json \
	--task-id timed-trajectory-4cb681e1dd1731a7 \
	--rl-sar-root /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar \
	--policy-key robot_lab_rear_r30o_s42_11497 \
	--scene /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml \
	--upper-controller floating_base_ocs2_mpc \
	--ocs2-root /home/simon/Projects/Simon/wbc_rl_mpc \
	--ocs2-task-profile native_ideal \
	--ocs2-task-file /home/simon/Projects/WBC/RoboDuet/tmp/experiments/identify_robotlab_v2/mpc/task_robot_lab_rear_r30o_s42_11497_selected.info \
	--ocs2-command-mode full \
	--ocs2-transport synchronous \
	--ocs2-reference-window-s 1.0 \
	--ocs2-reference-window-dt-s 0.02 \
	--viewer \
	--realtime \
	--output benchmark/results/manual_circle_identified_01
