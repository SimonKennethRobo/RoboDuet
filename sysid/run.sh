#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export PYTHONPATH=/home/simon/Projects/Simon/wbc_rl_mpc/RoboDuet:$PYTHONPATH


# 辨识指定policy
cd /home/simon/Projects/Simon/wbc_rl_mpc/RoboDuet
/opt/miniconda3/envs/base312/bin/python sysid/identify_policy.py \
	--policy /home/simon/Projects/Simon/wbc_rl_mpc/rl_sar/policy/go2_x5/robot_lab_gait_w30_s42_10698 \
	--output tmp/experiments/sysid/robot_lab_gait_w30_s42_10698

# 运行辨识后的 MPC：
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

# test multiple policies without identification
/opt/miniconda3/envs/base312/bin/python \
	sysid/run_policy_identification_benchmark.py \
	--policies I_Q NH_D_s11 robot_lab_rear_r30o_s42_11497 coord_I_26499 \
	--library benchmark/data/frozen_trajectory_library2 \
	--seed 20260915 \
	--count 10 \
	--identification-workers 1 \
	--evaluation-workers 4 \
	--output tmp/experiments/policy_identification_library2

# test multiple policies with identification
/opt/miniconda3/envs/base312/bin/python \
	sysid/run_policy_library_benchmark.py \
	--experiments \
	tmp/experiments/sysid/identify_iq_v2 \
	tmp/experiments/sysid/identify_NH_D_s11 \
	tmp/experiments/sysid/identify_robotlab_v2 \
	tmp/experiments/sysid/identify_coord_I_26499 \
	--library benchmark/data/frozen_trajectory_library2 \
	--seed 20260915 \
	--workers 10 \
	--output tmp/experiments/sysid/policy_sysid_library2_comparison
