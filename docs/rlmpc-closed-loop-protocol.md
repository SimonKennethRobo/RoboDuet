# Measured closed-loop response-model MPC screening

This experiment uses the actual RL-MPC_2 dog actor in IsaacGym physics. It does
not forward-simulate an MPC response model as the plant. No real robot, ROS
bridge or existing OCS2/MuJoCo deployment files are modified or started.

## Controlled comparison

- One fixed checkpoint: stage1_rlmpc_benchmark_2_223425, last dog weights.
- Two models: instantaneous unit-gain command tracking, versus identified
  first-order gain/bias/time-constant/delay for vx, vy, yaw rate, height, pitch.
- Identified parameters come only from the completed grouped-training chirps,
  not from closed-loop evaluation trajectories or future measurements.
- Same 1.0 s horizon, 0.1 s MPC period, SLSQP QP solver, costs, bounds and slew
  limits in both arms. EE kinematics are linearized from measured world-frame
  Jacobians at each solve, including the gripper-center offset.
- The same six arm joints are controlled in both arms, using joint-velocity
  decisions integrated into PD position targets. The trained dog observation
  width/history and its prescribed reference observation generator are retained.
- Stage-1 random arm action overrides and kinematic arm locking are disabled;
  explicit MPC arm targets use the canonical wrapper's Stage-1 arm action buffer.
- Planar velocity, height and pitch commands stay inside conservative subsets of
  checkpoint limits. Roll command is zero, as trained. Gait parameters remain
  at trained-range midpoints; the MPC does not optimize gait.
- Nominal flat-ground dynamics with training DR disabled. Paired seed and
  identical setup/warmup; actual initial states are logged for each trial.

## Tasks and outputs

Each of reach, line, circle and fixed-EE lateral-push rejection runs for 24 s
with seeds 17, 29 and 43, under both models: 24 trials total. Each starts with
3 s of identical stance/arm warmup. Target poses are anchored to the measured
post-warmup EE pose. Push rejection uses alternating 20 N lateral trunk forces
for 0.25 s at 6, 12 and 18 s, at identical simulation times in both arms.

Record position/orientation errors, full measured state, targets, commands,
arm position targets, environment resets and every QP solve status/timing.
An environment reset ends the trial as a failure; truncated RMSE must not be
used to claim improvement. Ten consecutive failed solves also end the trial.
All pairs, including failures, appear in the summary. Completed-pair RMSE
comparisons are explicitly conditional on both controllers completing.

Each trial saves manifest.json, config.json, initial_state.json, trajectory.npz,
solver.json and metrics.json. The pipeline writes summary.json and report.md
after each completed pair and skips already recorded trials on restart.

## Interpretation boundary

This is a local-linear, synchronous measured-physics MPC screening, not the
existing OCS2 nonlinear controller. It does not implement the paper's complete
self-collision or policy-conditioned feasibility envelope. Roll is unactuated;
arm motion uses an integrator prediction rather than an identified arm model.
Terrain/domain generalization and cross-arm-conditioned identification remain
separate experiments. Solver deadline misses are logged; the simulator does not
advance during optimization, so results do not establish real-time deployment.

Do not tune one controller on these tasks while leaving the other fixed.
Evidence for the response model is a paired improvement with acceptable
survival and solver behavior, not a low RMSE obtained by early termination.
