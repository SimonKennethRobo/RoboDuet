# E4 matched response-consistency ablation

## Question and controls

The existing robust and RL-MPC families differ in observation history, reset
recipes, and command/reference configuration. They cannot isolate the causal
effect of auxiliary consistency rewards. This batch tests those rewards while
keeping the reference-response tracking task identical.

| Variant | ref_tracking | phase_variance | steady_gain | domain_consistency |
| --- | ---: | ---: | ---: | ---: |
| reference_only | 2.0 | 0 | 0 | 0 |
| within_domain | 2.0 | -20.0 | -0.5 | 0 |
| full_consistency | 2.0 | -20.0 | -0.5 | -0.5 |

All three use the same immutable source archive from RoboDuet.tmp, seed 17,
Go2-X5, dynamic gait, benchmark DR, Stage 1 from scratch, 4096 environments,
24 rollout steps and 25,000 iterations. There is no Stage-2 training, policy
warm start, video collection, or per-variant architecture change. Checkpoints
are saved every 1000 iterations and at completion.

The current profile's response observations/history, excitation, nominal twins,
reset mixture, independent push curriculum, reference targets, response reward
curriculum and posture-reward handover remain enabled in ALL variants. In this
source snapshot the response stage boundaries are [3000, 6000, 12000], ramp
length 1000, and the handover floor is 0.15. The reference pitch target uses
omega_n=5.0 and rate_limit=0.8. Full response/grouping/profile values are stored
in each experiment_manifest.json and the checkpoint parameters.pkl.

Keeping ref_tracking avoids an invalid response-off control that still fades
the original pitch/height task rewards. This batch therefore tests auxiliary
consistency, NOT the whole reference-shaping package against vanilla robust.
The single paired seed is a mechanism screen, not a multi-seed significance
claim. Divergent policies will not maintain identical random trajectories
after resets even with a shared initial seed; evaluation must pair disturbances.

## Execution and durability

Use scripts/train_response_consistency_ablation.py and
tmp/submit_response_consistency_e4.sbatch. Submit one exclusive GPU allocation
per variant with the 4090 constraint, 10 CPUs, 30 GiB RAM and a 24-hour limit.
Existing jobs are not preempted; jobs may remain pending behind allocated GPUs.

Source and active outputs live in /scratch/$USER/rlmpc_ablation/<job>_<variant>.
Source archives, Slurm logs and periodically mirrored results live under the
batch's remote directory. Existing cluster source and Conda/SDK installations
are not modified. Outputs sync every five minutes and on job exit; scratch is
retained even on failure. Inspect exit_status.txt for both training and sync
status. A mirror made during a checkpoint write may be transient; final sync
is authoritative, and incomplete jobs must use an earlier complete checkpoint.

## Evaluation after training

1. Evaluate all three final policies under the corrected common physical-pose
   command convention and paired benchmark disturbances. Report survival,
   recovery, tracking, height/pitch authority and termination together.
2. Collect the balanced five-channel identification protocol with neutral roll,
   complete command logging and held-out nominal-twin groups. Use at least
   three evaluation seeds, keeping train/evaluation trajectories separate.
3. Compare reference residuals, phase variance, steady gain and cross-domain
   response dispersion. Report active-mask coverage and group desynchronization
   alongside masked consistency metrics, so reduced coverage is not a benefit.
4. Compare held-out 0.2/0.5/1.0-second prediction RMSE for instantaneous,
   first-order, second-order and gait-residual models. Reject apparently good
   prediction obtained by near-zero command gain or frequent termination.
5. Evaluate checkpoints around iterations 3000, 6000, 12000 and the final
   checkpoint to distinguish consistency-ramp effects from robustness recovery.
6. Only after matched improvements survive tracking/robustness checks, expand
   to independent training seeds and policy-aware MPC versus ideal-base MPC
   end-effector tracking. This batch alone cannot establish the MPC claim.
