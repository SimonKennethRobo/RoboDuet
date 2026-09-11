# Identification and P0 evaluation progress

## Completed

All ten original 120 s x 256-environment datasets are complete. The exploratory
chirp/Bode analysis originally failed on an empty moving-phase design matrix;
that case now reports an unavailable residual fit and retains the base model.
The analysis has completed for all datasets.

The grouped holdout predictor is in
`data/identification/prediction_horizons/report.md` and `summary.json`.
Version: `grouped-horizon-v2-body-command-contract`.

- Forty of fifty policy/channel combinations have enough independent training
  chirps and held-out groups. Ten are explicitly marked insufficient.
- Nominal-twin groups stay entirely on one side of the train/test split.
- Fits estimate gain, bias, bandwidth and effective delay per policy/channel.
- Forecasts at 0.2, 0.5 and 1.0 s use no future measured state, phase or speed.
- Velocity commands and measured responses retain the native body-frame
  contract. Legacy pitch commands are converted to physical positive RPY.
- This is five-channel validation, not full SE(3) or MPC closed-loop validation.

Example 1.0 s held-out RMSE, evaluated within each dataset:

| Run | Channel | Ideal | First order | Second + gait |
|---|---|---:|---:|---:|
| RL-MPC 2 | vx (m/s) | 0.12941 | 0.09103 | 0.09330 |
| RL-MPC 2 | height (m) | 0.04471 | 0.01837 | 0.01892 |
| RL-MPC 2 | pitch (rad) | 0.16500 | 0.06434 | 0.06452 |
| Robust 1 | vx (m/s) | 0.17679 | 0.12936 | 0.12906 |
| Robust 1 | height (m) | 0.12628 | 0.02262 | 0.02409 |
| Robust 1 | pitch (rad) | 0.12078 | 0.08029 | 0.08033 |

These examples support modeling each policy instead of assuming instantaneous
response. They do not establish a causal advantage of response-consistent
training: domain recipes, command distributions and training configurations
differ. Small model error must also be read with fitted gain and task execution;
a nearly unresponsive policy can be easy to predict.

## Running

`rlmpc-p0-benchmark.service` reruns the benchmark under a new output root,
`benchmark/results/rlmpc_p0_corrected`, with per-scenario resume:

- reference bandwidth/rate and observation switches included in compatibility;
- physical positive-RPY targets with negative legacy policy command conversion;
- projected-gravity metrics using the same physical positive convention;
- roll held at the common supported zero;
- configuration dispersion distinguished from repeated-seed uncertainty.

Old results are preserved with `VALIDITY_NOTICE.md`; do not mix the old and new
body-pose aggregates because their command support differs.

## Queued

`rlmpc-balanced-validation.service` waits for the corrected benchmark, then
collects ten supplemental datasets under `data/identification/balanced_validation`:

- 120 s, 256 environments, seed 17;
- 75 percent identification environments, equal channel weights;
- PRBS/chirp/ramp weights 0.2/0.6/0.2;
- zero roll command and complete command-vector recording;
- automatic grouped holdout prediction after collection.

This addresses sparse chirp coverage and missing roll-command records. Domain
recipes still come from each checkpoint; it is not a matched-domain reward
ablation. Data collection prints progress every 500 steps.

## Validation performed

Analytic checks passed for the exact second-order ZOH recurrence and first-order
finite-horizon prediction. Changing future measured phase/speed did not change
predictions. Checkpoint reference grouping resolves to [0,2] and [1,3]. Legacy
command conversion and physical positive-RPY metric checks passed. A 16-env,
100-step IsaacGym export confirmed all command columns, zero commanded roll and
75 percent identification allocation. Corrected benchmark scenario A completed;
remaining long-running stages have their own status files and service logs.

## Still required before paper claims

Matched physical commands and domain realizations, a same-configuration
reward-only training ablation, full SE(3) open-loop prediction, and sustained
walking EE tracking with independently measured progress and failures. No fitted
parameters have been automatically installed into the MPC configuration.
