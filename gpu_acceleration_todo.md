# Auto Train GPU Acceleration TODO

This note tracks valuable GPU acceleration work that was intentionally left out of the first low-risk cleanup pass.

## 1. Port command curriculum to torch

Current state:
- `go1_gym/envs/roboduet/curriculum.py` stores bins and samples commands with NumPy.
- `LeggedRobot._resample_commands()` still converts `env_ids` to CPU NumPy for `env_command_bins` and `env_command_categories`.
- `curriculum.update()` reads reward tensors through `.cpu()` / `.cpu().numpy()`.

Why it matters:
- Command resampling happens on reset/resample paths.
- With many environments, the CPU sync can become visible, especially when many envs reset in the same rollout.

Target design:
- Store curriculum bins, command ranges, bin weights, and env bin ids as torch tensors on `self.device`.
- Make `sample(batch_size)` return GPU tensors directly.
- Make `update()` consume GPU reward tensors without converting to NumPy.
- Preserve the existing bin update timing and success-threshold semantics.

Validation:
- Compare CPU and GPU implementations on a fixed command distribution.
- Check sampled command histograms, bin transitions, and zero-command probability.
- Do not expect bitwise seed equivalence after moving RNG from NumPy to torch.

## 2. Reduce observation/history allocation churn

Current state:
- `go1_gym/envs/roboduet/wbc_env.py` rolls observation history with `torch.cat`.
- Several observation builders repeatedly concatenate tensors every step.

Why it matters:
- This is already on GPU, but it allocates and copies large tensors every environment step.
- It can limit FPS after CPU sync points are reduced.

Target design:
- Preallocate history buffers and update them in place.
- Consider a ring buffer if shifting becomes expensive.
- Preallocate final observation tensors and fill slices instead of chaining `torch.cat`.

Validation:
- Confirm observation layout is byte-for-byte equivalent for a short rollout.
- Check policy input dimensions and history ordering.

## 3. Video/debug visualization status

Current state:
- Camera capture, OpenCV overlay, and trajectory projection are CPU-bound.
- This is acceptable for recording, but should be explicit when measuring training FPS.
- `auto_train.py` now exposes recording gates for video, text overlay, trajectory overlay, frame stride, and resolution.
- `LeggedRobot` now uses configured recording resolution, only renders camera sensors on captured frames, and skips overlay work when disabled.
- `Runner.save_io()` now writes with FPS adjusted by `recording_frame_stride` and avoids building an unused WandB video tensor.

Remaining target design:
- Keep overlay implementation CPU-based.
- Decide whether normal non-debug training should default to `--no_record_video` for maximum throughput.
- Optionally add the same recording controls to unified training if that path is still used.

Validation:
- FPS benchmark should report whether recording/overlay is enabled.
- Recorded video should preserve current command and trajectory annotations when enabled.

## 4. Batch remaining training metrics syncs

Current state:
- The first pass moved per-step reward/length buffer syncs to the iteration logging boundary.
- Other logging values can still trigger GPU sync through WandB or console formatting.

Target design:
- Keep rollout and PPO update metrics as tensors until a single logging flush.
- Convert all scalar/tensor metrics to Python floats in one logging helper.
- Keep `debug` mode behavior simple and readable.

Validation:
- Console and WandB values should remain numerically equivalent.
- Training storage, rewards, dones, and PPO losses used for gradients must not change.
