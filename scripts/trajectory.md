# Trajectory Curriculum

Velocity-sampled, length-controlled, **full-pose** end-effector reference
trajectories for the RoboDuet whole-body curriculum.

Two entry points:

| File | What it is |
|---|---|
| `scripts/trajectory_curriculum.py` | **Library.** `TrajectoryCurriculum` — per-env GPU buffers designed to drop into an IsaacGym training loop. |
| `scripts/test_velocity.py` | **Demo / visualizer.** Generates one batch per difficulty, dumps `.pt` + `.png` for inspection. |

Both share `VelocityTrajectorySimulator` (position) and
`pose_traj_gen.AttitudeSimulator` (orientation).

---

## 1. What the generator produces

The simulator first produces a `(K, 7)` sequence of EE poses in the
**arm-base (body-fixed) frame** — the same frame used by
`langevin_field_gen.py` (body box at the origin, ground at
`−base_z = −0.34 m`).  For whole-body RL, `TrajectoryCurriculum` can then
**anchor that sequence in world coordinates at reset** and convert it back to
the robot's current base frame each step.  That is the important part: base
translation/rotation then changes the command error, so the dog can actually
help track outreach targets.

| Field | Shape | Meaning |
|---|---|---|
| `positions` | `[N, K, 3]` | Cartesian EE position `(x, y, z)` (m) |
| `world_positions` | `[N, K, 3]` | World-anchored target positions after `anchor_world` / `resample(..., base_pos, base_quat)` |
| `velocities` | `[N, K, 3]` | EE linear velocity (m / s) — exact: `v = s(k)·dir(k)` |
| `quaternions` | `[N, K, 4]` | EE orientation `(x, y, z, w)` |
| `world_quaternions` | `[N, K, 4]` | World-anchored orientation targets after anchoring |
| `eulers` | `[N, K, 3]` | EE orientation `(roll, pitch, yaw)` (rad), pitch ≤ 60° |
| `abg` | `[N, K, 3]` | `(α, β, γ)` — the env reward's `quat_to_angle` readout |
| `target_len` | `[N]` | commanded path length (m), achieved to ≤ 0.001 mm |
| `stages` | `[N]` | `0 = point`, `1 = velocity`, `2 = outreach` |

`N` is the number of trajectories generated in one call; `K` is the number
of samples per trajectory; `dt` is the time per sample.  In the default config
`K=150, dt=0.02`, matching RoboDuet's 50 Hz control step for a 3 s trajectory.

Both position and attitude start and end at rest.

---

## 2. The curriculum — what `D` means

`D ∈ [0, 1]` is a single scalar that controls **length, curviness, XY span**
(plus the orientation spread).  Stages (set by `velocity.point_max` and
`velocity.outreach_min`):

```
D ≤ point_max            →  POINT     a single held pose
point < D ≤ outreach_min →  VELOCITY  inside the arm workspace
D > outreach_min         →  OUTREACH  roams freely; part of the path is
                                       deliberately out of the arm's reach
                                       or inside the robot body, so the
                                       base must move
```

**XY span grows with D.**  Every trajectory is assigned a fixed
per-trajectory XY drift direction at sample time; the wander is anchored to
*that direction* (rather than to the current heading).  So the heading
oscillates around the drift instead of accumulating away from it, and the
**net XY displacement scales with the path length**:

| D | length | net XY (50 Hz, K=150) | bbox diag |
|---|---|---|---|
| 0.2 (velocity) | 0.78 m | 0.51 m | 0.55 m |
| 0.5 (velocity) | 1.61 m | 0.53 m | 0.71 m |
| 0.7 (velocity) | 2.17 m | 0.53 m | 0.76 m   ← saturated by command cone |
| 0.9 (outreach) | 2.71 m | 1.13 m | 1.38 m   ← free roam |
| 1.0 (outreach) | 2.89 m | 1.12 m | 1.39 m |

In the velocity stage the XY span saturates at the intersection of the
workspace and the L/P/Y command cone (`l ∈ [0.30, 0.77] m`, `p ∈ ±0.45π`,
`y ∈ ±π/2`) — that's the geometric ceiling for a path that must stay
inside the arm's reachable command set.  In the outreach stage the outer
shell is moved to `outreach_max` past the workspace boundary, the command
cone is released, and the finite-grid cage keeps all samples inside the
precomputed SDF volume.  When the trajectory is world-anchored, the path
can sweep a large world-space arc and the **dog's base must translate/turn
to bring the current target back into arm reach**.

Without the drift anchor a long path folds in on itself (net XY ≈ 0.3 m
even at L = 3 m), and a fixed-base arm can track the entire EE target from
one spot — defeating the whole-body objective.

#### Length

```
t          = clamp((D − point_max) / (1 − point_max), 0, 1)
L_center   = lerp(length_min, length_max, t)
L          = L_center · (1 + uniform[−jitter, +jitter])    # per trajectory
```

L is **exact** — the simulator computes
`v_cruise = L / (Σ shape · dt)` and integrates `v = shape(k) · v_cruise`, so
the achieved length matches the commanded length to numerical precision
regardless of how much the avoidance turns the direction.

#### Curviness

A bounded direction-wander whose magnitude scales steeply with `D`:

```
turn_sigma = D^wander_exp · disturb       # wander_exp = 2 (default)
```

So:

| D | length | wander | result |
|---|---|---|---|
| 0.0–0.1 | — | — | a single held point |
| 0.2 | ≈ 0.5 m | ≈ 4 % of full | short, almost straight |
| 0.5 | ≈ 1.4 m | 25 % | medium, gentle curve |
| 0.8 | ≈ 2.4 m | 64 % | long, curvy |
| 1.0 | ≈ 3.0 m | 100 % | long, very curvy + outreach |

`wander_exp = 2` is what makes low D *genuinely* straight rather than just
"a little less curvy than D=1".

#### Outreach (D > outreach_min)

Outreach trajectories are NOT confined to the hard region — they roam
freely.  Only an **outer** bound (workspace SDF < `outreach_max`) keeps the
path inside base-translation range; a `grid_margin` cage keeps the path inside
the finite SDF/orientation grid; body-collision avoidance is OFF.

To guarantee that part of every outreach trajectory is unfollowable by the
arm alone, each one **starts** in a hard region — out of arm reach, or inside
the robot body — mixed by `body_fraction`.  The path freely crosses in and
out of arm reach / the body from there.

#### Command cone

For the point and velocity stages, the simulator also respects the RoboDuet
arm command cone:

```
l ∈ [0.30, 0.77] m
p ∈ [-0.45π, 0.45π]
y ∈ [-π/2, π/2]
```

This keeps arm-only curriculum targets in the same L/P/Y distribution used by
the existing random command sampler.  Outreach intentionally starts outside
that range or outside reach; after world anchoring, `get_target_lpy_world(...)`
recomputes the current base-relative command, so the target comes back into
range as the base moves.

#### Orientation

`pose_traj_gen.AttitudeSimulator` queries `field_orient` along the position
path and produces a per-position **feasible** orientation:

```
e(t) = smooth_t( e_mean(x(t)) + D · spread_scale · ξ · σ(x(t)) ),  clamped
```

- `D = 0`: every pose tracks the feasible mean at that position.
- `D = 1`: a per-particle, position-dependent offset within `±spread_scale·σ(x)`.

Because the goal is evaluated at the **current** position (not chased with a
2nd-order filter), there is no tracking lag, and the orientation never
drifts off the locally feasible set.

---

## 3. How the position simulator works  (the math, briefly)

A 1st-order kinematic model:

```
x[k+1] = x[k] + v[k] · dt
v[k]   = s[k] · dir[k]
```

- `s[k]` is the trapezoidal speed shape (smoothstep accel → cruise →
  smoothstep decel), pre-scaled so `Σ s · dt = L`.  Length is exact.
- `dir[k]` is built every step by **four stages that compute a target
  direction, then a single bounded rotation** moves `dir` toward it.  Because
  the bounded rotation is the only place `dir` is updated, the per-step
  heading change is *strictly* `≤ max_turn`.

**Stage 1 — drift-anchored OU wander** (the curviness + the XY span):
```
wnoise[k] = (1 − turn_decay) · wnoise[k−1] + turn_sigma · N(0, I)
anchor    = wander_scale · dir_drift  +  (1 − wander_scale) · dir[k−1]
wn_perp   = wnoise[k]  −  (wnoise[k] · anchor) · anchor          # perp(anchor)
d_target  = anchor + wander_scale · wn_perp
```
- `dir_drift` is a fixed, XY-only unit vector sampled once per trajectory.
  Anchoring on it (instead of on `dir[k−1]`) is what stops the wander from
  accumulating into a folded path.
- `wander_scale = (1 − ‖avoidance‖)+` fades the disturbance to zero near a
  barrier so it cannot fight the avoidance.
- Projecting `wnoise` onto `perp(anchor)` is a smoothness guarantee: a raw
  `anchor + wnoise` would *flip* whenever `wnoise` points anti-parallel to
  `anchor` with magnitude > 1.  The perpendicular component can only rotate,
  never flip — `angle(anchor, d_target) = atan(|wander_scale · wn_perp|) < 90°`.

**Stage 2 — bounded barrier rotation** (folded into the wander target):
```
safe     = avoidance / ‖avoidance‖
align    = d_target · safe
s_turn   = clamp(k_avoid · ‖avoidance‖ · (1 − align), 0, max_turn)
d_target = rotate(d_target, toward=safe, by=s_turn)
```

**Stage 3 — direction damping** (the last-line safety net, applied to
`d_target` *not* to `dir[k−1]`):
```
d_target = damp_forbidden(d_target)   # zero outward / inward / off-grid /
                                       #   out-of-cone components, then renorm
```
The damping projects out direction components that would push past a
forbidden boundary (workspace shell, body, ground, SDF-grid cage, L/P/Y
command cone).  Crucially it modifies the *target*, never `dir[k−1]`
directly — so the per-step turn cap in stage 4 still applies and no large
damping rotation can leak into `dir`.

**Stage 4 — single per-step rotation cap** (the smoothness):
```
align = dir[k−1] · d_target
turn  = clamp(acos(align), 0, max_turn)               # ≤ 12° per step
dir   = rotate(dir[k−1], toward=d_target, by=turn)
```
The per-step direction change is *strictly* bounded at `max_turn`.  Verified
numerically: at every D from 0.2 to 1.0, the per-step turn distribution is
`p99 = max = 12°`; the mean turn scales smoothly with D from 1.8° (D=0.2)
to 6.6° (D=1.0).

The avoidance itself is a sum of proximity-ramped fields:

| Force | Onset distance | When |
|---|---|---|
| Workspace shell (SDF < `shell_hi`) | `bound_margin` | always |
| Body collision | `body_margin` | velocity stage only (off for outreach) |
| Ground (z < `ground_z + margin`) | `ground_margin` | always |
| SDF grid cage | `grid_margin` | always |
| L/P/Y command cone | `command_margin_*` | point/velocity only |

`shell_hi = 0` for the velocity stage (stay inside the arm workspace) and
`shell_hi = outreach_max` for the outreach stage (the only constraint is a
distant outer bound).  The stage-3 damping removes direction components
that would move further through any of those boundaries; the speed
magnitude is unchanged, so the path length stays exact (`|err| ≤ 0.001 mm`
across the full D sweep).

---

## 4. The `TrajectoryCurriculum` API

```python
from scripts.trajectory_curriculum import TrajectoryCurriculum
```

#### Construction

```python
curr = TrajectoryCurriculum(
    num_envs    = 4096,                              # one buffer per IsaacGym env
    device      = 'cuda:0',
    config      = 'configs/trajectory_generation.yaml',   # self-contained YAML
    K           = 150,                               # samples per trajectory (None → cfg)
    dt          = 0.02,                              # seconds per sample   (None → cfg)
    fields_path = None,                              # None → cfg.generator.fields_path
)
```

The config is **fully self-contained** — no `base_config` chain, no
`traj_gen.yaml` dependency.  `K`, `dt`, and `fields_path` live in the
top-level `generator:` section of the YAML; constructor kwargs override
them.  `K · dt` is the trajectory duration; default `K=150, dt=0.02` → 3 s.
Pick `dt` to match your env control rate (50 Hz env → `dt=0.02`).

The constructor loads `precomputed_fields.pt`, builds the SDF
`FieldGenerator`, and allocates these GPU-resident buffers:

| Attribute | Shape | Notes |
|---|---|---|
| `positions` | `[N, K, 3]` | EE Cartesian (arm-base frame) |
| `world_positions` | `[N, K, 3]` | World target after anchoring |
| `quaternions` | `[N, K, 4]` | EE `(x, y, z, w)`; identity at init |
| `world_quaternions` | `[N, K, 4]` | World orientation target after anchoring |
| `velocities` | `[N, K, 3]` | EE linear velocity |
| `eulers` | `[N, K, 3]` | EE `(roll, pitch, yaw)` |
| `abg` | `[N, K, 3]` | env reward `(α, β, γ)` |
| `target_len` | `[N]` | commanded path length |
| `stages` | `[N]` long | `0` point · `1` velocity · `2` outreach |
| `cursor` | `[N]` long | per-env time index in `[0, K)` |

#### Per-reset entry point

```python
curr.resample(env_ids, D=0.5)
```

Generates fresh trajectories at scalar difficulty `D` for `env_ids` and
writes them into the buffers (cursor reset to 0).  `env_ids=None` means
all envs.  For **per-env D**, group envs by difficulty bucket and call
`resample` once per bucket — that costs one simulator pass per bucket
rather than per env.

For whole-body training, anchor the new trajectory in world coordinates at
reset by passing the current base state:

```python
curr.resample(
    env_ids,
    D=self.curriculum_D,
    base_pos=self.root_states[:, 0:3],
    base_quat=self.base_quat,
    measured_heights=self.measured_heights,
)
```

`base_pos`, `base_quat`, and `measured_heights` may be full env tensors
(`[num_envs, ...]`) or tensors already sliced to `env_ids`.

#### Per-step calls

```python
pos, quat = curr.get_target(env_ids)             # body-fixed reference
lpy, quat = curr.get_target_lpy(env_ids)         # body-fixed L/P/Y

lpy_w, quat_w = curr.get_target_lpy_world(       # world-anchored target,
    env_ids,                                     # converted to current base
    base_pos=self.root_states[:, 0:3],
    base_quat=self.base_quat,
    measured_heights=self.measured_heights,
)

pos, abg  = curr.get_target_abg(env_ids)         # body-fixed reward form
v         = curr.get_velocity(env_ids)           # [B,3]
done      = curr.is_done(env_ids)                # cursor reached K-1

curr.advance(env_ids, by=1)                      # cursor += by, clamped
curr.set_cursor(env_ids, k)                      # k: int or [B] long tensor
```

Use `get_target_lpy(...)` only if you deliberately want a target fixed in the
robot's body frame.  Use `get_target_lpy_world(...)` for the whole-body
curriculum, otherwise moving the base cannot reduce the target error.

#### Whole-trajectory access

```python
pos, quat, L = curr.get_full(env_ids)       # [B,K,3], [B,K,4], [B]
```

#### Coordinate helpers

```python
from scripts.trajectory_curriculum import (
    xyz_to_lpy, lpy_to_xyz,
    base_xyz_to_world, world_xyz_to_base,
)
lpy = xyz_to_lpy(xyz)                       # inverse of LeggedRobot._lpy_to_world_xyz
xyz = lpy_to_xyz(lpy)
target_world = base_xyz_to_world(xyz, base_pos, base_quat, measured_heights)
target_base  = world_xyz_to_base(target_world, base_pos, base_quat, measured_heights)
```

#### Inspection

```python
curr.difficulty_stage(D)                    # 'point' | 'velocity' | 'outreach'
```

---

## 5. IsaacGym integration

Drop-in pattern for `LeggedRobot`:

```python
# ── env __init__ ─────────────────────────────────────────────────────────────
from scripts.trajectory_curriculum import TrajectoryCurriculum

self.traj = TrajectoryCurriculum(
    num_envs = self.num_envs,
    device   = self.device,
    K        = 150,           # 3 s at 50 Hz
    dt       = self.dt,       # env control dt (e.g. 0.02 s)
)
self.curriculum_D = 0.0       # bumped by your curriculum scheduler

# ── env reset(env_ids) ───────────────────────────────────────────────────────
self.traj.resample(
    env_ids,
    D=self.curriculum_D,
    base_pos=self.root_states[:, 0:3],
    base_quat=self.base_quat,
    measured_heights=self.measured_heights,
)
# Assign the initial world-anchored target right away so obs are fresh.
lpy, quat = self.traj.get_target_lpy_world(
    env_ids,
    base_pos=self.root_states[:, 0:3],
    base_quat=self.base_quat,
    measured_heights=self.measured_heights,
)
self.commands_arm[env_ids, 0:3] = lpy
self.obj_quats[env_ids]         = quat
# … then run the same downstream postprocessing as _resample_arm_commands:
# commands_arm_obs, target_abg, rot6d/rpy obs, visual_rpy.

# ── env step() — once per control step ───────────────────────────────────────
self.traj.advance()                                # cursor += 1
lpy, quat = self.traj.get_target_lpy_world(
    base_pos=self.root_states[:, 0:3],
    base_quat=self.base_quat,
    measured_heights=self.measured_heights,
)
self.commands_arm[:, 0:3] = lpy
self.obj_quats[:]         = quat
# refresh the downstream obs / reward targets
self.target_abg = self.quat_to_angle(self.obj_quats)
# resample any envs that ran out of trajectory
done = self.traj.is_done()
if done.any():
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    self.traj.resample(
        ids,
        D=self.curriculum_D,
        base_pos=self.root_states[:, 0:3],
        base_quat=self.base_quat,
        measured_heights=self.measured_heights,
    )
```

#### Coordinate frames

The simulator works in the **arm-base (body-fixed) frame** of
`langevin_field_gen.py`: the body box is at the origin, the ground is at
`z = −base_z` (= −0.34 m).  The env's `commands_arm` is also base-relative,
but whole-body targets must not stay body-relative forever.  Anchor them once
at reset, then call `get_target_lpy_world(...)` each step so current base
motion changes the command.

`base_xyz_to_world` / `world_xyz_to_base` match
`LeggedRobot._lpy_to_world_xyz`: XY uses base yaw, and Z uses
`mean(measured_heights) + base_height` (`base_height=0.38` by default).

Reward angle errors should be wrapped to `[-π, π]` before taking absolute
value.  The trajectory branch updates the RoboDuet L/P/Y yaw and ABG reward
errors this way to avoid false large errors at the ±π discontinuity.

#### Per-env D (PLR-style)

```python
for d_value, ids in zip(unique_Ds, env_ids_per_D):
    curr.resample(ids, D=d_value)
```

Costs one generation per unique difficulty — typically 1–5 buckets, not
`num_envs` calls.

#### Resample cost

A single `resample(env_ids, D)` takes roughly 150–250 ms regardless of how
many envs are passed (the heavy work is per-difficulty SDF setup; the
batched simulator scales well to ~5 k envs at once).  At RoboDuet's 50 Hz
control rate, calling `resample` every step on a large fraction of the
buffer would burn most of the step budget.  Two cheap mitigations:

- **Stagger resets** so `done.sum()` per step is small — RoboDuet already
  does this for several buffers; reuse the same env-id batches.
- **Pool + sample-with-replacement.**  At each D bump, pre-generate a
  pool of trajectories and just copy from the pool on reset.  Only the
  *D bump* event pays the simulator cost.

#### Outreach learning curve

When an outreach trajectory starts inside the body box and the policy
walks the base *forward*, the world-anchored target moves to the *back*
of the dog in base coordinates (`bx = x_world − dx_walk`).  The dog has
to learn that the right escape direction depends on where the target was
anchored — which is non-monotone in the action.  Two ways to ease this:

- Train the *velocity* tiers (D ≤ outreach_min) for long enough that the
  policy reliably tracks in-cone targets before the outreach jump.
- For body-start outreach, bias `dir_drift` *outward* from the body so the
  target moves away from the start point along a single direction — this
  cuts the "walk toward target moves it behind you" inversion.

---

## 6. Configuration

`configs/trajectory_generation.yaml` is the canonical, **self-contained**
config — no `base_config` chain, no `traj_gen.yaml` dependency.  The library
reads everything it needs from this single file (constructor kwargs
override).

```yaml
generator:                # the library reads these
  K:  150                 # samples per trajectory
  dt: 0.02                # seconds per sample (set this = your env control dt)
  base_height: 0.38       # env z offset used by _lpy_to_world_xyz
  verbose: false          # keep RL reset logs quiet
  fields_path: runs/langevin/precomputed_fields.pt
  N:      5000            # visualiser-only — number of trajectories per D
  device: cuda            # visualiser-only — auto-falls-back to cpu

velocity:
  point_max:     0.1      # D ≤ this → point stage
  length_min:    0.5      # m  · path length at the bottom of the velocity stage
  length_max:    3.0      # m  · path length at D = 1
  length_jitter: 0.15     # ± fraction of per-traj length jitter
  ramp_frac:     0.20     # fraction of K used for accel + decel
  workspace:     dynamic  # 'static' or 'dynamic' SDF

  turn_decay: 0.05        # OU decay of the wander noise
  disturb:    0.15        # turn_sigma at D = 1 — 10× the un-anchored value,
                           # since the drift-anchored wander does not accumulate
  wander_exp: 2.0         # turn_sigma = D^this · disturb (≥ 2 ⇒ low D genuinely straight)

  k_avoid:        5.0     # avoidance steering gain
  max_turn_deg:   12.0    # cap on per-step heading change (wander + avoidance)
  bound_margin:   0.22    # m  · workspace-shell onset
  body_margin:    0.12    # m  · body-collision onset
  ground_margin:  0.12    # m  · ground onset
  grid_margin:    0.12    # m  · keep samples inside the finite SDF grid

  command_limits:          # point/velocity stages only
    l: [0.30, 0.77]
    p: [-1.4137167, 1.4137167]
    y: [-1.5707963, 1.5707963]
  command_margin_l:     0.08
  command_margin_angle: 0.15

  outreach_min:  0.7      # D > this → outreach stage
  outreach_max:  0.40     # m  · outer bound (workspace SDF < outreach_max)
  body_fraction: 0.4      # fraction of outreach trajectories starting inside the body

attitude:
  spread_scale: 1.0       # D=1 offset spans ±spread_scale · σ_orient(x)
  smooth_steps: 3.0       # temporal Gaussian σ on the Euler trajectory
```

`K` and `dt` come from the top-level `generator:` section; the constructor
kwargs `K=` / `dt=` / `fields_path=` override.  In RoboDuet, the policy/control
step is `control.decimation * sim.dt = 4 * 0.005 = 0.02 s`, so the canonical
config now uses `dt=0.02`.

---

## 7. Prerequisites

Before running anything, build the SDF / orientation fields once:

```bash
python scripts/langevin_field_gen.py
```

This produces `runs/langevin/precomputed_fields.pt`, which contains the
static and dynamic workspace SDFs, the body-collision SDF, and the per-voxel
reachable-orientation field used by the attitude simulator.

---

## 8. Quick verification

```bash
# Sanity-check the library on a few difficulties:
python scripts/trajectory_curriculum.py --num_envs 64

# Full visualisation per difficulty (writes .pt + .png per D):
python scripts/test_velocity.py

# Interactive 3D view of a single difficulty:
python scripts/test_velocity.py --show --D 0.8
```

Expected behaviour:

- **Length**: `|achieved − target| ≤ 0.001 mm` at every D.
- **Smoothness**: per-step heading change `p99 = max = 12°` at every D
  (strictly capped by the single bounded rotation in step 4 of the
  simulator).  Mean turn scales from ~1.8° at D=0.2 to ~6.6° at D=1.0.
- **Point/velocity command cone**: body-frame L/P/Y targets stay inside
  `l ∈ [0.30, 0.77]`, `p ∈ ±0.45π`, `y ∈ ±π/2` (aside from sub-mm boundary
  noise from the `command_margin_*` ramps).
- **Outreach grid bound**: samples stay inside the finite SDF grid
  (`outside SDF grid` should print ~0 %).
- **XY span** (50 Hz, K=150): grows with D, saturating around 0.5 m by the
  end of the velocity stage (cone+workspace ceiling), jumping to ~1.1 m
  in the outreach stage; sweeps wider once world-anchored and tracked by
  the moving base.
- **Outreach hard content**: every trajectory has ≥ 40 % of its samples
  in the hard region (out of arm reach or inside the body) at D ≥ 0.8,
  mean ≈ 90 %.
