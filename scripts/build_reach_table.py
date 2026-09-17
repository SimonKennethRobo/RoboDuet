"""Build the M2 direction-dependent reachability table R_max(u) for a robot.

This is the offline half of ``modules/reachability.py``: it supplies the
forward kinematics by brute force in IsaacGym and hands the samples to
``ReachabilityTable.build_2d`` / ``build_4d`` (project-design-v3.md §4.3).

Why IsaacGym rather than an analytic FK: the numbers that matter downstream
(rho, the reachability barrier, v_ff's ideal-shoulder distance) must agree with
what the *env* measures, and the env measures the grasp point as
``rigid_body_state[ee_idx]`` plus ``ee_local_pos``, with the shoulder taken
from the mount joint's URDF transform. Sampling the very same WBCEnv removes
any chance of the table and the runtime disagreeing about where the arm is.

The trunk is welded to the world (``asset.fix_base_link``) and gravity is off,
so a configuration set into the DOF state stays there; each batch is one
sim step, purely to let PhysX refresh the rigid-body/Jacobian/contact tensors.

    # default: 2M samples, 72x36 grid, self-collision + wrist-singularity filtered
    python scripts/build_reach_table.py --robot go2_x5

    # look at what came out before trusting it
    python scripts/build_reach_table.py --robot go2_x5 --inspect_only \
        --out resources/reach_tables/go2_x5_2d.pt
"""

import argparse
import dataclasses
import math
import os

import isaacgym  # noqa: F401  (must import before torch)
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul
import torch

from go1_gym.envs.config import build_roboduet_config
from go1_gym.envs.config.core import RoboDuetRuntimeOptions
from go1_gym.envs.roboduet.wbc_env import WBCEnv
from go1_gym.utils.math_utils import quat_conjugate

MINI_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(MINI_GYM_ROOT_DIR, "resources", "reach_tables")


def quat_rotate_inverse(q, v):
    return quat_apply(quat_conjugate(q), v)


def build_env(args):
    opts = dataclasses.replace(
        RoboDuetRuntimeOptions(num_envs=args.num_envs, robot=args.robot),
        traj_tracking=False,
    )
    cfg = build_roboduet_config(options=opts)
    cfg.env.num_envs = args.num_envs
    # Weld the trunk: we want the arm's reach relative to its own mount, with
    # no base motion and no legs to fall over.
    cfg.asset.fix_base_link = True
    # Park the welded robot far above the ground plane. R_max(u) is meant to be
    # a property of the ARM alone -- if the floor were in range it would clip
    # the downward directions, and the base height/pitch channel (§4.5) whose
    # whole job is to reach low targets would be double-counted against.
    cfg.init_state.pos = [0.0, 0.0, 5.0]
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    # The table describes the nominal robot; per-env mount jitter and actuator
    # randomization would only blur it.
    cfg.domain_rand.randomize_mount_position = False
    cfg.domain_rand.randomize_mount_rotation = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    for profile in (cfg.domain_rand.stage1_arm, cfg.domain_rand.stage2_arm):
        for field in list(vars(profile)) if hasattr(profile, "__dict__") else []:
            if field.startswith("randomize_"):
                setattr(profile, field, False)
    env = WBCEnv(sim_device=args.sim_device, headless=True, cfg=cfg, graphics_device_id=None)

    # Kill gravity. It cannot be done through cfg: LeggedRobot._randomize_gravity
    # writes `gravities + [0, 0, -9.8]` into the sim params during _create_envs,
    # ignoring cfg.sim.gravity entirely. Left on, every configuration sags a
    # little under its own weight between the state write and the read, biasing
    # the whole table downward.
    sim_params = env.gym.get_sim_params(env.sim)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    env.gym.set_sim_params(env.sim, sim_params)
    return env


class GymSampler:
    """Batched FK / Jacobian / self-collision oracle backed by a welded WBCEnv."""

    def __init__(self, env, trunk_margin=0.02):
        self.env = env
        self.n_env = env.num_envs
        self.arm_slice = slice(env.num_actions_loco, env.num_actions_loco + env.num_actions_arm)
        self._last_q = None
        self._checked = False

        # Trunk collision box, in BODY frame, read from the URDF so it tracks
        # the asset. Configurations whose grasp point falls inside it (plus a
        # margin) are rejected -- see fk_fn for why this is done geometrically
        # rather than with PhysX contacts.
        half, center = _trunk_box(env)
        self.trunk_half = (half + trunk_margin).to(env.device)
        self.trunk_center = center.to(env.device)
        print(f"[reach] trunk reject box (body frame): center {self.trunk_center.tolist()}, "
              f"half-extent {self.trunk_half.tolist()}")

        # Shoulder (arm mount) frame in world coordinates. The trunk is welded,
        # so this is constant -- compute it once from the base rigid body and
        # the mount joint's URDF transform, exactly as the env does at runtime.
        mount = env.arm_mount_tfs[:, :3]
        rpy = env.arm_mount_tfs[:, 3:6]
        base_pos = env.base_pos.clone()
        base_quat = env.base_quat.clone()
        self.shoulder_pos = base_pos + quat_apply(base_quat, mount)
        self.shoulder_quat = quat_mul(
            base_quat, quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        )

    @property
    def q_limits(self):
        lo = self.env.dof_pos_limits[self.arm_slice, 0].clone()
        hi = self.env.dof_pos_limits[self.arm_slice, 1].clone()
        return lo, hi

    def _settle(self, q):
        """Write q into the DOF state (and its position target, so the PD has
        nothing to correct) and take one sim step to refresh the tensors."""
        env = self.env
        env.dof_pos[:, self.arm_slice] = q
        env.dof_vel[:, self.arm_slice] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        targets = env.dof_pos.clone()
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.gym.simulate(env.sim)
        if env.device == "cpu":
            env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_jacobian_tensors(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        self._last_q = q

    def fk_fn(self, q):
        """(M, n_arm) joint angles -> grasp-point pose in the SHOULDER frame,
        plus a keep mask rejecting configurations that put the grasp point
        inside the trunk.

        Collision rejection is geometric rather than PhysX-based, deliberately.
        Enabling contacts (create_actor's filter=0) makes every *adjacent* link
        pair report permanent contact -- base<->x5_link2 and x5_link4<->link6
        fire on 100% of configurations -- which swamps the real signal, and it
        also lets the solver shove penetrating configurations away from the
        joint angles we asked for, corrupting the FK. Testing the grasp point
        against the trunk's own URDF collision box has neither problem.

        Scope of the rejection: only the trunk, whose pose relative to the
        SHOULDER frame is fixed, so it is the only obstacle that can honestly
        be baked into an R_max(u) expressed in that frame. The legs move, so
        freezing them into the table at their default stance would be wrong.
        Arm-link-vs-arm-link collisions are not covered.
        """
        assert q.shape[0] == self.n_env, "chunk must equal num_envs"
        env = self.env
        self._settle(q)

        rb = env.rigid_body_state.view(self.n_env, env.num_bodies, 13)
        ee_quat = rb[:, env.ee_idx, 3:7]
        ee_pos = rb[:, env.ee_idx, 0:3] + quat_apply(
            ee_quat, env.ee_local_offset.expand(self.n_env, -1)
        )

        p_sh = quat_rotate_inverse(self.shoulder_quat, ee_pos - self.shoulder_pos)
        R_sh = _quat_to_mat(quat_mul(quat_conjugate(self.shoulder_quat), ee_quat))

        p_body = quat_rotate_inverse(env.base_quat, ee_pos - env.base_pos)
        inside = ((p_body - self.trunk_center).abs() <= self.trunk_half).all(dim=-1)
        keep = ~inside

        if not self._checked:
            self._checked = True
            # The whole table is FK of `q`, so any drift between the
            # configuration we asked for and the one the sim holds would bias
            # every bucket.
            drift = (env.dof_pos[:, self.arm_slice] - q).abs().max()
            print(f"[reach] FK fidelity: max |dof_pos - q| = {drift:.2e} rad")
            if drift > 1e-3:
                raise RuntimeError(f"arm drifts {drift:.3e} rad from the requested configuration")
        return p_sh, R_sh, keep

    def jac_fn(self, q):
        """Rotational-singularity oracle. Reuses the Jacobian already refreshed
        by the fk_fn call for the same q (build_* always calls fk first)."""
        if self._last_q is None or not torch.equal(q, self._last_q):
            self._settle(q)
        J, _ = self.env._arm_jacobian()
        return J


def _trunk_box(env):
    """Half-extent and center of the trunk's box collision, in body frame,
    from the robot's own URDF. Falls back to the Go2 trunk if the asset has no
    plain box on its base link."""
    import xml.etree.ElementTree as ET

    path = env.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
    base_name = env.body_names[0]
    try:
        link = ET.parse(path).getroot().find(f"./link[@name='{base_name}']")
        for col in link.findall("collision"):
            box = col.find("geometry/box")
            if box is None:
                continue
            size = torch.tensor([float(v) for v in box.get("size").split()])
            org = col.find("origin")
            center = torch.tensor(
                [float(v) for v in (org.get("xyz", "0 0 0").split() if org is not None else "0 0 0".split())]
            )
            return size / 2.0, center
    except (OSError, ET.ParseError, AttributeError, ValueError) as exc:
        print(f"[reach] WARNING: could not read trunk collision box ({exc}); using Go2 default")
    return torch.tensor([0.3762, 0.0935, 0.114]) / 2.0, torch.zeros(3)


def _quat_to_mat(q):
    """xyzw quaternion -> (..., 3, 3) rotation matrix."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def inspect(table, reach_radius=None):
    """Print the slices §4.3's test criteria ask to eyeball: does the envelope
    have the arm's actual shape (far forward, near backward, etc.)?"""
    print(f"\ntable {tuple(table.table.shape)} mode={table.mode}")
    for k, v in sorted(table.meta.items()):
        print(f"  {k}: {v}")
    filled = table.table
    print(f"  R_max range: [{filled.min():.3f}, {filled.max():.3f}] m, mean {filled.mean():.3f}")
    if table.occupancy is not None:
        occ = table.occupancy.float()
        print(f"  bucket occupancy: min {int(occ.min())}, median {int(occ.median())}, max {int(occ.max())}")

    print("\n  horizontal slice (elevation 0deg), R_max vs azimuth:")
    for deg in range(-180, 180, 30):
        a = math.radians(deg)
        u = torch.tensor([[math.cos(a), math.sin(a), 0.0]], device=table.device)
        bar = "#" * int(table.query(u).item() * 60)
        print(f"    az {deg:+4d}deg  {table.query(u).item():.3f}  {bar}")

    print("\n  vertical slice (azimuth 0deg = straight ahead), R_max vs elevation:")
    for deg in range(-90, 91, 15):
        e = math.radians(deg)
        u = torch.tensor([[math.cos(e), 0.0, math.sin(e)]], device=table.device)
        bar = "#" * int(table.query(u).item() * 60)
        print(f"    el {deg:+4d}deg  {table.query(u).item():.3f}  {bar}")

    if reach_radius is not None:
        ratio = filled / reach_radius
        print(
            f"\n  vs the scalar sphere it replaces (reach_radius={reach_radius}): "
            f"R_max/{reach_radius} in [{ratio.min():.2f}, {ratio.max():.2f}] "
            f"-- a sphere misestimates reach by up to {max(abs(1-ratio.min()), abs(ratio.max()-1)):.0%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2_x5", choices=["go1", "go2", "go2_x5"])
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=4096, help="FK batch size")
    parser.add_argument("--n_samples", type=int, default=2_000_000)
    parser.add_argument("--mode", type=str, default="2d", choices=["2d", "4d"])
    parser.add_argument("--n_az", type=int, default=72)
    parser.add_argument("--n_el", type=int, default=36)
    parser.add_argument("--n_az_tool", type=int, default=24)
    parser.add_argument("--n_el_tool", type=int, default=12)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    # Measured on go2_x5 over 204800 random configurations:
    #     sigma_min(J_rot):  min 0.058  p0.1 0.089  p1 0.175  p50 0.882
    # so the design doc's 0.05 rejects nothing, and no plausible threshold
    # changes the table either: of the samples that actually SET each bucket's
    # 0.98 quantile, only 0.27% fall below 0.20 and 0.58% below 0.30 (median
    # 0.978 -- the reach-defining configurations are the most rotationally
    # dexterous ones, not the least). The singularity this arm really has at
    # full extension is a LINEAR one, which the 0.98 quantile already handles.
    # Raise this only with evidence; it is not a free tightening.
    parser.add_argument("--sigma_rot_min", type=float, default=0.05,
                        help="drop configs whose rotational Jacobian sigma_min is below this; "
                             "0 disables. Inert on go2_x5 -- see the comment above")
    parser.add_argument("--no_collision_filter", action="store_true",
                        help="skip the trunk-box rejection")
    parser.add_argument("--trunk_margin", type=float, default=0.02,
                        help="inflate the trunk reject box by this many metres")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--inspect_only", type=str, default=None,
                        help="path of an existing table to print instead of building")
    args = parser.parse_args()

    from modules.reachability import ReachabilityTable

    if args.inspect_only:
        table = ReachabilityTable.load(args.inspect_only)
        inspect(table)
        return

    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"{args.robot}_{args.mode}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = build_env(args)
    sampler = GymSampler(env, trunk_margin=args.trunk_margin)
    if args.no_collision_filter:
        raw_fk = sampler.fk_fn
        sampler.fk_fn = lambda q: raw_fk(q)[:2]
    q_lo, q_hi = sampler.q_limits
    print(f"[reach] arm joint limits lo={q_lo.tolist()}\n"
          f"[reach]                  hi={q_hi.tolist()}")

    def progress(done, total, kept):
        if done % (args.num_envs * 20) == 0 or done >= total:
            print(f"[reach] {done}/{total} sampled, {kept} kept ({kept / max(done,1):.1%})", flush=True)

    common = dict(
        fk_fn=sampler.fk_fn, q_lo=q_lo, q_hi=q_hi,
        jac_fn=None if args.sigma_rot_min <= 0 else sampler.jac_fn,
        n_samples=args.n_samples, sigma_rot_min=args.sigma_rot_min,
        quantile=args.quantile, smooth_sigma=args.smooth_sigma,
        chunk=args.num_envs, progress=progress,
        meta={"robot": args.robot, "ee_local_pos": env.ee_local_offset.tolist()},
    )
    if args.mode == "2d":
        table = ReachabilityTable.build_2d(n_az=args.n_az, n_el=args.n_el, **common)
    else:
        table = ReachabilityTable.build_4d(
            n_az_pos=args.n_az, n_el_pos=args.n_el,
            n_az_tool=args.n_az_tool, n_el_tool=args.n_el_tool, **common,
        )

    table.save(out)
    print(f"\n[reach] saved -> {out}")
    inspect(table, reach_radius=float(env.cfg.wbc.goal_reaching.reach_radius))


if __name__ == "__main__":
    main()
