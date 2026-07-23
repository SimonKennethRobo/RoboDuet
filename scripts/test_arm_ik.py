"""Standalone IK test for the ARX arm ALONE, using the arm-only URDF
(resources/robots/arx5p2Go1/urdf/arx5p2.urdf) -- no Go2 body, no WBCEnv, no
dog/arm-mount coupling. This is the simplest possible testbed for the DLS-IK
math (same formula as WBCEnv._solve_arm_dls_ik_step, see wbc_env.py) before
bringing the Go2+ARX combo URDF into the loop.

The arm base_link is welded to the world at the origin (asset_options.
fix_base_link=True), so the Jacobian only has 6 actuated columns (joint1..
joint6) and 8 rows (one non-base link each; the base link itself is dropped
from the fixed-base Jacobian, same convention documented in
WBCEnv._arm_jacobian). The two gripper finger joints (joint7/8, prismatic)
are held at a fixed position target throughout -- they're downstream of the
end-effector link (link6) so they don't affect the IK.

The stock arx5p2.urdf ships with mesh <geometry> filenames baked to an
absolute path from the machine that exported it
(/home/hz02/only_for_test/arm-these-way/resources/robots/arx5p2/meshes/...),
which doesn't exist on this machine. We rewrite those to the real mesh
location (resources/robots/arx5p2Go1/meshes/arx5p2_meshes/) into a scratch
copy at load time rather than touching the checked-in URDF.

Usage:
    python scripts/test_arm_ik.py                  # headless, prints PASS/FAIL per tier
    python scripts/test_arm_ik.py --viewer          # watch it converge
    python scripts/test_arm_ik.py --damping 0.1 --step_gain 0.3 --max_step_rad 0.05
"""

import argparse
import math
import os
import tempfile

import isaacgym  # noqa: F401  (must import before torch)
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_mul
import torch

from go1_gym.utils.math_utils import quat_error_axis_angle

MINI_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_URDF = os.path.join(MINI_GYM_ROOT_DIR, "resources/robots/arx5p2Go1/urdf/arx5p2.urdf")
MESH_DIR = os.path.join(MINI_GYM_ROOT_DIR, "resources/robots/arx5p2Go1/meshes/arx5p2_meshes")
BROKEN_MESH_PREFIX = "/home/hz02/only_for_test/arm-these-way/resources/robots/arx5p2/meshes"

ACTOR_NAME = "arx5p2_arm"
EE_LINK_NAME = "link6"  # last actuated link before the gripper fingers -- see module docstring
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]  # 6 actuated arm dof, matches num_actions_arm
GRIPPER_JOINT_NAMES = [f"joint{i}" for i in range(7, 9)]  # 2 prismatic finger dof, held fixed

# Same numbers as wbc.py's arm.control.{stiffness,damping}_arm (zarx_j1..j6),
# just re-keyed to this URDF's joint1..joint6 names.
ARM_STIFFNESS = {"joint1": 40.0, "joint2": 70.0, "joint3": 70.0, "joint4": 25.0, "joint5": 25.0, "joint6": 25.0}
ARM_DAMPING = {"joint1": 3.0, "joint2": 15.0, "joint3": 15.0, "joint4": 2.0, "joint5": 2.0, "joint6": 2.0}
GRIPPER_STIFFNESS = 50.0
GRIPPER_DAMPING = 20.0

DECIMATION = 4  # physics substeps per control step, matches typical WBCEnv control_freq/sim_freq ratio


def _write_fixed_urdf(scratch_dir):
    with open(SOURCE_URDF, "r") as f:
        text = f.read()
    fixed = text.replace(BROKEN_MESH_PREFIX, MESH_DIR)
    out_path = os.path.join(scratch_dir, "arx5p2_fixed.urdf")
    with open(out_path, "w") as f:
        f.write(fixed)
    return out_path


class ArmIKTestbed:
    def __init__(self, args):
        self.args = args
        self.num_envs = args.num_envs
        self.device = args.sim_device
        self.headless = not args.viewer

        self.gym = gymapi.acquire_gym()
        self._scratch_dir = tempfile.mkdtemp(prefix="arx5p2_ik_test_")
        self._build_sim()
        self._load_asset()
        self._create_envs()
        self._acquire_tensors()

    def _build_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = "cuda" in self.device
        sim_params.physx.use_gpu = "cuda" in self.device
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1

        compute_device_id = int(self.device.split(":")[-1]) if "cuda" in self.device else 0
        graphics_device_id = -1 if self.headless else compute_device_id
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self.viewer = None
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(1.2, 1.2, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _load_asset(self):
        fixed_urdf_path = _write_fixed_urdf(self._scratch_dir)
        asset_root, asset_file = os.path.split(fixed_urdf_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # weld base_link to the world -- see module docstring
        asset_options.collapse_fixed_joints = True
        # This test is about the DLS-IK/Jacobian math, not about gravity-compensation
        # PD tuning for the raw (unmounted) arm -- the combo URDF's arm PD gains are
        # tuned against the real mounted dynamics elsewhere. Guessed gains here are
        # too weak against gravity at some joints and the outer IK loop's per-step
        # target integration then winds up chasing a sag it can never close (verified:
        # the Jacobian direction itself is correct -- see the perturbation check this
        # script's dev history used). Disable gravity so pass/fail reflects the IK
        # controller, not this script's ad-hoc PD numbers; --gravity re-enables it.
        asset_options.disable_gravity = not self.args.gravity
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if self.gym.get_asset_rigid_body_count(self.asset) == 0:
            raise RuntimeError(f"Failed to load arm asset from {fixed_urdf_path}")

        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        dof_names = self.gym.get_asset_dof_names(self.asset)
        self.arm_dof_idx = torch.tensor(
            [dof_names.index(n) for n in ARM_JOINT_NAMES], dtype=torch.long, device=self.device
        )
        self.gripper_dof_idx = torch.tensor(
            [dof_names.index(n) for n in GRIPPER_JOINT_NAMES], dtype=torch.long, device=self.device
        )

        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        for i, name in enumerate(dof_names):
            self.dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name in ARM_STIFFNESS:
                self.dof_props["stiffness"][i] = ARM_STIFFNESS[name]
                self.dof_props["damping"][i] = ARM_DAMPING[name]
            else:
                self.dof_props["stiffness"][i] = GRIPPER_STIFFNESS
                self.dof_props["damping"][i] = GRIPPER_DAMPING

        body_names = self.gym.get_asset_rigid_body_names(self.asset)
        self.ee_body_idx = body_names.index(EE_LINK_NAME)
        print(f"[test_arm_ik] dof_names={dof_names}  body_names={body_names}  ee_link='{EE_LINK_NAME}'")

    def _create_envs(self):
        spacing = 1.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        per_row = max(1, int(self.num_envs**0.5))

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)  # elevate so the arm can't touch the ground plane
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.envs = []
        self.actor_handles = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, per_row)
            actor = self.gym.create_actor(env, self.asset, start_pose, ACTOR_NAME, i, 1, 0)
            self.gym.set_actor_dof_properties(env, actor, self.dof_props)
            self.envs.append(env)
            self.actor_handles.append(actor)

        self.gym.prepare_sim(self.sim)

    def _acquire_tensors(self):
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, ACTOR_NAME)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dof, 2)
        self.dof_pos = self.dof_state[..., 0]
        num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, num_bodies, 13)
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)

        self.dof_pos_target = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dof_pos_target[:, self.gripper_dof_idx] = 0.0

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    @property
    def ee_pos(self):
        return self.rigid_body_state[:, self.ee_body_idx, 0:3]

    @property
    def ee_quat(self):
        return self.rigid_body_state[:, self.ee_body_idx, 3:7]

    def arm_jacobian(self):
        """(num_envs, 6, 6) Jacobian of the EE body w.r.t. joint1..joint6.
        Fixed-base convention (base_link welded out): row = ee_body_idx - 1,
        columns = the 6 arm-joint dof indices. See WBCEnv._arm_jacobian for
        the row/column layout this mirrors."""
        row = self.ee_body_idx - 1
        return self.jacobian[:, row, :, : len(ARM_JOINT_NAMES)]

    def solve_dls_ik_step(self, pos_err, rot_err):
        """One damped-least-squares differential correction -- identical
        formula to WBCEnv._solve_arm_dls_ik_step."""
        J = self.arm_jacobian()
        err = torch.cat((pos_err, rot_err), dim=-1).unsqueeze(-1)  # (N, 6, 1)
        lam2 = self.args.damping**2
        JJt = torch.bmm(J, J.transpose(1, 2)) + lam2 * torch.eye(6, device=self.device).unsqueeze(0)
        delta_q = torch.bmm(J.transpose(1, 2), torch.linalg.solve(JJt, err)).squeeze(-1)  # (N, 6)
        delta_q = delta_q * self.args.step_gain
        norm = delta_q.norm(dim=-1, keepdim=True)
        max_step = self.args.max_step_rad
        return delta_q * torch.clamp(max_step / torch.clamp(norm, min=1e-8), max=1.0)

    def _draw_sphere_and_axes(self, position, quat, sphere_radius, color, axes_scale=0.05):
        """Cyan target / yellow EE overlay in the viewer, env 0 only -- same
        gymutil helpers as LeggedRobot.draw_sphere_and_axes."""
        sphere_geom = gymutil.WireframeSphereGeometry(sphere_radius, 6, 6, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(*position), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_pose = gymapi.Transform()
        axes_pose.r = gymapi.Quat(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())
        axes_geom = gymutil.AxesGeometry(axes_scale, axes_pose)
        gymutil.draw_lines(
            axes_geom, self.gym, self.viewer, self.envs[0], gymapi.Transform(gymapi.Vec3(*position), r=None)
        )

    def draw_overlays(self, target_pos, target_quat):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        self._draw_sphere_and_axes(target_pos[0].tolist(), target_quat[0], 0.02, (0, 1, 1))  # cyan: target
        self._draw_sphere_and_axes(self.ee_pos[0].tolist(), self.ee_quat[0], 0.02, (1, 1, 0))  # yellow: current EE

    def reset(self, q_init=None):
        dof_state = torch.zeros(self.num_envs, self.num_dof, 2, device=self.device)
        if q_init is not None:
            dof_state[:, self.arm_dof_idx, 0] = q_init
        self.dof_pos_target[:] = dof_state[..., 0]
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos_target))
        self._step_physics(1)  # one physics tick so rigid_body_state reflects the reset pose
        self._refresh()

    def _step_physics(self, n):
        for _ in range(n):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

    def control_step(self, target_pos, target_quat):
        self._refresh()
        pos_err = target_pos - self.ee_pos
        rot_err = quat_error_axis_angle(target_quat, self.ee_quat)
        delta_q = self.solve_dls_ik_step(pos_err, rot_err)
        current_arm_q = self.dof_pos[:, self.arm_dof_idx]
        self.dof_pos_target[:, self.arm_dof_idx] = current_arm_q + delta_q
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos_target))
        self.draw_overlays(target_pos, target_quat)
        self._step_physics(DECIMATION)
        self._refresh()
        return pos_err.norm(dim=-1), rot_err.norm(dim=-1)

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def run_tier(bed, name, target_pos, target_quat, steps, pos_threshold, rot_threshold, verbose):
    bed.reset()
    for step in range(steps):
        pos_err_norm, rot_err_norm = bed.control_step(target_pos, target_quat)
        if verbose and step % max(1, steps // 10) == 0:
            print(
                f"    step {step}: pos_err(env0)={pos_err_norm[0].item():.4f} rot_err(env0)={rot_err_norm[0].item():.4f}"
            )

    passed = (pos_err_norm < pos_threshold) & (rot_err_norm < rot_threshold)
    rate = 100 * passed.float().mean().item()
    status = "PASS" if rate == 100.0 else ("PARTIAL" if rate > 0 else "FAIL")
    print(
        f"[{status}] {name}: pos_err mean={pos_err_norm.mean().item():.4f} max={pos_err_norm.max().item():.4f}  "
        f"rot_err mean={rot_err_norm.mean().item():.4f} max={rot_err_norm.max().item():.4f}  pass_rate={rate:.0f}%"
    )
    return status == "PASS"


def sample_random_target(bed, base_pos, base_quat, pos_range, rot_max_rad):
    """One fresh random SE(3) offset per env from (base_pos, base_quat) --
    same idea as franka_reach.py's per-goal resampling (goalPositionRange/
    goalRotationRange) and WBCEnv._resample_arm_target's per-episode polar
    sampler, just parameterized as a cartesian box + axis-angle here since
    this testbed has no torso/base frame to sample "l/p/y" relative to."""
    pos_offset = (torch.rand(bed.num_envs, 3, device=bed.device) * 2 - 1) * pos_range
    target_pos = base_pos + pos_offset

    axis = torch.randn(bed.num_envs, 3, device=bed.device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angle = (torch.rand(bed.num_envs, device=bed.device) * 2 - 1) * rot_max_rad
    half = (angle / 2).unsqueeze(-1)
    delta_quat = torch.cat((axis * torch.sin(half), torch.cos(half)), dim=-1)
    target_quat = quat_mul(delta_quat, base_quat)
    return target_pos, target_quat


def run_resample_tier(
    bed, name, resample_count, hold_steps, pos_range, rot_max_rad, pos_threshold, rot_threshold, verbose
):
    """Tier 4: `resample_count` fresh random targets in a row, each held for
    `hold_steps` control steps (~2s of sim time by default) before the next
    one is drawn -- exercises the IK controller against resample diversity
    the way WBCEnv's mid-episode target resampling does, instead of a single
    fixed target per tier like tiers 1-3."""
    bed.reset()
    bed._refresh()
    base_pos = bed.ee_pos.clone()
    base_quat = bed.ee_quat.clone()

    per_command_pos_err, per_command_rot_err = [], []
    for cmd in range(resample_count):
        target_pos, target_quat = sample_random_target(bed, base_pos, base_quat, pos_range, rot_max_rad)
        for step in range(hold_steps):
            pos_err_norm, rot_err_norm = bed.control_step(target_pos, target_quat)
            if verbose and step % max(1, hold_steps // 4) == 0:
                print(
                    f"    command {cmd}: step {step}: pos_err(env0)={pos_err_norm[0].item():.4f} "
                    f"rot_err(env0)={rot_err_norm[0].item():.4f}"
                )
        per_command_pos_err.append(pos_err_norm)
        per_command_rot_err.append(rot_err_norm)

    all_pos = torch.stack(per_command_pos_err)  # (resample_count, num_envs)
    all_rot = torch.stack(per_command_rot_err)
    passed = (all_pos < pos_threshold) & (all_rot < rot_threshold)
    rate = 100 * passed.float().mean().item()
    status = "PASS" if rate == 100.0 else ("PARTIAL" if rate > 0 else "FAIL")
    print(
        f"[{status}] {name}: {resample_count} commands x {bed.num_envs} envs "
        f"({passed.numel()} trials)  pos_err mean={all_pos.mean().item():.4f} max={all_pos.max().item():.4f}  "
        f"rot_err mean={all_rot.mean().item():.4f} max={all_rot.max().item():.4f}  pass_rate={rate:.0f}%"
    )
    return status == "PASS"


def main(args):
    bed = ArmIKTestbed(args)
    print(
        f"arm-only IK test: damping={args.damping} step_gain={args.step_gain} max_step_rad={args.max_step_rad}\n"
        f"URDF={SOURCE_URDF} (arm alone, base fixed, no Go2/combo)\n"
    )

    results = {}

    # Tier 1: hold current pose -- target = current EE pose exactly. If this
    # fails, the bug is in the error/Jacobian bookkeeping, not target difficulty.
    bed.reset()
    bed._refresh()
    hold_pos = bed.ee_pos.clone()
    hold_quat = bed.ee_quat.clone()
    results["hold_current"] = run_tier(
        bed,
        "tier 1: hold current pose (target=current, trivial)",
        hold_pos,
        hold_quat,
        args.steps_easy,
        args.pos_threshold,
        args.rot_threshold,
        args.verbose,
    )

    # Tier 2: small 5cm/2cm cartesian nudge from the reset pose, orientation unchanged.
    bed.reset()
    bed._refresh()
    small_offset_pos = bed.ee_pos.clone() + torch.tensor([0.05, 0.0, 0.02], device=bed.device)
    small_offset_quat = bed.ee_quat.clone()
    results["small_offset"] = run_tier(
        bed,
        "tier 2: small 5cm/2cm cartesian offset",
        small_offset_pos,
        small_offset_quat,
        args.steps_easy,
        args.pos_threshold,
        args.rot_threshold,
        args.verbose,
    )

    # Tier 3: a larger 15cm/5cm/-10cm offset plus a ~20deg wrist rotation from
    # the reset pose -- stresses the IK harder than tier 2 while staying
    # anchored to this arm's actual default configuration (an absolute
    # far-away point like "0.35m forward" is a poor "moderate" target here:
    # the default pose is folded mostly vertical, ~0.1m forward, so an
    # absolute point can demand a near-full reconfiguration through
    # near-singular intermediate poses that has nothing to do with IK
    # correctness).
    bed.reset()
    bed._refresh()
    offset_rot = torch.zeros(bed.num_envs, 4, device=bed.device)
    theta = 0.35  # ~20deg
    offset_rot[:, 1] = torch.sin(torch.tensor(theta / 2.0))
    offset_rot[:, 3] = torch.cos(torch.tensor(theta / 2.0))
    moderate_pos = bed.ee_pos.clone() + torch.tensor([0.15, 0.05, -0.10], device=bed.device)
    moderate_quat = quat_mul(offset_rot, bed.ee_quat.clone())
    results["moderate_reach"] = run_tier(
        bed,
        "tier 3: larger 15/5/-10cm offset + ~20deg wrist rotation",
        moderate_pos,
        moderate_quat,
        args.steps_hard,
        args.pos_threshold,
        args.rot_threshold,
        args.verbose,
    )

    # Tier 4: resample_count random SE(3) targets in sequence, each held for
    # ~hold_time_s of sim time -- see run_resample_tier for why this differs
    # from tiers 1-3 (fixed single target) and how it maps to franka_reach.py
    # / WBCEnv's periodic resampling.
    hold_steps = max(1, round(args.hold_time_s * 60.0 / DECIMATION))
    results["resample_diversity"] = run_resample_tier(
        bed,
        f"tier 4: {args.resample_count} resampled targets, {args.hold_time_s}s each",
        args.resample_count,
        hold_steps,
        args.resample_pos_range,
        math.radians(args.resample_rot_max_deg),
        args.pos_threshold,
        args.rot_threshold,
        args.verbose,
    )

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print("\nAll tiers passed -- the DLS-IK controller itself is sound on the arm-only URDF.")
    else:
        print("\nSome tiers failed -- debug the IK controller/gains here before bringing in the Go2+ARX combo.")

    bed.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--viewer", action="store_true", default=False, help="run with a viewer instead of headless")
    parser.add_argument(
        "--gravity",
        action="store_true",
        default=False,
        help="keep gravity on (default off -- see module docstring: this script's PD gains "
        "aren't tuned for the raw arm-only URDF's gravity load, only for isolating the IK math)",
    )
    parser.add_argument("--damping", type=float, default=0.05, help="DLS damping lambda (see wbc.py arm.ik.damping)")
    parser.add_argument("--step_gain", type=float, default=0.5, help="per-step gain (see wbc.py arm.ik.step_gain)")
    parser.add_argument(
        "--max_step_rad", type=float, default=0.08, help="per-step clamp (see wbc.py arm.ik.max_step_rad)"
    )
    parser.add_argument("--steps_easy", type=int, default=100, help="control steps for tiers 1-2")
    parser.add_argument("--steps_hard", type=int, default=250, help="control steps for tier 3")
    parser.add_argument("--resample_count", type=int, default=8, help="tier 4: number of sequential resampled targets")
    parser.add_argument(
        "--hold_time_s",
        type=float,
        default=2.0,
        help="tier 4: sim seconds each resampled target is held before the next",
    )
    parser.add_argument(
        "--resample_pos_range", type=float, default=0.12, help="tier 4: +/- cartesian offset per axis, meters"
    )
    parser.add_argument(
        "--resample_rot_max_deg", type=float, default=25.0, help="tier 4: max random axis-angle rotation, degrees"
    )
    parser.add_argument("--pos_threshold", type=float, default=0.03)
    parser.add_argument("--rot_threshold", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
