"""Load a URDF in IsaacGym and drive every DoF from a slider GUI.

Usage (needs the isaacgym conda env on PATH/LD_LIBRARY_PATH, see repo notes):

    python scripts/view_urdf_joints.py
    python scripts/view_urdf_joints.py --urdf resources/robots/go2_x5_v3/urdf/go2_x5.urdf
    python scripts/view_urdf_joints.py --float-base --collapse-fixed-joints

The IsaacGym viewer has no widget toolkit, so the sliders live in a small
Tk window that is pumped from the same loop as the sim step (single threaded,
no locking needed).
"""

import argparse
import math
import os
import tkinter as tk

from isaacgym import gymapi, gymutil  # noqa: F401  (must precede torch import)
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="URDF path, absolute or relative to the repo root")
    p.add_argument("--float-base", action="store_true",
                   help="let the base fall under gravity (default: base is pinned)")
    p.add_argument("--collapse-fixed-joints", action="store_true",
                   help="merge fixed-joint links into their parent")
    p.add_argument("--flip-visual-attachments", action="store_true",
                   help="rotate visual meshes from y-up to z-up (needed by some .obj)")
    p.add_argument("--stiffness", type=float, default=200.0)
    p.add_argument("--damping", type=float, default=20.0)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true",
                   help="no 3D viewer (only useful to check that the asset loads)")
    return p.parse_args()


def make_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False  # tiny scene; CPU pipeline keeps this simple
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.use_gpu = args.sim_device.startswith("cuda")

    device_type, device_id = ("cuda", int(args.sim_device.split(":")[1])) \
        if ":" in args.sim_device else (args.sim_device, 0)
    graphics_device = -1 if args.headless else device_id
    sim = gym.create_sim(device_id, graphics_device, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("failed to create sim")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)
    return sim


def load_robot(gym, sim, args):
    urdf = args.urdf if os.path.isabs(args.urdf) else os.path.join(REPO_ROOT, args.urdf)
    if not os.path.isfile(urdf):
        raise FileNotFoundError(urdf)
    asset_root, asset_file = os.path.dirname(urdf), os.path.basename(urdf)

    opts = gymapi.AssetOptions()
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    opts.fix_base_link = not args.float_base
    opts.collapse_fixed_joints = args.collapse_fixed_joints
    opts.flip_visual_attachments = args.flip_visual_attachments
    opts.disable_gravity = False
    opts.armature = 0.01
    opts.thickness = 0.01
    opts.use_mesh_materials = True
    asset = gym.load_asset(sim, asset_root, asset_file, opts)
    if asset is None:
        raise RuntimeError(f"failed to load {urdf}")

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.55 if not args.float_base else 0.6)
    actor = gym.create_actor(env, asset, pose, "robot", 0, 0)

    props = gym.get_actor_dof_properties(env, actor)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(args.stiffness)
    props["damping"].fill(args.damping)
    gym.set_actor_dof_properties(env, actor, props)
    return env, actor, props


def dof_ranges(props, names):
    """Per-DoF (lo, hi, home) with sane fallbacks for unlimited/continuous joints."""
    out = []
    for i, _ in enumerate(names):
        lo, hi = float(props["lower"][i]), float(props["upper"][i])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -math.pi, math.pi
        out.append((lo, hi, float(np.clip(0.0, lo, hi))))
    return out


class SliderPanel:
    """Tk window with one slider per DoF, pumped manually from the sim loop."""

    def __init__(self, names, ranges, on_reset):
        self.root = tk.Tk()
        self.root.title("Joint control")
        self.alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.vars = []
        n_rows = (len(names) + 1) // 2
        for i, name in enumerate(names):
            lo, hi, home = ranges[i]
            col, row = divmod(i, n_rows)
            frame = tk.Frame(self.root)
            frame.grid(row=row, column=col, sticky="ew", padx=6, pady=1)
            tk.Label(frame, text=name, width=18, anchor="w").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=home)
            tk.Scale(frame, variable=var, from_=lo, to=hi, resolution=0.001,
                     orient=tk.HORIZONTAL, length=260, showvalue=True).pack(side=tk.LEFT)
            tk.Label(frame, text=f"[{lo:+.2f},{hi:+.2f}]", width=14).pack(side=tk.LEFT)
            self.vars.append(var)

        btns = tk.Frame(self.root)
        btns.grid(row=n_rows, column=0, columnspan=2, pady=6)
        tk.Button(btns, text="Reset to home", command=on_reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Quit", command=self._on_close).pack(side=tk.LEFT, padx=4)

    def _on_close(self):
        self.alive = False

    def targets(self):
        return np.array([v.get() for v in self.vars], dtype=np.float32)

    def set_targets(self, values):
        for var, v in zip(self.vars, values):
            var.set(float(v))

    def pump(self):
        if not self.alive:
            return False
        try:
            self.root.update()
        except tk.TclError:  # window destroyed out from under us
            self.alive = False
        return self.alive

    def destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim = make_sim(gym, args)
    env, actor, props = load_robot(gym, sim, args)

    names = gym.get_actor_dof_names(env, actor)
    ranges = dof_ranges(props, names)
    home = np.array([r[2] for r in ranges], dtype=np.float32)
    print(f"{len(names)} DoF: " + ", ".join(names))

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("failed to create viewer")
        gym.viewer_camera_look_at(viewer, None,
                                  gymapi.Vec3(1.6, 1.6, 1.0), gymapi.Vec3(0.0, 0.0, 0.4))

    panel = SliderPanel(names, ranges, on_reset=lambda: panel.set_targets(home))

    state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    state["pos"] = home
    state["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, home)

    try:
        while panel.pump():
            if viewer is not None and gym.query_viewer_has_closed(viewer):
                break
            gym.set_actor_dof_position_targets(env, actor, panel.targets())
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
    finally:
        panel.destroy()
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
