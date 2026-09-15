"""Official UMI-on-Legs actor adapter for the common Go2+X5 MuJoCo plant."""

from pathlib import Path
import pickle

import mujoco
import numpy as np
import torch


JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
               for part in ("hip", "thigh", "calf")] + [f"x5_joint{i}" for i in range(1, 7)]

# Home-pose transform T_umi_ee_x5_ee, measured from the checkpoint's archived
# go2_arx5_finray_x85_z94 URDF and the common Go2+X5 MJCF at the same 18-DoF
# controller offset. It exposes the original tool coordinates while leaving
# the scored x5_ee frame unchanged.
UMI_TO_X5_HOME = np.asarray([
    [0., -1., 0., 0.00100213],
    [0., 0., -1., -0.01293952],
    [1., 0., 0., -0.10821031],
    [0., 0., 0., 1.],
])


def _quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class UmiMujoco:
    controls_arm = True

    def __init__(self, checkpoint, scene, leg_action_limit=0.0, arm_action_limit=0.0,
                 tool_frame="native_x5"):
        self.config_path = Path(checkpoint).with_name("config.pkl")
        with self.config_path.open("rb") as stream:
            self.training_config = pickle.load(stream)["env"]
        cfg = self.training_config
        control = cfg["controller"]
        task = cfg["tasks"]["reaching"]
        if cfg["obs_history_len"] != 1 or task["target_relative_to_base"]:
            raise ValueError("unsupported UMI checkpoint observation contract")
        if task.get("position_obs_encoding", "linear") != "linear" or task.get("pose_latency_variability") is not None:
            raise ValueError("unsupported UMI task encoding/latency")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.n = 18
        joints = np.asarray([self.model.joint(name).id for name in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[joints].astype(int)
        self.vadr = self.model.jnt_dofadr[joints].astype(int)
        self.mapping = []
        for joint in joints:
            ids = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)
            if len(ids) != 1:
                raise ValueError(f"expected one actuator for joint id {joint}")
            self.mapping.append(int(ids[0]))
        self.default_dof_pos = np.asarray(control["offset"]["data"])
        self.action_scale = np.asarray(control["scale"]["data"])
        self.action_clip = float(cfg["max_action_value"])
        self.transfer_action_limits = np.asarray(
            [float(leg_action_limit)] * 12 + [float(arm_action_limit)] * 6)
        if (self.transfer_action_limits < 0).any():
            raise ValueError("UMI transfer action limits must be nonnegative")
        if tool_frame not in ("native_x5", "arx5_home"):
            raise ValueError(f"unsupported UMI tool frame {tool_frame}")
        self.tool_frame = tool_frame
        self.tool_transform = UMI_TO_X5_HOME.copy()
        limits = np.asarray(control["torque_limit"]["data"], dtype=np.float64)
        self.p = {"decimation": int(control["decimation_count"]),
                  "rl_kp": control["kp"]["data"],
                  "rl_kd": control["kd"]["data"],
                  "torque_limits": limits.tolist()}
        self.control_dt = float(cfg["cfg"]["sim"]["dt"])
        self.policy_dt, self.substeps = self.control_dt * self.p["decimation"], 1
        self.delay_steps = np.rint(np.asarray(cfg["ctrl_delay"]["data"]) / self.control_dt).astype(int)
        self.action_buffer = np.zeros((int(np.ceil(self.delay_steps.max() / self.p["decimation"])) + 1, 18))
        self.target_offsets = task["target_obs_times"]
        self.position_scale, self.orientation_scale = float(task["pos_obs_scale"]), float(task["orn_obs_scale"])
        self.pose_latency_frames = int(np.rint(task["pose_latency"] / task["sequence_sampler"]["dt"])) + 1
        self.pose_history = []
        saved = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
        actor_state = {k[len("actor."):]: v for k, v in saved.items()
                       if k.startswith("actor.")}
        linear_indices = sorted(int(key.split(".")[0]) for key in actor_state
                                if key.endswith(".weight"))
        layers = []
        for layer_number, index in enumerate(linear_indices):
            weight = actor_state[f"{index}.weight"]
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            if layer_number + 1 < len(linear_indices):
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers).eval()
        self.actor.load_state_dict(actor_state, strict=True)
        self.obs_dim = int(actor_state[f"{linear_indices[0]}.weight"].shape[1])
        self.checkpoint_path = Path(checkpoint)
        self.training_asset_path = (
            self.checkpoint_path.resolve().parents[3]
            / "resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf"
        )
        self.actions = np.zeros(18)
        self.history = np.zeros((1, 1))
        self.command = [0.] * 6
        self.gait_indices = 0.
        self.reference = None
        self.reference_time = 0.
        self.base = self.model.body("base_link").id
        self.ee = self.model.site("x5_ee").id

    def reset_policy(self):
        self.actions.fill(0.)
        self.action_buffer.fill(0.)
        self.pose_history = []
        self.after_control_step()

    def set_reference(self, reference, time_s):
        self.reference, self.reference_time = reference, time_s

    def read_state(self):
        q, dq = self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()
        quat = self.data.qpos[[4, 5, 6, 3]].copy()
        rot = self.data.xmat[self.base].reshape(3, 3)
        return q, dq, quat, self.data.qvel[3:6].copy(), self.data.qpos[:3].copy(), rot.T @ self.data.qvel[:3]

    def forward(self, q, dq, quat, gyro, base_pos, lin_vel):
        del base_pos, lin_vel
        if self.reference is None:
            raise RuntimeError("UMI reference was not set")
        base_rot = _quat_matrix(quat)
        # IsaacGymEnv sorts state_obs by key before concatenation.  The
        # checkpoint therefore sees dof_pos, dof_vel, gravity, then gyro --
        # not the YAML declaration order.
        state = np.r_[q - self.default_dof_pos, dq * .05,
                      base_rot.T @ [0., 0., -1.], gyro * .25]
        latency_idx = max(1, self.pose_latency_frames - 1)
        ee_pos, ee_rot = self.pose_history[max(0, len(self.pose_history) - latency_idx)]
        positions, rotations = [], []
        for offset in self.target_offsets:
            _arc, pos, target_q = self.reference.at(self.reference_time + offset)
            relative = np.eye(4)
            relative[:3, :3] = ee_rot.T @ _quat_matrix(target_q)
            relative[:3, 3] = ee_rot.T @ (pos - ee_pos)
            if getattr(self, "tool_frame", "native_x5") == "arx5_home":
                relative = UMI_TO_X5_HOME @ relative @ np.linalg.inv(UMI_TO_X5_HOME)
            # pytorch3d matrix_to_rotation_6d: first two matrix rows.
            positions.extend((relative[:3, 3] * self.position_scale).tolist())
            rotations.extend((relative[:2, :3].reshape(-1) * self.orientation_scale).tolist())
        # ReachingLinkTask.observe flattens positions for ALL horizons first,
        # then ALL rotation_6d blocks; it does not interleave per-horizon poses.
        obs = np.r_[state, positions, rotations, self.actions].astype(np.float32)
        if obs.shape != (getattr(self, "obs_dim", 96),) or not np.isfinite(obs).all():
            raise ValueError(f"invalid UMI observation {obs.shape}")
        with torch.inference_mode():
            self.actions = self.actor(torch.from_numpy(obs[None]))[0].numpy()
        transfer_limits = getattr(self, "transfer_action_limits", np.zeros(18))
        limits = np.where(transfer_limits > 0,
                          np.minimum(self.action_clip, transfer_limits), self.action_clip)
        self.actions = np.clip(self.actions, -limits, limits)
        self.action_buffer[1:] = self.action_buffer[:-1]
        self.action_buffer[0] = self.actions
        return self.actions.copy()

    def compute_output(self, actions):
        return self.default_dof_pos + self.action_scale * np.clip(actions, -self.action_clip, self.action_clip)

    def control_targets(self, decimation_step):
        # IsaacGymEnv.step indexes each joint independently: legs wait 20 ms,
        # arm waits 15 ms and receives the new action in the final 5 ms tick.
        indices = np.ceil((self.delay_steps - decimation_step) / self.p["decimation"]).astype(int)
        return self.compute_output(self.action_buffer[np.maximum(indices, 0), np.arange(18)])

    def after_control_step(self):
        mujoco.mj_forward(self.model, self.data)
        self.pose_history.append((self.data.site_xpos[self.ee].copy(),
                                  self.data.site_xmat[self.ee].reshape(3, 3).copy()))
        self.pose_history = self.pose_history[-self.pose_latency_frames:]
