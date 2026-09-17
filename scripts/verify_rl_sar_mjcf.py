"""Verify the rl_sar go2_x5 MuJoCo model against RoboDuet's training URDF.

The dog policy observes arm joint positions and velocities directly, so if the
MJCF's arm chain differs from the URDF IsaacGym trained on -- a flipped axis, a
frame convention, a mis-transcribed offset -- the policy is fed joint angles
that mean something different from what it learned, and no amount of tuning
will fix it. This script rules that out by comparing, for random joint
configurations:

  * every shared body's pose relative to the trunk (position + orientation)
  * per-link and total mass

against the same URDF MuJoCo compiles directly (meshes stripped, since MuJoCo
cannot read the .glb/.dae files the URDF references).

It also asserts the flat sensordata layout matches the offsets
RL_Sim::GetState() indexes in rl_sim_mujoco.cpp -- that mapping is positional
and silently wrong if the sensor block is reordered.

Usage::

    python scripts/verify_rl_sar_mjcf.py \
        --mjcf ../rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/go2_x5.xml
"""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = REPO_ROOT / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"

# Policy joint order (IsaacGym DoF order == URDF tree order).
POLICY_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "x5_joint1", "x5_joint2", "x5_joint3",
    "x5_joint4", "x5_joint5", "x5_joint6",
]
# Hardware order (Unitree SDK motor indices), which is what the MJCF's actuator
# and sensor blocks must follow so the exported joint_mapping works unchanged.
HARDWARE_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "x5_joint1", "x5_joint2", "x5_joint3",
    "x5_joint4", "x5_joint5", "x5_joint6",
]
# MuJoCo's URDF importer welds a fixed-base root link into `world` and folds
# fixed-joint children (x5_base_link) into their parent, so the URDF model's
# world frame *is* the trunk frame. The MJCF trunk is a floating body, hence
# the two different names.
TRUNK = {"mjcf": "base_link", "urdf": "world"}


def strip_meshes(urdf_path):
    """MuJoCo cannot load .glb/.dae, and we only need the kinematic tree, so
    drop every <visual>/<collision> block and keep the joints and inertials."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    # A URDF root link is welded to the world by MuJoCo's importer; that is
    # fine because we only ever compare trunk-relative poses.
    handle = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
    tree.write(handle.name)
    return handle.name


def pose_relative_to(model, data, body_id, trunk_id):
    """Body pose expressed in the trunk frame."""
    trunk_pos = data.xpos[trunk_id]
    trunk_mat = data.xmat[trunk_id].reshape(3, 3)
    rel_pos = trunk_mat.T @ (data.xpos[body_id] - trunk_pos)
    rel_mat = trunk_mat.T @ data.xmat[body_id].reshape(3, 3)
    return rel_pos, rel_mat


def joint_qpos_address(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint '{name}' not found")
    return model.jnt_qposadr[jid]


def check_kinematics(mjcf_model, urdf_model, samples, seed, atol):
    mjcf_data = mujoco.MjData(mjcf_model)
    urdf_data = mujoco.MjData(urdf_model)

    mjcf_trunk = mujoco.mj_name2id(mjcf_model, mujoco.mjtObj.mjOBJ_BODY, TRUNK["mjcf"])
    urdf_trunk = mujoco.mj_name2id(urdf_model, mujoco.mjtObj.mjOBJ_BODY, TRUNK["urdf"])
    if mjcf_trunk < 0 or urdf_trunk < 0:
        raise SystemExit("trunk body not found in one of the models")

    # Bodies present in both models, matched by name. The two trunk bodies are
    # the reference frame for the comparison, so they are excluded.
    shared = []
    for bid in range(mjcf_model.nbody):
        name = mujoco.mj_id2name(mjcf_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name in (None, TRUNK["mjcf"], TRUNK["urdf"], "world"):
            continue
        other = mujoco.mj_name2id(urdf_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if other >= 0:
            shared.append((name, bid, other))

    mjcf_adr = [joint_qpos_address(mjcf_model, n) for n in POLICY_JOINTS]
    urdf_adr = [joint_qpos_address(urdf_model, n) for n in POLICY_JOINTS]

    # Sample inside each joint's range so configurations stay physical.
    lo, hi = [], []
    for name in POLICY_JOINTS:
        jid = mujoco.mj_name2id(mjcf_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_lo, joint_hi = mjcf_model.jnt_range[jid]
        lo.append(joint_lo)
        hi.append(joint_hi)
    lo, hi = np.array(lo), np.array(hi)

    rng = np.random.default_rng(seed)
    worst = {name: (0.0, 0.0) for name, _, _ in shared}
    for sample in range(samples):
        q = lo + (hi - lo) * rng.random(len(POLICY_JOINTS)) if sample else np.zeros(len(POLICY_JOINTS))

        mujoco.mj_resetData(mjcf_model, mjcf_data)
        mujoco.mj_resetData(urdf_model, urdf_data)
        for adr, value in zip(mjcf_adr, q):
            mjcf_data.qpos[adr] = value
        for adr, value in zip(urdf_adr, q):
            urdf_data.qpos[adr] = value
        mujoco.mj_kinematics(mjcf_model, mjcf_data)
        mujoco.mj_kinematics(urdf_model, urdf_data)

        for name, bid, other in shared:
            pos_a, mat_a = pose_relative_to(mjcf_model, mjcf_data, bid, mjcf_trunk)
            pos_b, mat_b = pose_relative_to(urdf_model, urdf_data, other, urdf_trunk)
            pos_err = float(np.max(np.abs(pos_a - pos_b)))
            rot_err = float(np.max(np.abs(mat_a - mat_b)))
            best_pos, best_rot = worst[name]
            worst[name] = (max(best_pos, pos_err), max(best_rot, rot_err))

    print(f"\nkinematics: {len(shared)} shared bodies, {samples} configurations "
          f"(first is the zero pose), tolerance {atol}\n")
    failures = 0
    for name, (pos_err, rot_err) in sorted(worst.items()):
        ok = pos_err <= atol and rot_err <= atol
        failures += 0 if ok else 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<16s} pos {pos_err:.3e} m   rot {rot_err:.3e}")
    return failures


def check_mass(mjcf_model, urdf_path, atol):
    """Compare against the URDF XML rather than the compiled URDF model: MuJoCo
    folds fixed-joint links (the trunk, x5_base_link) into their parent, so
    their mass is not recoverable from the compiled model."""
    root = ET.parse(urdf_path).getroot()
    urdf_mass = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        urdf_mass[link.get("name")] = float(inertial.find("mass").get("value"))
    # The rl_sar go2 model calls the trunk base_link; the URDF calls it base.
    aliases = {"base_link": "base"}

    print("\nmass (vs URDF XML):\n")
    failures = 0
    total_mjcf = total_urdf = 0.0
    unmatched = []
    for bid in range(1, mjcf_model.nbody):
        name = mujoco.mj_id2name(mjcf_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        urdf_name = aliases.get(name, name)
        if urdf_name not in urdf_mass:
            unmatched.append(name)
            continue
        a, b = mjcf_model.body_mass[bid], urdf_mass[urdf_name]
        total_mjcf += a
        total_urdf += b
        if abs(a - b) > atol:
            failures += 1
            print(f"  [FAIL] {name:<16s} mjcf {a:.4f} kg   urdf {b:.4f} kg")
    print(f"  [{'ok  ' if not failures else 'FAIL'}] {mjcf_model.nbody - 1 - len(unmatched)} "
          f"matched bodies: mjcf {total_mjcf:.3f} kg   urdf {total_urdf:.3f} kg")
    if unmatched:
        print(f"         not in URDF (expected: gripper pads etc.): {', '.join(unmatched)}")
    print(f"         full model mass: {mjcf_model.body_subtreemass[0]:.3f} kg")
    return failures


def check_rl_sar_layout(model):
    """rl_sim_mujoco.cpp indexes sensordata and ctrl positionally."""
    n = len(HARDWARE_JOINTS)
    print(f"\nrl_sar layout (num_of_dofs = {n}):\n")
    failures = 0

    def sensor_adr(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return -1 if sid < 0 else model.sensor_adr[sid]

    # jointpos[i], jointvel[i + N], jointactuatorfrc[i + 2N] for hardware index i
    for i, joint in enumerate(HARDWARE_JOINTS):
        for suffix, expected in ((("_pos"), i), (("_vel"), i + n), (("_torque"), i + 2 * n)):
            adr = sensor_adr(joint.replace("_joint", "") + suffix if joint.startswith(("FR", "FL", "RR", "RL"))
                             else joint + suffix)
            if adr != expected:
                failures += 1
                print(f"  [FAIL] sensor for {joint}{suffix}: adr {adr}, GetState() reads {expected}")
    if not failures:
        print(f"  [ok  ] jointpos / jointvel / jointactuatorfrc occupy 0..{3 * n - 1} "
              f"in hardware order")

    expectations = [
        ("imu_quat", 3 * n, 4, "framequat -> imu.quaternion"),
        ("imu_gyro", 3 * n + 4, 3, "gyro -> imu.gyroscope"),
        ("imu_acc", 3 * n + 7, 3, "accelerometer"),
        ("frame_pos", 3 * n + 10, 3, "framepos -> base.position (WORLD)"),
        ("frame_vel", 3 * n + 13, 3, "framelinvel -> base.lin_vel (rotated to BODY)"),
    ]
    for name, expected, dim, description in expectations:
        adr = sensor_adr(name)
        ok = adr == expected
        failures += 0 if ok else 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<10s} adr {adr:>3d} (expected {expected:>3d})"
              f"  {description}")

    if model.nsensordata != 3 * n + 16:
        failures += 1
        print(f"  [FAIL] nsensordata {model.nsensordata}, expected {3 * n + 16}")

    # ctrl[i] must drive hardware joint i
    for i, joint in enumerate(HARDWARE_JOINTS):
        aid = model.actuator_trnid[i, 0]
        actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, aid)
        if actual != joint:
            failures += 1
            print(f"  [FAIL] ctrl[{i}] drives '{actual}', expected '{joint}'")
    if model.nu != n:
        failures += 1
        print(f"  [FAIL] nu {model.nu}, expected {n}")
    print(f"  [{'ok  ' if model.nu == n else 'FAIL'}] {model.nu} actuators in hardware order")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mjcf", type=str, required=True)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    mjcf_model = mujoco.MjModel.from_xml_path(args.mjcf)
    urdf_model = mujoco.MjModel.from_xml_path(strip_meshes(args.urdf))

    failures = check_kinematics(mjcf_model, urdf_model, args.samples, args.seed, args.atol)
    failures += check_mass(mjcf_model, args.urdf, 1e-4)
    failures += check_rl_sar_layout(mjcf_model)

    if failures:
        print(f"\n{failures} check(s) failed -- the MJCF does not match training.")
        return 1
    print("\nMJCF matches the training URDF and rl_sar's expected layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
