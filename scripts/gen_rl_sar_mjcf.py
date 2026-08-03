"""Generate rl_sar's go2_x5 MuJoCo model from RoboDuet's training URDF.

Grafts the ARX5 arm onto the canonical rl_sar go2 model, so the legs, actuator
order and sensor block already follow rl_sar's conventions exactly, and every
arm number -- link poses, joint axes, joint ranges, inertials, the arm mount
offset -- is read straight out of the URDF IsaacGym trained against. The Go2
leg/trunk inertials are retargeted to the same URDF too, since the stock model
ships MuJoCo Menagerie's values which differ by ~0.5 kg over the four legs.

Nothing here is hand-transcribed, so the model cannot silently drift from
training. Verify the result with scripts/verify_rl_sar_mjcf.py, which compares
forward kinematics and mass against the URDF body by body.

Usage::

    python scripts/gen_rl_sar_mjcf.py --zoo <rl_sar>/src/rl_sar_zoo
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--zoo", type=str, required=True,
                    help="path to rl_sar/src/rl_sar_zoo")
parser.add_argument("--urdf", type=str,
                    default=str(REPO_ROOT / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"),
                    help="RoboDuet training URDF the arm numbers come from")
parser.add_argument("--base_model", type=str, default="go2_description/mjcf/go2.xml",
                    help="stock rl_sar model to graft the arm onto, relative to --zoo")
parser.add_argument("--out", type=str, default="go2_x5_description/mjcf/go2_x5.xml",
                    help="output model, relative to --zoo")
args = parser.parse_args()

ZOO = Path(args.zoo)
URDF = Path(args.urdf)
SRC = ZOO / args.base_model
DST = ZOO / args.out

ARM_JOINTS = [f"x5_joint{i}" for i in range(1, 7)]
# The reference ARX5 .obj meshes are authored in a frame rotated 180 deg about
# X relative to the URDF link frame for these links. Same kinematics either way;
# we keep the URDF frame (so the XML reads 1:1 against training) and rotate the
# visual mesh instead.
MESH_FLIPPED = {"x5_link3", "x5_link4", "x5_link5"}

root = ET.parse(URDF).getroot()
joints = {j.get("name"): j for j in root.findall("joint")}
links = {l.get("name"): l for l in root.findall("link")}


def fmt(values):
    return " ".join(f"{v:.8g}" for v in values)


def origin(joint_name):
    o = joints[joint_name].find("origin")
    return [float(v) for v in (o.get("xyz") or "0 0 0").split()]


def axis(joint_name):
    a = joints[joint_name].find("axis")
    return [float(v) for v in a.get("xyz").split()]


def limits(joint_name):
    lim = joints[joint_name].find("limit")
    return float(lim.get("lower")), float(lim.get("upper")), float(lim.get("effort"))


def inertial(link_name):
    ine = links[link_name].find("inertial")
    o = ine.find("origin")
    pos = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
    mass = float(ine.find("mass").get("value"))
    i = ine.find("inertia")
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    full = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    return pos, mass, full


def inertial_xml(link_name, indent):
    pos, mass, full = inertial(link_name)
    return (f'{indent}<inertial pos="{fmt(pos)}" mass="{mass:.8g}" '
            f'fullinertia="{fmt(full)}"/>')


def body_open(link_name, joint_name, indent, mesh, collision=""):
    pos = origin(joint_name)
    lo, hi, _ = limits(joint_name)
    ax = axis(joint_name)
    mesh_quat = ' quat="0 1 0 0"' if link_name in MESH_FLIPPED else ""
    out = [f'{indent}<body name="{link_name}" pos="{fmt(pos)}">']
    out.append(f'{indent}  <joint name="{joint_name}" axis="{fmt(ax)}" '
               f'range="{lo:.6g} {hi:.6g}" class="arx5_motor"/>')
    out.append(inertial_xml(link_name, indent + "  "))
    out.append(f'{indent}  <geom mesh="{mesh}" class="arx5_visual"{mesh_quat}/>')
    if collision:
        out.append(f"{indent}  {collision}")
    return out


def build_arm():
    mount = origin("arm_mount_joint")
    ind = "      "
    lines = [
        f"{ind}<!-- ======================= ARX5 arm =======================",
        f"{ind}     Every pose, axis, joint range and inertial below is copied",
        f"{ind}     verbatim from RoboDuet's training URDF",
        f"{ind}     resources/robots/go2_x5_v3/urdf/go2_x5.urdf, including the",
        f"{ind}     arm_mount_joint offset. Do not 'tidy' these numbers: the dog",
        f"{ind}     policy observes arm joint positions/velocities directly and",
        f"{ind}     was trained against this exact kinematic chain.",
        f"{ind}-->",
        f'{ind}<body name="x5_base_link" pos="{fmt(mount)}" childclass="arx5">',
        inertial_xml("x5_base_link", ind + "  "),
        f'{ind}  <geom mesh="x5_base_link" class="arx5_visual"/>',
        f'{ind}  <geom class="arx5_collision" type="box" size="0.035 0.035 0.03" pos="0 0 0.03"/>',
    ]

    meshes = {f"x5_link{i}": f"x5_link{i}" for i in range(1, 9)}
    # Approximate collision capsules along the two long links plus the wrist, so
    # the arm cannot sink through the floor or the dog's own body. These are
    # deliberately coarse -- they exist for plausibility, not contact fidelity.
    collisions = {
        "x5_link2": '<geom class="arx5_collision" type="capsule" size="0.035" fromto="0 0 0 -0.264 0 0"/>',
        "x5_link3": '<geom class="arx5_collision" type="capsule" size="0.032" fromto="0 0 0.05 0.245 0 0.05"/>',
        "x5_link5": '<geom class="arx5_collision" type="capsule" size="0.035" fromto="0 0 0 0.03 0 -0.085"/>',
        "x5_link6": '<geom class="arx5_collision" type="capsule" size="0.03" fromto="0 0 0 0.09 0 0"/>',
    }

    depth = 1
    for name, joint in zip([f"x5_link{i}" for i in range(1, 7)], ARM_JOINTS):
        indent = ind + "  " * depth
        lines += body_open(name, joint, indent, meshes[name], collisions.get(name, ""))
        depth += 1

    # Gripper fingers: welded at the closed position. RoboDuet's policy neither
    # observes nor drives them (num_of_dofs counts 12 legs + 6 arm), so giving
    # them slide joints here would add DoFs rl_sar does not know about.
    indent = ind + "  " * depth
    for link, joint in (("x5_link7", "x5_gripper_joint"), ("x5_link8", "x5_joint8")):
        pos = origin(joint)
        lines.append(f'{indent}<body name="{link}" pos="{fmt(pos)}">')
        lines.append(inertial_xml(link, indent + "  "))
        lines.append(f'{indent}  <geom mesh="{link}" class="arx5_visual"/>')
        lines.append(f"{indent}</body>")
    # End-effector reference frame (cfg.arm.ik.ee_local_pos), for debugging only.
    lines.append(f'{indent}<site name="x5_ee" pos="0.1424 0 0.0001057" size="0.01"/>')

    for depth in range(depth - 1, -1, -1):
        lines.append(ind + "  " * depth + "</body>")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Leg / trunk inertials
# ---------------------------------------------------------------------------
# The stock rl_sar go2.xml carries MuJoCo Menagerie's Go2 inertials, which
# differ from RoboDuet's URDF by up to ~90 g per link (0.5 kg over the four
# legs). The kinematic frames are identical -- verified body-by-body by
# scripts/verify_rl_sar_mjcf.py -- so the URDF inertials can be dropped in
# as-is, making the sim2sim model dynamically match what the policy trained
# against. Perturb these deliberately if you want a robustness test; do not let
# them drift by accident.
import re

LEG_LINKS = {"base_link": "base"}
for side in ("FL", "FR", "RL", "RR"):
    for part in ("hip", "thigh", "calf"):
        LEG_LINKS[f"{side}_{part}"] = f"{side}_{part}"
# The stock model declares the foot as an empty (massless) frame body and folds
# the foot mass into the calf. The URDF keeps them separate, so give the foot
# bodies their own inertial and the calf keeps only the calf's own mass --
# otherwise retargeting the calf alone would silently drop 4 x 40 g.
FOOT_LINKS = {f"{side}_foot": f"{side}_foot" for side in ("FL", "FR", "RL", "RR")}


def retarget_inertials(text):
    for mjcf_name, urdf_name in LEG_LINKS.items():
        pattern = re.compile(
            r'(<body name="' + re.escape(mjcf_name) + r'"[^>]*>\s*)<inertial\b.*?/>',
            re.S,
        )
        replacement = lambda m: m.group(1) + inertial_xml(urdf_name, "").strip()
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise SystemExit(f"could not rewrite inertial for {mjcf_name}")

    for mjcf_name, urdf_name in FOOT_LINKS.items():
        pattern = re.compile(r'( *)<body name="' + re.escape(mjcf_name) + r'"([^>]*?)/>')
        match = pattern.search(text)
        if match is None:
            raise SystemExit(f"could not find empty foot body {mjcf_name}")
        indent = match.group(1)
        body = (f'{indent}<body name="{mjcf_name}"{match.group(2).rstrip()}>\n'
                f'{inertial_xml(urdf_name, indent + "  ")}\n'
                f'{indent}</body>')
        text = text[:match.start()] + body + text[match.end():]
    return text


ARM_DEFAULTS = """
    <!-- ARX5 arm.
         armature: reflected rotor inertia. RoboDuet's URDF declares none, and
         IsaacGym gets away with that because control_type="M" drives the arm
         through an *implicit* position drive. rl_sar instead computes the PD
         explicitly at 1/dt and writes a torque, which is only stable while
         kd*dt/I < 2. With the exported wrist gains (kd=1..2.5) and the bare
         link inertias (I ~ 5e-4) that ratio reaches ~10 and joints 4-6 ring at
         several tenths of a radian. A geared arm genuinely has this inertia --
         the canonical rl_sar go2 model already carries armature="0.01" on
         every leg joint for the same reason -- so declaring it here moves the
         model towards the real arm, not away from training.
         damping/frictionloss stay at 0: those the URDF really does model as
         absent, and rl_kd already supplies the damping term. -->
    <default class="arx5">
      <joint damping="0" armature="0.01" frictionloss="0"/>
      <default class="arx5_motor">
        <joint/>
      </default>
      <default class="arx5_visual">
        <geom type="mesh" material="arx5" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="arx5_collision">
        <geom group="3" rgba="0.9 0.3 0.3 0.4"/>
      </default>
    </default>
"""

ARM_ASSETS = """
    <material name="arx5" rgba="0.75 0.75 0.78 1" />
    <mesh name="x5_base_link" file="base_link.obj" />
    <mesh name="x5_link1" file="link1.obj" />
    <mesh name="x5_link2" file="link2.obj" />
    <mesh name="x5_link3" file="link3.obj" />
    <mesh name="x5_link4" file="link4.obj" />
    <mesh name="x5_link5" file="link5.obj" />
    <mesh name="x5_link6" file="link6.obj" />
    <mesh name="x5_link7" file="link7.obj" />
    <mesh name="x5_link8" file="link8.obj" />
"""

ARM_CONTACT = """
  <contact>
    <exclude body1="base_link" body2="x5_base_link"/>
    <exclude body1="base_link" body2="x5_link1"/>
    <exclude body1="base_link" body2="x5_link2"/>
    <exclude body1="x5_base_link" body2="x5_link1"/>
    <exclude body1="x5_base_link" body2="x5_link2"/>
    <exclude body1="x5_link1" body2="x5_link2"/>
    <exclude body1="x5_link2" body2="x5_link3"/>
    <exclude body1="x5_link3" body2="x5_link4"/>
    <exclude body1="x5_link4" body2="x5_link5"/>
    <exclude body1="x5_link5" body2="x5_link6"/>
    <exclude body1="x5_link6" body2="x5_link7"/>
    <exclude body1="x5_link6" body2="x5_link8"/>
  </contact>
"""


def arm_actuators():
    lines = ["\n    <!-- ARX5 arm, ctrlrange from the URDF effort limits -->"]
    for joint in ARM_JOINTS:
        _, _, effort = limits(joint)
        lines.append(f'    <motor name="{joint}" joint="{joint}" '
                     f'ctrlrange="-{effort:g} {effort:g}"/>')
    return "\n".join(lines)


def arm_sensors(kind):
    tag = {"pos": "jointpos", "vel": "jointvel", "torque": "jointactuatorfrc"}[kind]
    suffix = {"pos": "_pos", "vel": "_vel", "torque": "_torque"}[kind]
    extra = ' noise="0.01"' if kind == "torque" else ""
    return "\n".join(f'    <{tag} name="{j}{suffix}" joint="{j}"{extra} />'
                     for j in ARM_JOINTS)


text = SRC.read_text()

text = text.replace('<mujoco model="go2">', '<mujoco model="go2_x5">', 1)
text = retarget_inertials(text)

# defaults / assets
text = text.replace("    </default>\n  </default>", "    </default>\n" + ARM_DEFAULTS + "  </default>", 1)
text = text.replace('    <mesh file="foot.obj" />', '    <mesh file="foot.obj" />\n' + ARM_ASSETS, 1)

# Arm body: appended as the LAST child of the trunk, after all four legs, so
# MuJoCo's qpos/qvel order becomes [free(7), legs(12), arm(6)] -- the same
# ordering RoboDuet's IsaacGym DoF indices use. Sensor and actuator order is
# declared explicitly further down and is unaffected by this, but matching the
# training order here makes every qpos dump directly comparable.
anchor = "    </body>\n  </worldbody>"
if text.count(anchor) != 1:
    raise SystemExit("could not uniquely locate the trunk closing tag in go2.xml")
text = text.replace(anchor, "\n" + build_arm() + "\n    </body>\n  </worldbody>", 1)

# The stock keyframe only covers 12 leg joints. Rewrite it with RoboDuet's
# default_dof_pos (policy order FL, FR, RL, RR) plus the arm's zero pose, and
# zero ctrl -- these are torque motors, so anything else applies a step input.
KEYFRAME = """  <keyframe>
    <key name="home"
      qpos="0 0 0.34 1 0 0 0
            0.1 0.8 -1.5   -0.1 0.8 -1.5   0.1 1.0 -1.5   -0.1 1.0 -1.5
            0 0 0 0 0 0"
      ctrl="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" />
  </keyframe>"""
start = text.index("  <keyframe>")
end = text.index("</keyframe>") + len("</keyframe>")
text = text[:start] + KEYFRAME + text[end:]

# contact exclusions before the actuator block
text = text.replace("  <actuator>", ARM_CONTACT + "\n  <actuator>", 1)

# actuators appended (hardware order: FR, FL, RR, RL, then arm)
text = text.replace('    <motor class="knee" name="RL_calf" joint="RL_calf_joint" />',
                    '    <motor class="knee" name="RL_calf" joint="RL_calf_joint" />\n' + arm_actuators(), 1)

# sensors: arm entries go at the end of each of the three per-joint blocks so
# the flat sensordata layout stays [legs..., arm...] per quantity
text = text.replace('    <jointpos name="RL_calf_pos" joint="RL_calf_joint" />',
                    '    <jointpos name="RL_calf_pos" joint="RL_calf_joint" />\n' + arm_sensors("pos"), 1)
text = text.replace('    <jointvel name="RL_calf_vel" joint="RL_calf_joint" />',
                    '    <jointvel name="RL_calf_vel" joint="RL_calf_joint" />\n' + arm_sensors("vel"), 1)
text = text.replace('    <jointactuatorfrc name="RL_calf_torque" joint="RL_calf_joint" noise="0.01" />',
                    '    <jointactuatorfrc name="RL_calf_torque" joint="RL_calf_joint" noise="0.01" />\n' + arm_sensors("torque"), 1)

# The arm raises the standing height; drop the spawn height so the feet start
# just above the floor rather than the robot falling 10 cm on reset.
text = text.replace('<body name="base_link" pos="0 0 0.445"', '<body name="base_link" pos="0 0 0.40"', 1)

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(text)
print(f"wrote {DST} ({len(text.splitlines())} lines)")
