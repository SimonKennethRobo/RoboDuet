from typing import Any


def config_asset(Cnfg: Any):

    Cnfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf"
    Cnfg.asset.self_collisions = 1  # 0 to enable self-collision, 1 to disable

    Cnfg.asset.penalize_contacts_on = ["base", "trunk", "wrist", "thigh", "calf", "Head"]
    # Cnfg.asset.arm_contact_bodies = ["zarx_body", "gripper"]
    Cnfg.asset.arm_contact_bodies = []

    Cnfg.asset.terminate_after_contacts_on = [""]

    Cnfg.asset.hip_joints = {"hip"}

    Cnfg.control.stiffness = {"joint": 35.0, "widow": 5.0, "zarx": 5.0, "zarx_j3": 20}  # [N*m/rad]

    Cnfg.arm.control.stiffness_arm = {
        "zarx": 50,
        "zarx_j1": 40,
        "zarx_j2": 70,
        "zarx_j3": 70,
        "zarx_j4": 25,
        "zarx_j5": 25,
        "zarx_j6": 25,
        "zarx_j7": 50,
        "zarx_j8": 50,
    }  # [N*m/rad]
    Cnfg.arm.control.damping_arm = {
        "zarx": 20,
        "zarx_j1": 3,
        "zarx_j2": 15,
        "zarx_j3": 15,
        "zarx_j4": 2,
        "zarx_j5": 2,
        "zarx_j6": 2,
        "zarx_j7": 20,
        "zarx_j8": 20,
    }  # [N*m*s/rad]

    Cnfg.dog.control.stiffness_leg = {"joint": 35.0}  # [N*m/rad]
    Cnfg.dog.control.damping_leg = {"joint": 1.0}  # [N*m*s/rad]

    Cnfg.asset.render_sphere = True  # NOTE no use in headless

    Cnfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
        "FL_hip_joint": 0.1,  # [rad]
        "RL_hip_joint": 0.1,  # [rad]
        "FR_hip_joint": -0.1,  # [rad]
        "RR_hip_joint": -0.1,  # [rad]
        "FL_thigh_joint": 0.8,  # [rad]
        "RL_thigh_joint": 1.0,  # [rad]
        "FR_thigh_joint": 0.8,  # [rad]
        "RR_thigh_joint": 1.0,  # [rad]
        "FL_calf_joint": -1.5,  # [rad]
        "RL_calf_joint": -1.5,  # [rad]
        "FR_calf_joint": -1.5,  # [rad]
        "RR_calf_joint": -1.5,  # [rad]
        "widow_waist": 0.0,
        "widow_shoulder": 0.0,
        "widow_elbow": 0.0,
        "forearm_roll": 0.0,
        "widow_wrist_angle": 0.0,
        "widow_wrist_rotate": 0.0,
        "widow_forearm_roll": 0.0,
        "gripper": 0.0,
        "widow_left_finger": 0.0,
        "widow_right_finger": 0.0,
        "zarx_j1": 0.0,
        "zarx_j2": 0.0,
        "zarx_j3": 0.0,
        "zarx_j4": 0.0,
        "zarx_j5": 0.0,
        "zarx_j6": 0.0,
        "zarx_j7": 0.0,
        "zarx_j8": 0.0,
    }
