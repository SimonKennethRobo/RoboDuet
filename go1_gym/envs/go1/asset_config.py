from typing import Any


def config_asset(Cnfg: Any):
    Cnfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
    Cnfg.asset.foot_name = "foot"
    Cnfg.asset.penalize_contacts_on = ["thigh", "calf"]
    Cnfg.asset.terminate_after_contacts_on = ["base"]
    Cnfg.asset.self_collisions = 0
    Cnfg.asset.flip_visual_attachments = False
    Cnfg.asset.fix_base_link = False
    Cnfg.asset.hip_joints = {"hip"}
    Cnfg.asset.render_sphere = False

    Cnfg.control.stiffness = {"joint": 35.0}
    Cnfg.control.damping = {"joint": 1.0}

    Cnfg.init_state.default_joint_angles = {
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "FR_thigh_joint": 0.8,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }
