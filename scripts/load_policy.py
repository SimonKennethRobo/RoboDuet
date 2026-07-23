import torch
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
import os.path as osp
import pickle as pkl
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.envs.config import (
    RoboDuetRuntimeOptions,
    apply_config_snapshot,
    build_roboduet_config,
    recompute_observation_dims,
)
from go1_gym.envs.config.wbc import ROBOT_ASSET_FILES


def _ensure_asset_file(cfg, robot=None, checkpoint_asset_file=None):
    if checkpoint_asset_file:
        return

    asset_file = getattr(cfg.asset, "file", "")
    robot = robot or "go2"
    if robot not in ROBOT_ASSET_FILES:
        if asset_file:
            return
        raise ValueError(f"Unknown robot '{robot}', expected one of {sorted(ROBOT_ASSET_FILES)}")
    cfg.asset.file = ROBOT_ASSET_FILES[robot]

    print(f"[RoboDuet] checkpoint asset file was empty; using {robot} asset: {cfg.asset.file}")


def load_dog_policy(logdir, ckpt_id, cfg):
    actor_critic = DogActorCritic(cfg.dog.dog_num_observations,
                                cfg.dog.dog_num_privileged_obs,
                                cfg.dog.dog_num_obs_history,
                                cfg.dog.dog_actions,
                                use_adaptation_module=getattr(cfg.dog, "use_adaptation_module", True),
                                ).to("cpu")
    device = torch.device("cpu")
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id + '_dog'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(logdir + f'/checkpoints_dog/ac_weights_{str(ckpt_id_)}.pt', map_location=device)
    # for key, value in ckpt.items():
    #     print(key, value.shape)
    actor_critic.load_state_dict(ckpt)

    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body

    def policy(obs, info={}):
        i = 0
        actor_input = (obs["obs_history"].to('cpu'),)
        if adaptation_module is not None:
            latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
            actor_input = (obs["obs_history"].to('cpu'), latent)
            info['latent'] = latent
        action = body.forward(torch.cat(actor_input, dim=-1))
        return action

    return policy

def load_arm_policy(logdir, ckpt_id, cfg):
    actor_critic = ArmActorCritic(
        cfg.arm.arm_num_observations,
        cfg.arm.arm_num_privileged_obs,
        cfg.arm.arm_num_obs_history,
        cfg.arm.num_actions_arm_cd,
        use_adaptation_module=getattr(cfg.arm, "use_adaptation_module", False),
        device='cpu'
    ).to('cpu')

    device = torch.device("cpu")
    if ckpt_id == 'last':
        ckpt_id_ = ckpt_id +'_arm'
    else:
        ckpt_id_ = ckpt_id.zfill(6)
    ckpt = torch.load(logdir + f'/checkpoints_arm/ac_weights_{str(ckpt_id_)}.pt', map_location=device)
    actor_critic.load_state_dict(ckpt)

    actor_critic.eval()
    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body
    actor_his = actor_critic.actor_history_encoder

    def policy(obs, info={}):
        hist = actor_his.forward(obs["obs_history"].to('cpu')[..., :-cfg.arm.arm_num_observations])
        actor_input = (obs["obs"].to('cpu'), hist)
        if adaptation_module is not None:
            latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
            actor_input = (obs["obs"].to('cpu'), latent, hist)
            info['latent'] = latent
        action = body.forward(torch.cat(actor_input, dim=-1))
        return action

    return policy

def load_env(logdir, wrapper, headless=False, device='cuda:0', robot=None):
    print('*'*10, logdir)
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=1, robot=robot or "go2"))

    checkpoint_asset_file = None
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        snapshot = pkl_cfg["Cfg"]
        checkpoint_asset_file = snapshot.get("asset", {}).get("file")
        apply_config_snapshot(cfg, snapshot)

    _ensure_asset_file(cfg, robot=robot, checkpoint_asset_file=checkpoint_asset_file)
    recompute_observation_dims(cfg)

    cfg.terrain.mesh_type = "plane"
    if cfg.terrain.mesh_type == "plane":
      cfg.terrain.teleport_robots = False

    # turn off DR for evaluation script
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_ground_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_com_displacement = False

    cfg.domain_rand.randomize_end_effector_force = False

    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = 1
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = False
    cfg.asset.render_sphere = True
    cfg.env.episode_length_s = 10000
    cfg.commands.resampling_time = 10000
    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    cfg.control.control_type = "M"
    cfg.rewards.use_terminal_body_height = False
    cfg.rewards.use_terminal_roll = False
    cfg.rewards.use_terminal_pitch = False
    cfg.wbc.rewards.use_terminal_body_height = False
    cfg.wbc.rewards.use_terminal_roll = False
    cfg.wbc.rewards.use_terminal_pitch = False
    # Cfg.sim.physx["num_position_iterations"] = 8
    # Cfg.sim.physx["num_velocity_iterations"] = 8


    env = wrapper(sim_device=device, headless=headless, cfg=cfg)
    env = HistoryWrapper(env)
    # load policy



    return env, cfg
