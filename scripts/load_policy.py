from dataclasses import fields

import torch
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
import os.path as osp
import pickle as pkl
from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
from go1_gym.envs.roboduet.wbc_env_config import (
    ROBODUET_DEFAULTS,
    RoboDuetCfg as Cfg,
    configure_external_recipes,
    env_obs_dim_parts,
    arm_obs_dim_parts,
    dog_obs_dim_parts,
    sum_dim_parts,
)


def _ensure_asset_file(cfg, robot=None, checkpoint_asset_file=None):
    if checkpoint_asset_file:
        return

    asset_file = getattr(cfg.asset, "file", "")
    robot = robot or "go2"
    if robot == "go1":
        cfg.asset.file = ROBODUET_DEFAULTS.asset.go1_file
    elif robot == "go2":
        cfg.asset.file = ROBODUET_DEFAULTS.asset.go2_file
    else:
        if asset_file:
            return
        raise ValueError(f"Unknown robot '{robot}', expected 'go1' or 'go2'")

    print(f"[RoboDuet] checkpoint asset file was empty; using {robot} asset: {cfg.asset.file}")


def _ensure_missing_dataclass_fields(target, defaults):
    for field in fields(defaults):
        try:
            getattr(target, field.name)
        except AttributeError:
            setattr(target, field.name, getattr(defaults, field.name))


def _ensure_play_cfg_defaults(cfg):
    defaults = ROBODUET_DEFAULTS
    for target, source in (
        (cfg.hybrid, defaults.hybrid),
        (cfg.hybrid.rewards, defaults.hybrid.rewards),
        (cfg.hybrid.reward_scales, defaults.hybrid.reward_scales),
        (cfg.arm, defaults.arm),
        (cfg.arm.commands, defaults.arm.commands),
        (cfg.arm.trajectory, defaults.arm.trajectory),
        (cfg.dog, defaults.dog),
        (cfg.env, defaults.env),
        (cfg.commands, defaults.commands),
        (cfg.control, defaults.control),
        (cfg.domain_rand, defaults.domain_rand),
        (cfg.rewards, defaults.rewards),
        (cfg.reward_scales, defaults.reward_scales),
    ):
        _ensure_missing_dataclass_fields(target, source)


def _recompute_play_dims(cfg):
    cfg.env.num_observations = sum_dim_parts(env_obs_dim_parts(cfg))
    cfg.env.num_obs_history = cfg.env.num_observation_history * cfg.env.num_observations
    cfg.arm.arm_num_observations = sum_dim_parts(arm_obs_dim_parts(cfg))
    cfg.arm.arm_num_obs_history = cfg.arm.arm_num_observation_history * cfg.arm.arm_num_observations
    cfg.dog.dog_num_observations = sum_dim_parts(dog_obs_dim_parts(cfg))
    cfg.dog.dog_num_obs_history = cfg.dog.dog_num_observation_history * cfg.dog.dog_num_observations

def load_dog_policy(logdir, ckpt_id, Cfg):
    actor_critic = DogActorCritic(Cfg.dog.dog_num_observations,
                                Cfg.dog.dog_num_privileged_obs,
                                Cfg.dog.dog_num_obs_history,
                                Cfg.dog.dog_actions,
                                use_adaptation_module=getattr(Cfg.dog, "use_adaptation_module", True),
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

def load_arm_policy(logdir, ckpt_id, Cfg):
    actor_critic = ArmActorCritic(
        Cfg.arm.arm_num_observations,
        Cfg.arm.arm_num_privileged_obs,
        Cfg.arm.arm_num_obs_history,
        Cfg.arm.num_actions_arm_cd,
        use_adaptation_module=getattr(Cfg.arm, "use_adaptation_module", False),
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
        hist = actor_his.forward(obs["obs_history"].to('cpu')[..., :-Cfg.arm.arm_num_observations])
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
    configure_external_recipes(Cfg)

    checkpoint_asset_file = None
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        # print(pkl_cfg.keys())
        # print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                if key in ["dog", "arm", "hybrid"]:

                    for key2, value2 in cfg[key].items():
                        if not isinstance(cfg[key][key2], dict):
                            setattr(getattr(Cfg, key), key2, value2)
                        else:
                            for key3, value3 in cfg[key][key2].items():
                                setattr(getattr(getattr(Cfg, key), key2), key3, value3)

                else:
                    if isinstance(cfg[key], dict):
                        for key2, value2 in cfg[key].items():
                            if key == "asset" and key2 == "file":
                                checkpoint_asset_file = value2
                            setattr(getattr(Cfg, key), key2, value2)
                    else:
                        setattr(Cfg, key, cfg[key])

    _ensure_play_cfg_defaults(Cfg)
    _ensure_asset_file(Cfg, robot=robot, checkpoint_asset_file=checkpoint_asset_file)
    _recompute_play_dims(Cfg)

    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
      Cfg.terrain.teleport_robots = False

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.domain_rand.randomize_end_effector_force = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.asset.render_sphere = True
    Cfg.env.episode_length_s = 10000
    Cfg.commands.resampling_time = 10000
    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "M"
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_roll = False
    Cfg.rewards.use_terminal_pitch = False
    Cfg.hybrid.rewards.use_terminal_body_height = False
    Cfg.hybrid.rewards.use_terminal_roll = False
    Cfg.hybrid.rewards.use_terminal_pitch = False
    Cfg.arm.commands.T_traj = [20000, 30000]
    # Cfg.sim.physx["num_position_iterations"] = 8
    # Cfg.sim.physx["num_velocity_iterations"] = 8


    env = wrapper(sim_device=device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    # load policy



    return env, Cfg
