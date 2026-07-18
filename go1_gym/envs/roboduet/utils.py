import torch


DOG_PLAY_COMMAND_LIMIT_ATTRS = {
    "x_vel": "limit_vel_x",
    "y_vel": "limit_vel_y",
    "yaw_vel": "limit_vel_yaw",
    "body_pitch": "limit_body_pitch",
    "body_roll": "limit_body_roll",
    "body_height_delta": "limit_body_height",
    "body_height": "limit_body_height",
    "gait_freq": "limit_gait_frequency",
    "gait_frequency": "limit_gait_frequency",
    "footswing_height": "limit_footswing_height",
    "stance_width": "limit_stance_width",
    "stance_length": "limit_stance_length",
    "gait_duration": "limit_gait_duration",
}

ARM_PLAY_COMMAND_LIMIT_ATTRS = {
    "arm_x": "l",
    "arm_z": "p",
    "arm_y": "y",
    "arm_roll": "roll_ee",
    "arm_pitch": "pitch_ee",
    "arm_yaw": "yaw_ee",
}


def as_limit_pair(value):
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) != 2:
            return None
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def get_play_command_limits(cfg):
    limits = {"dog": {}, "arm": {}}
    for command_key, attr_name in DOG_PLAY_COMMAND_LIMIT_ATTRS.items():
        pair = as_limit_pair(getattr(cfg.commands, attr_name, None))
        if pair is not None:
            limits["dog"][command_key] = pair
    for command_key, attr_name in ARM_PLAY_COMMAND_LIMIT_ATTRS.items():
        pair = as_limit_pair(getattr(cfg.arm.commands, attr_name, None))
        if pair is not None:
            limits["arm"][command_key] = pair
    return limits


def get_play_command_limit(cfg, target, command_key, fallback):
    pair = get_play_command_limits(cfg).get(target, {}).get(command_key)
    return pair if pair is not None else fallback


def apply_wbc_reward_settings(cfg):
    for key, value in vars(cfg.wbc.rewards).items():
        setattr(cfg.rewards, key, value)


def clip_observation(env, obs):
    clip_obs = env.cfg.normalization.clip_observations
    return torch.clip(obs, -clip_obs, clip_obs)


class StageSchedule:
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    TWO_STAGE = "two_stage"

    def __init__(
        self,
        train_stage,
        num_learning_iterations,
        default_switch_iteration,
        debug=False,
        debug_switch_iteration=20,
    ):
        self.train_stage = train_stage
        self.num_learning_iterations = num_learning_iterations
        self.default_switch_iteration = default_switch_iteration
        self.debug = debug
        self.debug_switch_iteration = debug_switch_iteration

    @property
    def starts_in_stage2(self):
        return self.train_stage == self.STAGE2

    def _stage1_learning_iterations(self):
        if self.train_stage == self.STAGE1:
            return self.num_learning_iterations
        if self.train_stage == self.STAGE2:
            return 1
        return self.default_switch_iteration

    def configure(self, global_switch):
        global_switch.count = 0
        global_switch.stage1_count = 0
        global_switch.switch_flag = False
        global_switch.stage1_arm_ramp_iterations = max(1, int(self._stage1_learning_iterations()))

        if self.train_stage == self.STAGE1:
            global_switch.pretrained_to_wbc_start = self.num_learning_iterations + 1
            global_switch.pretrained_to_wbc_end = global_switch.pretrained_to_wbc_start + 1
            return

        if self.train_stage == self.STAGE2:
            global_switch.pretrained_to_wbc_start = -1
            global_switch.pretrained_to_wbc_end = 0
            global_switch.count = global_switch.pretrained_to_wbc_end
            global_switch.open_switch()
            return

        global_switch.pretrained_to_wbc_start = self.default_switch_iteration
        global_switch.pretrained_to_wbc_end = global_switch.pretrained_to_wbc_start + 0
        if self.debug and global_switch.pretrained_to_wbc_start > 0:
            global_switch.pretrained_to_wbc_start = self.debug_switch_iteration
            global_switch.pretrained_to_wbc_end = global_switch.pretrained_to_wbc_start + 2
            global_switch.stage1_arm_ramp_iterations = self.debug_switch_iteration

    def maybe_switch(self, iteration, global_switch, env, message):
        if global_switch.switch_open or iteration != global_switch.pretrained_to_wbc_start:
            return False

        print(message)
        global_switch.open_switch()
        apply_wbc_reward_settings(env.cfg)
        return True


class ObservationBuilder:
    def __init__(self, env, name, expected_dim):
        self.env = env
        self.name = name
        self.expected_dim = expected_dim
        self.terms = []

    def add(self, *terms):
        for term in terms:
            if term is None:
                continue
            if term.shape[-1] == 0:
                continue
            self.terms.append(term)

    def build(self, clip=True):
        if self.terms:
            obs = torch.cat(self.terms, dim=-1)
        else:
            obs = torch.empty(self.env.num_envs, 0, dtype=torch.float, device=self.env.device)

        if obs.shape[1] != self.expected_dim:
            raise AssertionError(
                f"{self.name} num_observations ({self.expected_dim}) != the number of observations ({obs.shape[1]})"
            )

        if clip:
            clip_obs = self.env.cfg.normalization.clip_observations
            obs = torch.clip(obs, -clip_obs, clip_obs)
        return obs
