"""Training-stage scheduling helpers.

This module keeps the legacy `global_switch` API as the runtime source of
truth, but centralizes how train modes initialize and transition it.
"""


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
        debug_switch_iteration=2,
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
            global_switch.pretrained_to_hybrid_start = self.num_learning_iterations + 1
            global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 1
            return

        if self.train_stage == self.STAGE2:
            global_switch.pretrained_to_hybrid_start = -1
            global_switch.pretrained_to_hybrid_end = 0
            global_switch.count = global_switch.pretrained_to_hybrid_end
            global_switch.open_switch()
            return

        global_switch.pretrained_to_hybrid_start = self.default_switch_iteration
        global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 0
        if self.debug and global_switch.pretrained_to_hybrid_start > 0:
            global_switch.pretrained_to_hybrid_start = self.debug_switch_iteration
            global_switch.pretrained_to_hybrid_end = global_switch.pretrained_to_hybrid_start + 2
            global_switch.stage1_arm_ramp_iterations = self.debug_switch_iteration

    def maybe_switch(self, iteration, global_switch, env, message):
        if global_switch.switch_open or iteration != global_switch.pretrained_to_hybrid_start:
            return False

        print(message)
        global_switch.open_switch()
        apply_hybrid_reward_settings(env.cfg)
        return True


def apply_hybrid_reward_settings(cfg):
    for key, value in vars(cfg.hybrid.rewards).items():
        setattr(cfg.rewards, key, value)
