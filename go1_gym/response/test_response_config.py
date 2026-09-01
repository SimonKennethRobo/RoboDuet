"""Config -> ReferenceModel wiring (step 3).

Checks that what the env builds at runtime is the model R2 specifies, without
needing IsaacGym.
"""

import argparse

import pytest

from go1_gym.envs.config import build_roboduet_config
from go1_gym.response import (
    DECISION_CHANNEL_NAMES,
    DECISION_CHANNEL_UNITS,
    DEFAULT_CHANNELS,
    ReferenceModel,
    build_channels,
)


def _args(**overrides):
    base = dict(
        robot="go2_x5",
        num_envs=4096,
        dyna_gait=True,
        goal_reaching=False,
        traj_tracking=False,
        arm_action_mode=None,
        no_reach_table=False,
        dyna_gait_min_frequency=0.0,
        video=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def cfg():
    return build_roboduet_config(_args())


def _channels_from_cfg(cfg):
    """Exactly what LeggedRobot._init_buffers does."""
    return build_channels(
        cfg.response.channel_order, cfg.response.omega_n, cfg.response.rate_limit
    )


def test_config_reproduces_the_r2_starting_values(cfg):
    channels = _channels_from_cfg(cfg)
    assert channels == DEFAULT_CHANNELS


def test_r2_starting_values_match_the_requirements_table(cfg):
    """Guard against a silent edit of the table the calibration will replace."""
    expected = {
        "vx": (8.0, 1.2),
        "vy": (6.0, 0.8),
        "wyaw": (8.0, 3.0),
        "height": (7.0, 0.25),
        "pitch": (5.0, 0.8),
    }
    for name, (omega_n, rate_limit) in expected.items():
        assert cfg.response.omega_n[name] == omega_n
        assert cfg.response.rate_limit[name] == rate_limit


def test_channel_order_is_the_response_channel_order(cfg):
    assert tuple(cfg.response.channel_order) == DECISION_CHANNEL_NAMES


def test_command_indices_are_not_configurable(cfg):
    """cmd_index is a layout fact, not a tunable -- it must not appear in cfg."""
    assert not hasattr(cfg.response, "cmd_index")
    channels = _channels_from_cfg(cfg)
    assert tuple(c.cmd_index for c in channels) == (0, 1, 2, 5, 3)


def test_every_channel_has_a_metric_unit():
    """The perf metrics name themselves after these; a missing entry would
    silently drop the unit suffix."""
    for name in DECISION_CHANNEL_NAMES:
        assert DECISION_CHANNEL_UNITS.get(name)


def test_unknown_channel_name_is_rejected():
    with pytest.raises(ValueError, match="unknown response channel"):
        build_channels(["vx", "roll"], {"vx": 8.0, "roll": 5.0}, {"vx": 1.2, "roll": 0.4})


def test_config_survives_a_serialisation_round_trip(cfg):
    """parameters.pkl must carry the reference model, or a checkpoint cannot be
    replayed against the reference it was trained on."""
    from go1_gym.envs.config import cfg_to_dict

    snapshot = cfg_to_dict(cfg)
    assert "response" in snapshot
    assert snapshot["response"]["omega_n"]["pitch"] == 5.0
    assert snapshot["response"]["channel_order"] == list(DECISION_CHANNEL_NAMES)


def test_model_built_from_config_steps(cfg):
    import torch

    model = ReferenceModel(_channels_from_cfg(cfg), num_envs=4, dt=0.02)
    commands_dog = torch.zeros(4, cfg.dog.dog_num_commands)
    commands_dog[:, 3] = 0.3   # body pitch
    commands_dog[:, 5] = 0.1   # body height
    for _ in range(50):
        model.step(model.gather_commands(commands_dog))

    names = list(model.channel_names)
    assert model.xi[0, names.index("pitch")].item() == pytest.approx(0.3, abs=0.02)
    assert model.xi[0, names.index("height")].item() == pytest.approx(0.1, abs=0.02)
    assert model.xi[0, names.index("vx")].item() == 0.0


def test_the_ripple_window_can_resolve_the_gait_band_it_is_watching(cfg):
    """R8.2's diagnostic window has to be long enough for the slowest gait
    frequency R1 samples, or the alarm is silent by construction rather than
    because the reference model is well calibrated."""
    dt = cfg.sim.dt * cfg.control.decimation
    window_s = int(cfg.response.diagnostics.ripple_window_steps) * dt
    slowest_hz = float(cfg.commands.limit_gait_frequency[0])
    assert window_s * slowest_hz >= 4.0
    # And the band has to be wide enough to survive the FFT resolution: the
    # detector floors it at one bin, so this asserts the floor is not carrying
    # the whole check on its own.
    bin_hz = 1.0 / window_s
    assert bin_hz <= 2.0 * float(cfg.response.diagnostics.ripple_tolerance) * slowest_hz
