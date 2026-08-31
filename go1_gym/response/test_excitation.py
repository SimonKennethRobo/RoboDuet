"""R6 acceptance: rich excitation signals.

The criteria being pinned here, from
``docs/project-design-rlmpc-v3-coding.md``:

* identification envs see >= 20 command jumps per episode;
* the chirp env's command spectrum covers 0.1-2.0 Hz;
* the posture channels get the larger share of the excitation budget.

Plus the two invariants that are cheap to state and expensive to discover
later: the excited channel is the *only* one this object writes, and the
evaluation environments are never excited.
"""

import math

import pytest
import torch

from go1_gym.response import DEFAULT_CHANNELS, ExcitationSampler
from go1_gym.response.excitation import CHIRP, PRBS, RAMP

DT = 0.02
EPISODE_STEPS = 1000  # 20 s
CHANNEL_LOW = [-0.5, -0.3, -1.0, -0.2, -0.4]
CHANNEL_HIGH = [0.5, 0.3, 1.0, 0.3, 0.4]


def make(num_envs=256, seed=0, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        low=CHANNEL_LOW,
        high=CHANNEL_HIGH,
        env_fraction=0.25,
        signal_weights={"prbs": 0.5, "chirp": 0.3, "ramp": 0.2},
        channel_weights={"vx": 0.15, "vy": 0.15, "wyaw": 0.15, "height": 0.275, "pitch": 0.275},
        prbs_hold_s=(0.5, 3.0),
        chirp_hz=(0.1, 2.0),
        chirp_duration_s=20.0,
        ramp_slope_multiple=(0.2, 3.0),
        generator=generator,
    )
    options.update(kwargs)
    return ExcitationSampler(DEFAULT_CHANNELS, num_envs, DT, **options)


def rollout(sampler, steps=EPISODE_STEPS, num_commands=11):
    """Run one episode; return the command history and the per-env jump count."""
    commands = torch.zeros(sampler.num_envs, num_commands)
    history = torch.empty(steps, sampler.num_envs, num_commands)
    jumps = torch.zeros(sampler.num_envs)
    for k in range(steps):
        jumps += sampler.step(commands).float()
        history[k] = commands
    return history, jumps


# --- environment partition -------------------------------------------------


def test_identification_block_is_the_tail_of_the_training_pool():
    """Eval envs live past ``num_train_envs`` and must never be excited."""
    sampler = make(num_envs=100, pool_envs=80)
    ids = sampler.identification_ids
    assert ids.numel() == 20
    assert int(ids.min()) == 60 and int(ids.max()) == 79
    assert not sampler.is_identification[80:].any()


def test_single_env_build_is_a_no_op():
    """A play/eval build must not need a special case to turn this off."""
    sampler = make(num_envs=1)
    assert not sampler.active
    commands = torch.zeros(1, 11)
    assert not sampler.step(commands).any()
    assert torch.equal(commands, torch.zeros(1, 11))


def test_env_fraction_is_honoured():
    for fraction in (0.0, 0.1, 0.25, 0.5):
        sampler = make(num_envs=400, env_fraction=fraction)
        assert sampler.identification_ids.numel() == int(400 * fraction)


# --- R6 acceptance: jump count ---------------------------------------------


def test_chirp_and_ramp_envs_change_command_every_step():
    sampler = make(num_envs=512)
    _, jumps = rollout(sampler)
    ids = sampler.identification_ids
    signal = sampler.signal[ids]
    continuous = ids[(signal == CHIRP) | (signal == RAMP)]
    assert continuous.numel() > 0
    assert torch.all(jumps[continuous] == EPISODE_STEPS)


def test_identification_group_clears_twenty_jumps_per_episode():
    """The R6 acceptance gate, read at the level of the identification group."""
    sampler = make(num_envs=512)
    _, jumps = rollout(sampler)
    assert float(jumps[sampler.identification_ids].mean()) >= 20.0


def test_prbs_switch_count_matches_the_specified_hold_interval():
    """The one place the requirements document contradicts itself.

    The table asks for a hold of ``U(0.5, 3.0)`` s and the acceptance criterion
    asks for >= 20 jumps in a 20 s episode.  Mean hold is 1.75 s, so a PRBS env
    gets ~11.4 switches -- the two cannot both hold.  The interval is
    implemented as specified; this test pins the resulting count so the
    discrepancy is a recorded number rather than a surprise, and fails loudly if
    the hold distribution is ever changed without revisiting it.
    """
    sampler = make(num_envs=1024)
    _, jumps = rollout(sampler)
    ids = sampler.identification_ids
    prbs = ids[sampler.signal[ids] == PRBS]
    assert prbs.numel() > 50
    mean_switches = float(jumps[prbs].mean())
    expected = EPISODE_STEPS * DT / 1.75
    assert mean_switches == pytest.approx(expected, rel=0.15)
    assert mean_switches < 20.0  # documented shortfall, not a passing gate


def test_narrowing_the_hold_interval_clears_twenty():
    """The one-line fix, verified, so the trade-off is a config choice."""
    sampler = make(num_envs=1024, prbs_hold_s=(0.5, 1.5))
    _, jumps = rollout(sampler)
    ids = sampler.identification_ids
    prbs = ids[sampler.signal[ids] == PRBS]
    assert float(jumps[prbs].mean()) >= 20.0


# --- R6 acceptance: chirp spectrum -----------------------------------------


def test_chirp_sweeps_the_full_band():
    """Instantaneous frequency must span 0.1-2.0 Hz over one episode."""
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    chirp = ids[sampler.signal[ids] == CHIRP]
    assert chirp.numel() > 0
    history, _ = rollout(sampler)

    env = int(chirp[0])
    column = int(sampler.cmd_index[sampler.channel[env]])
    signal = history[:, env, column] - float(sampler.center[env])

    # Zero crossings give the realised frequency sweep directly and, unlike an
    # FFT, do not smear a nonstationary signal across the whole band.
    sign = torch.sign(signal)
    crossings = (sign[1:] * sign[:-1] < 0).nonzero().flatten().float() * DT
    assert crossings.numel() > 10
    # The sweep starts at sin(0) = 0, which this test's sign-change detector
    # does not count, so the first *measured* half-cycle is already the second
    # one.  The low-frequency end therefore has to be read off the time the
    # first crossing takes to arrive: half a cycle of a sweep starting at
    # 0.1 Hz lands at 2.36 s, against 0.9 s if the sweep started at 0.5 Hz.
    assert float(crossings[0]) > 2.0
    half_periods = crossings[1:] - crossings[:-1]
    instantaneous_hz = 0.5 / half_periods
    assert float(instantaneous_hz.min()) < 0.4
    assert float(instantaneous_hz.max()) > 1.8


def test_chirp_energy_covers_the_band_in_the_fft():
    """The acceptance criterion as literally written: FFT coverage."""
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    chirp = ids[sampler.signal[ids] == CHIRP]
    history, _ = rollout(sampler)

    env = int(chirp[0])
    column = int(sampler.cmd_index[sampler.channel[env]])
    signal = history[:, env, column] - float(sampler.center[env])

    spectrum = torch.fft.rfft(signal).abs()
    freqs = torch.fft.rfftfreq(signal.numel(), d=DT)
    in_band = (freqs >= 0.1) & (freqs <= 2.0)
    assert float(spectrum[in_band].sum() / spectrum.sum()) > 0.9
    # every octave of the band carries energy, i.e. no gaps in the sweep
    for lo in (0.1, 0.25, 0.5, 1.0):
        octave = (freqs >= lo) & (freqs < 2 * lo)
        assert float(spectrum[octave].max()) > 0.02 * float(spectrum.max())


def test_chirp_never_asks_the_reference_model_to_saturate():
    """Constant-slew taper: |du/dt| stays under the channel's rate limit."""
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    chirp = ids[sampler.signal[ids] == CHIRP]
    history, _ = rollout(sampler)
    for env in chirp[:20].tolist():
        channel = int(sampler.channel[env])
        column = int(sampler.cmd_index[channel])
        signal = history[:, env, column]
        rate = (signal[1:] - signal[:-1]).abs() / DT
        assert float(rate.max()) <= float(sampler.rate_limit[channel]) * 1.01


# --- ramp ------------------------------------------------------------------


def test_ramp_slope_is_within_the_configured_multiple_of_the_rate_limit():
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    ramp = ids[sampler.signal[ids] == RAMP]
    assert ramp.numel() > 0
    history, _ = rollout(sampler)
    for env in ramp[:20].tolist():
        channel = int(sampler.channel[env])
        column = int(sampler.cmd_index[channel])
        limit = float(sampler.rate_limit[channel])
        rate = ((history[1:, env, column] - history[:-1, env, column]).abs() / DT)
        moving = rate[rate > 1e-9]
        assert float(moving.max()) <= 3.0 * limit * 1.01
        assert float(moving.min()) >= 0.2 * limit * 0.99


def test_ramp_reflects_instead_of_parking_at_the_limit():
    """A clamped ramp stops exciting; the sample starvation R6 exists to fix."""
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    ramp = ids[sampler.signal[ids] == RAMP]
    history, _ = rollout(sampler)
    for env in ramp[:20].tolist():
        column = int(sampler.cmd_index[sampler.channel[env]])
        trace = history[:, env, column]
        # last 20% of the episode must still be moving
        tail = (trace[-200:] - trace[-201:-1]).abs()
        assert float(tail.max()) > 1e-6


# --- invariants ------------------------------------------------------------


def test_only_the_excited_channel_is_written():
    """SISO invariant: the other four decision channels are untouched."""
    sampler = make(num_envs=256)
    baseline = torch.randn(sampler.num_envs, 11)
    commands = baseline.clone()
    for _ in range(200):
        sampler.step(commands)
    ids = sampler.identification_ids
    excited_column = sampler.cmd_index[sampler.channel]
    for env in ids.tolist():
        for column in range(11):
            if column == int(excited_column[env]):
                continue
            assert commands[env, column] == baseline[env, column]


def test_non_identification_envs_are_never_touched():
    sampler = make(num_envs=256)
    baseline = torch.randn(sampler.num_envs, 11)
    commands = baseline.clone()
    for _ in range(200):
        sampler.step(commands)
    passive = ~sampler.is_identification
    assert torch.equal(commands[passive], baseline[passive])


def test_commands_stay_inside_the_configured_band():
    sampler = make(num_envs=512)
    history, _ = rollout(sampler)
    ids = sampler.identification_ids
    for env in ids.tolist():
        channel = int(sampler.channel[env])
        column = int(sampler.cmd_index[channel])
        trace = history[:, env, column]
        assert float(trace.min()) >= CHANNEL_LOW[channel] - 1e-5
        assert float(trace.max()) <= CHANNEL_HIGH[channel] + 1e-5


def test_posture_channels_get_the_larger_share():
    """R6: body height and pitch are the channels the original setup starves."""
    sampler = make(num_envs=4096)
    counts = sampler.channel_counts()
    posture = counts["height"] + counts["pitch"]
    velocity = counts["vx"] + counts["vy"] + counts["wyaw"]
    assert posture > velocity
    assert min(counts["height"], counts["pitch"]) > max(
        counts["vx"], counts["vy"], counts["wyaw"]
    )


def test_signal_mix_matches_the_configured_weights():
    sampler = make(num_envs=8192)
    counts = sampler.signal_counts()
    total = sum(counts.values())
    assert counts["prbs"] / total == pytest.approx(0.5, abs=0.03)
    assert counts["chirp"] / total == pytest.approx(0.3, abs=0.03)
    assert counts["ramp"] / total == pytest.approx(0.2, abs=0.03)


def test_plan_only_redraws_the_requested_envs():
    sampler = make(num_envs=256)
    before = sampler.signal.clone(), sampler.channel.clone()
    subset = sampler.identification_ids[:5]
    sampler.plan(subset)
    untouched = torch.ones(sampler.num_envs, dtype=torch.bool)
    untouched[subset] = False
    assert torch.equal(sampler.signal[untouched], before[0][untouched])
    assert torch.equal(sampler.channel[untouched], before[1][untouched])


def test_plan_ignores_non_identification_envs():
    sampler = make(num_envs=256)
    signal_before = sampler.signal.clone()
    sampler.plan(torch.arange(0, 10))
    assert torch.equal(sampler.signal, signal_before)


def test_chirp_restarts_the_sweep_on_plan():
    sampler = make(num_envs=256)
    for _ in range(300):
        sampler.step(torch.zeros(sampler.num_envs, 11))
    ids = sampler.identification_ids
    assert float(sampler.chirp_t[ids].max()) > 0
    sampler.plan(ids)
    assert float(sampler.chirp_t[ids].max()) == 0.0


# --- configuration guards --------------------------------------------------


def test_chirp_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        make(chirp_hz=(0.1, 30.0))


def test_bad_bands_are_rejected():
    with pytest.raises(ValueError, match="high > low"):
        make(low=[0.5] * 5, high=[0.5] * 5)
    with pytest.raises(ValueError, match="one entry per channel"):
        make(low=[0.0, 1.0], high=[1.0, 2.0])


def test_unknown_or_degenerate_weights_are_rejected():
    with pytest.raises(ValueError, match="no channel weight given"):
        make(channel_weights={"vx": 1.0})
    with pytest.raises(ValueError, match="all zero"):
        make(signal_weights={"prbs": 0.0, "chirp": 0.0, "ramp": 0.0})


def test_env_fraction_bounds():
    with pytest.raises(ValueError, match="env_fraction"):
        make(env_fraction=1.0)
    with pytest.raises(ValueError, match="env_fraction"):
        make(env_fraction=-0.1)


def test_determinism_under_a_seeded_generator():
    a, _ = rollout(make(num_envs=128, seed=7), steps=200)
    b, _ = rollout(make(num_envs=128, seed=7), steps=200)
    assert torch.equal(a, b)
    c, _ = rollout(make(num_envs=128, seed=8), steps=200)
    assert not torch.equal(a, c)


def test_chirp_phase_is_continuous_across_the_sweep():
    """No discontinuity where the instantaneous frequency changes."""
    sampler = make(num_envs=256)
    ids = sampler.identification_ids
    chirp = ids[sampler.signal[ids] == CHIRP]
    history, _ = rollout(sampler)
    env = int(chirp[0])
    column = int(sampler.cmd_index[sampler.channel[env]])
    trace = history[:, env, column]
    jump = (trace[1:] - trace[:-1]).abs().max()
    amplitude = float(sampler.amplitude[env])
    # a 2 Hz sine at dt=0.02 moves at most sin(2*pi*2*0.02) = 25% of amplitude
    assert float(jump) < 0.3 * amplitude + 1e-6
    assert math.isfinite(float(jump))


# --- R5 interaction: group-shared excitation --------------------------------


def test_block_multiple_rounds_the_identification_block_to_whole_groups():
    """Half a group excited and half not would break R5's shared command."""
    sampler = make(num_envs=100, pool_envs=100, env_fraction=0.25, block_multiple=4)
    assert sampler.identification_ids.numel() == 24     # 25 -> 24
    assert int(sampler.identification_ids.min()) % 4 == 0
    with pytest.raises(ValueError, match="block_multiple"):
        make(block_multiple=0)


def _group_share(num_envs, group_size=4):
    index = torch.arange(num_envs)
    return (index // group_size) * group_size


def test_shared_envs_receive_an_identical_command_every_step():
    share = _group_share(256)
    sampler = make(num_envs=256, block_multiple=4, share_with=share)
    history, _ = rollout(sampler, steps=400)
    ids = sampler.identification_ids
    for env in ids.tolist():
        leader = int(share[env])
        assert torch.equal(history[:, env], history[:, leader]), f"env {env} drifted"


def test_sharing_survives_a_mid_rollout_replan():
    share = _group_share(256)
    sampler = make(num_envs=256, block_multiple=4, share_with=share)
    commands = torch.zeros(256, 11)
    for step in range(300):
        if step == 150:
            sampler.plan(sampler.identification_ids)
        sampler.step(commands)
        if step > 150:
            for env in sampler.identification_ids.tolist():
                assert commands[env].tolist() == commands[int(share[env])].tolist()


def test_sharing_leaves_the_signal_mix_intact():
    """Members inherit the leader's signal, so counts collapse onto leaders."""
    share = _group_share(4096)
    sampler = make(num_envs=4096, block_multiple=4, share_with=share)
    ids = sampler.identification_ids
    leaders = ids[sampler.is_identification[ids] & (share[ids] == ids)]
    assert leaders.numel() * 4 == ids.numel()
    for env in ids.tolist():
        assert int(sampler.signal[env]) == int(sampler.signal[int(share[env])])
        assert int(sampler.channel[env]) == int(sampler.channel[int(share[env])])


def test_jump_mask_is_shared_too():
    """R4.2/4.3's settle counter must reset for the whole group at once."""
    share = _group_share(64)
    sampler = make(num_envs=64, block_multiple=4, share_with=share)
    commands = torch.zeros(64, 11)
    for _ in range(200):
        jumped = sampler.step(commands)
        for env in sampler.identification_ids.tolist():
            assert bool(jumped[env]) == bool(jumped[int(share[env])])


def test_bad_share_with_shape_rejected():
    with pytest.raises(ValueError, match="one entry per env"):
        make(num_envs=64, share_with=torch.arange(32))


def test_plan_generation_distinguishes_a_redraw_of_the_same_signal():
    """(signal, channel) is not a plan identity; the counter is."""
    sampler = make(num_envs=64)
    ids = sampler.identification_ids
    before = sampler.plan_generation[ids].clone()
    sampler.plan(ids)
    assert torch.all(sampler.plan_generation[ids] == before + 1)
    # forcing an identical redraw must still change the generation
    sampler.signal[ids] = PRBS
    sampler.channel[ids] = 0
    marker = sampler.plan_generation[ids].clone()
    sampler.plan(ids)
    assert torch.all(sampler.plan_generation[ids] > marker)


def test_plan_generation_is_shared_within_a_group():
    share = _group_share(64)
    sampler = make(num_envs=64, block_multiple=4, share_with=share)
    sampler.plan(sampler.identification_ids)
    for env in sampler.identification_ids.tolist():
        assert int(sampler.plan_generation[env]) == int(
            sampler.plan_generation[int(share[env])]
        )
