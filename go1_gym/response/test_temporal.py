"""R7.3: the structural constraints that keep a TCN a drop-in.

Cheap to satisfy now, expensive to retrofit, so pinned here rather than left as
a comment.  The one that matters most is the history layout: an interleaved
buffer would not fail, it would silently make the encoder's reshape wrong.
"""

import pytest
import torch
import torch.nn as nn

from go1_gym.response.temporal import EncoderDefaults, TemporalEncoder

NUM_OBS = 112
STEPS = 50
HIDDEN = 512


class Args(EncoderDefaults):
    pass


def build(mode, **kwargs):
    class Local(EncoderDefaults):
        temporal_encoder = mode
    for key, value in list(kwargs.items()):
        if hasattr(EncoderDefaults, key):
            setattr(Local, key, kwargs.pop(key))
    return TemporalEncoder(
        num_obs=NUM_OBS, num_history_steps=STEPS, out_dim=HIDDEN,
        activation=nn.ELU(), args=Local, **kwargs
    )


def test_public_signature_is_flat_in_flat_out_for_both_modes():
    """The deployment contract is jit.script over one flat tensor, so the
    reshape has to be the encoder's business, not the caller's."""
    x = torch.randn(7, STEPS * NUM_OBS)
    for mode in ("flat", "tcn"):
        assert build(mode)(x).shape == (7, HIDDEN)


def test_dimensions_come_from_config_not_from_literals():
    encoder = TemporalEncoder(num_obs=7, num_history_steps=13, out_dim=5,
                              activation=nn.ELU())
    assert encoder.flat_dim == 91
    assert encoder(torch.randn(3, 91)).shape == (3, 5)


def test_flat_mode_is_exactly_the_linear_it_replaces():
    encoder = build("flat")
    assert isinstance(encoder.body[0], nn.Linear)
    assert encoder.body[0].weight.shape == (HIDDEN, STEPS * NUM_OBS)


def test_tcn_is_far_smaller_than_the_flat_first_layer():
    flat = sum(p.numel() for p in build("flat").parameters())
    tcn = sum(p.numel() for p in build("tcn").parameters())
    assert tcn < flat / 5


def test_tcn_receptive_field_covers_the_history():
    encoder = build("tcn")
    assert encoder.receptive_field >= STEPS
    with pytest.raises(ValueError, match="receptive field"):
        build("tcn", tcn_dilations=[1, 2])


def test_tcn_preserves_sequence_length_so_the_last_step_is_the_last_step():
    encoder = build("tcn").eval()
    x = torch.randn(1, STEPS * NUM_OBS)
    sequence = encoder.body(x.view(1, STEPS, NUM_OBS).transpose(1, 2))
    assert sequence.shape[-1] == STEPS


def test_tcn_is_causal_and_reads_the_newest_step():
    encoder = build("tcn").eval()
    x = torch.randn(1, STEPS * NUM_OBS)
    with torch.no_grad():
        base = encoder(x)
        newest = x.clone()
        newest[0, (STEPS - 1) * NUM_OBS:] += 5.0
        assert not torch.allclose(encoder(newest), base, atol=1e-5)


def test_tcn_rejects_the_adaptation_module_layout():
    """A concatenated privileged vector is not part of the (T, C) grid."""
    with pytest.raises(ValueError, match=r"pure \(T, C\) history"):
        build("tcn", extra_dim=106)


def test_flat_mode_accepts_the_adaptation_module_layout():
    encoder = build("flat", extra_dim=106)
    assert encoder(torch.randn(2, STEPS * NUM_OBS + 106)).shape == (2, HIDDEN)


def test_unknown_encoder_name_is_rejected():
    with pytest.raises(ValueError, match="unknown temporal_encoder"):
        build("lstm")


def test_both_modes_survive_jit_script():
    """R7.3 constraint 3: the export path must not learn which mode is in use."""
    for mode in ("flat", "tcn"):
        scripted = torch.jit.script(build(mode).eval())
        with torch.no_grad():
            assert scripted(torch.randn(2, STEPS * NUM_OBS)).shape == (2, HIDDEN)


# --- R7.3 constraint 1: the history buffer layout ---------------------------


def test_history_ring_buffer_layout_matches_the_encoder_reshape():
    """The constraint the TCN silently depends on.

    ``HistoryWrapper`` maintains the window as
    ``cat((history[:, C:], newest), -1)`` -- oldest step first, each step's C
    values contiguous.  ``TemporalEncoder.view(B, T, C)`` is only correct for
    exactly that layout.  An interleaved buffer (all of channel 0 across time,
    then all of channel 1) would not raise: it would reshape happily and feed
    the convolution garbage.  This reproduces the wrapper's update rule and
    asserts the two agree.
    """
    steps, channels = 4, 3
    history = torch.zeros(1, steps * channels)
    for step in range(steps):
        newest = torch.full((1, channels), float(step + 1))
        history = torch.cat((history[:, channels:], newest), dim=-1)

    grid = history.view(1, steps, channels)
    # newest step last, and each row is one time step
    assert grid[0, -1].tolist() == [4.0, 4.0, 4.0]
    assert grid[0, 0].tolist() == [1.0, 1.0, 1.0]
    # ...and the flat buffer really is step-major, not channel-major
    assert history[0].tolist() == [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3


def test_encoder_reads_the_newest_step_from_the_end_of_the_buffer():
    """Ties the layout above to the encoder: zeroing everything except the
    newest step must still produce a response that depends on it."""
    encoder = build("tcn").eval()
    x = torch.zeros(1, STEPS * NUM_OBS)
    with torch.no_grad():
        quiet = encoder(x)
        x[0, (STEPS - 1) * NUM_OBS:] = 1.0
        assert not torch.allclose(encoder(x), quiet, atol=1e-6)
