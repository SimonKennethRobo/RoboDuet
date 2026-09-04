"""R5 acceptance, on the grouping primitive alone.

The env-level assertions (same command vector per group, synchronised
``gait_indices``, nominal twin domain parameters) need IsaacGym and live in
``scripts/check_response_runtime.py --check r5``.  What is testable here is the
bookkeeping those assertions rely on: who is whose twin, when a group is due,
and -- the part R5 calls an invariant -- that a reset invalidates the group for
a settling window afterwards.
"""

import pytest
import torch

from go1_gym.response import EnvGrouping


def make(num_envs=64, group_size=4, pool_envs=None, settle_steps=50):
    return EnvGrouping(
        num_envs, group_size=group_size, pool_envs=pool_envs, settle_steps=settle_steps
    )


# --- layout ----------------------------------------------------------------


def test_twin_is_the_group_leader():
    g = make(16)
    assert g.num_groups == 4
    assert g.twin_of.tolist() == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
    assert g.is_twin.nonzero().flatten().tolist() == [0, 4, 8, 12]


def test_partial_trailing_group_is_left_ungrouped():
    """A group whose members do not all exist has no usable twin."""
    g = make(14)
    assert g.num_groups == 3
    assert g.num_grouped == 12
    assert g.is_grouped.tolist() == [True] * 12 + [False] * 2
    assert g.group_of[12:].tolist() == [-1, -1]
    # ungrouped envs are their own "twin", so broadcast is the identity there
    assert g.twin_of[12:].tolist() == [12, 13]


def test_eval_envs_are_excluded_via_pool():
    g = make(20, pool_envs=16)
    assert g.num_grouped == 16
    assert not g.is_grouped[16:].any()
    assert not g.is_twin[16:].any()


def test_group_size_one_makes_everything_its_own_twin():
    g = make(8, group_size=1)
    assert g.is_twin.all()
    assert g.twin_of.tolist() == list(range(8))
    # ...and therefore nothing is ever a valid comparison
    assert g.valid.sum() == 0


def test_bad_group_size_rejected():
    with pytest.raises(ValueError, match="group_size"):
        make(8, group_size=0)
    with pytest.raises(ValueError, match="pool_envs"):
        make(8, pool_envs=9)
    with pytest.raises(ValueError, match="settle_steps"):
        make(8, settle_steps=0)


# --- clock -----------------------------------------------------------------


def test_every_group_is_due_on_the_very_first_step():
    """Otherwise each group holds zero commands for a whole interval."""
    g = make(16)
    assert g.groups_due(500).tolist() == [0, 1, 2, 3]


def test_groups_come_due_on_the_shared_interval():
    g = make(16)
    fired = []
    for step in range(1, 1102):
        due = g.groups_due(500)
        if due.numel():
            assert due.tolist() == [0, 1, 2, 3]
            fired.append(step)
            g.resync(due)
        g.advance()
    assert fired == [1, 501, 1001]


def test_resync_restarts_the_clock_exactly():
    """Interval must stay exact across resyncs, not drift by a step."""
    g = make(8)
    fired = []
    for step in range(1, 62):
        due = g.groups_due(20)
        if due.numel():
            fired.append(step)
            g.resync(due)
        g.advance()
    assert fired == [1, 21, 41, 61]


def test_envs_of_groups_expands_and_sorts():
    g = make(16)
    assert g.envs_of_groups(torch.tensor([2, 0])).tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert g.envs_of_groups(torch.zeros(0, dtype=torch.long)).numel() == 0


# --- the R5 invariant: termination asymmetry -------------------------------


def test_a_fall_invalidates_the_whole_group_for_the_settling_window():
    g = make(16, settle_steps=5)
    assert g.valid[1:4].tolist() == [1.0, 1.0, 1.0]

    g.mark_desync(torch.tensor([2]))          # one member falls
    assert g.valid[0:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert g.valid[4:8].tolist() == [0.0, 1.0, 1.0, 1.0]   # neighbours unaffected

    for _ in range(4):
        g.advance()
        assert g.valid[1:4].tolist() == [0.0, 0.0, 0.0]
    g.advance()
    assert g.valid[1:4].tolist() == [1.0, 1.0, 1.0]


def test_a_resample_does_not_cut_the_settling_window_short():
    """The command is re-adopted at the reset instant; what the window waits out
    is the robot's own start-up transient, which a resample does nothing about.
    """
    g = make(16, settle_steps=5)
    g.mark_desync(torch.tensor([2]))
    g.advance()
    g.resync(torch.tensor([0]))               # shared resample lands mid-window
    assert g.valid[1:4].tolist() == [0.0, 0.0, 0.0]
    assert g.group_clock[0] == 0              # ...but the clock did restart
    for _ in range(4):
        g.advance()
    assert g.valid[1:4].tolist() == [1.0, 1.0, 1.0]


def test_a_second_reset_restarts_the_window_rather_than_accumulating():
    g = make(16, settle_steps=5)
    g.mark_desync(torch.tensor([1]))
    for _ in range(3):
        g.advance()
    g.mark_desync(torch.tensor([3]))          # a second member falls
    for _ in range(4):
        g.advance()
        assert g.valid[1:4].tolist() == [0.0, 0.0, 0.0]
    g.advance()
    assert g.valid[1:4].tolist() == [1.0, 1.0, 1.0]


def test_availability_does_not_depend_on_the_resample_period():
    """The point of the change: the mask's duty cycle is set by settle_steps and
    the reset rate, not by how often the group happens to resample."""
    opened = []
    for interval in (20, 500):
        g = make(16, settle_steps=5)
        open_steps = 0
        for step in range(1, 401):
            due = g.groups_due(interval)
            if due.numel():
                g.resync(due)
            if step % 50 == 0:                # one env in the group resets
                g.mark_desync(torch.tensor([2]))
            open_steps += int(g.valid[1:4].sum().item() > 0)
            g.advance()
        opened.append(open_steps)
    assert opened[0] == opened[1]


def test_the_twin_falling_invalidates_its_group():
    """R5: the twin is the alignment target, so losing it loses the group."""
    g = make(16)
    g.mark_desync(torch.tensor([4]))          # the twin of group 1
    assert g.valid[4:8].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert g.valid[9] == 1.0


def test_the_twin_never_scores_against_itself():
    g = make(16)
    assert g.valid[g.is_twin].sum() == 0
    assert g.valid.sum() == 12   # 4 groups x 3 non-twin members


def test_ungrouped_envs_never_score():
    g = make(20, pool_envs=16)
    assert g.valid[16:].sum() == 0


def test_marking_an_ungrouped_env_is_a_no_op():
    g = make(20, pool_envs=16)
    g.mark_desync(torch.tensor([17, 19]))
    assert not g.group_desync.any()
    assert g.valid.sum() == 12


def test_mark_desync_accepts_an_empty_selection():
    g = make(16)
    g.mark_desync(torch.zeros(0, dtype=torch.long))
    assert not g.group_desync.any()


# --- broadcast -------------------------------------------------------------


def test_broadcast_replaces_members_with_their_twin():
    g = make(12)
    commands = torch.arange(12, dtype=torch.float).unsqueeze(-1).repeat(1, 3)
    shared = g.broadcast_from_twin(commands)
    assert shared[:, 0].tolist() == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8]
    # every group is now internally identical, dimension by dimension
    for group in range(3):
        block = shared[group * 4:(group + 1) * 4]
        assert torch.equal(block, block[0].expand_as(block))


def test_broadcast_leaves_ungrouped_rows_alone():
    g = make(10, pool_envs=8)
    values = torch.arange(10, dtype=torch.float)
    shared = g.broadcast_from_twin(values)
    assert shared[8:].tolist() == [8.0, 9.0]


def test_broadcast_handles_extra_dimensions():
    g = make(8)
    values = torch.randn(8, 5, 2)
    shared = g.broadcast_from_twin(values)
    assert shared.shape == values.shape
    assert torch.equal(shared[3], values[0])
    assert torch.equal(shared[7], values[4])
