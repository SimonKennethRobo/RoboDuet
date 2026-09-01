"""R8.1: the four-stage weight schedule.

The property that matters most is the safe direction of every edge case.  Every
consistency term has a measured failure mode at full weight on an untrained
policy -- the reward drops by ~1000x and never recovers -- so "off" has to be
what happens when anything is unclear: before the stage starts, with the
curriculum disabled, or with a term nobody scheduled.
"""

import pytest

from go1_gym.response.curriculum import (
    ResponseCurriculum,
    dominant_frequency,
    gait_frequency_ripple,
)

BOUNDARIES = [3000, 6000, 12000]
TERM_STAGE = {
    "ref_tracking": 2,
    "phase_variance": 3,
    "steady_gain": 3,
    "domain_consistency": 3,
}
SCALES = {
    "ref_tracking": 2.0,
    "phase_variance": -20.0,
    "steady_gain": -0.5,
    "domain_consistency": -3.0,
    "tracking_lin_vel": 1.0,
}


def make(**kwargs):
    options = dict(
        stage_boundaries=BOUNDARIES, ramp_iterations=1000, term_stage=TERM_STAGE
    )
    options.update(kwargs)
    return ResponseCurriculum(**options)


# --- stages ----------------------------------------------------------------


def test_stage_boundaries():
    curriculum = make()
    assert curriculum.num_stages == 4
    for iteration, stage in [
        (0, 1), (2999, 1), (3000, 2), (5999, 2),
        (6000, 3), (11999, 3), (12000, 4), (100000, 4),
    ]:
        assert curriculum.stage(iteration) == stage, iteration


def test_stage_start_is_the_boundary():
    curriculum = make()
    assert curriculum.stage_start(1) == 0
    assert curriculum.stage_start(2) == 3000
    assert curriculum.stage_start(3) == 6000
    assert curriculum.stage_start(4) == 12000


# --- the safe direction ----------------------------------------------------


def test_stage_one_runs_the_original_rewards_only():
    """R4.1 included: R8.2 calibrates from the stage-1 checkpoint, and a policy
    already shaped by reference tracking is not a neutral measurement of it."""
    weights = make().multiplier(0)
    assert weights == {name: 0.0 for name in TERM_STAGE}


def test_disabled_curriculum_freezes_at_stage_one():
    """Disabling must not mean 'every term at full weight from iteration 0' --
    that is the one configuration measured to destroy the reward."""
    curriculum = make(enabled=False)
    assert curriculum.stage(1_000_000) == 1
    assert all(v == 0.0 for v in curriculum.multiplier(1_000_000).values())


def test_a_term_nobody_scheduled_is_untouched():
    scaled = make().apply(SCALES, 20000)
    assert scaled["tracking_lin_vel"] == 1.0


def test_apply_does_not_mutate_the_caller_dict():
    """get_reward_scales() may hand back the trainer's own dict; scaling it in
    place would compound the multiplier once per control step."""
    original = dict(SCALES)
    curriculum = make()
    curriculum.apply(original, 6500)
    assert original == SCALES


# --- ramps -----------------------------------------------------------------


def test_ref_tracking_ramps_over_stage_two():
    curriculum = make()
    assert curriculum.multiplier(2999)["ref_tracking"] == 0.0
    assert curriculum.multiplier(3000)["ref_tracking"] == 0.0
    assert curriculum.multiplier(3500)["ref_tracking"] == pytest.approx(0.5)
    assert curriculum.multiplier(4000)["ref_tracking"] == 1.0
    assert curriculum.multiplier(50000)["ref_tracking"] == 1.0


def test_the_consistency_terms_ramp_together_over_stage_three():
    curriculum = make()
    for iteration, expected in [(5999, 0.0), (6000, 0.0), (6500, 0.5), (7000, 1.0)]:
        weights = curriculum.multiplier(iteration)
        for name in ("phase_variance", "steady_gain", "domain_consistency"):
            assert weights[name] == pytest.approx(expected), (name, iteration)


def test_applied_weights_reach_exactly_the_configured_target():
    """The configured number is the end-of-ramp target, so the ramp must land
    on it rather than near it."""
    scaled = make().apply(SCALES, 7000)
    assert scaled["phase_variance"] == -20.0
    assert scaled["steady_gain"] == -0.5
    assert scaled["domain_consistency"] == -3.0
    assert scaled["ref_tracking"] == 2.0


def test_stage_four_freezes_the_weights():
    curriculum = make()
    assert curriculum.multiplier(12000) == curriculum.multiplier(50000)


def test_zero_ramp_is_a_step():
    curriculum = make(ramp_iterations=0)
    assert curriculum.multiplier(5999)["phase_variance"] == 0.0
    assert curriculum.multiplier(6000)["phase_variance"] == 1.0


# --- reporting -------------------------------------------------------------


def test_report_exposes_the_stage_and_every_weight():
    report = make().report(6500)
    assert report["curriculum_stage"] == 3.0
    assert report["curriculum_weight_phase_variance"] == pytest.approx(0.5)
    assert report["curriculum_weight_ref_tracking"] == 1.0


def test_checkpoint_marks_cover_the_pareto_pair():
    """R8 requires the end of stage 3 and the end of stage 4 as two points on
    the robustness/predictability front."""
    marks = make().checkpoint_iterations()
    assert marks["stage1_end"] == 3000
    assert marks["stage3_end"] == 12000


# --- configuration guards --------------------------------------------------


def test_bad_configuration_rejected():
    with pytest.raises(ValueError, match="ascending"):
        make(stage_boundaries=[3000, 1000])
    with pytest.raises(ValueError, match="non-negative"):
        make(stage_boundaries=[-1, 10])
    with pytest.raises(ValueError, match="ramp_iterations"):
        make(ramp_iterations=-1)
    with pytest.raises(ValueError, match="outside 1..4"):
        make(term_stage={"ref_tracking": 9})


# --- R8.2's diagnostic -----------------------------------------------------


def test_dominant_frequency_finds_a_planted_tone():
    import math

    dt = 0.5
    series = [math.sin(2 * math.pi * 0.1 * k * dt) for k in range(512)]
    assert dominant_frequency(series, dt) == pytest.approx(0.1, abs=0.01)


def test_gait_frequency_ripple_detects_the_documented_signature():
    """R8.2: a reward rippling at the gait frequency means the posture channels
    were calibrated without binning by gait phase."""
    import math

    dt = 0.02
    gait = 3.0
    rippled = [1.0 + 0.2 * math.sin(2 * math.pi * gait * k * dt) for k in range(1024)]
    assert gait_frequency_ripple(rippled, dt, gait)

    clean = [1.0 + 0.2 * math.sin(2 * math.pi * 0.3 * k * dt) for k in range(1024)]
    assert not gait_frequency_ripple(clean, dt, gait)


def test_ripple_detector_is_quiet_on_degenerate_input():
    assert dominant_frequency([1.0, 2.0], 0.02) is None
    assert dominant_frequency([1.0] * 64, 0.02) is None
    assert not gait_frequency_ripple([1.0] * 64, 0.02, 3.0)
    assert not gait_frequency_ripple([1.0, 2.0, 1.0] * 40, 0.02, 0.0)


# --- R8.1's other two columns ----------------------------------------------


def test_randomization_opens_over_stage_three_and_never_starts_at_zero():
    curriculum = make()
    assert curriculum.randomization_intensity(0) == pytest.approx(0.3)
    assert curriculum.randomization_intensity(5999) == pytest.approx(0.3)
    assert curriculum.randomization_intensity(6500) == pytest.approx(0.65)
    assert curriculum.randomization_intensity(7000) == pytest.approx(1.0)
    assert curriculum.randomization_intensity(50000) == pytest.approx(1.0)


def test_the_randomization_floor_is_configurable_and_bounded():
    assert make(randomization_floor=0.0).randomization_intensity(0) == 0.0
    assert make(randomization_floor=1.0).randomization_intensity(0) == 1.0
    with pytest.raises(ValueError, match="randomization_floor"):
        make(randomization_floor=1.5)


def test_disturbance_is_off_until_stage_four():
    """Stage 4 must not overlap the stage-3 weight ramp, or a robustness change
    and a weight change land together and neither can be attributed."""
    curriculum = make()
    for iteration in (0, 6000, 7000, 11999):
        assert curriculum.disturbance_intensity(iteration) == 0.0
    assert curriculum.disturbance_intensity(12500) == pytest.approx(0.5)
    assert curriculum.disturbance_intensity(13000) == pytest.approx(1.0)


def test_the_stage_the_ramps_attach_to_is_validated():
    with pytest.raises(ValueError, match="randomization_stage"):
        make(randomization_stage=0)
    with pytest.raises(ValueError, match="disturbance_stage"):
        make(disturbance_stage=99)


def test_disabling_the_curriculum_freezes_randomization_at_the_floor():
    """Safe direction again: disabled must not mean full randomisation plus
    full disturbance from iteration 0."""
    curriculum = make(enabled=False)
    assert curriculum.randomization_intensity(1_000_000) == pytest.approx(0.3)
    assert curriculum.disturbance_intensity(1_000_000) == 0.0


def test_report_carries_both_intensities():
    report = make().report(12500)
    assert report["curriculum_randomization"] == pytest.approx(1.0)
    assert report["curriculum_disturbance"] == pytest.approx(0.5)
