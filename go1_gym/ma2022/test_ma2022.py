"""Mathematical and learning-contract checks without IsaacGym."""

from dataclasses import replace

import torch

from go1_gym.envs.config.ma2022 import MaTrainingConfig, build_ma_config
from go1_gym.ma2022.models import Teacher, Student, DeployedStudent, distillation_losses
from go1_gym.ma2022.training import gae, export_student
from go1_gym.ma2022.wrench import WrenchSequence, world_to_body


def fixture():
    cfg = replace(MaTrainingConfig(), hidden_dim=16, embedding_dim=8)
    dims = dict(proprio=76, wrench=39, scan=25, privileged=49)
    obs = {key: torch.randn(3, dim) for key, dim in dims.items()}
    obs["applied_wrench"], obs["gain"] = torch.randn(3, 6), torch.randn(3, 2)
    return cfg, dims, obs


def test_quadratic_interpolation_and_shift():
    sequence = WrenchSequence(3, MaTrainingConfig())
    expected = sequence.knots.clone()
    torch.testing.assert_close(sequence.evaluate([0, 1, 2]), expected)
    shifted = sequence.evaluate([0.02, 1.02])
    sequence.advance()
    torch.testing.assert_close(sequence.knots[:, :2], shifted)
    assert ((sequence.knots[:, 2] >= sequence.low) & (sequence.knots[:, 2] <= sequence.high)).all()


def test_wrench_prediction_body_frame_and_no_privileged_leak():
    cfg = replace(MaTrainingConfig(), prediction_noise_std=0., prediction_bias_std=0., prediction_scale_std=0.)
    sequence = WrenchSequence(1, cfg)
    sequence.knots.zero_()
    sequence.knots[..., 0] = 10
    q = torch.tensor([[0., 0., 2**-0.5, 2**-0.5]])
    prediction = sequence.prediction(q).view(1, 5, 6)
    torch.testing.assert_close(prediction[..., 0], torch.zeros(1, 5), atol=1e-6, rtol=0.)
    torch.testing.assert_close(prediction[..., 1], torch.full((1, 5), -0.5))
    before = sequence.prediction(q)
    sequence.gains.fill_(100)
    torch.testing.assert_close(sequence.prediction(q, noisy=True), before)


def test_episode_reset_is_selective_and_ball_sampling_is_bounded():
    cfg = MaTrainingConfig()
    sequence = WrenchSequence(1000, cfg)
    assert (sequence.gains[:, :3].norm(dim=-1) <= cfg.force_gain_radius + 1e-6).all()
    assert (sequence.gains[:, 3:].norm(dim=-1) <= cfg.torque_gain_radius + 1e-6).all()
    before = sequence.knots.clone()
    sequence.reset(torch.tensor([2, 5]))
    torch.testing.assert_close(sequence.knots[0], before[0])
    assert not torch.equal(sequence.knots[2], before[2])


def test_unobserved_disturbance_acceleration_and_noise():
    cfg = replace(MaTrainingConfig(), disturbance_std=(0.,)*6)
    sequence = WrenchSequence(2, cfg)
    sequence.gains.fill_(2)
    torch.testing.assert_close(sequence.disturbance(torch.ones(2, 6)*3), torch.ones(2, 6)*6)


def test_student_reset_and_decoder_stream_isolation():
    cfg, dims, obs = fixture()
    model = Student(dims, cfg)
    h = torch.randn(3, cfg.hidden_dim)
    reset = torch.tensor([True, False, True])
    a = model(obs["proprio"], obs["wrench"], obs["scan"], h, h, reset)
    b = model(obs["proprio"], obs["wrench"] + 10, obs["scan"], h, h, reset)
    for index in (2, 4, 5, 6, 7, 8):
        torch.testing.assert_close(a[index], b[index])
    assert not torch.equal(a[1], b[1])
    zero = torch.zeros_like(h)
    c = model(obs["proprio"], obs["wrench"], obs["scan"], zero, zero, reset)
    torch.testing.assert_close(a[0][reset], c[0][reset])
    # Decoder loss must have no computational path to wrench_rnn.
    a[7].square().mean().backward()
    assert model.wrench_rnn.weight_ih.grad is None
    assert model.belief_rnn.weight_ih.grad is not None


def test_distillation_all_losses_backpropagate_and_teacher_stays_frozen():
    cfg, dims, obs = fixture()
    teacher = Teacher(dims, cfg).requires_grad_(False)
    student = Student(dims, cfg)
    h = torch.zeros(3, cfg.hidden_dim)
    result = student(obs["proprio"], obs["wrench"], obs["scan"], h, h, torch.ones(3, dtype=torch.bool))
    losses = distillation_losses(result, teacher(obs), obs)
    assert set(losses) == {"action", "embedding", "privileged", "scan", "w1", "w2"}
    sum(losses.values()).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_timeout_bootstrap_and_no_advantage_across_reset():
    rewards = torch.tensor([[1., 1.], [100., 100.]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10., 10.], [0., 0.]])
    dones = torch.ones(2, 2, dtype=torch.bool)
    timeouts = torch.tensor([[True, False], [False, False]])
    advantage, returns = gae(rewards, values, next_values, dones, timeouts, .9, .95)
    torch.testing.assert_close(returns[0], torch.tensor([10., 1.]))
    torch.testing.assert_close(advantage, returns)


def test_torchscript_export_parity_and_state(tmp_path):
    cfg, dims, obs = fixture()
    model = Student(dims, cfg).eval()
    path = tmp_path / "student.pt"
    export_student(model, path)
    loaded = torch.jit.load(str(path))
    h = torch.zeros(3, cfg.hidden_dim)
    inputs = (obs["proprio"], obs["wrench"], obs["scan"], h, h, torch.ones(3, dtype=torch.bool))
    expected = DeployedStudent(model)(*inputs)
    actual = loaded(*inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_config_is_independent_and_disables_arm_and_response_training():
    a, b = build_ma_config(4), build_ma_config(4)
    assert a.env.num_actions == 12 and a.arm.num_actions_arm == 0
    assert not a.env.arm_policy_enabled
    assert not a.response.grouping.enabled
    assert not a.response.excitation.enabled
    assert not a.response.curriculum.enabled
    assert not a.domain_rand.push_robots
    assert "arx" not in a.asset.file
    a.reward_scales.orientation = 123
    assert b.reward_scales.orientation == -5.
