"""Unit tests for FlashSAC learner and actor interfaces."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from uni_rl.algos.flash_sac import learner as flash_sac_module
from uni_rl.algos.flash_sac.learner import FlashSACLearner, RewardNormalizer
from uni_rl.algos.flash_sac.update import compute_categorical_td_target
from uni_rl.logging.metric_schema import normalize_metric_map


def _make_batch(batch_size: int = 32) -> dict[str, torch.Tensor]:
    obs = torch.randn(batch_size, 98)
    critic = torch.randn(batch_size, 101)
    actions = torch.tanh(torch.randn(batch_size, 29))
    rewards = torch.randn(batch_size)
    next_obs = torch.randn(batch_size, 98)
    next_critic = torch.randn(batch_size, 101)
    dones = torch.zeros(batch_size)
    truncated = torch.zeros(batch_size)
    return {
        "obs": obs,
        "critic": critic,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "next_critic": next_critic,
        "dones": dones,
        "truncated": truncated,
    }


def _make_small_learner(**kwargs: Any) -> FlashSACLearner:
    defaults = {
        "obs_dim": 4,
        "action_dim": 2,
        "critic_obs_dim": 6,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
        "num_atoms": 5,
        "device": "cpu",
        "use_compile": False,
    }
    defaults.update(kwargs)
    return FlashSACLearner(**defaults)


def _make_small_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(batch_size, 4),
        "critic": torch.randn(batch_size, 6),
        "actions": torch.tanh(torch.randn(batch_size, 2)),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, 4),
        "next_critic": torch.randn(batch_size, 6),
        "dones": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }


def test_flashsac_learner_exposes_expected_dims():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")

    assert learner.obs_dim == 98
    assert learner.critic_obs_dim == 101
    assert learner.action_dim == 29


def test_flashsac_learner_declares_inference_startup_scratch():
    learner = _make_small_learner()

    assert learner.inference_startup_memory_categories(2048) == {
        "persistent_exploration_scratch": 2048 * (2 * 4 + 2 * 4)
    }


def test_flashsac_cuda_adam_optimizers_are_capturable(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only optimizer kwargs require a CUDA-enabled torch build")

    calls: list[dict[str, Any]] = []

    class _FakeAdam:
        def __init__(self, _params, **kwargs):
            calls.append(kwargs)
            self.param_groups = [{"lr": kwargs["lr"]}]

    class _FakeLambdaLR:
        def __init__(self, optimizer, lr_lambda):
            self.optimizer = optimizer
            self.lr_lambda = lr_lambda

    monkeypatch.setattr(torch.optim, "Adam", _FakeAdam)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", _FakeLambdaLR)
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: lambda fn: fn,
    )
    monkeypatch.setattr(
        FlashSACLearner, "_materialize_capturable_optimizer_state", lambda _self: None
    )
    monkeypatch.setattr(FlashSACLearner, "_compile_training_methods", lambda _self: None)

    FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cuda", use_compile=True)

    assert len(calls) == 3
    assert all(call["fused"] for call in calls)
    assert all(call["capturable"] for call in calls)


def test_flashsac_gradient_sync_leaves_whole_cycle_mode_for_dp() -> None:
    learner = _make_small_learner()
    learner._compile_full_update_cycle = True

    learner.set_gradient_sync(lambda _parameters: None)

    assert learner._compile_full_update_cycle is False
    assert learner.use_update_cycle is False


def test_flashsac_update_cycle_rejects_compatibility_fallback() -> None:
    learner = _make_small_learner()
    batch = _make_small_batch(8)

    with pytest.raises(RuntimeError, match="requires the NVIDIA CUDA whole-cycle path"):
        learner.update_cycle(
            batch,
            updates_per_step=1,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=False,
        )


def test_flashsac_update_cycle_eager_fallback_after_capture_failure() -> None:
    """Once capture has failed, update_cycle runs the core eagerly (no graph)."""
    torch.manual_seed(123)
    learner = _make_small_learner()
    learner._compile_full_update_cycle = True
    learner._update_cycle_capture_failed = True
    torch.manual_seed(123)
    reference = _make_small_learner()
    large_batch = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        for key, value in _make_small_batch(8).items()
    }
    rng_state = torch.random.get_rng_state()

    learner.update_cycle(
        large_batch,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )

    assert learner._update_cycle_graph is None
    assert learner._pending_cycle_metric_values is not None
    torch.random.set_rng_state(rng_state)
    reference._run_update_cycle_core(
        large_batch,
        updates_per_step=2,
        policy_frequency=1,
        target_frequency=1,
        policy_before_critic=False,
    )
    for module_name in ("actor", "critic", "target_critic", "temperature"):
        torch.testing.assert_close(
            getattr(learner, module_name).state_dict(),
            getattr(reference, module_name).state_dict(),
        )
    assert learner.critic_scheduler.last_epoch == reference.critic_scheduler.last_epoch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture failure path required")
def test_flashsac_capture_failure_warns_and_falls_back(monkeypatch) -> None:
    """A failed CUDA graph capture warns once and completes the cycle eagerly."""

    class _FailingGraph:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> None:
            raise torch.AcceleratorError("simulated capture failure")

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "graph", _FailingGraph)
    learner = _make_small_learner(device="cuda:0")
    learner._compile_full_update_cycle = True
    large_batch = {
        key: value.repeat(2, *([1] * (value.ndim - 1))).cuda()
        for key, value in _make_small_batch(8).items()
    }

    with pytest.warns(UserWarning, match="falling back to eager"):
        learner.update_cycle(
            large_batch,
            updates_per_step=2,
            policy_frequency=1,
            target_frequency=1,
            policy_before_critic=False,
        )

    assert learner._update_cycle_capture_failed is True
    assert learner._update_cycle_graph is None
    assert learner._pending_cycle_metric_values is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_flashsac_nvidia_cuda_honors_compile_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: pytest.fail("explicit opt-out queried torch.compile"),
    )
    monkeypatch.setattr(
        FlashSACLearner, "_materialize_capturable_optimizer_state", lambda _self: None
    )
    monkeypatch.setattr(FlashSACLearner, "_compile_training_methods", lambda _self: None)

    learner = _make_small_learner(
        device="cuda:0",
        use_compile=False,
        compile_full_objectives=True,
        use_whole_cycle_cuda_graph=True,
    )

    assert learner.use_compile is False
    assert learner.compile_full_objectives is False
    assert learner.use_update_cycle is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_flashsac_nvidia_cuda_keeps_accelerated_default(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: lambda fn, **_compile_kwargs: fn,
    )
    monkeypatch.setattr(
        FlashSACLearner, "_materialize_capturable_optimizer_state", lambda _self: None
    )
    monkeypatch.setattr(FlashSACLearner, "_compile_training_methods", lambda _self: None)

    learner = _make_small_learner(device="cuda:0", use_compile=None)

    assert learner.use_compile is True
    assert learner.compile_full_objectives is True
    assert learner.use_update_cycle is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_flashsac_nvidia_cuda_graph_opt_out_keeps_objective_compile(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: lambda fn, **_compile_kwargs: fn,
    )
    monkeypatch.setattr(FlashSACLearner, "_compile_training_methods", lambda _self: None)

    learner = _make_small_learner(
        device="cuda:0",
        use_compile=True,
        compile_full_objectives=True,
        use_whole_cycle_cuda_graph=False,
    )

    assert learner.use_compile is True
    assert learner.compile_full_objectives is True
    assert learner.use_update_cycle is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_flashsac_whole_cycle_does_not_override_compile_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: lambda fn, **_compile_kwargs: fn,
    )
    monkeypatch.setattr(
        FlashSACLearner, "_materialize_capturable_optimizer_state", lambda _self: None
    )
    monkeypatch.setattr(FlashSACLearner, "_compile_training_methods", lambda _self: None)

    learner = _make_small_learner(
        device="cuda:0",
        use_compile=True,
        compile_full_objectives=False,
        use_whole_cycle_cuda_graph=True,
    )

    assert learner.use_compile is True
    assert learner.compile_full_objectives is False
    assert learner.use_update_cycle is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime selection required")
def test_flashsac_hip_cuda_keeps_compatible_compile_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "simulated-rocm", raising=False)

    learner = _make_small_learner(device="cuda:0", use_compile=False)

    assert learner.use_compile is False
    assert learner.use_update_cycle is False
    learner.set_gradient_sync(lambda _parameters: None)
    assert learner._gradient_sync is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA runtime required")
def test_flashsac_nvidia_cuda_fails_closed_without_inductor(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module, "get_torch_compile_for_cuda", lambda *_args, **_kwargs: None
    )

    with pytest.raises(RuntimeError, match="requires CUDA Inductor/Triton"):
        FlashSACLearner(
            obs_dim=4,
            action_dim=2,
            critic_obs_dim=6,
            device="cuda:0",
            use_compile=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA runtime required")
def test_flashsac_nvidia_cuda_fails_closed_for_graph_incompatible_options(monkeypatch) -> None:
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: lambda fn: fn,
    )

    with pytest.raises(ValueError, match="fp16 GradScaler"):
        _make_small_learner(device="cuda:0", use_compile=True, use_amp=True, amp_dtype="fp16")

    with pytest.raises(ValueError, match="obs normalization"):
        _make_small_learner(device="cuda:0", use_compile=True, obs_normalization=True)


def test_flashsac_obs_normalizer_uses_local_batch_moments() -> None:
    learner = _make_small_learner(obs_normalization=True)
    learner._update_obs_normalizer(
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        )
    )

    normalizer = learner.obs_normalizer
    assert not isinstance(normalizer, torch.nn.Identity)
    torch.testing.assert_close(normalizer.count, torch.tensor(2))
    torch.testing.assert_close(
        normalizer.mean,
        torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )


def test_flashsac_compile_targets_training_hot_paths(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    learner.device = torch.device("cuda")
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACActor.get_mean_and_std",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "FlashSACLearner._critic_loss_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "FlashSACLearner._actor_loss_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
    ]


def test_flashsac_compile_full_objectives_targets_complete_forward_graphs(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _make_small_learner()
    learner.device = torch.device("cuda")
    learner.compile_full_objectives = True
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACLearner._critic_objective_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
        (
            "FlashSACLearner._actor_objective_tensors",
            {"dynamic": False, "options": {"triton.cudagraphs": True}},
        ),
    ]


def test_flashsac_whole_cycle_uses_max_autotune_without_nested_graphs(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn: Callable, **kwargs):
        calls.append((fn.__qualname__, kwargs))
        return fn

    learner = _make_small_learner()
    learner.device = torch.device("cuda")
    learner.compile_full_objectives = True
    learner._compile_full_update_cycle = True
    monkeypatch.setattr(
        flash_sac_module,
        "get_torch_compile_for_cuda",
        lambda *_args, **_kwargs: fake_compile,
    )

    learner._compile_training_methods()

    assert calls == [
        (
            "FlashSACLearner._critic_objective_tensors",
            {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        ),
        (
            "FlashSACLearner._actor_objective_tensors",
            {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        ),
    ]


def test_flashsac_amp_dtype_resolution_and_scaler_rules() -> None:
    assert FlashSACLearner._resolve_amp_dtype("auto", "cuda") is torch.bfloat16
    assert FlashSACLearner._resolve_amp_dtype("auto", "xpu") is torch.bfloat16
    assert FlashSACLearner._resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert FlashSACLearner._resolve_amp_dtype("bf16", "cuda") is torch.bfloat16

    assert FlashSACLearner._should_use_grad_scaler(True, "cuda", torch.float16)
    assert not FlashSACLearner._should_use_grad_scaler(True, "cuda", torch.bfloat16)
    assert not FlashSACLearner._should_use_grad_scaler(True, "xpu", torch.bfloat16)
    assert not FlashSACLearner._should_use_grad_scaler(False, "cuda", torch.float16)

    with pytest.raises(ValueError, match="amp_dtype"):
        FlashSACLearner._resolve_amp_dtype("tf32", "cuda")


def test_flashsac_actor_explore_and_forward_shapes():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    obs = torch.randn(4, 98)

    actions = learner.actor.explore(obs, deterministic=False)
    deterministic_actions = learner.actor.explore(obs, deterministic=True)
    sampled_actions, info = learner.actor(obs, training=True)

    assert actions.shape == (4, 29)
    assert deterministic_actions.shape == (4, 29)
    assert sampled_actions.shape == (4, 29)
    assert info["log_prob"].shape == (4,)


def test_flashsac_export_module_matches_deterministic_policy():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    obs = torch.randn(4, 98)

    export_module = learner.actor.as_export_module()

    with torch.inference_mode():
        exported_once = export_module(obs)
        exported_twice = export_module(obs)
        deterministic_actions = learner.actor.explore(obs, deterministic=True)

    torch.testing.assert_close(exported_once, exported_twice)
    torch.testing.assert_close(exported_once, deterministic_actions)


def test_flashsac_update_steps_run_on_cpu():
    torch.manual_seed(17)
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()
    pre_update_temperature = float(learner.temperature().detach())

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    learner.soft_update_target()

    assert "Loss/critic" in critic_metrics
    assert "Train/reward_scale_std" in critic_metrics
    assert "Loss/actor" in actor_metrics
    assert "Policy/temperature" in actor_metrics
    assert actor_metrics["Policy/temperature"] != pytest.approx(pre_update_temperature)
    assert actor_metrics["Policy/temperature"] == pytest.approx(
        float(learner.temperature().detach())
    )
    normalize_metric_map({**critic_metrics, **actor_metrics})


def test_flashsac_omits_constant_reward_scale_when_normalization_is_disabled():
    learner = FlashSACLearner(
        obs_dim=98,
        action_dim=29,
        critic_obs_dim=101,
        device="cpu",
        normalize_reward=False,
    )

    critic_metrics = learner.update_critic(_make_batch())

    assert "Train/reward_scale_std" not in critic_metrics
    normalize_metric_map(critic_metrics)


def test_flashsac_actor_update_does_not_accumulate_critic_grads() -> None:
    learner = _make_small_learner()
    batch = _make_small_batch()
    learner.critic_optimizer.zero_grad(set_to_none=True)

    learner.update_actor(batch)

    assert all(parameter.grad is None for parameter in learner.critic.parameters())


def test_flashsac_deferred_actor_metrics_read_once_at_cycle_end() -> None:
    learner = _make_small_learner()
    metrics = learner.update_actor(_make_small_batch(), read_metrics=False)

    assert metrics == {}
    deferred = learner.read_deferred_actor_metrics()
    assert set(deferred) == {
        "Loss/actor",
        "Loss/entropy",
        "Policy/temperature",
        "Loss/temperature",
    }


@pytest.mark.parametrize(
    ("policy_frequency", "target_frequency", "policy_before_critic"),
    [(1, 1, True), (2, 1, False), (2, 3, True), (3, 2, False)],
)
def test_flashsac_update_cycle_matches_runner_composition(
    policy_frequency: int,
    target_frequency: int,
    policy_before_critic: bool,
) -> None:
    torch.manual_seed(123)
    actual = _make_small_learner()
    torch.manual_seed(123)
    expected = _make_small_learner()
    small_batch = _make_small_batch(8)
    large_batch = {
        key: value.repeat(4, *([1] * (value.ndim - 1))) for key, value in small_batch.items()
    }
    rng_state = torch.random.get_rng_state()

    torch.random.set_rng_state(rng_state)
    actual._run_update_cycle_core(
        large_batch,
        updates_per_step=4,
        policy_frequency=policy_frequency,
        target_frequency=target_frequency,
        policy_before_critic=policy_before_critic,
    )
    torch.random.set_rng_state(rng_state)
    for update_idx in range(4):
        start = update_idx * 8
        batch = {key: value[start : start + 8] for key, value in large_batch.items()}
        do_actor_update = update_idx % policy_frequency == 0
        if policy_before_critic and do_actor_update:
            expected.update_actor(batch)
        expected.update_critic(batch)
        if not policy_before_critic and do_actor_update:
            expected.update_actor(batch)
        if update_idx % target_frequency == 0:
            expected.soft_update_target()

    for module_name in ("actor", "critic", "target_critic", "temperature"):
        torch.testing.assert_close(
            getattr(actual, module_name).state_dict(),
            getattr(expected, module_name).state_dict(),
        )
    assert actual.critic_scheduler.last_epoch == expected.critic_scheduler.last_epoch
    assert actual.actor_scheduler.last_epoch == expected.actor_scheduler.last_epoch
    assert actual.temperature_scheduler.last_epoch == expected.temperature_scheduler.last_epoch


def test_flashsac_update_cycle_defers_metrics_until_one_read() -> None:
    learner = _make_small_learner()
    large_batch = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        for key, value in _make_small_batch(8).items()
    }

    learner._run_update_cycle_core(
        large_batch,
        updates_per_step=2,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    metrics = learner.read_deferred_cycle_metrics()

    assert set(metrics) == {
        "Loss/critic",
        "Train/reward_scale_std",
        "Loss/actor",
        "Loss/entropy",
        "Policy/temperature",
        "Loss/temperature",
    }
    assert learner.read_deferred_cycle_metrics() == {}


def test_flashsac_state_dict_round_trip():
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()
    learner.update_critic(batch)
    learner.update_actor(batch)
    state_dict = learner.get_state_dict()

    restored = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    restored.load_state_dict(state_dict)

    assert restored.get_state_dict()["update_count"] == learner.get_state_dict()["update_count"]


def test_reward_normalizer_tracks_discounted_returns() -> None:
    normalizer = RewardNormalizer(gamma=0.5, g_max=5.0, device=torch.device("cpu"))

    normalizer.update_from_transitions(
        rewards=torch.tensor([[2.0, 1.0], [4.0, 3.0]]),
        dones=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    )

    torch.testing.assert_close(normalizer.g_r, torch.tensor([5.0, 3.5]))
    torch.testing.assert_close(normalizer.g_r_max, torch.tensor(5.0))


def test_reward_normalizer_keeps_stable_storage_for_whole_cycle_graphs() -> None:
    normalizer = RewardNormalizer(gamma=0.5, g_max=5.0, device=torch.device("cpu"))
    normalizer.update_from_transitions(
        rewards=torch.tensor([[1.0, 2.0]]),
        dones=torch.zeros(1, 2),
    )
    pointers = {
        "mean": normalizer.rms.mean.data_ptr(),
        "var": normalizer.rms.var.data_ptr(),
        "count": normalizer.rms.count.data_ptr(),
        "g_r": normalizer.g_r.data_ptr(),
        "g_r_max": normalizer.g_r_max.data_ptr(),
    }

    normalizer.update_from_transitions(
        rewards=torch.tensor([[2.0, 1.0]]),
        dones=torch.zeros(1, 2),
    )
    normalizer.load_state_dict(normalizer.state_dict())

    assert pointers == {
        "mean": normalizer.rms.mean.data_ptr(),
        "var": normalizer.rms.var.data_ptr(),
        "count": normalizer.rms.count.data_ptr(),
        "g_r": normalizer.g_r.data_ptr(),
        "g_r_max": normalizer.g_r_max.data_ptr(),
    }


def test_flashsac_critic_update_does_not_advance_reward_stats_from_sampled_batch() -> None:
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    learner.update_reward_stats(
        rewards=torch.tensor([[1.0, 2.0]]),
        dones=torch.zeros(1, 2),
    )
    assert learner.reward_normalizer is not None
    before = learner.reward_normalizer.g_r.clone()

    learner.update_critic(_make_batch())

    torch.testing.assert_close(learner.reward_normalizer.g_r, before)


def test_flashsac_critic_requires_truncated_field() -> None:
    learner = FlashSACLearner(obs_dim=98, action_dim=29, critic_obs_dim=101, device="cpu")
    batch = _make_batch()
    batch.pop("truncated")

    try:
        learner.update_critic(batch)
    except KeyError as exc:
        assert exc.args == ("truncated",)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("FlashSAC learner must require replay 'truncated'")


def test_flashsac_td_target_treats_dones_as_combined_done_with_truncation_bootstrap() -> None:
    support = torch.tensor([0.0, 1.0, 2.0])
    target_log_probs = torch.log(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ).clamp_min(1e-8)
    )

    targets = compute_categorical_td_target(
        support=support,
        target_log_probs=target_log_probs,
        reward=torch.zeros(3),
        dones=torch.tensor([0.0, 1.0, 1.0]),
        truncated=torch.tensor([0.0, 1.0, 0.0]),
        actor_entropy=torch.zeros(3),
        gamma=1.0,
    )

    # Continuing rows and truncated rows bootstrap to support value 2.0.
    torch.testing.assert_close(targets[0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(targets[1], torch.tensor([0.0, 0.0, 1.0]))
    # True terminal rows do not bootstrap and project to reward-only value 0.0.
    torch.testing.assert_close(targets[2], torch.tensor([1.0, 0.0, 0.0]))


def test_flashsac_td_target_failure_timeout_overlap_does_not_bootstrap() -> None:
    """uni_rl #29 four-state table against collector-encoded replay columns."""
    support = torch.tensor([0.0, 1.0, 2.0])
    target_log_probs = torch.log(torch.tensor([[0.0, 0.0, 1.0]] * 4).clamp_min(1e-8))

    # Collector writes dones = terminated | truncated and truncated =
    # truncated & ~terminated: normal, failure, timeout, failure+timeout.
    targets = compute_categorical_td_target(
        support=support,
        target_log_probs=target_log_probs,
        reward=torch.zeros(4),
        dones=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        truncated=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        actor_entropy=torch.zeros(4),
        gamma=1.0,
    )

    # Normal and timeout-only rows bootstrap; failure-only and the
    # failure+timeout overlap project to the reward-only support value 0.0.
    torch.testing.assert_close(targets[0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(targets[1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(targets[2], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(targets[3], torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression")
def test_flashsac_td_target_is_inductor_cuda_graph_safe() -> None:
    support = torch.linspace(-5.0, 5.0, 11, device="cuda")
    log_probs = torch.log_softmax(torch.randn(8, 11, device="cuda"), dim=-1)
    reward = torch.randn(8, device="cuda")
    dones = torch.zeros(8, device="cuda")
    truncated = torch.zeros(8, device="cuda")
    entropy = torch.randn(8, device="cuda")
    static_output = torch.empty(8, 11, device="cuda")

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static_output.copy_(
            compute_categorical_td_target(
                support,
                log_probs,
                reward,
                dones,
                truncated,
                entropy,
                0.99,
            )
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(static_output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only whole-cycle graph")
def test_flashsac_update_cycle_raw_graph_replays_with_stable_inputs_and_metrics() -> None:
    torch.manual_seed(123)
    learner = FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cuda:0",
        use_compile=True,
    )
    batch = _make_small_batch(16)
    batch = {key: value.to("cuda:0") for key, value in batch.items()}

    torch.manual_seed(321)
    learner.update_cycle(
        batch,
        updates_per_step=4,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    first_metrics = learner.read_deferred_cycle_metrics()

    assert learner._update_cycle_graph is not None
    assert learner._update_cycle_static_batch is not None
    static_addresses = {
        key: value.data_ptr() for key, value in learner._update_cycle_static_batch.items()
    }
    changed_batch = {key: value * 0.5 for key, value in batch.items()}
    learner.update_cycle(
        changed_batch,
        updates_per_step=4,
        policy_frequency=2,
        target_frequency=1,
        policy_before_critic=False,
    )
    second_metrics = learner.read_deferred_cycle_metrics()
    assert static_addresses == {
        key: value.data_ptr() for key, value in learner._update_cycle_static_batch.items()
    }
    assert set(first_metrics) == set(second_metrics)
    assert first_metrics
    assert all(math.isfinite(value) for value in second_metrics.values())


def _legacy_rms_update(rms, x: torch.Tensor) -> None:
    """Pre-row-schema RunningMeanStd.update: whole-batch scalar statistics."""
    x = x.reshape(-1).to(dtype=torch.float32)
    if x.numel() == 0:
        return
    batch_mean = x.mean()
    batch_var = x.var(unbiased=False)
    batch_count = torch.tensor(float(x.numel()), device=x.device, dtype=torch.float32)
    delta = batch_mean - rms.mean
    total_count = rms.count + batch_count
    new_mean = rms.mean + delta * batch_count / total_count
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    correction = delta.pow(2) * rms.count * batch_count / total_count
    new_var = (m_a + m_b + correction) / total_count
    rms.mean.copy_(new_mean)
    rms.var.copy_(new_var)
    rms.count.copy_(total_count)


def test_running_mean_std_single_head_is_bitwise_legacy() -> None:
    from uni_rl.algos.flash_sac.learner import RunningMeanStd

    torch.manual_seed(0)
    legacy = RunningMeanStd.create(torch.device("cpu"))
    current = RunningMeanStd.create(torch.device("cpu"))
    for _ in range(25):
        x = torch.randn(37)
        _legacy_rms_update(legacy, x)
        current.update(x)

    assert torch.equal(legacy.mean, current.mean)
    assert torch.equal(legacy.var, current.var)
    assert torch.equal(legacy.count, current.count)


def _legacy_normalizer_step_state(
    gamma: float,
    rewards: torch.Tensor,
    dones: torch.Tensor,
):
    """Pre-row-schema per-env discounted-return statistics (single head)."""
    from uni_rl.algos.flash_sac.learner import RunningMeanStd

    rms = RunningMeanStd.create(torch.device("cpu"))
    g_r = torch.zeros(rewards.shape[1])
    g_r_max = torch.tensor(0.0)
    done = torch.clamp(dones, min=0.0, max=1.0)
    for step in range(rewards.shape[0]):
        g_r.mul_(gamma * (1.0 - done[step])).add_(rewards[step])
        g_r_max.copy_(torch.maximum(g_r_max, g_r.abs().max()))
        _legacy_rms_update(rms, g_r)
    return rms, g_r, g_r_max


def test_reward_normalizer_single_head_is_bitwise_legacy() -> None:
    torch.manual_seed(1)
    rewards = torch.randn(20, 5)
    dones = (torch.rand(20, 5) > 0.9).float()

    legacy_rms, legacy_g_r, legacy_g_r_max = _legacy_normalizer_step_state(0.99, rewards, dones)
    normalizer = RewardNormalizer(gamma=0.99, g_max=5.0, device=torch.device("cpu"))
    normalizer.update_from_transitions(rewards, dones)

    # Historical shapes are preserved exactly: (num_envs,) g_r, scalar g_r_max.
    assert normalizer.g_r.shape == (5,)
    assert normalizer.g_r_max.ndim == 0
    assert torch.equal(normalizer.g_r, legacy_g_r)
    assert torch.equal(normalizer.g_r_max, legacy_g_r_max)
    assert torch.equal(normalizer.rms.mean, legacy_rms.mean)
    assert torch.equal(normalizer.rms.var, legacy_rms.var)

    sample = torch.randn(7)
    denominator = torch.maximum(
        torch.sqrt(legacy_rms.var + normalizer.eps),
        legacy_g_r_max / 5.0,
    )
    assert torch.equal(normalizer.normalize(sample), sample / denominator)


def test_reward_normalizer_per_head_matches_independent_single_heads() -> None:
    torch.manual_seed(2)
    rewards = torch.randn(20, 5, 2)
    dones = (torch.rand(20, 5) > 0.9).float()

    normalizer = RewardNormalizer(gamma=0.99, g_max=5.0, device=torch.device("cpu"), num_heads=2)
    normalizer.update_from_transitions(rewards, dones)

    assert normalizer.g_r.shape == (5, 2)
    assert normalizer.g_r_max.shape == (2,)
    assert normalizer.rms.mean.shape == (2,)
    for head in range(2):
        legacy_rms, legacy_g_r, legacy_g_r_max = _legacy_normalizer_step_state(
            0.99, rewards[..., head], dones
        )
        assert torch.equal(normalizer.g_r[:, head], legacy_g_r)
        assert torch.equal(normalizer.g_r_max[head], legacy_g_r_max)
        assert torch.equal(normalizer.rms.mean[head], legacy_rms.mean[0])
        assert torch.equal(normalizer.rms.var[head], legacy_rms.var[0])

    sample = torch.randn(7, 2)
    normalized = normalizer.normalize(sample)
    assert normalized.shape == (7, 2)
    for head in range(2):
        denominator = torch.maximum(
            torch.sqrt(normalizer.rms.var[head] + normalizer.eps),
            normalizer.g_r_max[head] / 5.0,
        )
        assert torch.equal(normalized[:, head], sample[:, head] / denominator)


def test_reward_normalizer_rejects_head_shape_mismatch() -> None:
    normalizer = RewardNormalizer(gamma=0.99, g_max=5.0, device=torch.device("cpu"), num_heads=2)
    with pytest.raises(ValueError, match="require num_heads=1"):
        normalizer.update_from_transitions(torch.randn(4, 3), torch.zeros(4, 3))
    with pytest.raises(ValueError, match="num_heads=2"):
        normalizer.update_from_transitions(torch.randn(4, 3, 3), torch.zeros(4, 3))
    with pytest.raises(ValueError, match="num_heads"):
        RewardNormalizer(gamma=0.99, g_max=5.0, device=torch.device("cpu"), num_heads=0)


def _legacy_head_critic_loss(
    support, next_q, next_q_log_probs, pred_log_probs, reward, dones, truncated, entropy, gamma
):
    from uni_rl.algos.flash_sac.update import (
        compute_categorical_td_target,
        select_min_q_log_probs,
    )

    next_log_probs = select_min_q_log_probs(next_q, next_q_log_probs)
    target_probs = compute_categorical_td_target(
        support=support,
        target_log_probs=next_log_probs,
        reward=reward,
        dones=dones,
        truncated=truncated,
        actor_entropy=entropy,
        gamma=gamma,
    )
    return -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean()


def _critic_loss_inputs(batch_size: int = 16, atoms: int = 11):
    support = torch.linspace(-5, 5, atoms)
    dones = (torch.rand(batch_size) > 0.9).float()
    truncated = (torch.rand(batch_size) > 0.95).float()
    entropy = torch.randn(batch_size)
    return support, dones, truncated, entropy


def test_critic_loss_single_head_is_bitwise_legacy() -> None:
    torch.manual_seed(3)
    batch_size, atoms = 16, 11
    learner = FlashSACLearner(obs_dim=4, action_dim=2, critic_obs_dim=4, device="cpu")
    support, dones, truncated, entropy = _critic_loss_inputs(batch_size, atoms)
    rewards = torch.randn(batch_size)
    next_q = torch.randn(2, batch_size)
    next_q_log_probs = torch.randn(2, batch_size, atoms)
    pred_log_probs = torch.randn(2, batch_size, atoms)

    loss = learner._critic_loss_tensors(
        next_q, next_q_log_probs, support, rewards, dones, truncated, entropy, pred_log_probs, 0.99
    )
    legacy = _legacy_head_critic_loss(
        support,
        next_q,
        next_q_log_probs,
        pred_log_probs,
        rewards,
        dones,
        truncated,
        entropy,
        0.99,
    )
    assert torch.equal(loss, legacy)


def test_critic_loss_expands_per_head_and_averages() -> None:
    torch.manual_seed(4)
    batch_size, atoms = 16, 11
    learner = FlashSACLearner(
        obs_dim=4, action_dim=2, critic_obs_dim=4, device="cpu", reward_head_count=2
    )
    support, dones, truncated, entropy = _critic_loss_inputs(batch_size, atoms)
    rewards = torch.randn(batch_size, 2)
    next_q = torch.randn(4, batch_size)
    next_q_log_probs = torch.randn(4, batch_size, atoms)
    pred_log_probs = torch.randn(4, batch_size, atoms)

    loss = learner._critic_loss_tensors(
        next_q, next_q_log_probs, support, rewards, dones, truncated, entropy, pred_log_probs, 0.99
    )
    reference = torch.stack(
        [
            _legacy_head_critic_loss(
                support,
                next_q[2 * head : 2 * head + 2],
                next_q_log_probs[2 * head : 2 * head + 2],
                pred_log_probs[2 * head : 2 * head + 2],
                rewards[:, head],
                dones,
                truncated,
                entropy,
                0.99,
            )
            for head in range(2)
        ]
    ).mean()
    assert torch.equal(loss, reference)


def test_critic_loss_rejects_member_count_without_2h_critic() -> None:
    torch.manual_seed(5)
    batch_size, atoms = 8, 11
    learner = FlashSACLearner(
        obs_dim=4, action_dim=2, critic_obs_dim=4, device="cpu", reward_head_count=2
    )
    support, dones, truncated, entropy = _critic_loss_inputs(batch_size, atoms)

    with pytest.raises(ValueError, match="2x reward heads 2"):
        learner._critic_loss_tensors(
            torch.randn(2, batch_size),
            torch.randn(2, batch_size, atoms),
            support,
            torch.randn(batch_size, 2),
            dones,
            truncated,
            entropy,
            torch.randn(2, batch_size, atoms),
            0.99,
        )


def test_learner_defaults_to_single_reward_head() -> None:
    learner = FlashSACLearner(obs_dim=4, action_dim=2, critic_obs_dim=4, device="cpu")
    assert learner.reward_head_count == 1
    assert learner.reward_normalizer is not None
    assert learner.reward_normalizer.num_heads == 1
    with pytest.raises(ValueError, match="reward_head_count"):
        FlashSACLearner(
            obs_dim=4, action_dim=2, critic_obs_dim=4, device="cpu", reward_head_count=0
        )


def test_critic_zero_dim_falls_back_to_actor_observation() -> None:
    torch.manual_seed(6)
    learner = FlashSACLearner(obs_dim=8, action_dim=3, critic_obs_dim=0, device="cpu")
    assert learner.critic_obs_dim == 0
    assert learner.critic_input_obs_dim == 8
    batch = {
        "obs": torch.randn(8, 8),
        "next_obs": torch.randn(8, 8),
        "actions": torch.tanh(torch.randn(8, 3)),
        "rewards": torch.randn(8),
        "dones": torch.zeros(8),
        "truncated": torch.zeros(8),
    }

    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)

    assert "Loss/critic" in critic_metrics
    assert "Loss/actor" in actor_metrics
