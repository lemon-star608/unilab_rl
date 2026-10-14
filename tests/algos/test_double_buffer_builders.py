"""Forwarding tests for the double-buffer runner builder helpers.

The builders are the only public construction path for
``DoubleBufferOffPolicyRunner`` from Hydra owner configs; every runner kwarg
the builders accept must reach the runner unchanged (issue #1481 added
``backend_device_binder`` after UniLab had to set it post-construction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf


class _FakeEnv:
    obs_groups_spec = {"obs": 4, "critic": 6}
    action_space = SimpleNamespace(shape=(2,))

    def init_state(self):
        return SimpleNamespace(obs={"obs": torch.zeros((1, 4)), "critic": torch.zeros((1, 6))})

    def close(self):
        return None


def _fake_env_factory(num_envs, env_cfg_override):
    del num_envs, env_cfg_override
    return _FakeEnv()


class _FakeLearner:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        del args
        type(self).last_kwargs = kwargs


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs


def _binder(backend: str) -> str | None:
    del backend
    return None


def _learner_name(algo: str) -> str:
    return "SACLearner" if algo == "sac" else "FlashSACLearner"


def _training_cfg() -> dict[str, Any]:
    return {
        "task_name": "FakeTask",
        "sim_backend": "fake",
        "env_steps_per_sync": 1,
        "use_amp": False,
        "trace_enabled": False,
        "trace_output_dir": "logs",
        "trace_thread_time": False,
        "trace_cuda_events": False,
        "log_interval": 3,
    }


def _algo_cfg(extra: dict[str, Any]) -> dict[str, Any]:
    base = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "batch_size": 4,
        "learning_starts": 4,
        "updates_per_step": 1,
        "policy_frequency": 1,
        "seed": 1,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "num_atoms": 1,
        "obs_normalization": False,
    }
    base.update(extra)
    return base


def _sac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "use_layer_norm": False,
                    "algo_params": {
                        "alpha_lr": 1e-3,
                        "alpha_init": 1.0,
                        "target_entropy_ratio": 1.0,
                        "max_grad_norm": 1.0,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                    },
                }
            ),
        }
    )


def _flashsac_cfg() -> Any:
    return OmegaConf.create(
        {
            "training": _training_cfg(),
            "algo": _algo_cfg(
                {
                    "algo_params": {
                        "actor_num_blocks": 1,
                        "critic_num_blocks": 1,
                        "critic_min_v": -10.0,
                        "critic_max_v": 10.0,
                        "temp_initial_value": 1.0,
                        "temp_target_sigma": 1.0,
                        "temp_target_entropy": 1.0,
                        "actor_bc_alpha": 1.0,
                        "actor_noise_zeta_mu": 0.0,
                        "actor_noise_zeta_max": 0.0,
                        "learning_rate_init": 1e-3,
                        "learning_rate_peak": 1e-3,
                        "learning_rate_end": 1e-3,
                        "learning_rate_warmup_steps": 1,
                        "learning_rate_decay_steps": 1,
                        "normalize_reward": False,
                        "normalized_g_max": 1.0,
                        "n_step": 1,
                        "amp_dtype": "bf16",
                        "use_compile": False,
                        "compile_full_objectives": True,
                        "use_whole_cycle_cuda_graph": False,
                    },
                }
            ),
        }
    )


@pytest.mark.parametrize("with_binder", [False, True])
def test_sac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.algos.sac.double_buffer as module

    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_sac_double_buffer_runner(
        _sac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == 4


def test_sac_builder_forwards_runtime_iteration_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.sac.double_buffer as module

    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    cfg = _sac_cfg()
    cfg.algo.runtime_resolver = f"{__name__}:resolve_iteration_runtime"
    runner = module.build_sac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    assert runner.kwargs["iteration_hook"] is _iteration_hook


@pytest.mark.parametrize("with_binder", [False, True])
def test_flashsac_builder_forwards_backend_device_binder(
    monkeypatch: pytest.MonkeyPatch, with_binder: bool
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    # CPU-only test host: bypass the CUDA/MPS replay-device gate and seeding.
    monkeypatch.setattr(
        module,
        "require_offpolicy_replay_device",
        lambda device: device,
        raising=False,
    )
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)

    kwargs: dict[str, Any] = {}
    if with_binder:
        kwargs["backend_device_binder"] = _binder
    runner = module.build_flashsac_double_buffer_runner(
        _flashsac_cfg(),
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
        **kwargs,
    )

    assert runner.kwargs["backend_device_binder"] is (_binder if with_binder else None)
    assert runner.kwargs["log_interval"] == 3
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"
    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 2
    assert settings.replay_ingress_slot_rows == 4
    assert _FakeLearner.last_kwargs["compile_full_objectives"] is True
    assert _FakeLearner.last_kwargs["use_whole_cycle_cuda_graph"] is False
    assert _FakeLearner.last_kwargs["actor_normalize_parameters"] is True
    assert _FakeLearner.last_kwargs["critic_normalize_parameters"] is True
    assert runner.kwargs["target_frequency"] == 1
    assert runner.kwargs["policy_before_critic"] is False
    assert runner.kwargs["replay_pipeline_factory"].keywords == {
        "decay_step": 0,
        "min_weight": 0.1,
        "num_buckets": 2000,
    }


def test_flashsac_builder_forwards_runtime_iteration_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    _patch_flashsac_builder(module, monkeypatch)
    cfg = _flashsac_cfg()
    cfg.algo.runtime_resolver = f"{__name__}:resolve_iteration_runtime"
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["iteration_hook"] is _iteration_hook


def _iteration_hook(runner, iteration, env_steps, log_dir):
    del runner, iteration, env_steps, log_dir
    return {"Eval/owner_metric": 1.0}


def resolve_iteration_runtime(rl_cfg):
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    del rl_cfg
    return OffPolicyRuntime(iteration_hook=_iteration_hook)


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_forward_tensor_runtime_settings(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import uni_rl.algos.sac.double_buffer as module

        cfg = _sac_cfg()
        learner_name = "SACLearner"
    else:
        import uni_rl.algos.flash_sac.double_buffer as module

        cfg = _flashsac_cfg()
        learner_name = "FlashSACLearner"

    monkeypatch.setattr(module, learner_name, _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    if algo == "flashsac":
        monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    if algo == "flashsac":
        monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)
    cfg.env = {"tensor_runtime": False}
    cfg.training.inference_slot_capacity = 3
    cfg.training.collector_metrics_interval = 7
    cfg.training.replay_ingress_depth = 4
    cfg.training.replay_ingress_slot_rows = 2

    if algo == "sac":
        runner = module.build_sac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )
    else:
        runner = module.build_flashsac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )

    settings = runner.kwargs["tensor_runtime_settings"]
    assert settings.inference_slot_capacity == 3
    assert settings.collector_metrics_interval == 7
    assert settings.replay_ingress_depth == 4
    assert settings.replay_ingress_slot_rows == 2
    assert settings.batch_size == 4
    assert settings.updates_per_step == 1
    assert settings.learner_sample_count == 4


@pytest.mark.parametrize("algo", ["sac", "flashsac"])
def test_double_buffer_builders_reject_runtime_bounds_before_env_probe(
    monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    if algo == "sac":
        import uni_rl.algos.sac.double_buffer as module

        cfg = _sac_cfg()
        build = module.build_sac_double_buffer_runner
    else:
        import uni_rl.algos.flash_sac.double_buffer as module

        cfg = _flashsac_cfg()
        build = module.build_flashsac_double_buffer_runner
        monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
        monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)
    cfg.env = {"tensor_runtime": False}
    cfg.training.replay_ingress_depth = 17

    def _fail_factory(*args, **kwargs):
        raise AssertionError("invalid bounds must fail before env probing")

    monkeypatch.setattr(module, _learner_name(algo), _FakeLearner)
    with pytest.raises(ValueError, match="replay_ingress_depth.*17"):
        build(
            cfg,
            env_factory=_fail_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )


def test_flashsac_builder_resolves_tensor_runtime_before_collector_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)

    cfg = _flashsac_cfg()
    cfg.env = {"tensor_runtime": False}
    runner = module.build_flashsac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )
    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"


def test_sac_builder_resolves_tensor_runtime_before_collector_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.sac.double_buffer as module

    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    cfg = _sac_cfg()
    cfg.env = {"tensor_runtime": False}

    runner = module.build_sac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    assert runner.kwargs["inference_placement"].collector_tensor_native is False
    assert runner.kwargs["inference_placement"].mode.value == "cpu"


def test_sac_builder_resolves_transport_on_cpu_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.sac.double_buffer as module

    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    cfg = _sac_cfg()

    runner = module.build_sac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )

    assert runner.kwargs["inference_placement"].mode.value == "cpu"
    assert runner.kwargs["inference_placement"].collector_tensor_native is False


def test_flashsac_builder_rejects_tensor_runtime_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    cfg = _flashsac_cfg()
    cfg.env = {"tensor_runtime": True}

    with pytest.raises(
        ValueError,
        match="Off-policy training requires a CUDA or MPS learner device",
    ):
        module.build_flashsac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )


def test_sac_builder_forwards_custom_runtime_preparation_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.sac.double_buffer as module
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    def prepare_hook(learner, context):
        del learner, context

    runtime = OffPolicyRuntime(learner_cls=_FakeLearner, learner_prepare_hook=prepare_hook)
    monkeypatch.setattr(module, "SACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    monkeypatch.setattr(module, "resolve_custom_offpolicy_runtime", lambda rl_cfg: runtime)

    with pytest.warns(DeprecationWarning, match="inference_request_timeout_sec is deprecated"):
        cfg = _sac_cfg()
        cfg.training.inference_request_timeout_sec = 17.0
        runner = module.build_sac_double_buffer_runner(
            cfg,
            env_factory=_fake_env_factory,
            env_cfg_override=None,
            replay_prefetch_mode="one_tick",
            device="cpu",
        )

    assert runner.kwargs["learner_prepare_hook"] is prepare_hook
    assert "inference_request_timeout_sec" not in runner.kwargs


class _StubFlashSACLearner(_FakeLearner):
    """Stand-in for an externally injected FlashSAC learner subclass."""


def _stub_prepare_hook(learner: Any, context: Any) -> None:
    del learner, context


def _stub_replay_pipeline_factory(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise AssertionError("the builder must forward, not invoke, the replay pipeline factory")


def resolve_flashsac_stub_runtime(rl_cfg: dict[str, Any]) -> Any:
    """runtime_resolver entry point exercised through real dotted-path resolution."""
    del rl_cfg
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    return OffPolicyRuntime(
        learner_cls=_StubFlashSACLearner,
        algo_type="flashsac_stub",
        actor_kwargs={"stub_marker": "injected"},
        actor_adapter_modules=(__name__,),
        learner_prepare_hook=_stub_prepare_hook,
        replay_pipeline_factory=_stub_replay_pipeline_factory,
    )


def _patch_flashsac_builder(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "FlashSACLearner", _FakeLearner)
    monkeypatch.setattr(module, "DoubleBufferOffPolicyRunner", _FakeRunner)
    # CPU-only test host: bypass the CUDA/MPS replay-device gate and seeding.
    monkeypatch.setattr(module, "require_offpolicy_replay_device", lambda device: device)
    monkeypatch.setattr(module, "apply_training_seed", lambda *args, **kwargs: None)


def _build_flashsac_runner(cfg: Any) -> Any:
    import uni_rl.algos.flash_sac.double_buffer as module

    return module.build_flashsac_double_buffer_runner(
        cfg,
        env_factory=_fake_env_factory,
        env_cfg_override=None,
        replay_prefetch_mode="one_tick",
        device="cpu",
    )


def test_flashsac_builder_consumes_custom_offpolicy_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    def _unexpected_default_learner(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("default learner_cls must not be built when one is injected")

    monkeypatch.setattr(module, "FlashSACLearner", _unexpected_default_learner)

    cfg = _flashsac_cfg()
    cfg.algo.runtime_resolver = f"{__name__}:resolve_flashsac_stub_runtime"
    runner = _build_flashsac_runner(cfg)

    assert type(runner.kwargs["learner"]) is _StubFlashSACLearner
    assert runner.kwargs["algo_type"] == "flashsac_stub"
    assert runner.kwargs["replay_pipeline_factory"] is _stub_replay_pipeline_factory
    assert runner.kwargs["learner_prepare_hook"] is _stub_prepare_hook
    assert runner.kwargs["actor_adapter_modules"] == (__name__,)
    # Runtime-provided actor kwargs merge into the learner constructor kwargs.
    assert _StubFlashSACLearner.last_kwargs["stub_marker"] == "injected"
    assert _StubFlashSACLearner.last_kwargs["obs_dim"] == 4
    assert _StubFlashSACLearner.last_kwargs["n_step"] == 1


def test_flashsac_builder_forwards_owner_actor_adapter_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.actor_adapter_modules = [__name__]
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["actor_adapter_modules"] == (__name__,)
    assert type(runner.kwargs["learner"]) is _FakeLearner
    assert runner.kwargs["algo_type"] == "flashsac"


def test_flashsac_builder_without_runtime_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module
    from uni_rl.algos.flash_sac.replay import AgeBiasedReplayPipeline

    _patch_flashsac_builder(module, monkeypatch)

    runner = _build_flashsac_runner(_flashsac_cfg())

    assert type(runner.kwargs["learner"]) is _FakeLearner
    assert runner.kwargs["algo_type"] == "flashsac"
    assert runner.kwargs["actor_adapter_modules"] == ()
    assert runner.kwargs["learner_prepare_hook"] is None
    assert runner.kwargs["replay_storage_owner"] == "gpu"
    assert runner.kwargs["n_step"] == 1
    factory = runner.kwargs["replay_pipeline_factory"]
    assert factory.func is AgeBiasedReplayPipeline
    assert factory.keywords == {
        "decay_step": 0,
        "min_weight": 0.1,
        "num_buckets": 2000,
    }
    # The default row schema reproduces the historical packed fp32 layout:
    # one slab, legacy column offsets, scalar reward, no aux columns.
    row_schema = runner.kwargs["row_schema"]
    assert row_schema.reward_heads == 1
    assert row_schema.aux_width == 0
    assert row_schema.float_dtype is torch.float32
    assert len(row_schema.slabs) == 1
    assert row_schema.slabs[0].dtype is torch.float32
    assert row_schema.storage_width == 2 * 4 + 2 + 3 + 2 * 6
    assert row_schema.storage_row_bytes == row_schema.storage_width * 4
    assert row_schema.field("obs").offset == 0
    assert row_schema.field("next_obs").offset == 4
    assert row_schema.field("actions").offset == 8
    assert row_schema.field("rewards").offset == 10
    assert row_schema.field("dones").offset == 11
    assert row_schema.field("truncated").offset == 12
    assert row_schema.field("critic").offset == 13
    assert row_schema.field("next_critic").offset == 19
    # The learner constructor kwargs are exactly the pre-row-schema set; the
    # only additive key is reward_head_count, matching the learner default.
    assert _FakeLearner.last_kwargs == {
        "device": "cpu",
        "obs_dim": 4,
        "action_dim": 2,
        "critic_obs_dim": 6,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "actor_num_blocks": 1,
        "critic_num_blocks": 1,
        "num_atoms": 1,
        "critic_min_v": -10.0,
        "critic_max_v": 10.0,
        "temp_initial_value": 1.0,
        "temp_target_sigma": 1.0,
        "temp_target_entropy": 1.0,
        "actor_bc_alpha": 1.0,
        "actor_noise_zeta_mu": 0.0,
        "actor_noise_zeta_max": 0.0,
        "learning_rate_init": 1e-3,
        "learning_rate_peak": 1e-3,
        "learning_rate_end": 1e-3,
        "learning_rate_warmup_steps": 1,
        "learning_rate_decay_steps": 1,
        "normalize_reward": False,
        "normalized_g_max": 1.0,
        "n_step": 1,
        "reward_head_count": 1,
        "obs_normalization": False,
        "use_amp": False,
        "amp_dtype": "bf16",
        "use_compile": False,
        "compile_full_objectives": True,
        "use_whole_cycle_cuda_graph": False,
        "actor_normalize_parameters": True,
        "critic_normalize_parameters": True,
    }


def test_flashsac_builder_builds_row_schema_from_owner_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.reward_head_count = 2
    cfg.algo.algo_params.replay_aux_width = 2
    cfg.algo.algo_params.replay_float_dtype = "fp16"
    cfg.algo.algo_params.replay_action_float_dtype = "fp32"
    cfg.algo.algo_params.random_action_warmup = True
    cfg.algo.algo_params.random_action_warmup_range = [-1.0, 1.0]
    runner = _build_flashsac_runner(cfg)

    row_schema = runner.kwargs["row_schema"]
    assert row_schema.reward_heads == 2
    assert row_schema.aux_width == 2
    assert row_schema.float_dtype is torch.float16
    assert row_schema.action_float_dtype is torch.float32
    assert row_schema.obs_dim == 4
    assert row_schema.action_dim == 2
    assert row_schema.critic_dim == 6
    assert [slab.dtype for slab in row_schema.slabs] == [
        torch.float16,
        torch.float32,
        torch.float16,
        torch.int32,
    ]
    # obs/next_obs | actions+rewards+flags | critic/next_critic | aux
    assert runner.kwargs["random_action_warmup"] is True
    assert runner.kwargs["random_action_warmup_range"] == [-1.0, 1.0]
    assert _FakeLearner.last_kwargs["reward_head_count"] == 2


def test_flashsac_builder_accepts_zero_width_shared_critic_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    cfg.algo.algo_params.replay_obs_dim = 2
    cfg.algo.algo_params.replay_critic_dim = 0
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["row_schema"].critic_dim == 0
    assert not runner.kwargs["row_schema"].has_field("critic")


def test_flashsac_builder_rejects_invalid_row_schema_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_float_dtype = "bf16"
    with pytest.raises(ValueError, match="fp32"):
        _build_flashsac_runner(cfg)


def stub_sample_rebuilder(batch: dict[str, Any], *, indices: Any, device: Any) -> dict[str, Any]:
    del indices, device
    return batch


def stub_ingress_transformer_factory(params: Any) -> Any:
    del params

    def transform(transition: dict[str, Any]) -> dict[str, Any]:
        out = dict(transition)
        for key in (
            "obs",
            "next_obs",
            "critic",
            "next_critic",
            "terminal_next_obs",
            "terminal_next_critic",
        ):
            value = out.get(key)
            if value is not None:
                out[key] = value[..., ::2]
        return out

    return transform


def test_flashsac_builder_selects_layered_replay_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module
    from uni_rl.algos.flash_sac.replay import LayeredAgeBiasedReplayPipeline

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_storage = "cpu_layered"
    cfg.algo.algo_params.replay_layered_transport = "h2d"
    cfg.algo.algo_params.replay_sample_rebuilder = f"{__name__}:stub_sample_rebuilder"
    cfg.algo.algo_params.decay_step = 5
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["replay_storage_owner"] == "cpu"
    factory = runner.kwargs["replay_pipeline_factory"]
    assert factory.func is LayeredAgeBiasedReplayPipeline
    assert factory.keywords["transport"] == "h2d"
    assert factory.keywords["sample_rebuilder"] is stub_sample_rebuilder
    assert factory.keywords["decay_step"] == 5
    assert factory.keywords["min_weight"] == 0.1
    assert factory.keywords["num_buckets"] == 2000


def test_flashsac_builder_rejects_rebuilder_without_layered_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_sample_rebuilder = f"{__name__}:stub_sample_rebuilder"
    with pytest.raises(ValueError, match="cpu_layered"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_layered_storage_with_custom_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module
    from uni_rl.offpolicy.runtime import OffPolicyRuntime

    _patch_flashsac_builder(module, monkeypatch)

    def _custom_factory(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unused")

    runtime = OffPolicyRuntime(replay_pipeline_factory=_custom_factory)
    monkeypatch.setattr(module, "resolve_custom_offpolicy_runtime", lambda rl_cfg: runtime)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_storage = "cpu_layered"
    with pytest.raises(ValueError, match="exactly one mechanism"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_invalid_replay_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_storage = "ssd"
    with pytest.raises(ValueError, match="replay_storage"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_forwards_n_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.n_step = 5
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["n_step"] == 5
    assert _FakeLearner.last_kwargs["n_step"] == 5


@pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
def test_flashsac_builder_rejects_invalid_n_step(
    monkeypatch: pytest.MonkeyPatch, bad: object
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.n_step = bad
    with pytest.raises(ValueError, match="n_step must be a positive integer"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_wires_next_obs_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module
    from uni_rl.algos.flash_sac.replay import LayeredAgeBiasedReplayPipeline

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.num_envs = 8
    cfg.algo.algo_params.n_step = 5
    cfg.algo.algo_params.replay_storage = "cpu_layered"
    cfg.algo.algo_params.replay_reconstruct_next_obs = True
    runner = _build_flashsac_runner(cfg)

    assert runner.kwargs["reconstruct_next_obs"] is True
    assert runner.kwargs["n_step"] == 5
    factory = runner.kwargs["replay_pipeline_factory"]
    assert factory.func is LayeredAgeBiasedReplayPipeline
    assert factory.keywords["reconstruct_next_obs"] is True
    assert factory.keywords["next_obs_alias_rows"] == 40


def test_flashsac_builder_rejects_reconstruction_without_layered_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_reconstruct_next_obs = True
    with pytest.raises(ValueError, match="cpu_layered"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_wires_replay_ingress_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    cfg.algo.algo_params.replay_obs_dim = 2
    cfg.algo.algo_params.replay_critic_dim = 3
    cfg.algo.algo_params.replay_aux_width = 2
    runner = _build_flashsac_runner(cfg)

    # The env keeps the full width (learner/ring stay 4/6); replay rows store
    # the compact projection declared by the owner config.
    assert runner.kwargs["row_schema"].obs_dim == 2
    assert runner.kwargs["row_schema"].critic_dim == 3
    assert runner.kwargs["row_schema"].aux_width == 2
    assert _FakeLearner.last_kwargs["obs_dim"] == 4
    assert _FakeLearner.last_kwargs["critic_obs_dim"] == 6
    assert (
        runner.kwargs["replay_ingress_transformer"]
        == f"{__name__}:stub_ingress_transformer_factory"
    )
    assert runner.kwargs["replay_ingress_transformer_params"]["replay_obs_dim"] == 2
    assert runner.kwargs["replay_ingress_transformer_params"]["replay_critic_dim"] == 3


def test_flashsac_builder_rejects_transformer_without_replay_obs_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    with pytest.raises(ValueError, match="replay_obs_dim"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_replay_width_without_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_obs_dim = 2
    with pytest.raises(ValueError, match="replay_ingress_transformer"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_wider_replay_obs_than_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    cfg.algo.algo_params.replay_obs_dim = 5
    cfg.algo.algo_params.replay_critic_dim = 3
    with pytest.raises(ValueError, match=r"replay_obs_dim must be in \(0, 4\]"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_transformer_without_replay_critic_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    cfg.algo.algo_params.replay_obs_dim = 2
    with pytest.raises(ValueError, match="replay_critic_dim"):
        _build_flashsac_runner(cfg)


def test_flashsac_builder_rejects_wider_replay_critic_than_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.algos.flash_sac.double_buffer as module

    _patch_flashsac_builder(module, monkeypatch)

    cfg = _flashsac_cfg()
    cfg.algo.algo_params.replay_ingress_transformer = f"{__name__}:stub_ingress_transformer_factory"
    cfg.algo.algo_params.replay_obs_dim = 2
    cfg.algo.algo_params.replay_critic_dim = 7
    with pytest.raises(ValueError, match=r"replay_critic_dim must be an integer in \[0, 6\]"):
        _build_flashsac_runner(cfg)
