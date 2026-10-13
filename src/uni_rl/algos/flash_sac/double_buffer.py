"""FlashSAC builder for the device-authoritative replay path."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from uni_rl.algos.flash_sac.learner import FlashSACLearner
from uni_rl.algos.flash_sac.replay import AgeBiasedReplayPipeline, LayeredAgeBiasedReplayPipeline
from uni_rl.env_contract import EnvFactory
from uni_rl.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device
from uni_rl.ipc.row_schema import RowSchema
from uni_rl.offpolicy.actor_adapter import import_actor_adapter_modules
from uni_rl.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from uni_rl.offpolicy.runtime import (
    resolve_actor_adapter_modules,
    resolve_custom_offpolicy_runtime,
    resolve_dotted_callable,
)
from uni_rl.utils.device import get_default_device
from uni_rl.utils.nan_guard import NanGuardCfg
from uni_rl.utils.observations import get_obs_dims
from uni_rl.utils.seed import apply_training_seed
from uni_rl.utils.tensor_runtime import (
    InferencePlacement,
    resolve_inference_transport,
    resolve_tensor_runtime_settings,
)

if TYPE_CHECKING:
    from uni_rl.ipc.dp_sync import DpParameterSync


def _algo_param(algo_cfg: DictConfig, name: str, default: Any) -> Any:
    params = algo_cfg.get("algo_params", {})
    if name in params:
        return params[name]
    return algo_cfg.get(name, default)


def _validate_flashsac_double_buffer_runtime(
    cfg: DictConfig,
    *,
    replay_prefetch_mode: str,
) -> None:
    if replay_prefetch_mode != "one_tick":
        raise ValueError("FlashSAC device replay requires replay_prefetch_mode='one_tick'")
    n_step = cfg.algo.algo_params.n_step
    if isinstance(n_step, bool) or not isinstance(n_step, int) or n_step < 1:
        raise ValueError(f"FlashSAC n_step must be a positive integer, got {n_step!r}")


def build_flashsac_double_buffer_runner(
    cfg: DictConfig,
    *,
    env_factory: EnvFactory,
    env_cfg_override: dict[str, Any] | None,
    replay_prefetch_mode: str,
    device: str | None = None,
    nan_guard_cfg: NanGuardCfg | None = None,
    nan_guard_factory: Callable[[NanGuardCfg, int, bool], Any] | None = None,
    torch_thread_runtime: dict[str, Any] | None = None,
    collector_cpu_ids: list[int] | None = None,
    dp_sync: DpParameterSync | None = None,
    backend_device_binder: Callable[[str], str | None] | None = None,
) -> Any:
    """Build FlashSAC with the bounded-ingress device replay pipeline."""
    device = require_offpolicy_replay_device(device or get_default_device())
    apply_training_seed(cfg.algo.seed, torch_runtime=True, cuda=True)
    _validate_flashsac_double_buffer_runtime(
        cfg,
        replay_prefetch_mode=replay_prefetch_mode,
    )
    rl_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.algo, resolve=True))
    custom_runtime = resolve_custom_offpolicy_runtime(rl_cfg)
    # Import registration modules before the learner builds its actor so
    # custom adapter registrations are live in this process; the runner
    # forwards the list to the spawn collector.
    actor_adapter_modules = resolve_actor_adapter_modules(rl_cfg, custom_runtime)
    import_actor_adapter_modules(actor_adapter_modules)
    # Resolve public tensor-runtime bounds before constructing a one-env probe.
    # Invalid training knobs must fail closed without materializing a backend.
    tensor_runtime_settings = resolve_tensor_runtime_settings(
        cfg,
        algo_name="FlashSAC",
        num_envs=cfg.algo.num_envs,
    )
    probe_env = env_factory(1, env_cfg_override)
    try:
        probe_state = probe_env.init_state()
        probe_observation = probe_state.obs.get("obs")
        env_tensor_native = isinstance(probe_observation, torch.Tensor) and str(
            probe_observation.device
        ).startswith("cuda")
    finally:
        probe_env.close()
    inference_placement = resolve_inference_transport(
        cfg,
        device=device,
        algo_name="FlashSAC",
        env_tensor_native=env_tensor_native,
    )

    if "inference_request_timeout_sec" in cfg.training:
        warnings.warn(
            "training.inference_request_timeout_sec is deprecated and ignored; "
            "remove it from owner YAML.",
            DeprecationWarning,
            stacklevel=2,
        )
    env = env_factory(1, env_cfg_override)
    try:
        obs_dim, critic_obs_dim = get_obs_dims(dict(env.obs_groups_spec))
        action_shape = env.action_space.shape
        assert action_shape is not None
        action_dim = int(action_shape[0])
    finally:
        env.close()

    learner_cls: type[Any] = FlashSACLearner
    algo_type = "flashsac"
    learner_extra_kwargs: dict[str, Any] = {}
    if custom_runtime is not None:
        learner_extra_kwargs = cast(
            dict[str, Any],
            custom_runtime.build_model_kwargs(
                obs_dim=int(obs_dim),
                critic_obs_dim=int(critic_obs_dim),
            ),
        )
        if custom_runtime.learner_cls is not None:
            learner_cls = custom_runtime.learner_cls
        if custom_runtime.algo_type is not None:
            algo_type = str(custom_runtime.algo_type)

    # Optional collector-side replay-ingress compaction: the env/inference
    # ring keeps the full observation width while replay rows store a compact
    # projection produced by the transformer factory (resolved in the collector
    # process; the sample-side rebuilder expands rows again before the learner).
    ingress_transformer_path = _algo_param(cfg.algo, "replay_ingress_transformer", None)
    replay_obs_dim = _algo_param(cfg.algo, "replay_obs_dim", None)
    replay_critic_dim = _algo_param(cfg.algo, "replay_critic_dim", None)
    if ingress_transformer_path not in (None, ""):
        resolve_dotted_callable(str(ingress_transformer_path))
        if replay_obs_dim is None:
            raise ValueError(
                "algo.algo_params.replay_ingress_transformer requires "
                "algo.algo_params.replay_obs_dim"
            )
        if isinstance(replay_obs_dim, bool) or not isinstance(replay_obs_dim, int):
            raise ValueError(f"replay_obs_dim must be an integer, got {replay_obs_dim!r}")
        if not 0 < replay_obs_dim <= obs_dim:
            raise ValueError(
                f"replay_obs_dim must be in (0, {obs_dim}] (env obs width), got {replay_obs_dim!r}"
            )
        if critic_obs_dim > 0:
            if replay_critic_dim is None:
                raise ValueError(
                    "algo.algo_params.replay_ingress_transformer with a critic "
                    "observation requires algo.algo_params.replay_critic_dim"
                )
            if (
                isinstance(replay_critic_dim, bool)
                or not isinstance(replay_critic_dim, int)
                or not 0 <= replay_critic_dim <= critic_obs_dim
            ):
                raise ValueError(
                    f"replay_critic_dim must be an integer in [0, {critic_obs_dim}], "
                    f"got {replay_critic_dim!r}"
                )
        elif replay_critic_dim not in (None, 0):
            raise ValueError(
                "algo.algo_params.replay_critic_dim requires the env to expose a "
                "separate critic observation"
            )
    elif replay_obs_dim is not None or replay_critic_dim is not None:
        raise ValueError(
            "algo.algo_params.replay_obs_dim/replay_critic_dim are replay-row "
            "widths and require algo.algo_params.replay_ingress_transformer"
        )

    # Row layout is generated from owner YAML and travels with the run config:
    # scalar reward (H=1), no aux columns, and fp32 storage reproduce the
    # historical packed layout bit for bit.
    row_schema = RowSchema(
        obs_dim=obs_dim if replay_obs_dim is None else int(replay_obs_dim),
        action_dim=action_dim,
        critic_dim=critic_obs_dim if replay_critic_dim is None else int(replay_critic_dim),
        reward_heads=int(_algo_param(cfg.algo, "reward_head_count", 1)),
        aux_width=int(_algo_param(cfg.algo, "replay_aux_width", 0)),
        float_dtype=_algo_param(cfg.algo, "replay_float_dtype", "fp32"),
        action_float_dtype=_algo_param(cfg.algo, "replay_action_float_dtype", None),
    )

    learner_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "critic_obs_dim": critic_obs_dim,
        "gamma": cfg.algo.gamma,
        "tau": cfg.algo.tau,
        "actor_lr": cfg.algo.actor_lr,
        "critic_lr": cfg.algo.critic_lr,
        "actor_hidden_dim": cfg.algo.actor_hidden_dim,
        "critic_hidden_dim": cfg.algo.critic_hidden_dim,
        "actor_num_blocks": cfg.algo.algo_params.actor_num_blocks,
        "critic_num_blocks": cfg.algo.algo_params.critic_num_blocks,
        "num_atoms": cfg.algo.num_atoms,
        "critic_min_v": cfg.algo.algo_params.critic_min_v,
        "critic_max_v": cfg.algo.algo_params.critic_max_v,
        "temp_initial_value": cfg.algo.algo_params.temp_initial_value,
        "temp_target_sigma": cfg.algo.algo_params.temp_target_sigma,
        "temp_target_entropy": cfg.algo.algo_params.temp_target_entropy,
        "actor_bc_alpha": cfg.algo.algo_params.actor_bc_alpha,
        "actor_noise_zeta_mu": cfg.algo.algo_params.actor_noise_zeta_mu,
        "actor_noise_zeta_max": cfg.algo.algo_params.actor_noise_zeta_max,
        "learning_rate_init": cfg.algo.algo_params.learning_rate_init,
        "learning_rate_peak": cfg.algo.algo_params.learning_rate_peak,
        "learning_rate_end": cfg.algo.algo_params.learning_rate_end,
        "learning_rate_warmup_steps": cfg.algo.algo_params.learning_rate_warmup_steps,
        "learning_rate_decay_steps": cfg.algo.algo_params.learning_rate_decay_steps,
        "normalize_reward": cfg.algo.algo_params.normalize_reward,
        "normalized_g_max": cfg.algo.algo_params.normalized_g_max,
        "n_step": cfg.algo.algo_params.n_step,
        "reward_head_count": row_schema.reward_heads,
        "obs_normalization": cfg.algo.obs_normalization,
        "use_amp": cfg.training.use_amp,
        "amp_dtype": cfg.algo.algo_params.amp_dtype,
        "use_compile": cfg.algo.algo_params.use_compile,
        "compile_full_objectives": _algo_param(cfg.algo, "compile_full_objectives", None),
        "use_whole_cycle_cuda_graph": _algo_param(cfg.algo, "use_whole_cycle_cuda_graph", None),
        "actor_normalize_parameters": bool(
            _algo_param(cfg.algo, "actor_normalize_parameters", True)
        ),
        "critic_normalize_parameters": bool(
            _algo_param(cfg.algo, "critic_normalize_parameters", True)
        ),
    }
    learner_kwargs.update(learner_extra_kwargs)
    learner = learner_cls(device=device, **learner_kwargs)

    replay_storage = str(_algo_param(cfg.algo, "replay_storage", "gpu"))
    if replay_storage not in ("gpu", "cpu_layered"):
        raise ValueError(
            "algo.algo_params.replay_storage must be 'gpu' or 'cpu_layered', "
            f"got {replay_storage!r}"
        )
    layered_transport = str(_algo_param(cfg.algo, "replay_layered_transport", "auto"))
    if layered_transport not in ("auto", "uva", "h2d"):
        raise ValueError(
            "algo.algo_params.replay_layered_transport must be 'auto', 'uva', or "
            f"'h2d', got {layered_transport!r}"
        )
    rebuilder_path = _algo_param(cfg.algo, "replay_sample_rebuilder", None)
    sample_rebuilder = None
    if rebuilder_path not in (None, ""):
        if replay_storage != "cpu_layered":
            raise ValueError(
                "algo.algo_params.replay_sample_rebuilder requires replay_storage='cpu_layered'"
            )
        sample_rebuilder = resolve_dotted_callable(str(rebuilder_path))
    reconstruct_next_obs = bool(_algo_param(cfg.algo, "replay_reconstruct_next_obs", False))
    if reconstruct_next_obs and replay_storage != "cpu_layered":
        raise ValueError(
            "algo.algo_params.replay_reconstruct_next_obs requires replay_storage='cpu_layered'"
        )

    if custom_runtime is not None and custom_runtime.replay_pipeline_factory is not None:
        if replay_storage != "gpu":
            raise ValueError(
                "algo.runtime_resolver provided a replay_pipeline_factory, which "
                "conflicts with algo.algo_params.replay_storage='cpu_layered'; "
                "select the replay storage through exactly one mechanism"
            )
        replay_pipeline_factory = custom_runtime.replay_pipeline_factory
        replay_storage_owner = "gpu"
    elif replay_storage == "cpu_layered":
        replay_pipeline_factory = partial(
            LayeredAgeBiasedReplayPipeline,
            decay_step=int(_algo_param(cfg.algo, "decay_step", 0)),
            min_weight=float(_algo_param(cfg.algo, "replay_min_weight", 0.1)),
            num_buckets=int(_algo_param(cfg.algo, "replay_num_buckets", 2000)),
            transport=layered_transport,
            sample_rebuilder=sample_rebuilder,
            reconstruct_next_obs=reconstruct_next_obs,
            next_obs_alias_rows=(
                int(cfg.algo.algo_params.n_step) * int(cfg.algo.num_envs)
                if reconstruct_next_obs
                else None
            ),
        )
        replay_storage_owner = "cpu"
    else:
        replay_pipeline_factory = partial(
            AgeBiasedReplayPipeline,
            decay_step=int(_algo_param(cfg.algo, "decay_step", 0)),
            min_weight=float(_algo_param(cfg.algo, "replay_min_weight", 0.1)),
            num_buckets=int(_algo_param(cfg.algo, "replay_num_buckets", 2000)),
        )
        replay_storage_owner = "gpu"

    return DoubleBufferOffPolicyRunner(
        learner=learner,
        env_name=cfg.training.task_name,
        algo_type=algo_type,
        env_factory=env_factory,
        num_envs=cfg.algo.num_envs,
        replay_buffer_n=cfg.algo.replay_buffer_n,
        batch_size=tensor_runtime_settings.batch_size,
        learning_starts=cfg.algo.learning_starts,
        updates_per_step=tensor_runtime_settings.updates_per_step,
        policy_frequency=cfg.algo.policy_frequency,
        target_frequency=int(_algo_param(cfg.algo, "target_frequency", 1)),
        policy_before_critic=bool(_algo_param(cfg.algo, "policy_before_critic", False)),
        env_steps_per_sync=cfg.training.env_steps_per_sync,
        device=device,
        obs_normalization=cfg.algo.obs_normalization,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
        seed=cfg.algo.seed,
        trace_enabled=cfg.training.trace_enabled,
        trace_output_dir=cfg.training.trace_output_dir,
        trace_thread_time=cfg.training.trace_thread_time,
        trace_cuda_events=cfg.training.trace_cuda_events,
        replay_prefetch_mode=replay_prefetch_mode,
        nan_guard_cfg=nan_guard_cfg,
        nan_guard_factory=nan_guard_factory,
        torch_thread_runtime=torch_thread_runtime,
        collector_cpu_ids=collector_cpu_ids,
        dp_sync=dp_sync,
        learner_prepare_hook=(
            custom_runtime.learner_prepare_hook if custom_runtime is not None else None
        ),
        backend_device_binder=backend_device_binder,
        actor_adapter_modules=actor_adapter_modules,
        inference_placement=inference_placement,
        tensor_runtime_settings=tensor_runtime_settings,
        log_interval=int(cfg.training.log_interval),
        row_schema=row_schema,
        random_action_warmup=bool(_algo_param(cfg.algo, "random_action_warmup", False)),
        random_action_warmup_range=list(
            _algo_param(cfg.algo, "random_action_warmup_range", (-1.0, 1.0))
        ),
        replay_storage_owner=replay_storage_owner,
        replay_ingress_transformer=(
            None if ingress_transformer_path in (None, "") else str(ingress_transformer_path)
        ),
        replay_ingress_transformer_params=(
            None
            if ingress_transformer_path in (None, "")
            else cast(
                dict[str, Any],
                OmegaConf.to_container(cfg.algo.algo_params, resolve=True),
            )
        ),
        n_step=int(cfg.algo.algo_params.n_step),
        reconstruct_next_obs=reconstruct_next_obs,
        replay_pipeline_factory=replay_pipeline_factory,
    )
