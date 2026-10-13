from __future__ import annotations

import queue
import statistics
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import uni_rl.offpolicy.worker as worker_module
from uni_rl.algos.common.collector_timing import extract_env_step_breakdown_timing_ms
from uni_rl.algos.flash_sac.update import compute_categorical_td_target
from uni_rl.env_contract import REPLAY_AUX_INFO_KEY
from uni_rl.ipc.inference_ring import SharedInferenceRing
from uni_rl.logging.metrics_drain import RewardComponentWindow, drain_collector_metrics
from uni_rl.logging.offpolicy import OffPolicyLogger
from uni_rl.offpolicy.tensor_metrics import TensorCollectorMetrics
from uni_rl.offpolicy.worker import (
    _collector_action_numpy,
    _publish_collector_ready,
    _publish_inference_tick,
    _wait_for_inference_tick,
    resolve_offpolicy_actor_priv_info,
    sample_offpolicy_actions,
)


@pytest.mark.parametrize("actions_is_tensor", [False, True])
def test_cpu_collector_action_conversion_preserves_legacy_contract(
    actions_is_tensor: bool,
) -> None:
    values = [[1.0, 2.0], [3.0, 4.0]]
    actions = (
        torch.tensor(values, dtype=torch.float32)
        if actions_is_tensor
        else np.asarray(values, dtype=np.float32)
    )

    converted = _collector_action_numpy(actions)

    assert isinstance(converted, np.ndarray)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, values)


def test_tensor_collector_metrics_resets_done_episodes_and_flushes_one_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = TensorCollectorMetrics(num_envs=3, interval=4, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    metrics.update(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        done=torch.tensor([False, True, False]),
        timeout=torch.tensor([False, False, False]),
    )
    metrics.update(
        rewards=torch.tensor([4.0, 5.0, 6.0]),
        done=torch.tensor([True, False, True]),
        timeout=torch.tensor([False, False, False]),
    )
    metrics.update(
        rewards=torch.tensor([7.0, 8.0, 9.0]),
        done=torch.tensor([False, False, False]),
        timeout=torch.tensor([False, False, False]),
    )
    assert calls == 0
    metrics.update(
        rewards=torch.tensor([10.0, 11.0, 12.0]),
        done=torch.tensor([True, True, True]),
        timeout=torch.tensor([True, False, True]),
    )

    flushed = metrics.flush()

    assert flushed.rewards == [2.0, 5.0, 9.0, 17.0, 24.0, 21.0]
    assert flushed.lengths == [1, 2, 2, 2, 3, 2]
    assert flushed.done_count == 6
    assert flushed.timeout_count == 2
    assert calls == 1
    assert metrics.ready is False
    assert torch.count_nonzero(metrics.current_rewards) == 0
    assert torch.count_nonzero(metrics.current_lengths) == 0


def test_tensor_collector_metrics_supports_no_done_window() -> None:
    metrics = TensorCollectorMetrics(num_envs=2, interval=2, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([3.0, 4.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )

    flushed = metrics.flush()

    assert flushed.rewards == []
    assert flushed.lengths == []
    assert flushed.done_count == 0
    assert flushed.timeout_count == 0
    assert metrics.current_rewards.tolist() == [4.0, 6.0]
    assert metrics.current_lengths.tolist() == [2, 2]


def test_tensor_collector_final_flush_handles_empty_and_partial_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = TensorCollectorMetrics(num_envs=2, interval=3, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    empty_before_updates = metrics.final_flush()
    empty_after_finalize = metrics.final_flush()

    assert empty_before_updates == empty_after_finalize
    assert calls == 0

    metrics = TensorCollectorMetrics(num_envs=2, interval=3, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    no_completed = metrics.final_flush()

    assert no_completed.rewards == []
    assert no_completed.lengths == []
    assert no_completed.done_count == 0
    assert no_completed.timeout_count == 0
    assert calls == 1
    assert metrics.current_rewards.tolist() == [1.0, 10.0]
    assert metrics.current_lengths.tolist() == [1, 1]


def test_tensor_collector_final_flush_emits_only_completed_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = TensorCollectorMetrics(num_envs=2, interval=3, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([True, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0, 20.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )

    flushed = metrics.final_flush()

    assert flushed.rewards == [1.0]
    assert flushed.lengths == [1]
    assert flushed.done_count == 1
    assert flushed.timeout_count == 0
    assert calls == 1
    assert metrics.current_rewards.tolist() == [2.0, 30.0]
    assert metrics.current_lengths.tolist() == [1, 2]


def test_tensor_collector_final_flush_emits_multiple_completed_episodes() -> None:
    metrics = TensorCollectorMetrics(num_envs=2, interval=4, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0, 10.0]),
        done=torch.tensor([True, False]),
        timeout=torch.tensor([False, False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0, 20.0]),
        done=torch.tensor([True, True]),
        timeout=torch.tensor([False, True]),
    )

    flushed = metrics.final_flush()

    assert flushed.rewards == [1.0, 2.0, 30.0]
    assert flushed.lengths == [1, 1, 2]
    assert flushed.done_count == 3
    assert flushed.timeout_count == 1


def test_tensor_collector_final_flush_after_normal_flush_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = TensorCollectorMetrics(num_envs=1, interval=2, device="cpu")
    calls = 0
    original_cpu = torch.Tensor.cpu

    def counting_cpu(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    metrics.update(
        rewards=torch.tensor([1.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    normal_flush = metrics.flush()
    final_flush = metrics.final_flush()

    assert normal_flush.rewards == [1.0, 2.0]
    assert normal_flush.lengths == [1, 1]
    assert final_flush.rewards == []
    assert final_flush.lengths == []
    assert final_flush.done_count == 0
    assert final_flush.timeout_count == 0
    assert calls == 1


def test_tensor_collector_metrics_preserves_new_episode_after_done() -> None:
    metrics = TensorCollectorMetrics(num_envs=1, interval=3, device="cpu")
    metrics.update(
        rewards=torch.tensor([1.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([2.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([3.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )

    first_flush = metrics.flush()

    assert first_flush.rewards == [1.0]
    assert first_flush.lengths == [1]
    assert metrics.current_rewards.tolist() == [5.0]
    assert metrics.current_lengths.tolist() == [2]

    metrics.update(
        rewards=torch.tensor([4.0]),
        done=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    metrics.update(
        rewards=torch.tensor([0.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    metrics.update(
        rewards=torch.tensor([0.0]),
        done=torch.tensor([False]),
        timeout=torch.tensor([False]),
    )
    second_flush = metrics.flush()

    assert second_flush.rewards == [9.0]
    assert second_flush.lengths == [3]


def test_tensor_collector_metrics_enforces_window_and_input_contract() -> None:
    metrics = TensorCollectorMetrics(num_envs=2, interval=1, device="cpu")

    with pytest.raises(RuntimeError, match="window is not ready"):
        metrics.flush()
    with pytest.raises(ValueError, match="positive environment count"):
        TensorCollectorMetrics(num_envs=0, interval=1, device="cpu")
    with pytest.raises(ValueError, match="positive interval"):
        TensorCollectorMetrics(num_envs=1, interval=True, device="cpu")
    with pytest.raises(TypeError, match="rewards must be Torch tensors"):
        metrics.update(
            rewards=np.asarray([1.0, 2.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(TypeError, match="rewards must have a floating dtype"):
        metrics.update(
            rewards=torch.tensor([1, 2]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(TypeError, match="done and timeout values must be Torch tensors"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=np.asarray([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(ValueError, match="environment shape"):
        metrics.update(
            rewards=torch.tensor([1.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )
    with pytest.raises(ValueError, match="timeout values must have environment shape"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False]),
        )
    with pytest.raises(TypeError, match="boolean tensors"):
        metrics.update(
            rewards=torch.tensor([1.0, 2.0]),
            done=torch.tensor([0, 0]),
            timeout=torch.tensor([False, False]),
        )

    metrics.update(
        rewards=torch.tensor([1.0, 2.0]),
        done=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
    )
    with pytest.raises(RuntimeError, match="window was not flushed"):
        metrics.update(
            rewards=torch.tensor([3.0, 4.0]),
            done=torch.tensor([False, False]),
            timeout=torch.tensor([False, False]),
        )


def test_collector_publishes_ready_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})
            events.append("env_init")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            return True

    def publish_ready(coordination_queue, event) -> bool:
        assert not metrics_queue.empty()
        events.append("ready")
        event.set()
        assert _publish_collector_ready(coordination_queue, None)

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_publish_collector_ready", publish_ready)
    monkeypatch.setattr(
        worker_module,
        "_publish_inference_tick",
        lambda *args, **kwargs: events.append("inference_tick"),
    )

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=1,
        replay_buffer=_ReplayBuffer(),
        inference_slot=None,
        inference_request_queue=inference_request_queue,
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=None,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert events == ["env_init", "ready"]
    assert metrics_queue.get_nowait()["runtime_manifest"]["inference_owner"] == "learner"
    assert inference_request_queue.get_nowait() == worker_module.COLLECTOR_READY_TICK


def _run_ready_collector(
    monkeypatch: pytest.MonkeyPatch,
    env: Any,
    metrics_queue: queue.Queue,
) -> None:
    stop_event = threading.Event()

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            return True

    def publish_ready(coordination_queue, event) -> bool:
        del coordination_queue, event
        # Returning False exits right after the manifest publish, before the
        # collection loop and its teardown path.
        return False

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_publish_collector_ready", publish_ready)

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: env,
        num_envs=1,
        replay_buffer=_ReplayBuffer(),
        inference_slot=None,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=None,
        trace_enabled=False,
        trace_thread_time=False,
    )


def test_collector_manifest_publishes_backend_runtime_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

        def close(self) -> None:
            pass

        def get_backend_runtime_diagnostics(self) -> dict:
            return {
                "available": True,
                "backend_type": "mjwarp",
                "diagnostics": {
                    "cuda_graph": {
                        "requested": True,
                        "enabled": False,
                        "disable_reason": "graph capture failed",
                    }
                },
            }

    _run_ready_collector(monkeypatch, _Env(), metrics_queue)

    manifest = metrics_queue.get_nowait()["runtime_manifest"]
    assert manifest["backend_diagnostics"] == {
        "available": True,
        "backend_type": "mjwarp",
        "diagnostics": {
            "cuda_graph": {
                "requested": True,
                "enabled": False,
                "disable_reason": "graph capture failed",
            }
        },
    }


def test_collector_manifest_records_unavailable_backend_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_queue: queue.Queue[dict] = queue.Queue()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

        def close(self) -> None:
            pass

    _run_ready_collector(monkeypatch, _Env(), metrics_queue)

    manifest = metrics_queue.get_nowait()["runtime_manifest"]
    assert manifest["backend_diagnostics"] == {
        "available": False,
        "unavailable_reason": "env does not expose backend runtime diagnostics",
    }


def test_collector_requires_batched_replay_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()

    class _Env:
        state = None

        def init_state(self) -> None:
            self.state = SimpleNamespace(obs={"obs": torch.zeros((1, 2))}, info={})

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False

        def attach_stop_event(self, stop) -> None:
            del stop

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)

    with pytest.raises(AttributeError, match="add_batch"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=1,
            replay_buffer=_ReplayBuffer(),
            inference_slot=None,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="legacy_tensor",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_binds_backend_device_before_env_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, ...]] = []

    class _EnvMaterializedError(RuntimeError):
        pass

    def bind_device(device: str) -> str:
        events.append(("bind", device))
        return device

    def env_factory(num_envs, env_cfg_override=None):
        del env_cfg_override
        events.append(("make", str(num_envs)))
        raise _EnvMaterializedError

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *args, **kwargs: None)

    with pytest.raises(_EnvMaterializedError):
        worker_module._run_collector(
            stop_event=None,
            env_factory=env_factory,
            num_envs=2,
            replay_buffer=None,
            inference_slot=None,
            inference_request_queue=None,
            inference_response_queue=None,
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=None,
            inference_transport="cpu",
            sim_backend="mjwarp",
            backend_device="cuda:3",
            env_cfg_override=None,
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
            backend_device_binder=bind_device,
        )

    # The real configure_backend_process_device resolves mjwarp -> cuda:3 and
    # must bind it before the env factory runs.
    assert events == [
        ("bind", "cuda:3"),
        ("make", "2"),
    ]


def test_tensor_collector_reports_bounded_inference_scheduling_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    inference_response_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    events: list[str] = []
    replay_calls: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    initial_obs = {"obs": torch.zeros((2, 2), dtype=torch.float32)}
    next_obs = {"obs": torch.ones((2, 2), dtype=torch.float32)}

    class _State:
        obs = initial_obs
        reward = torch.ones(2)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        final_observation = None
        info = {"timing": {}}

    class _Env:
        state = _State()

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            events.append("step")
            stop_event.set()
            self.state = SimpleNamespace(
                obs=next_obs,
                reward=torch.ones(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def close(self):
            events.append("close")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("replay_add")

        def add_batch(self, *args, **kwargs) -> bool:
            replay_calls.append((args, kwargs))
            del args, kwargs
            events.append("replay_add_batch")
            return True

        def release_ipc(self) -> None:
            events.append("replay_release")

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=4)

    def serve_learner_action(
        coordination_queue,
        tick_id,
        stop,
        **kwargs,
    ):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    original_consume_action = ring.consume_action

    def consume_tensor_action(**kwargs):
        actions, policy_version = original_consume_action(**kwargs)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        return actions, policy_version

    ring.consume_action = consume_tensor_action

    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=2,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    manifest_message = metrics_queue.get_nowait()
    manifest = manifest_message["runtime_manifest"]
    assert manifest["inference_queue_capacity"] == 4
    assert manifest["inference_scheduling_policy"] == "sequential_transition_dependency"
    assert manifest["inference_legal_max_in_flight"] == 1
    assert manifest["inference_dependency_graph"]["transition_to_next_observation"] == (
        "env.step(action[t]) -> observation[t+1]"
    )

    report_message = metrics_queue.get_nowait()
    report = report_message["collector_inference"]
    assert report["queue_depth"] == 0
    assert report["action_backlog"] == 0
    assert report["max_action_backlog"] == 0
    assert report["in_flight"] == 0
    assert report["max_in_flight"] == 1
    assert report["wait_time_ms"] >= 0.0
    assert report["publication_lag"] == 0
    assert report["max_publication_lag"] == 1
    manifest_flight = report_message["runtime_manifest"]["inference_flight"]
    assert manifest_flight == report
    assert events == ["step", "replay_add_batch", "close", "replay_release"]
    assert len(replay_calls) == 1
    replay_args, replay_kwargs = replay_calls[0]
    assert len(replay_args) == 6
    assert tuple(tensor.shape[0] for tensor in replay_args) == (2, 2, 2, 2, 2, 2)
    assert replay_args[1].dtype == torch.float32
    for key in (
        "terminal_mask",
        "terminal_next_obs",
        "critic",
        "next_critic",
        "terminal_next_critic",
    ):
        assert key in replay_kwargs
    terminal_mask = replay_kwargs["terminal_mask"]
    critic = replay_kwargs["critic"]
    next_critic = replay_kwargs["next_critic"]
    assert isinstance(terminal_mask, torch.Tensor)
    assert isinstance(critic, torch.Tensor)
    assert isinstance(next_critic, torch.Tensor)
    assert terminal_mask.shape == (2,)
    assert critic.shape == (2, 2)
    assert next_critic.shape == (2, 2)
    assert inference_request_queue.get_nowait() == worker_module.COLLECTOR_READY_TICK
    assert inference_request_queue.get_nowait() == 0


def test_worker_shutdown_flushes_partial_metrics_before_abnormal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    inference_response_queue: queue.Queue[dict] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    events: list[str] = []

    class _State:
        obs = {"obs": torch.zeros((2, 2), dtype=torch.float32)}
        reward = torch.tensor([4.0, 2.0])
        terminated = torch.tensor([True, False])
        truncated = torch.tensor([False, False])
        final_observation = None
        info = {"timing": {}}

    class _Env:
        state = _State()

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((2, 2)))
            events.append("step")
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": torch.ones((2, 2), dtype=torch.float32)},
                reward=torch.tensor([4.0, 2.0]),
                terminated=torch.tensor([True, False]),
                truncated=torch.tensor([False, False]),
                final_observation=None,
                info={"timing": {}},
            )
            return self.state

        def cleanup(self):
            events.append("cleanup")
            raise RuntimeError("abnormal shutdown after partial window")

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.tensor([2], dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("replay_add")

        def add_batch(self, *args, **kwargs) -> bool:
            del args, kwargs
            events.append("replay_add")
            return True

        def release_ipc(self) -> None:
            events.append("replay_release")

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *a, **kw: None)
    ring = SharedInferenceRing(2, 2, 2, capacity=4)

    def serve_learner_action(
        coordination_queue,
        tick_id,
        stop,
        **kwargs,
    ):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((2, 2)),
            dones=torch.empty(2),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((2, 2)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    original_consume_action = ring.consume_action

    def consume_tensor_action(**kwargs):
        actions, policy_version = original_consume_action(**kwargs)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        return actions, policy_version

    ring.consume_action = consume_tensor_action

    with pytest.raises(RuntimeError, match="abnormal shutdown after partial window"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=2,
            replay_buffer=_ReplayBuffer(),
            inference_slot=ring,
            inference_request_queue=inference_request_queue,
            inference_response_queue=inference_response_queue,
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=metrics_queue,
            inference_transport="legacy_tensor",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override=None,
            inference_epoch=0,
            collector_metrics_interval=3,
            seed=1,
            trace_enabled=False,
            trace_thread_time=False,
        )

    assert metrics_queue.get_nowait()["runtime_manifest"]["tensor_native_env"] is True
    final_message = metrics_queue.get_nowait()
    assert final_message["metric_flush"] == "final"
    assert final_message["total_steps"] == 2
    assert final_message["return_mean_ep100"] == 4.0
    assert final_message["mean_episode_length"] == 1.0
    assert final_message["timeout_rate"] == 0.0
    assert metrics_queue.empty()

    reward_history: deque[float] = deque(maxlen=100)
    logger = OffPolicyLogger(log_backend="none")
    # Requeue the captured messages so the shared drain under test sees the
    # exact worker publication sequence.
    metrics_queue.put_nowait({"runtime_manifest": {"tensor_native_env": True}})
    metrics_queue.put_nowait(final_message)
    drain_collector_metrics(
        metrics_queue,
        reward_history,
        RewardComponentWindow(),
        logger,
        runner_label="test",
        raise_on_collector_error=True,
        require_buffer_size=True,
    )

    scalars = logger._build_backend_scalars(
        iteration=1,
        metrics=None,
        return_mean_ep100=statistics.mean(reward_history),
        reward_components={},
    )
    assert list(reward_history) == [4.0]
    assert logger._total_steps == 2
    assert scalars["Train/mean_reward"] == 4.0
    assert scalars["Train/mean_episode_length"] == 1.0
    assert scalars["Episode/timeout_rate"] == 0.0
    assert events == ["step", "replay_add", "cleanup"]


def test_inference_request_publish_timeout_is_explicit() -> None:
    requests: queue.Queue[int] = queue.Queue(maxsize=1)
    requests.put_nowait(0)

    with pytest.raises(TimeoutError, match="publishing off-policy inference tick 1"):
        _publish_inference_tick(requests, 1, threading.Event(), timeout=0.01)


def test_inference_response_wait_timeout_is_explicit() -> None:
    with pytest.raises(TimeoutError, match="waiting for off-policy inference tick 0"):
        _wait_for_inference_tick(queue.Queue(), 0, threading.Event(), timeout=0.01)


def test_inference_response_wait_allows_healthy_busy_learner_beyond_deadline() -> None:
    from uni_rl.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    response_queue: queue.Queue[int] = queue.Queue()
    threading.Timer(0.04, response_queue.put, args=(0,)).start()

    assert _wait_for_inference_tick(
        response_queue,
        0,
        threading.Event(),
        learner_coordination=state,
        learner_pid=None,
        timeout=0.01,
    )


def test_inference_response_wait_detects_stopped_or_dead_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uni_rl.offpolicy.coordination import LearnerCoordinationState

    stopped = LearnerCoordinationState()
    stopped.mark_stopped()
    with pytest.raises(RuntimeError, match="Learner stopped before inference tick 0"):
        _wait_for_inference_tick(
            queue.Queue(),
            0,
            threading.Event(),
            learner_coordination=stopped,
            learner_pid=None,
            timeout=0.01,
        )

    alive_phase = LearnerCoordinationState()
    alive_phase.mark_busy()
    monkeypatch.setattr(worker_module, "learner_pid_is_alive", lambda pid: False)
    with pytest.raises(RuntimeError, match="Learner process died before inference tick 0"):
        _wait_for_inference_tick(
            queue.Queue(),
            0,
            threading.Event(),
            learner_coordination=alive_phase,
            learner_pid=12345,
            timeout=0.01,
        )


def test_inference_response_wait_uses_waiting_progress_not_latency() -> None:
    from uni_rl.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_waiting()
    response_queue: queue.Queue[int] = queue.Queue()

    def progress_then_publish() -> None:
        state.mark_progress()
        response_queue.put(0)

    threading.Timer(0.02, progress_then_publish).start()

    assert _wait_for_inference_tick(
        response_queue,
        0,
        threading.Event(),
        learner_coordination=state,
        learner_pid=None,
        timeout=0.01,
    )


def test_learner_stop_releases_inference_response_wait() -> None:
    from uni_rl.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    stop_event = threading.Event()
    threading.Timer(0.01, stop_event.set).start()

    assert not _wait_for_inference_tick(
        queue.Queue(),
        0,
        stop_event,
        learner_coordination=state,
        learner_pid=None,
        timeout=10.0,
    )


def test_stopped_phase_with_stop_request_releases_wait_gracefully() -> None:
    """Normal training completion sets the stop event before STOPPED.

    A collector already inside the wait loop must observe the completed run as
    a graceful release, not as a "Learner stopped" crash.
    """
    from uni_rl.offpolicy.coordination import LearnerCoordinationState

    state = LearnerCoordinationState()
    state.mark_busy()
    stop_event = threading.Event()

    def shutdown() -> None:
        # Mirrors DoubleBufferRunner._shutdown_collector ordering.
        stop_event.set()
        state.mark_stopped()

    threading.Timer(0.01, shutdown).start()

    assert not _wait_for_inference_tick(
        queue.Queue(),
        0,
        stop_event,
        learner_coordination=state,
        learner_pid=None,
        timeout=10.0,
    )


class _DummyActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        assert dones is not None
        self.calls.append((obs.clone(), dones.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def test_extract_env_step_breakdown_timing_ms_maps_env_owned_keys_only() -> None:
    timing = extract_env_step_breakdown_timing_ms(
        {
            "timing": {
                "step_core_ms": 1.5,
                "update_state_ms": 2.5,
                "reset_done_ms": 0.25,
                "action_validate_ms": 0.1,
                "apply_action_ms": 9.0,
            }
        }
    )

    assert timing == {
        "env_step_backend_ms": 1.5,
        "env_step_update_state_ms": 2.5,
        "env_step_reset_done_ms": 0.25,
        "env_step_action_validate_ms": 0.1,
        "env_step_apply_action_ms": 9.0,
    }


@pytest.mark.parametrize("algo_type", ["sac", "flashsac"])
def test_sample_offpolicy_actions_uses_actor_explore(algo_type: str) -> None:
    actor = _DummyActor()
    obs = torch.zeros(4, 5)
    dones = torch.zeros(4)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type=algo_type,
        obs_torch=obs,
        prev_dones_torch=dones,
    )

    assert len(actor.calls) == 1
    assert actor.calls[0][2] is False
    assert actions.shape == (4, 3)


def test_sample_offpolicy_actions_rejects_unknown_algo() -> None:
    actor = _DummyActor()

    with pytest.raises(ValueError, match="learner action sampling"):
        sample_offpolicy_actions(
            actor=actor,
            algo_type="unknown",
            obs_torch=torch.zeros(2, 4),
            prev_dones_torch=torch.zeros(2),
        )


class _DummyPrivInfoActor:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.calls.append((obs.clone(), priv_info.clone(), deterministic))
        return torch.ones(obs.shape[0], 3, dtype=obs.dtype)


def _register_dummy_priv_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from uni_rl.offpolicy import actor_adapter as actor_adapter_module

    def resolve_priv_info(obs_np, critic_np, info):
        if info is not None and info.get("critic_info") is not None:
            return np.asarray(info["critic_info"], dtype=np.float32)
        return np.asarray(critic_np[:, obs_np.shape[1] :], dtype=np.float32)

    monkeypatch.setitem(
        actor_adapter_module._ADAPTERS,
        "dummy_priv_sac",
        actor_adapter_module.OffPolicyActorAdapter(
            algo_type="dummy_priv_sac",
            sample_actions=lambda actor, obs, dones, priv: actor.explore(
                obs, priv, deterministic=False
            ),
            resolve_priv_info=resolve_priv_info,
        ),
    )


def test_sample_offpolicy_actions_uses_adapter_sample_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_dummy_priv_adapter(monkeypatch)
    actor = _DummyPrivInfoActor()
    obs = torch.zeros(4, 5)
    priv_info = torch.randn(4, 2)

    actions = sample_offpolicy_actions(
        actor=actor,
        algo_type="dummy_priv_sac",
        obs_torch=obs,
        prev_dones_torch=torch.zeros(4),
        priv_info_torch=priv_info,
    )

    assert actions.shape == (4, 3)
    assert len(actor.calls) == 1
    torch.testing.assert_close(actor.calls[0][1], priv_info)


def test_resolve_offpolicy_actor_priv_info_prefers_explicit_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    _register_dummy_priv_adapter(monkeypatch)
    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.ones((2, 2), dtype=np.float32)
    critic = np.concatenate([obs, critic_tail], axis=1)
    explicit = np.full((2, 2), 7.0, dtype=np.float32)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="dummy_priv_sac",
        obs_np=obs,
        critic_np=critic,
        info={"critic_info": explicit},
    )

    np.testing.assert_allclose(resolved, explicit)


def test_resolve_offpolicy_actor_priv_info_uses_critic_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    _register_dummy_priv_adapter(monkeypatch)
    obs = np.zeros((2, 3), dtype=np.float32)
    critic_tail = np.arange(4, dtype=np.float32).reshape(2, 2)
    critic = np.concatenate([obs, critic_tail], axis=1)

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="dummy_priv_sac",
        obs_np=obs,
        critic_np=critic,
        info={},
    )

    np.testing.assert_allclose(resolved, critic_tail)


def test_resolve_offpolicy_actor_priv_info_returns_none_without_adapter() -> None:
    import numpy as np

    resolved = resolve_offpolicy_actor_priv_info(
        algo_type="sac",
        obs_np=np.zeros((2, 3), dtype=np.float32),
        critic_np=np.zeros((2, 5), dtype=np.float32),
        info=None,
    )

    assert resolved is None


def test_collector_rejects_transport_metadata_in_env_override() -> None:
    stop_event = threading.Event()
    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(ValueError, match="Collector env override must not set"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: SimpleNamespace(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="cpu",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={"inference_transport": "gpu"},
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


def test_collector_rejects_cuda_transport_with_cpu_ring() -> None:
    stop_event = threading.Event()
    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(ValueError, match="CUDA collector inference requires a CUDA inference ring"):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: SimpleNamespace(),
            num_envs=1,
            replay_buffer=SimpleNamespace(),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={},
            inference_transport="cuda",
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


class _CudaStubDeviceTensor(torch.Tensor):
    """CPU tensor shaped enough for early CUDA-device validation on CPU CI."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")


def test_explicit_cpu_transport_rejects_cuda_tensor_observations() -> None:
    stop_event = threading.Event()

    class _State:
        obs = {
            "obs": torch.zeros((1, 2), device="cuda")
            if torch.cuda.is_available()
            else _CudaStubDeviceTensor()
        }
        info = {}

    class _Env:
        state = _State()

    ring = SharedInferenceRing(1, 2, 2, device="cpu")

    with pytest.raises(
        ValueError, match="CPU inference transport received CUDA tensor observations"
    ):
        worker_module._run_collector(
            stop_event=stop_event,
            env_factory=lambda num_envs, env_cfg_override=None: _Env(),
            num_envs=1,
            replay_buffer=SimpleNamespace(
                trace_recorder=None,
                trace_thread_time=False,
                attach_stop_event=lambda stop: None,
            ),
            inference_slot=ring,
            inference_request_queue=queue.Queue(),
            inference_response_queue=queue.Queue(),
            algo_type="sac",
            actor_adapter_modules=None,
            metrics_queue=queue.Queue(),
            inference_transport="cpu",
            sim_backend="mujoco",
            backend_device=None,
            env_cfg_override={},
            seed=None,
            trace_enabled=False,
            trace_thread_time=False,
        )


@pytest.mark.parametrize("as_tensor", [False, True])
def test_per_env_metric_rewards_passthrough_and_head_mean(as_tensor: bool) -> None:
    from uni_rl.offpolicy.worker import _per_env_metric_rewards

    flat_values = [1.0, 2.0, 3.0]
    flat = torch.tensor(flat_values) if as_tensor else np.asarray(flat_values, dtype=np.float32)
    assert _per_env_metric_rewards(flat, 3) is flat

    head_values = [[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]]
    heads = torch.tensor(head_values) if as_tensor else np.asarray(head_values, dtype=np.float32)
    reduced = _per_env_metric_rewards(heads, 3)
    torch.testing.assert_close(
        torch.as_tensor(np.asarray(reduced)),
        torch.tensor([2.0, 3.0, 6.0]),
    )


def test_tensor_collector_folds_n_step_rows_and_publishes_raw_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scripted 8-step rollout through _run_collector with n_step=3.

    The fake env drives a deterministic reward/done script across two envs;
    captured replay rows must equal the reference n-step fold, and the raw
    stats ring must hold every raw per-step reward.
    """
    stop_event = threading.Event()
    inference_request_queue: queue.Queue[int] = queue.Queue()
    inference_response_queue: queue.Queue[int] = queue.Queue()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=8)
    replay_rows: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    n_envs, n_steps, gamma, obs_dim = 2, 3, 0.5, 2
    reward_script = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [4.0, 40.0],
            [8.0, 80.0],
            [16.0, 160.0],
            [32.0, 320.0],
            [64.0, 640.0],
            [128.0, 1280.0],
        ]
    )
    done_script = torch.zeros(8, n_envs)
    done_script[4, 0] = 1.0
    final_obs_script = torch.full((8, n_envs, obs_dim), 99.0)
    env_obs = [torch.full((n_envs, obs_dim), float(t)) for t in range(9)]

    class _Env:
        def __init__(self) -> None:
            self.t = 0
            self.state = SimpleNamespace(
                obs={"obs": env_obs[0]},
                reward=torch.zeros(n_envs),
                terminated=torch.zeros(n_envs, dtype=torch.bool),
                truncated=torch.zeros(n_envs, dtype=torch.bool),
                final_observation=None,
                info={"timing": {}},
            )

        def step(self, actions):
            del actions
            t = self.t
            dones = done_script[t]
            self.state = SimpleNamespace(
                obs={"obs": env_obs[t + 1]},
                reward=reward_script[t],
                terminated=dones.bool(),
                truncated=torch.zeros(n_envs, dtype=torch.bool),
                final_observation=({"obs": final_obs_script[t]} if bool(dones.any()) else None),
                info={"timing": {}},
            )
            self.t += 1
            if self.t >= 8:
                stop_event.set()
            return self.state

        def close(self):
            return None

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            replay_rows.append((args, kwargs))
            return True

        def release_ipc(self) -> None:
            return None

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *a, **kw: None)
    ring_ipc = SharedInferenceRing(n_envs, obs_dim, obs_dim, capacity=4)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring_ipc.copy_observation_to(
            tick_id=tick_id,
            observations=torch.zeros(n_envs, obs_dim),
            dones=torch.zeros(n_envs),
            epoch=0,
        )
        ring_ipc.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros(n_envs, obs_dim),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    from uni_rl.ipc.reward_stats_ring import RawRewardStatsRing

    stats_ring = RawRewardStatsRing(num_envs=n_envs, reward_heads=1, depth=16)
    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=n_envs,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring_ipc,
        inference_request_queue=inference_request_queue,
        inference_response_queue=inference_response_queue,
        algo_type="flashsac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
        n_step=n_steps,
        n_step_gamma=gamma,
        reward_stats_ring=stats_ring,
    )

    # 8 env steps -> 6 folded rows (the first n-1 steps emit nothing).
    assert len(replay_rows) == 8 - n_steps + 1

    def ref_reward(env: int, start: int) -> torch.Tensor:
        r = reward_script[start + 2, env]
        for index in reversed(range(start, start + 2)):
            d = done_script[index, env]
            r = reward_script[index, env] + gamma * r * (1.0 - d)
        return r

    for emit_index, (args, kwargs) in enumerate(replay_rows):
        obs, actions, rewards, next_obs, dones, truncated = args
        step = emit_index
        assert torch.equal(obs, env_obs[step])
        for env in range(n_envs):
            assert torch.equal(rewards[env], ref_reward(env, step))
        # Span-end done flags and next_obs rows.
        span_done = done_script[step : step + n_steps, :].any(dim=0)
        for env in range(n_envs):
            first_done_rel = (
                int(torch.nonzero(done_script[step : step + n_steps, env]).min())
                if span_done[env]
                else None
            )
            if first_done_rel is not None:
                end_step = step + first_done_rel
                assert dones[env] == 1.0
                # Timeout/terminated span end keeps the true final obs via the
                # terminal patch fields (final_obs = 99.0 here).
                terminal_next_obs = kwargs["terminal_next_obs"]
                assert terminal_next_obs is not None
                assert torch.equal(terminal_next_obs[env], final_obs_script[end_step][env])
                # No critic group: the critic row falls back to the actor obs,
                # so its terminal patch must carry the same final obs.
                terminal_next_critic = kwargs["terminal_next_critic"]
                assert terminal_next_critic is not None
                assert torch.equal(terminal_next_critic[env], final_obs_script[end_step][env])
            else:
                assert dones[env] == 0.0
                assert torch.equal(next_obs[env], env_obs[step + n_steps][env])

    # Raw per-step rewards reached the stats ring untouched by the fold.
    raw_rewards, raw_dones, read = stats_ring.drain()
    assert read == 8
    assert raw_rewards.shape == (8, n_envs, 1)
    assert torch.equal(raw_rewards[:, 0, 0], reward_script[:, 0])
    assert torch.equal(raw_rewards[:, 1, 0], reward_script[:, 1])
    assert torch.equal(raw_dones, done_script)


@pytest.mark.parametrize("tensor_env", [True, False])
def test_collector_encodes_bootstrap_eligible_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tensor_env: bool,
) -> None:
    """uni_rl #29: failure+timeout overlap must reach replay as truncated=0.

    Four envs cover the full termination table — normal, failure-only,
    timeout-only, and failure+timeout — through the real collector loop and
    the n_step=1 fold. Episode ``dones`` stays ``terminated | truncated``;
    replay's ``truncated`` column is bootstrap-eligible truncation
    (``truncated & ~terminated``). The captured replay row is then fed
    straight into the FlashSAC categorical TD target to close the
    collector -> replay -> learner chain: only normal and timeout-only rows
    bootstrap; both failure variants project to the reward-only support.
    """
    stop_event = threading.Event()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
    replay_rows: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    n_envs, obs_dim = 4, 2
    rewards = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    terminated = np.asarray([False, True, False, True], dtype=bool)
    truncated = np.asarray([False, False, True, True], dtype=bool)
    obs0 = np.zeros((n_envs, obs_dim), dtype=np.float32)
    obs1 = np.ones((n_envs, obs_dim), dtype=np.float32)
    final_obs = np.asarray([[10.0 + env, 20.0 + env] for env in range(n_envs)], dtype=np.float32)

    def as_env_value(value: np.ndarray):
        return torch.from_numpy(value.copy()) if tensor_env else value

    class _Env:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                obs={"obs": as_env_value(obs0)},
                reward=as_env_value(np.zeros(n_envs, dtype=np.float32)),
                terminated=as_env_value(np.zeros(n_envs, dtype=bool)),
                truncated=as_env_value(np.zeros(n_envs, dtype=bool)),
                final_observation=None,
                info={"timing": {}},
            )

        def step(self, actions):
            del actions
            self.state = SimpleNamespace(
                obs={"obs": as_env_value(obs1)},
                reward=as_env_value(rewards),
                terminated=as_env_value(terminated),
                truncated=as_env_value(truncated),
                final_observation={"obs": as_env_value(final_obs)},
                info={"timing": {}},
            )
            stop_event.set()
            return self.state

        def close(self):
            return None

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            replay_rows.append((args, kwargs))
            return True

        def release_ipc(self) -> None:
            return None

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *a, **kw: None)
    ring_ipc = SharedInferenceRing(n_envs, obs_dim, obs_dim, capacity=4)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring_ipc.copy_observation_to(
            tick_id=tick_id,
            observations=torch.zeros(n_envs, obs_dim),
            dones=torch.zeros(n_envs),
            epoch=0,
        )
        ring_ipc.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros(n_envs, obs_dim),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=n_envs,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring_ipc,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="flashsac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    assert len(replay_rows) == 1
    args, kwargs = replay_rows[0]
    obs, actions, replay_rewards, next_obs, dones, replay_truncated = args
    del obs, actions, replay_rewards
    assert torch.equal(next_obs, torch.ones((n_envs, obs_dim)))
    assert torch.equal(dones, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(replay_truncated, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    # The terminal patch preserves the true final observation for every done
    # row (failure and timeout alike); bootstrap eligibility is carried only
    # by the encoded `truncated` column.
    assert torch.equal(kwargs["terminal_mask"], torch.tensor([False, True, True, True]))
    assert torch.equal(kwargs["terminal_next_obs"], torch.from_numpy(final_obs))
    # The env publishes no critic group, so replay's critic row is the actor
    # obs; a timeout row's next_critic must be patched with the same final
    # obs, not left as the post-reset observation of the next episode.
    assert torch.equal(kwargs["next_critic"], torch.ones((n_envs, obs_dim)))
    assert torch.equal(kwargs["terminal_next_critic"], torch.from_numpy(final_obs))

    support = torch.tensor([0.0, 1.0, 2.0])
    target_log_probs = torch.log(torch.tensor([[0.0, 0.0, 1.0]] * n_envs).clamp_min(1e-8))
    targets = compute_categorical_td_target(
        support=support,
        target_log_probs=target_log_probs,
        reward=torch.zeros(n_envs),
        dones=dones,
        truncated=replay_truncated,
        actor_entropy=torch.zeros(n_envs),
        gamma=1.0,
    )
    torch.testing.assert_close(targets[0], torch.tensor([0.0, 0.0, 1.0]))  # normal
    torch.testing.assert_close(targets[1], torch.tensor([1.0, 0.0, 0.0]))  # failure
    torch.testing.assert_close(targets[2], torch.tensor([0.0, 0.0, 1.0]))  # timeout
    torch.testing.assert_close(targets[3], torch.tensor([1.0, 0.0, 0.0]))  # overlap


def _halving_ingress_transformer_factory(params):
    del params

    def transform(transition):
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


@pytest.mark.parametrize("n_step", [1, 2])
def test_collector_compacts_full_obs_before_replay_ingress(
    monkeypatch: pytest.MonkeyPatch,
    n_step: int,
) -> None:
    """The ingress transformer sees full env obs; replay rows store compact obs.

    The env and inference ring keep the full 6/4-dimensional actor/critic
    observation; the transformer factory (resolved in the collector process)
    halves observation-bearing columns before the n-step fold and the replay
    write. Rewards, flags, aux, and raw reward-stats accounting must pass
    through untouched.
    """
    from uni_rl.ipc.reward_stats_ring import RawRewardStatsRing

    stop_event = threading.Event()
    metrics_queue: queue.Queue[dict] = queue.Queue(maxsize=8)
    replay_rows: list[tuple[tuple[torch.Tensor, ...], dict[str, object]]] = []

    n_envs, full_obs_dim, full_critic_dim = 2, 6, 4
    n_steps = n_step + 1
    reward_script = torch.arange(1.0, float(n_steps) + 1.0).unsqueeze(1).repeat(1, n_envs)
    aux_script = torch.stack(
        [
            torch.arange(n_steps, dtype=torch.int64).unsqueeze(1).expand(n_steps, n_envs),
            torch.arange(100, 100 + n_steps, dtype=torch.int64)
            .unsqueeze(1)
            .expand(n_steps, n_envs),
        ],
        dim=-1,
    )
    env_obs = [torch.full((n_envs, full_obs_dim), float(t)) for t in range(n_steps + 1)]
    env_critic = [torch.full((n_envs, full_critic_dim), float(t) + 0.5) for t in range(n_steps + 1)]
    final_obs = torch.full((n_envs, full_obs_dim), 99.0)

    class _Env:
        def __init__(self) -> None:
            self.t = 0
            self.state = SimpleNamespace(
                obs={"obs": env_obs[0], "critic": env_critic[0]},
                reward=torch.zeros(n_envs),
                terminated=torch.zeros(n_envs, dtype=torch.bool),
                truncated=torch.zeros(n_envs, dtype=torch.bool),
                final_observation=None,
                info={},
            )

        def step(self, actions):
            del actions
            t = self.t
            done = t == n_steps - 1
            self.state = SimpleNamespace(
                obs={"obs": env_obs[t + 1], "critic": env_critic[t + 1]},
                reward=reward_script[t],
                terminated=torch.full((n_envs,), done, dtype=torch.bool),
                truncated=torch.zeros(n_envs, dtype=torch.bool),
                final_observation={"obs": final_obs, "critic": final_obs[..., :full_critic_dim]},
                info={REPLAY_AUX_INFO_KEY: aux_script[t]},
            )
            self.t += 1
            if self.t >= n_steps:
                stop_event.set()
            return self.state

        def close(self):
            return None

    class _ReplayBuffer:
        trace_recorder = None
        trace_thread_time = False
        size = torch.zeros(1, dtype=torch.int64)

        def attach_stop_event(self, stop) -> None:
            del stop

        def add_batch(self, *args, **kwargs) -> bool:
            replay_rows.append((args, kwargs))
            return True

        def release_ipc(self) -> None:
            return None

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_training_seed", lambda *a, **kw: None)
    ring_ipc = SharedInferenceRing(n_envs, full_obs_dim, full_obs_dim, capacity=4)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring_ipc.copy_observation_to(
            tick_id=tick_id,
            observations=torch.zeros(n_envs, full_obs_dim),
            dones=torch.zeros(n_envs),
            epoch=0,
        )
        ring_ipc.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros(n_envs, full_obs_dim),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    stats_ring = RawRewardStatsRing(num_envs=n_envs, reward_heads=1, depth=16)
    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=n_envs,
        replay_buffer=_ReplayBuffer(),
        inference_slot=ring_ipc,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="flashsac",
        actor_adapter_modules=None,
        metrics_queue=metrics_queue,
        inference_transport="legacy_tensor",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
        n_step=n_step,
        n_step_gamma=0.99,
        reward_stats_ring=stats_ring,
        replay_ingress_transformer=_halving_ingress_transformer_factory,
        replay_ingress_transformer_params={"replay_obs_dim": full_obs_dim // 2},
    )

    assert len(replay_rows) == n_steps - n_step + 1
    for args, kwargs in replay_rows:
        obs, actions, rewards, next_obs, dones, truncated = args
        del actions, dones, truncated
        assert obs.shape == (n_envs, full_obs_dim // 2)
        assert next_obs.shape == (n_envs, full_obs_dim // 2)
        assert kwargs["critic"].shape == (n_envs, full_critic_dim // 2)
        assert kwargs["next_critic"].shape == (n_envs, full_critic_dim // 2)
        assert kwargs["terminal_next_obs"].shape == (n_envs, full_obs_dim // 2)
        assert kwargs["terminal_next_critic"].shape == (n_envs, full_critic_dim // 2)
        assert kwargs["aux"].shape == (n_envs, 2)
        assert kwargs["aux"].dtype == torch.int64
        # Rewards and flags pass through the compaction boundary unchanged.
        assert rewards.ndim == 1

    # Raw reward stats see every full-width env step, never compact columns.
    raw_rewards, raw_dones, read = stats_ring.drain()
    assert read == n_steps
    assert raw_rewards.shape == (n_steps, n_envs, 1)
    assert torch.equal(raw_rewards[:, :, 0], reward_script)
    assert raw_dones.sum() == float(n_envs)
