from __future__ import annotations

import shutil

import pytest

from uni_rl.ipc.inference_ring import estimate_inference_ring_bytes
from uni_rl.ipc.memory_budget import (
    CUDA_INFERENCE_EVENT_RESERVE_BYTES,
    CUDA_INFERENCE_MIN_OVERHEAD_BYTES,
    CUDA_INFERENCE_TIMING_EVENT_COUNT,
    estimate_cuda_inference_ipc_bytes,
    estimate_cuda_tensor_runtime_bytes,
    estimate_offpolicy_bytes,
    raise_if_cuda_memory_over_budget,
    raise_if_shared_memory_over_budget,
)


def _tensor_runtime_budget(*, collector_tensor_native: bool) -> dict[str, int | str]:
    return estimate_cuda_tensor_runtime_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        inference_obs_dim=4,
        sample_count=8,
        inference_slot_capacity=1,
        replay_ingress_depth=2,
        replay_ingress_slot_rows=2,
        collector_tensor_native=collector_tensor_native,
    )


def test_offpolicy_memory_budget_notes_native_exclusions() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=5120,
        replay_buffer_n=1024,
        obs_dim=98,
        action_dim=29,
        critic_dim=101,
    )

    breakdown = str(estimate["breakdown"])
    assert "MuJoCo BatchEnvPool" in breakdown
    assert "CUDA pinned/shared" in breakdown
    assert "driver memory" in breakdown


def test_device_replay_host_budget_is_independent_of_replay_capacity() -> None:
    estimates = [
        estimate_offpolicy_bytes(
            num_envs=10,
            replay_buffer_n=replay_buffer_n,
            obs_dim=2,
            action_dim=1,
            critic_dim=3,
            ingress_depth=2,
        )
        for replay_buffer_n in (4, 4000)
    ]

    assert estimates[0]["replay_buffer"] == 0
    assert estimates[0]["bounded_ingress_slots"] > 0
    assert estimates[0]["total"] == estimates[1]["total"]
    assert "authoritative on the learner device" in str(estimates[0]["breakdown"])


def test_device_replay_host_budget_excludes_gpu_collector_ingress() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        ingress_on_device=True,
    )

    assert estimate["bounded_ingress_slots"] == 0
    assert estimate["total"] == 0
    assert "device-resident" in str(estimate["breakdown"])


def test_host_replay_budget_uses_effective_ingress_slot_rows() -> None:
    default_rows = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
    )
    custom_rows = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        ingress_slot_rows=1,
    )

    assert default_rows["ingress_slot_rows"] == 4
    assert custom_rows["ingress_slot_rows"] == 1
    assert custom_rows["bounded_ingress_slots"] * 4 == default_rows["bounded_ingress_slots"]
    assert "1 rows" in str(custom_rows["breakdown"])


@pytest.mark.parametrize("value", [0, -1, True, "1", 1.0, 5])
def test_host_replay_budget_rejects_invalid_ingress_slot_rows(value: object) -> None:
    with pytest.raises(ValueError, match="ingress_slot_rows"):
        estimate_offpolicy_bytes(
            num_envs=4,
            replay_buffer_n=8,
            obs_dim=4,
            action_dim=2,
            critic_dim=5,
            ingress_slot_rows=value,
        )


def test_host_budget_includes_cpu_inference_ring_storage() -> None:
    estimate = estimate_offpolicy_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        inference_ring_bytes=168,
    )

    assert estimate["inference_ring"] == 168
    assert estimate["total"] == estimate["bounded_ingress_slots"] + 168
    assert "Inference ring" in str(estimate["breakdown"])
    assert estimate_inference_ring_bytes(2, 4, 2, capacity=3) == 168


def test_cuda_inference_budget_accounts_for_all_startup_categories() -> None:
    persistent_scratch = 2 * (2 * 4 + 2 * 4)
    estimate = estimate_cuda_inference_ipc_bytes(
        2,
        4,
        2,
        capacity=3,
        ring_on_device=True,
        persistent_exploration_scratch=persistent_scratch,
    )
    exact = 168 + 2 * 4 * 4 + 2 * 4 + 2 * 2 * 4 + persistent_scratch

    assert estimate["ring_storage"] == 168
    assert estimate["observation_scratch"] == 32
    assert estimate["done_scratch"] == 8
    assert estimate["action_output"] == 16
    assert estimate["persistent_exploration_scratch"] == persistent_scratch
    assert estimate["ipc_event_count"] == 9
    assert estimate["timing_event_count"] == CUDA_INFERENCE_TIMING_EVENT_COUNT
    assert estimate["cuda_event_count"] == 11
    assert estimate["cuda_events"] == 11 * CUDA_INFERENCE_EVENT_RESERVE_BYTES
    assert estimate["conservative_overhead"] == CUDA_INFERENCE_MIN_OVERHEAD_BYTES
    assert estimate["total"] == exact + estimate["cuda_events"] + estimate["conservative_overhead"]
    assert "CUDA inference IPC budget" in str(estimate["breakdown"])

    cpu_ring = estimate_cuda_inference_ipc_bytes(
        2,
        4,
        2,
        capacity=3,
        ring_on_device=False,
    )
    assert cpu_ring["ring_storage"] == 0
    assert cpu_ring["ipc_event_count"] == 0
    assert cpu_ring["cuda_events"] == (
        CUDA_INFERENCE_TIMING_EVENT_COUNT * CUDA_INFERENCE_EVENT_RESERVE_BYTES
    )


def test_cuda_budget_threshold_and_actionable_error() -> None:
    estimate = estimate_cuda_inference_ipc_bytes(2, 4, 2, capacity=3, ring_on_device=True)

    raise_if_cuda_memory_over_budget(
        estimate,
        label="Off-policy (sac)",
        available_bytes=int(estimate["total"]) * 10,
        threshold=0.8,
    )

    with pytest.raises(MemoryError) as excinfo:
        raise_if_cuda_memory_over_budget(
            estimate,
            label="Off-policy (sac)",
            available_bytes=int(estimate["total"]),
            threshold=0.8,
        )

    message = str(excinfo.value)
    assert "Off-policy (sac)" in message
    assert "conservative CUDA inference IPC budget" in message
    assert "80% limit" in message
    assert "device free" in message
    assert "training.inference_slot_capacity" in message
    assert "Inference ring storage" in message


def test_cuda_tensor_runtime_budget_combines_inference_and_replay() -> None:
    native = _tensor_runtime_budget(collector_tensor_native=True)
    host_bridge = _tensor_runtime_budget(collector_tensor_native=False)
    row_width = 23
    replay_storage = 4 * 2 * row_width * 4
    learner_batches = 2 * 8 * row_width * 4
    native_ingress = 2 * 2 * row_width * 4

    assert native["replay_storage"] == replay_storage
    assert native["learner_batch_slots"] == learner_batches
    assert native["replay_ingress"] == native_ingress
    assert host_bridge["replay_ingress"] == 0
    assert native["total"] > int(native["inference_total"])
    assert "CUDA tensor-runtime budget" in str(native["breakdown"])
    assert "before any tensor-runtime allocation" in str(native["breakdown"])


def test_cuda_tensor_runtime_budget_scales_with_dimensions() -> None:
    baseline = _tensor_runtime_budget(collector_tensor_native=True)
    changed = estimate_cuda_tensor_runtime_bytes(
        num_envs=2,
        replay_buffer_n=4,
        obs_dim=5,
        action_dim=2,
        critic_dim=6,
        inference_obs_dim=6,
        sample_count=16,
        inference_slot_capacity=1,
        replay_ingress_depth=2,
        replay_ingress_slot_rows=2,
        collector_tensor_native=True,
    )

    assert changed["replay_row_width"] == 27
    assert changed["replay_storage"] == 4 * 2 * 27 * 4
    assert changed["learner_batch_slots"] == 2 * 16 * 27 * 4
    assert changed["replay_ingress"] == 2 * 2 * 27 * 4
    assert changed["replay_storage"] > baseline["replay_storage"]
    assert changed["learner_batch_slots"] > baseline["learner_batch_slots"]
    assert changed["replay_ingress"] > baseline["replay_ingress"]
    assert changed["inference_total"] > baseline["inference_total"]


def test_cuda_tensor_runtime_budget_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="no greater than num_envs"):
        estimate_cuda_tensor_runtime_bytes(
            num_envs=2,
            replay_buffer_n=4,
            obs_dim=4,
            action_dim=2,
            critic_dim=5,
            inference_obs_dim=4,
            sample_count=8,
            inference_slot_capacity=1,
            replay_ingress_depth=2,
            replay_ingress_slot_rows=3,
            collector_tensor_native=True,
        )


@pytest.mark.parametrize("depth", [0, -1, True, "2", 1.0])
def test_cuda_tensor_runtime_budget_rejects_invalid_ingress_depth(depth: object) -> None:
    with pytest.raises((TypeError, ValueError), match="dimensions must be"):
        estimate_cuda_tensor_runtime_bytes(
            num_envs=2,
            replay_buffer_n=4,
            obs_dim=4,
            action_dim=2,
            critic_dim=5,
            inference_obs_dim=4,
            sample_count=8,
            inference_slot_capacity=1,
            replay_ingress_depth=depth,
            replay_ingress_slot_rows=2,
            collector_tensor_native=True,
        )


def test_shared_memory_budget_unknown_available_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: (_ for _ in ()).throw(OSError()))
    estimate = {"total": 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", path="/missing-shm")


def test_shared_memory_budget_allows_within_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 80 * 1024, "breakdown": "test"}

    raise_if_shared_memory_over_budget(estimate, label="test", threshold=0.8)


def test_shared_memory_budget_raises_before_over_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Usage:
        free = 100 * 1024

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _Usage())
    estimate = {"total": 81 * 1024, "breakdown": "test"}

    with pytest.raises(MemoryError) as excinfo:
        raise_if_shared_memory_over_budget(estimate, label="Off-policy (sac)", threshold=0.8)

    message = str(excinfo.value)
    assert "Off-policy (sac)" in message
    assert "/dev/shm" in message
    assert "estimated" in message
    assert "available" in message
    assert "algo.num_envs" in message
    assert "algo.replay_buffer_n" in message


def test_budget_estimates_follow_row_schema_storage_bytes() -> None:
    from uni_rl.ipc.row_schema import RowSchema

    default_estimate = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
    )
    default_schema = RowSchema(obs_dim=4, action_dim=2, critic_dim=5)
    schema_estimate = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        row_schema=default_schema,
    )
    # The default schema reproduces the legacy fp32 row bytes exactly.
    assert schema_estimate["bounded_ingress_slots"] == default_estimate["bounded_ingress_slots"]

    fp16_schema = RowSchema(obs_dim=4, action_dim=2, critic_dim=5, float_dtype="fp16")
    fp16_estimate = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        row_schema=fp16_schema,
    )
    assert 0 < fp16_estimate["bounded_ingress_slots"] < default_estimate["bounded_ingress_slots"]

    wide_schema = RowSchema(obs_dim=4, action_dim=2, critic_dim=5, reward_heads=2, aux_width=2)
    wide_estimate = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        row_schema=wide_schema,
    )
    expected_row_bytes = (2 * 4 + 2 + 3 + 2 * 5 + 1) * 4 + 2 * 4
    assert wide_schema.storage_row_bytes == expected_row_bytes
    assert wide_estimate["bounded_ingress_slots"] == 2 * 4 * expected_row_bytes


def test_cuda_tensor_runtime_budget_follows_row_schema_storage_bytes() -> None:
    from uni_rl.ipc.row_schema import RowSchema

    base_kwargs = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "obs_dim": 4,
        "action_dim": 2,
        "critic_dim": 5,
        "inference_obs_dim": 4,
        "sample_count": 8,
        "inference_slot_capacity": 1,
        "replay_ingress_depth": 2,
        "replay_ingress_slot_rows": 2,
        "collector_tensor_native": True,
    }
    default_estimate = estimate_cuda_tensor_runtime_bytes(**base_kwargs)
    default_schema = RowSchema(obs_dim=4, action_dim=2, critic_dim=5)
    schema_estimate = estimate_cuda_tensor_runtime_bytes(**base_kwargs, row_schema=default_schema)
    assert schema_estimate["replay_storage"] == default_estimate["replay_storage"]
    assert schema_estimate["total"] == default_estimate["total"]

    fp16_schema = RowSchema(obs_dim=4, action_dim=2, critic_dim=5, float_dtype="fp16")
    fp16_estimate = estimate_cuda_tensor_runtime_bytes(**base_kwargs, row_schema=fp16_schema)
    expected_storage = (4 * 8) * fp16_schema.storage_row_bytes
    assert fp16_estimate["replay_storage"] == expected_storage
    assert fp16_estimate["replay_storage"] < default_estimate["replay_storage"]


def test_budget_accounts_cpu_layered_replay_on_host() -> None:
    base = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
    )
    layered = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=4,
        action_dim=2,
        critic_dim=5,
        ingress_depth=2,
        replay_storage_owner="cpu",
    )
    row_bytes = (2 * 4 + 2 + 3 + 2 * 5) * 4
    assert base["replay_buffer"] == 0
    assert layered["replay_buffer"] == 4 * 8 * row_bytes
    assert layered["total"] == base["total"] + 4 * 8 * row_bytes
    with pytest.raises(ValueError, match="replay_storage_owner"):
        estimate_offpolicy_bytes(
            num_envs=4,
            replay_buffer_n=8,
            obs_dim=4,
            action_dim=2,
            critic_dim=5,
            replay_storage_owner="ssd",
        )


def test_cuda_budget_excludes_cpu_layered_replay_storage() -> None:
    base_kwargs = {
        "num_envs": 4,
        "replay_buffer_n": 8,
        "obs_dim": 4,
        "action_dim": 2,
        "critic_dim": 5,
        "inference_obs_dim": 4,
        "sample_count": 8,
        "inference_slot_capacity": 1,
        "replay_ingress_depth": 2,
        "replay_ingress_slot_rows": 2,
        "collector_tensor_native": True,
    }
    resident = estimate_cuda_tensor_runtime_bytes(**base_kwargs)
    layered = estimate_cuda_tensor_runtime_bytes(**base_kwargs, replay_storage_owner="cpu")
    assert layered["replay_storage"] == 0
    assert layered["learner_batch_slots"] == resident["learner_batch_slots"]
    assert layered["replay_ingress"] == resident["replay_ingress"]
    assert layered["total"] < resident["total"]


def test_host_memory_gate_fails_closed_and_skips_without_meminfo(monkeypatch) -> None:
    import uni_rl.ipc.replay_pipelines.layered as layered_module
    from uni_rl.ipc.memory_budget import raise_if_host_memory_over_budget

    monkeypatch.setattr(layered_module, "host_available_memory_bytes", lambda: 10 * 1024**3)
    with pytest.raises(MemoryError, match="host RAM"):
        raise_if_host_memory_over_budget({"total": 9 * 1024**3}, label="test", threshold=0.8)
    raise_if_host_memory_over_budget({"total": 7 * 1024**3}, label="test", threshold=0.8)

    monkeypatch.setattr(layered_module, "host_available_memory_bytes", lambda: None)
    with pytest.warns(UserWarning, match="budget check skipped"):
        raise_if_host_memory_over_budget({"total": 999 * 1024**3}, label="test")


def test_budget_drops_next_obs_groups_for_reconstruction() -> None:
    from uni_rl.ipc.row_schema import RowSchema

    schema = RowSchema(obs_dim=8, action_dim=3, critic_dim=3, reward_heads=2, aux_width=2)
    dense = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=8,
        action_dim=3,
        critic_dim=3,
        replay_storage_owner="cpu",
        row_schema=schema,
    )
    recon = estimate_offpolicy_bytes(
        num_envs=4,
        replay_buffer_n=8,
        obs_dim=8,
        action_dim=3,
        critic_dim=3,
        replay_storage_owner="cpu",
        row_schema=schema,
        reconstruct_next_obs=True,
    )
    ring_schema = schema.without_next_obs()
    assert recon["replay_buffer"] == 4 * 8 * ring_schema.storage_row_bytes
    # Ingress slots stay dense; only the pinned ring shrinks.
    assert recon["bounded_ingress_slots"] == dense["bounded_ingress_slots"]
    assert recon["replay_buffer"] < dense["replay_buffer"]
