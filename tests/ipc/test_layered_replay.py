"""Tests for the CPU-pinned layered (UVA/H2D) replay pipeline."""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from uni_rl.ipc.replay_buffer import ReplayBuffer
from uni_rl.ipc.replay_pipelines.layered import (
    LayeredReplayPipeline,
    host_available_memory_bytes,
)
from uni_rl.ipc.row_schema import RowSchema

_HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")

_OBS_DIM = 8
_ACTION_DIM = 3
_CRITIC_DIM = 3
_ROWS = 64


def _schema(**overrides) -> RowSchema:
    kwargs: dict[str, Any] = {
        "obs_dim": _OBS_DIM,
        "action_dim": _ACTION_DIM,
        "critic_dim": _CRITIC_DIM,
        "reward_heads": 2,
        "aux_width": 2,
    }
    kwargs.update(overrides)
    return RowSchema(**kwargs)


def _make_replay(*, ingress_device: str = "cpu", schema: RowSchema | None = None) -> ReplayBuffer:
    schema = schema or _schema()
    return ReplayBuffer(
        capacity=_ROWS,
        obs_dim=schema.obs_dim,
        action_dim=schema.action_dim,
        device="cuda",
        critic_dim=schema.critic_dim,
        ingress_slot_rows=_ROWS,
        ingress_device=ingress_device,
        row_schema=schema,
    )


def _pattern(n: int, schema: RowSchema) -> dict[str, torch.Tensor]:
    rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    return {
        "obs": rows * 10 + torch.arange(schema.obs_dim, dtype=torch.float32),
        "next_obs": rows * 100 + torch.arange(schema.obs_dim, dtype=torch.float32),
        "actions": rows * 1000 + torch.arange(schema.action_dim, dtype=torch.float32),
        "rewards": rows * 7 + torch.arange(schema.reward_heads, dtype=torch.float32),
        "dones": torch.zeros(n),
        "truncated": torch.ones(n),
        "critic": rows * 5 + torch.arange(schema.critic_dim, dtype=torch.float32),
        "next_critic": rows * 6 + torch.arange(schema.critic_dim, dtype=torch.float32),
        "aux": (rows * 3 + torch.arange(schema.aux_width, dtype=torch.float32)).to(torch.int64),
    }


def _fill(rb: ReplayBuffer, pattern: dict[str, torch.Tensor], *, device: str = "cpu") -> None:
    tensors = {key: value.to(device) for key, value in pattern.items()}
    assert rb.add(
        tensors["obs"],
        tensors["actions"],
        tensors["rewards"],
        tensors["next_obs"],
        tensors["dones"],
        tensors["truncated"],
        critic=tensors["critic"],
        next_critic=tensors["next_critic"],
        aux=tensors["aux"] if rb._schema.aux_width > 0 else None,
    )


def _wait_visible(pipeline: LayeredReplayPipeline, ptr: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while pipeline._visible_ptr < ptr:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Layered replay stalled: visible_ptr={pipeline._visible_ptr} < {ptr}"
            )
        time.sleep(0.005)


@pytest.fixture
def pipeline_factory():
    created = []

    def _make(rb, **kwargs):
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("sample_count", 16)
        pipeline = LayeredReplayPipeline(rb, **kwargs)
        created.append(pipeline)
        return pipeline

    yield _make
    for pipeline in created:
        pipeline.close()


def test_host_available_memory_bytes_reads_meminfo() -> None:
    available = host_available_memory_bytes()
    assert available is None or available > 0


def test_layered_transport_string_is_validated() -> None:
    with pytest.raises(ValueError, match="transport"):
        LayeredReplayPipeline(
            _make_replay(),
            device="cuda" if _HAS_CUDA else "cpu",
            sample_count=4,
            transport="bogus",
        )


@cuda_only
def test_host_ram_budget_gate_fails_closed(pipeline_factory, monkeypatch) -> None:
    import uni_rl.ipc.replay_pipelines.layered as layered_module

    monkeypatch.setattr(
        layered_module,
        "host_available_memory_bytes",
        lambda: 1024,  # 1 KiB: any layered storage exceeds the gate
    )
    with pytest.raises(RuntimeError, match="does not fit in host RAM"):
        pipeline_factory(_make_replay(), sample_count=4)


@cuda_only
def test_uva_transport_requested_but_unavailable_fails_closed(monkeypatch) -> None:
    import uni_rl.ipc.replay_pipelines.layered as layered_module

    monkeypatch.setattr(layered_module, "load_extension", lambda: None)
    monkeypatch.setattr(layered_module, "get_diagnostic", lambda: "test-unavailable")
    with pytest.raises(RuntimeError, match="test-unavailable"):
        LayeredReplayPipeline(_make_replay(), device="cuda", sample_count=4, transport="uva")


@cuda_only
def test_auto_transport_falls_back_to_h2d(monkeypatch) -> None:
    import uni_rl.ipc.replay_pipelines.layered as layered_module

    monkeypatch.setattr(layered_module, "load_extension", lambda: None)
    monkeypatch.setattr(layered_module, "get_diagnostic", lambda: "test-unavailable")
    pipeline = LayeredReplayPipeline(
        _make_replay(), device="cuda", sample_count=4, transport="auto"
    )
    assert pipeline.sampling_transport == "h2d"
    pipeline.close()


@cuda_only
class TestLayeredReplayPipeline:
    @pytest.mark.parametrize("transport", ["uva", "h2d"])
    def test_schema_roundtrip_sampling_and_committed_fields(
        self, pipeline_factory, transport: str
    ) -> None:
        schema = _schema(float_dtype="fp16")
        rb = _make_replay(schema=schema)
        pattern = _pattern(_ROWS, schema)
        _fill(rb, pattern)
        pipeline = pipeline_factory(rb, transport=transport)
        assert pipeline.sampling_transport == transport
        _wait_visible(pipeline, _ROWS)

        batch = pipeline.sample_large_batch(0, 16)
        assert batch["rewards"].shape == (16, 2)
        assert batch["rewards"].dtype == torch.float32
        assert batch["obs"].dtype == torch.float32
        assert batch["aux"].shape == (16, 2)
        assert batch["aux"].dtype == torch.int32
        assert batch["dones"].shape == (16,)

        row_ids = (batch["rewards"][:, 0].cpu() / 7).long()
        assert torch.equal(
            batch["rewards"].cpu(), row_ids.unsqueeze(1) * 7 + torch.arange(2, dtype=torch.float32)
        )
        expected_obs = (row_ids.unsqueeze(1) * 10 + torch.arange(_OBS_DIM)).to(torch.float16)
        assert torch.equal(batch["obs"].cpu(), expected_obs.to(torch.float32))
        assert torch.equal(
            batch["aux"].cpu(),
            (row_ids.unsqueeze(1) * 3 + torch.arange(2)).to(torch.int32),
        )
        # Repeat calls for one tick return the same (possibly rebuilt) batch.
        assert pipeline.sample_large_batch(0, 16) is batch
        pipeline.after_tick()

        end_ptr, fields = pipeline.read_committed_fields(("rewards", "dones"), start_ptr=0)
        assert end_ptr == _ROWS
        assert torch.equal(fields["rewards"], pattern["rewards"])
        assert torch.equal(fields["dones"], pattern["dones"])

        manifest = pipeline.transfer_manifest
        assert manifest["storage_owner"] == "cpu"
        assert manifest["sampling_transport"] == transport
        assert manifest["storage_bytes"] == _ROWS * pipeline._row_bytes
        assert manifest["row_schema"]["float_dtype"] == "float16"
        rb.close()

    def test_cuda_ingress_pumps_d2h(self, pipeline_factory) -> None:
        schema = _schema()
        rb = _make_replay(ingress_device="cuda", schema=schema)
        pattern = _pattern(_ROWS, schema)
        _fill(rb, pattern, device="cuda")
        pipeline = pipeline_factory(rb, transport="h2d")
        _wait_visible(pipeline, _ROWS)

        batch = pipeline.sample_large_batch(0, 16)
        row_ids = (batch["rewards"][:, 0].cpu() / 7).long()
        assert torch.equal(batch["obs"].cpu(), row_ids.unsqueeze(1) * 10 + torch.arange(_OBS_DIM))
        rb.close()

    def test_rebuilder_hook_runs_once_per_prepared_batch(self, pipeline_factory) -> None:
        rb = _make_replay()
        _fill(rb, _pattern(_ROWS, _schema()))
        pipeline = pipeline_factory(rb, transport="h2d")
        _wait_visible(pipeline, _ROWS)
        calls: list[tuple[int, tuple[int, ...], str]] = []

        def hook(batch, *, indices, device):
            calls.append((batch["obs"].shape[0], tuple(indices.shape), str(device)))
            rebuilt = dict(batch)
            rebuilt["rewards"] = batch["rewards"] + 1.0
            return rebuilt

        pipeline._sample_rebuilder = hook
        batch = pipeline.sample_large_batch(0, 16)
        assert calls == [(16, (16,), "cuda:0")]
        baseline = pipeline.sample_large_batch(0, 16)
        assert baseline is batch
        assert len(calls) == 1
        # The hook result is what the learner sees.
        assert torch.all(batch["rewards"] >= 1.0 - 1e-6)
        rb.close()

    @pytest.mark.parametrize("transport", ["uva", "h2d"])
    def test_warmup_uses_scratch_gather_only(self, pipeline_factory, transport: str) -> None:
        rb = _make_replay()
        pipeline = pipeline_factory(rb, transport=transport)
        batch = pipeline.warmup()
        assert batch["obs"].shape == (16, _OBS_DIM)
        assert batch["obs"].dtype == torch.float32
        assert pipeline._visible_ptr == 0
        rb.close()

    def test_age_biased_layered_sampler(self, pipeline_factory) -> None:
        from uni_rl.algos.flash_sac.replay import LayeredAgeBiasedReplayPipeline

        rb = _make_replay()
        _fill(rb, _pattern(_ROWS, _schema()))
        pipeline = LayeredAgeBiasedReplayPipeline(
            rb,
            device="cuda",
            sample_count=16,
            base_seed=0,
            transport="uva",
            decay_step=8,
        )
        _wait_visible(pipeline, _ROWS)
        batch = pipeline.sample_large_batch(3, 16)
        assert batch["obs"].shape == (16, _OBS_DIM)
        pipeline.close()
        rb.close()

    def test_default_single_slab_pipeline_keeps_bare_packed_tensor(self, pipeline_factory) -> None:
        schema = RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, critic_dim=_CRITIC_DIM)
        rb = _make_replay(schema=schema)
        _fill(rb, {**_pattern(_ROWS, _schema()), "rewards": torch.randn(_ROWS)})
        pipeline = pipeline_factory(rb, transport="h2d")
        _wait_visible(pipeline, _ROWS)
        batch = pipeline.sample_large_batch(0, 16)
        assert batch["rewards"].shape == (16,)
        assert batch["obs"].shape == (16, _OBS_DIM)
        rb.close()
