"""Supply-rate benchmark for the CPU-pinned layered replay pipeline.

Slow benchmark (deselected by default): run with

    uv run pytest -m slow tests/ipc/test_layered_replay_benchmark.py -s

It fills a 10M-row pinned ring with fp16-compressed rows and measures the
sustainable sampling supply (rows/s) for the UVA and H2D transports, in both
serial (prepare-then-consume) and pipelined (single-batch lookahead, the
runner's production pattern) modes. Numbers are reported, not asserted.
"""

from __future__ import annotations

import time

import pytest
import torch

from uni_rl.ipc.replay_buffer import ReplayBuffer
from uni_rl.ipc.replay_pipelines.layered import LayeredReplayPipeline
from uni_rl.ipc.row_schema import RowSchema

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# FDR-iso dense row: obs 386 fp16 (772B) + next_obs 772B + action 36 fp32
# (144B) + reward 2 fp32 (8B) + flags 8B + aux int32x2 (8B) = 1712B/row.
_ISO = {"obs_dim": 386, "action_dim": 36, "reward_heads": 2, "aux_width": 2}
# FDR full-dims dense row (with next_obs): obs 772 fp16 (1544B) -> 3256B/row.
_FULL = {"obs_dim": 772, "action_dim": 36, "reward_heads": 2, "aux_width": 2}

_FILL_CHUNK = 8192
_SAMPLE_COUNT = 4096
_UTD = 2


def _fill_ring(rb: ReplayBuffer, schema: RowSchema, rows: int) -> float:
    """Fill the ring with deterministic rows; return wall seconds."""
    obs = torch.zeros(_FILL_CHUNK, schema.obs_dim)
    nobs = torch.zeros(_FILL_CHUNK, schema.obs_dim)
    act = torch.zeros(_FILL_CHUNK, schema.action_dim)
    rew = torch.zeros(_FILL_CHUNK, schema.reward_heads)
    dones = torch.zeros(_FILL_CHUNK)
    trunc = torch.zeros(_FILL_CHUNK)
    aux = torch.zeros(_FILL_CHUNK, schema.aux_width, dtype=torch.int32)
    start = time.perf_counter()
    written = 0
    while written < rows:
        count = min(_FILL_CHUNK, rows - written)
        assert rb.add_batch(
            obs[:count],
            act[:count],
            rew[:count],
            nobs[:count],
            dones[:count],
            trunc[:count],
            aux=aux[:count],
        )
        written += count
    return time.perf_counter() - start


def _drain_visible(pipeline: LayeredReplayPipeline, rows: int, timeout: float = 600.0) -> float:
    deadline = time.monotonic() + timeout
    while pipeline._visible_ptr < rows:
        if time.monotonic() > deadline:
            raise TimeoutError(f"layered pump stalled at {pipeline._visible_ptr} < {rows}")
        time.sleep(0.01)
    return pipeline._visible_ptr


def _measure_serial(pipeline: LayeredReplayPipeline, iterations: int) -> float:
    """prepare -> consume serialized; returns rows/s."""
    start = time.perf_counter()
    for tick in range(iterations):
        pipeline.start_prepare(tick, _SAMPLE_COUNT)
        batch = pipeline.sample_large_batch(tick, _SAMPLE_COUNT)
        pipeline.after_tick()
        del batch
    return iterations * _SAMPLE_COUNT / (time.perf_counter() - start)


def _measure_pipelined(pipeline: LayeredReplayPipeline, iterations: int) -> float:
    """Single-batch lookahead (the runner's production pattern); rows/s."""
    pipeline.start_prepare(0, _SAMPLE_COUNT)
    start = time.perf_counter()
    for tick in range(iterations):
        batch = pipeline.sample_large_batch(tick, _SAMPLE_COUNT)
        pipeline.after_tick()
        pipeline.start_prepare(tick + 1, _SAMPLE_COUNT)
        del batch
    return iterations * _SAMPLE_COUNT / (time.perf_counter() - start)


def _run_config(
    name: str,
    schema_kwargs: dict,
    capacity: int,
    transports: list[str],
    *,
    reconstruct: bool = False,
) -> None:
    schema = RowSchema(**schema_kwargs, float_dtype="fp16")
    row_bytes = (
        schema.storage_row_bytes if not reconstruct else schema.without_next_obs().storage_row_bytes
    )
    gib = capacity * row_bytes / 2**30
    print(f"\n=== {name}: {capacity / 1e6:.1f}M rows x {row_bytes}B = {gib:.2f} GiB pinned ===")
    rb = ReplayBuffer(
        capacity=capacity,
        obs_dim=schema.obs_dim,
        action_dim=schema.action_dim,
        device="cuda",
        critic_dim=schema.critic_dim,
        ingress_slot_rows=_FILL_CHUNK,
        ingress_depth=2,
        row_schema=schema,
    )
    for transport in transports:
        pipeline = LayeredReplayPipeline(
            rb,
            device="cuda",
            sample_count=_SAMPLE_COUNT,
            base_seed=0,
            transport=transport,
            reconstruct_next_obs=reconstruct,
            next_obs_alias_rows=(5 * 4096) if reconstruct else None,
        )
        fill_s = _fill_ring(rb, schema, capacity)
        visible = _drain_visible(pipeline, capacity)
        assert visible == capacity
        fill_rows_s = capacity / fill_s
        print(
            f"[{transport}] fill: {fill_s:.1f}s -> {fill_rows_s / 1e6:.2f}M rows/s "
            f"({capacity * row_bytes / fill_s / 2**30:.2f} GiB/s incl. collector write)"
        )
        for warmup_tick in range(20):
            pipeline.start_prepare(warmup_tick, _SAMPLE_COUNT)
            pipeline.sample_large_batch(warmup_tick, _SAMPLE_COUNT)
            pipeline.after_tick()

        serial_rows_s = _measure_serial(pipeline, 400)
        pipelined_rows_s = _measure_pipelined(pipeline, 1200)
        for label, rows_s in (("serial", serial_rows_s), ("pipelined", pipelined_rows_s)):
            gbps = rows_s * row_bytes / 1e9
            max_sps = rows_s / (_UTD * _SAMPLE_COUNT)
            print(
                f"[{transport}] {label:9s}: {rows_s / 1e6:.2f}M rows/s "
                f"({gbps:.1f} GB/s) -> max SPS at UTD={_UTD}, batch={_SAMPLE_COUNT}: "
                f"{max_sps:,.0f} env-steps/s"
            )
        pipeline.progress(wait=True)
        pipeline.close()
        # Fresh buffer per transport: ingress is single-consumer.
        rb.close()
        rb = ReplayBuffer(
            capacity=capacity,
            obs_dim=schema.obs_dim,
            action_dim=schema.action_dim,
            device="cuda",
            critic_dim=schema.critic_dim,
            ingress_slot_rows=_FILL_CHUNK,
            ingress_depth=2,
            row_schema=schema,
        )


def test_layered_replay_supply_rate_10m() -> None:
    transports = ["uva", "h2d"]
    _run_config("FDR-iso dense (1640B rows)", _ISO, capacity=10_000_000, transports=transports)
    _run_config(
        "FDR full-dims dense (3256B rows)", _FULL, capacity=5_000_000, transports=transports
    )
    # Reconstructed next_obs: the ring drops next_obs/next_critic (868B/row
    # vs 1640B dense, x1.89 capacity saving). Bus traffic is ~neutral: the
    # alias gather reads the obs group twice per sampled row, so supply
    # (rows/s) is expected to match the dense path within overhead.
    _run_config(
        "FDR-iso reconstruct next_obs (868B rows)",
        _ISO,
        capacity=10_000_000,
        transports=transports,
        reconstruct=True,
    )
