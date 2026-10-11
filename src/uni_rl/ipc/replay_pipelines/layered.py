"""CPU-pinned authoritative replay with layered UVA/H2D sampling.

The layered pipeline keeps the full replay ring in host memory pinned with
``cudaHostRegister(PORTABLE|MAPPED)`` (exact pinning — the torch pinned
allocator rounds blocks larger than 32 GiB up to a power of two) and serves
samples to the learner device on demand:

- ``uva`` (preferred): a native kernel gathers rows straight out of the
  host-locked ring into device hot slots; bytes cross the bus once.
- ``h2d`` (fallback): rows are gathered host-side into pinned staging slots,
  then copied to device hot slots on a side stream (double-buffered).

fp16/int32 row groups stay compressed through storage and transport; batch
views cast float groups back to fp32, so the learner contract is unchanged.
The collector protocol is untouched: the same bounded ingress, backpressure,
and commit ordering as the device-resident pipeline.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch

from uni_rl.ipc.replay_buffer import ReplayBuffer, _as_slab_list
from uni_rl.ipc.replay_pipelines.base import ReplayTickMetadata
from uni_rl.ipc.replay_pipelines.gpu_resident import (
    _ring_spans,
    require_offpolicy_replay_device,
)
from uni_rl.ipc.replay_pipelines.layered_uva import get_diagnostic, load_extension

# cudaHostRegisterPortable | cudaHostRegisterMapped: the mapping makes the
# host pages dereferenceable from device kernels (UVA) on every device.
_HOST_REGISTER_FLAGS = 0x01 | 0x02

# torch's pinned allocator rounds blocks larger than 32 GiB up to a power of
# two; at and above this threshold the pipeline pins pageable allocations
# exactly with cudaHostRegister instead.
_EXACT_REGISTER_THRESHOLD_BYTES = 32 * 2**30

#: Post-sample rebuild hook: called with the sampled batch view, the absolute
#: ring indices that produced it, and the learner device; must return the
#: batch dict to hand to the learner (identity for the default path). Runs
#: after sampling and before the batch reaches the learner, so multi-head /
#: compressed-row tasks can expand aux-driven columns on the fly. Implementations
#: used with whole-cycle CUDA graphs must be capture-safe.
ReplaySampleRebuilder = Callable[..., Dict[str, torch.Tensor]]


def host_available_memory_bytes() -> int | None:
    """Best-effort MemAvailable snapshot (Linux); None when unreadable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class LayeredReplayPipeline:
    """Authoritative CPU-pinned replay ring with layered UVA/H2D sampling."""

    _POLL_INTERVAL_S = 0.0005
    _HOST_MEMORY_HEADROOM = 0.8

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        device: str,
        sample_count: int,
        base_seed: int = 0,
        trace_recorder=None,
        trace_cuda_events: bool = True,
        transport: str = "auto",
        sample_rebuilder: ReplaySampleRebuilder | None = None,
        copy_workers: int = 4,
    ) -> None:
        self._replay_buffer = replay_buffer
        if transport not in ("auto", "uva", "h2d"):
            raise ValueError(
                f"layered replay transport must be 'auto', 'uva', or 'h2d'; got {transport!r}"
            )
        self._device = torch.device(require_offpolicy_replay_device(device))
        if self._device.type != "cuda":
            raise ValueError(
                "LayeredReplayPipeline requires a CUDA device for host-pinned "
                f"UVA/H2D replay; got {self._device.type!r}"
            )
        if self._device.index is None:
            self._device = torch.device("cuda", index=torch.cuda.current_device())
        self._learner_thread_id = threading.get_ident()
        self._sample_count = int(sample_count)
        self._base_seed = int(base_seed)
        self._trace_recorder = trace_recorder
        self._trace_cuda_events = bool(trace_cuda_events)
        self._schema = replay_buffer.row_schema
        self._slabs = self._schema.slabs
        self._capacity = int(replay_buffer.capacity)
        self._storage_width = self._schema.storage_width
        self._row_bytes = self._schema.storage_row_bytes
        self._sample_rebuilder = sample_rebuilder
        self._copy_workers = max(1, int(copy_workers))

        # -- host RAM guard (hard fail: pinned pages are non-pageable) --
        storage_bytes = self._capacity * self._row_bytes
        staging_bytes = 2 * self._sample_count * self._row_bytes
        self._raise_if_host_over_budget(storage_bytes + staging_bytes)

        self._cudart: Any = torch.cuda.cudart()
        if self._cudart is None:
            raise RuntimeError("torch.cuda.cudart() is required for layered replay pinning")
        self._registered_ptrs: list[int] = []
        self._sync_stream: Any = torch.cuda.Stream(device=self._device)
        self._slot_events: list[Any] = [torch.cuda.Event() for _ in range(2)]
        self._uva_ext: Any | None = None
        # Resolve the transport before pinning storage so a failed resolution
        # leaves no registered pages behind.
        self._transport = self._resolve_transport(transport)
        try:
            self._host_slabs = self._allocate_host_slabs(self._capacity)
            # Pinned staging slots for the H2D path (hot/cold double buffer).
            self._staging_slabs = self._allocate_host_slabs(self._sample_count, count=2)
        except BaseException:
            self._unregister_all()
            raise
        self._gpu_packed_slabs = [
            [
                torch.empty(
                    (self._sample_count, slab.width),
                    dtype=slab.dtype,
                    device=self._device,
                )
                for _ in range(2)
            ]
            for slab in self._slabs
        ]

        self._gather_in_flight: Any | None = None
        self._copy_pool = ThreadPoolExecutor(
            max_workers=self._copy_workers,
            thread_name_prefix="replay_layered_copy",
        )
        self._copy_lock = threading.Lock()
        self._prepare_condition = threading.Condition()
        self._visible_ptr = 0
        self._ingress_high_water = replay_buffer.ingress_diagnostics()["high_water_occupancy"]
        self.last_incremental_h2d_time_s = 0.0

        self._hot = 0
        self._cold = 1
        self._has_hot_batch = False
        self._hot_batch: Dict[str, torch.Tensor] | None = None
        self._hot_metadata: ReplayTickMetadata | None = None
        self._hot_indices: torch.Tensor | None = None
        self._prepared_metadata: ReplayTickMetadata | None = None
        self._prepared_indices: torch.Tensor | None = None
        self._prepare_tick_id: int | None = None
        self._prepare_required_ptr = 0
        self._prepare_state = "idle"
        self._prepare_error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="replay_layered_worker",
            daemon=True,
        )
        self._worker.start()

    # -- construction helpers --------------------------------------------------

    def _raise_if_host_over_budget(self, required_bytes: int) -> None:
        available = host_available_memory_bytes()
        if available is None:
            warnings.warn(
                "Layered replay host budget check skipped: /proc/meminfo "
                "MemAvailable is unreadable on this platform; "
                "cudaHostRegister remains the fail-closed gate."
            )
            return
        usable = int(available * self._HOST_MEMORY_HEADROOM)
        if required_bytes > usable:
            raise RuntimeError(
                "Layered CPU replay does not fit in host RAM: requires "
                f"{required_bytes / 2**30:.2f} GiB pinned "
                f"(storage {self._capacity * self._row_bytes / 2**30:.2f} + "
                f"staging {2 * self._sample_count * self._row_bytes / 2**30:.2f}), "
                f"host MemAvailable {available / 2**30:.2f} GiB "
                f"({self._HOST_MEMORY_HEADROOM:.0%} of available RAM may be pinned)"
            )

    def _pin_tensor(self, tensor: torch.Tensor) -> None:
        result = self._cudart.cudaHostRegister(
            int(tensor.data_ptr()),
            int(tensor.numel() * tensor.element_size()),
            _HOST_REGISTER_FLAGS,
        )
        if result != self._cudart.cudaError.success:
            raise RuntimeError(
                "cudaHostRegister(PORTABLE|MAPPED) failed for layered replay "
                f"storage ({tensor.numel() * tensor.element_size() / 2**30:.2f} GiB): {result}"
            )
        self._registered_ptrs.append(int(tensor.data_ptr()))

    def _allocate_pinned_tensor(self, rows: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        nbytes = rows * width * dtype.itemsize
        if nbytes < _EXACT_REGISTER_THRESHOLD_BYTES:
            # torch's pinned allocator is exact below 32 GiB; its blocks are
            # page-locked and UVA-accessible on every modern CUDA system.
            return torch.empty((rows, width), dtype=dtype, pin_memory=True)
        # Above 32 GiB the torch pinned allocator rounds up to a power of two
        # (77 GB -> 128 GiB), so large rings are pageable-allocated and pinned
        # exactly with cudaHostRegister(PORTABLE|MAPPED). Blocks this size come
        # from a fresh page-aligned mapping, so no other tensor shares pages.
        tensor = torch.empty((rows, width), dtype=dtype)
        self._pin_tensor(tensor)
        return tensor

    def _allocate_host_slabs(self, rows: int, count: int = 1) -> list[Any]:
        try:
            tensors = [
                self._allocate_pinned_tensor(rows, slab.width, slab.dtype)
                for slab in self._slabs
                for _ in range(count)
            ]
        except BaseException:
            self._unregister_all()
            raise
        if count == 1:
            return tensors
        # Regroup into slab-major slot lists for count > 1.
        return [tensors[index * count : (index + 1) * count] for index in range(len(self._slabs))]

    def _unregister_all(self) -> None:
        while self._registered_ptrs:
            ptr = self._registered_ptrs.pop()
            try:
                self._cudart.cudaHostUnregister(ptr)
            except Exception:
                pass

    def _resolve_transport(self, requested: str) -> str:
        if requested == "h2d":
            return "h2d"
        ext = load_extension()
        reason = get_diagnostic()
        if ext is not None:
            try:
                self._self_test_uva(ext)
            except BaseException as exc:
                ext = None
                reason = f"UVA self-test failed: {exc}"
        if ext is None:
            if requested == "uva":
                raise RuntimeError(
                    "layered replay transport 'uva' requested but the native UVA "
                    f"gather is unavailable: {reason}"
                )
            print(
                f"[LayeredReplay] UVA unavailable ({reason}); "
                "falling back to pinned staging + side-stream H2D.",
                flush=True,
            )
            return "h2d"
        self._uva_ext = ext
        return "uva"

    def _self_test_uva(self, ext: Any) -> None:
        src = torch.arange(64, dtype=torch.float32, pin_memory=True).reshape(8, 8)
        indices = torch.tensor([5, 1, 0], dtype=torch.int64, device=self._device)
        dst = torch.empty((3, 8), dtype=torch.float32, device=self._device)
        ext.gather_rows_uva(int(src.data_ptr()), dst, indices, self._sync_stream.cuda_stream)
        self._slot_events[0].record(self._sync_stream)
        self._slot_events[0].synchronize()
        expected = src[indices.cpu()].to(self._device)
        if not torch.equal(dst, expected):
            raise RuntimeError("UVA gather returned wrong rows")

    # -- diagnostics -----------------------------------------------------------

    @property
    def h2d_submitter(self) -> str:
        return "layered_cpu_pump"

    @property
    def sampling_transport(self) -> str:
        return self._transport

    @property
    def transfer_manifest(self) -> dict[str, object]:
        return {
            "pipeline": "layered_cpu_pinned",
            "storage_owner": "cpu",
            "sampling_transport": self._transport,
            "device": str(self._device),
            "device_family": "cuda",
            "host_memory_kind": "cudaHostRegister(portable|mapped)",
            "host_pinned": True,
            "storage_rows": self._capacity,
            "storage_width": self._storage_width,
            "storage_bytes": int(
                sum(slab.numel() * slab.element_size() for slab in self._host_slabs)
            ),
            "row_schema": self._schema.summary(),
            "ingress_depth": self._replay_buffer._ingress_depth,
            "ingress_slot_rows": self._replay_buffer._ingress_slot_rows,
            "ingress_diagnostics": self.ingress_diagnostics(),
            "h2d_submitter": self.h2d_submitter,
            "ring_depth": 2,
        }

    def ingress_diagnostics(self) -> dict[str, int | float]:
        diagnostics = self._replay_buffer.ingress_diagnostics()
        self._ingress_high_water = max(
            int(self._ingress_high_water),
            int(diagnostics["high_water_occupancy"]),
            int(diagnostics["occupancy"]),
        )
        diagnostics["high_water_occupancy"] = self._ingress_high_water
        return diagnostics

    # -- ingress pump ----------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            if self._closed:
                return
            try:
                did_work = self._pump_once()
                did_work |= self._service_pending_prepare()
            except BaseException as exc:
                with self._prepare_condition:
                    self._prepare_error = exc
                    self._prepare_condition.notify_all()
                return
            if not did_work:
                # Wake immediately on prepare requests (start_prepare notifies)
                # so sampling supply is not throttled by the poll interval.
                with self._prepare_condition:
                    if self._closed:
                        return
                    self._prepare_condition.wait(timeout=self._POLL_INTERVAL_S)

    def _pump_once(self) -> bool:
        copy_begin_ns = time.perf_counter_ns()
        with self._copy_lock:
            # The lock serializes every pumper (worker thread and synchronous
            # progress() callers) so slots commit strictly in publish order.
            ingress = self._replay_buffer.take_published_ingress()
            if ingress is None:
                return False
            slot, start, count, source = ingress
            if start != self._visible_ptr:
                raise RuntimeError(
                    "Layered replay ingress publication is not contiguous: "
                    f"pumped ptr {self._visible_ptr}, slot start {start}"
                )
            # CPU span writes must not overlap an in-flight gather reading the
            # wrapped ring region; gathers are rare relative to spans, so gate
            # the copy on the last submitted gather.
            if self._gather_in_flight is not None:
                self._gather_in_flight.synchronize()
                self._gather_in_flight = None
            self._copy_span_to_ring(source, start, count)
            self._replay_buffer.commit_ingress(slot=slot, start=start, count=count)
            self.last_incremental_h2d_time_s = (time.perf_counter_ns() - copy_begin_ns) / 1e9
            with self._prepare_condition:
                self._visible_ptr = start + count
                self._prepare_condition.notify_all()
        return True

    def _copy_span_to_ring(
        self,
        source: torch.Tensor | list[torch.Tensor],
        start: int,
        count: int,
    ) -> None:
        source_slabs = _as_slab_list(source)
        if source_slabs[0].is_cuda:
            # GPU-centric ingress: D2H copy on the side stream, committed only
            # after the copy completes (pump thread blocks; spans are rare and
            # the bus is the limit anyway).
            with torch.cuda.stream(self._sync_stream):
                for source_slab, host_slab in zip(source_slabs, self._host_slabs, strict=True):
                    source_offset = 0
                    for offset, length in _ring_spans(start, start + count, self._capacity):
                        host_slab[offset : offset + length].copy_(
                            source_slab[source_offset : source_offset + length],
                            non_blocking=True,
                        )
                        source_offset += length
            done_event = torch.cuda.Event()
            done_event.record(self._sync_stream)
            done_event.synchronize()
            return
        futures = []
        for source_slab, host_slab in zip(source_slabs, self._host_slabs, strict=True):
            workers = min(self._copy_workers, count)
            bounds = [(index * count) // workers for index in range(workers + 1)]
            for chunk_index in range(workers):
                row_lo = bounds[chunk_index]
                row_hi = bounds[chunk_index + 1]
                if row_hi == row_lo:
                    continue
                futures.append(
                    self._copy_pool.submit(
                        self._copy_rows_wrapped,
                        source_slab,
                        host_slab,
                        start + row_lo,
                        start + row_hi,
                        row_lo,
                    )
                )
        for future in futures:
            future.result()

    def _copy_rows_wrapped(
        self,
        source_slab: torch.Tensor,
        host_slab: torch.Tensor,
        abs_lo: int,
        abs_hi: int,
        source_offset: int,
    ) -> None:
        for offset, length in _ring_spans(abs_lo, abs_hi, self._capacity):
            host_slab[offset : offset + length].copy_(
                source_slab[source_offset : source_offset + length]
            )
            source_offset += length

    # -- sampling --------------------------------------------------------------

    def _draw_indices(
        self,
        *,
        visible_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Uniform physical row indices; subclasses may override the sampler."""
        return torch.randint(
            0,
            visible_size,
            (self._sample_count,),
            generator=generator,
            device=device,
        )

    def _sample_indices(self, *, visible_size: int, sample_seed: int) -> torch.Tensor:
        index_device = self._device if self._transport == "uva" else torch.device("cpu")
        generator = torch.Generator(device=index_device)
        generator.manual_seed(sample_seed)
        return self._draw_indices(
            visible_size=visible_size,
            device=index_device,
            generator=generator,
        )

    def _submit_gather(self, indices: torch.Tensor, *, slot: int) -> torch.Tensor:
        """Submit the cold-slot gather; returns device-resident batch indices."""
        with torch.cuda.stream(self._sync_stream):
            if self._transport == "uva":
                assert self._uva_ext is not None
                for host_slab, packed_slabs in zip(
                    self._host_slabs, self._gpu_packed_slabs, strict=True
                ):
                    self._uva_ext.gather_rows_uva(
                        int(host_slab.data_ptr()),
                        packed_slabs[slot],
                        indices,
                        self._sync_stream.cuda_stream,
                    )
                device_indices = indices
            else:
                for host_slab, staging_slabs, packed_slabs in zip(
                    self._host_slabs, self._staging_slabs, self._gpu_packed_slabs, strict=True
                ):
                    torch.index_select(host_slab, 0, indices, out=staging_slabs[slot])
                    packed_slabs[slot].copy_(staging_slabs[slot], non_blocking=True)
                device_indices = indices.to(self._device, non_blocking=True)
            self._slot_events[slot].record(self._sync_stream)
        self._gather_in_flight = self._slot_events[slot]
        return device_indices

    def _service_pending_prepare(self) -> bool:
        with self._prepare_condition:
            if self._prepare_state != "preparing" or self._prepare_tick_id is None:
                return False
            tick_id = self._prepare_tick_id
            required_ptr = self._prepare_required_ptr
            slot = self._cold
        visible_size = min(self._visible_ptr, self._capacity)
        if self._visible_ptr < required_ptr or visible_size <= 0:
            return False
        sample_seed = self._base_seed + int(tick_id)
        indices = self._sample_indices(visible_size=visible_size, sample_seed=sample_seed)
        device_indices = self._submit_gather(indices, slot=slot)
        metadata = ReplayTickMetadata(
            tick_id=int(tick_id),
            snapshot_ptr=int(self._visible_ptr),
            snapshot_size=visible_size,
            sample_seed=sample_seed,
            sample_count=self._sample_count,
            batch_host_slot=None,
            batch_gpu_slot=slot,
        )
        with self._prepare_condition:
            self._prepared_metadata = metadata
            self._prepared_indices = device_indices
            self._prepare_state = "gather_submitted"
            self._prepare_condition.notify_all()
        return True

    # -- batch views -----------------------------------------------------------

    def _read_batch_field(
        self,
        packed: torch.Tensor | list[torch.Tensor],
        field_name: str,
    ) -> torch.Tensor:
        view = self._replay_buffer.field_view(packed, field_name)
        if view.is_floating_point() and view.dtype != torch.float32:
            return view.to(torch.float32)
        return view

    def _packed_batch_view(
        self, packed: torch.Tensor | list[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        rb = self._replay_buffer
        batch = {
            "obs": self._read_batch_field(packed, "obs"),
            "next_obs": self._read_batch_field(packed, "next_obs"),
            "actions": self._read_batch_field(packed, "actions"),
            "rewards": self._read_batch_field(packed, "rewards"),
            "dones": self._read_batch_field(packed, "dones"),
            "truncated": self._read_batch_field(packed, "truncated"),
        }
        if rb._critic_dim > 0:
            batch["critic"] = self._read_batch_field(packed, "critic")
            batch["next_critic"] = self._read_batch_field(packed, "next_critic")
        if rb._schema.aux_width > 0:
            batch["aux"] = self._read_batch_field(packed, "aux")
        return batch

    def _packed_slot(self, slot: int) -> torch.Tensor | list[torch.Tensor]:
        payloads = [slab_slots[slot] for slab_slots in self._gpu_packed_slabs]
        return payloads[0] if len(payloads) == 1 else payloads

    def _large_batch_view(self, slot: int) -> Dict[str, torch.Tensor]:
        return self._packed_batch_view(self._packed_slot(slot))

    # -- public API ------------------------------------------------------------

    def _validate_sample_count(self, sample_count: int) -> None:
        if int(sample_count) != int(self._sample_count):
            raise ValueError("sample_count must match the value used to allocate the double buffer")

    def _refresh_prepare_state(self) -> None:
        if self._prepare_error is not None:
            raise self._prepare_error
        if self._prepared_metadata is not None:
            slot = self._prepared_metadata.batch_gpu_slot
            if slot is not None and self._slot_events[slot].query():
                self._prepare_state = "ready"

    def progress(self, *, wait: bool = False) -> bool:
        """Pump published ingress synchronously when wait is requested."""
        if not wait:
            return False
        did_work = False
        while self._pump_once():
            did_work = True
        return did_work

    def read_committed_fields(
        self,
        field_names: tuple[str, ...],
        *,
        start_ptr: int,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        """Return a stable ordered field snapshot through the committed pointer."""
        published_snapshot = self._replay_buffer.published_ptr
        deadline = time.monotonic() + 60.0
        with self._prepare_condition:
            while self._visible_ptr < published_snapshot:
                if self._prepare_error is not None:
                    raise self._prepare_error
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Layered replay pump stalled before committed-field snapshot: "
                        f"visible {self._visible_ptr} < published {published_snapshot}"
                    )
                self._prepare_condition.wait(timeout=0.1)
        with self._copy_lock:
            # No span copy is in flight while the snapshot is taken, and the
            # pump blocks on this lock before starting the next one.
            end_ptr = self._visible_ptr
            count = min(max(end_ptr - start_ptr, 0), self._capacity)
            field_start = end_ptr - count
            index = field_start % self._capacity
            fields: dict[str, torch.Tensor] = {}
            for field_name in field_names:
                source = self._replay_buffer.field_view(self._host_slabs, field_name)
                if index + count <= self._capacity:
                    fields[field_name] = source[index : index + count].clone()
                    continue
                split = self._capacity - index
                fields[field_name] = torch.cat(
                    [source[index:], source[: count - split]],
                    dim=0,
                )
        return end_ptr, fields

    def warmup(self) -> dict[str, torch.Tensor]:
        """Compile/warm the owner's gather path without replay state.

        This scratch gather writes only the current cold output slot and does
        not advance replay pointers, prepare metadata, RNG used by training, or
        any learner state.
        """
        if self._closed:
            raise RuntimeError("Cannot warm replay pipeline after pipeline.close()")
        slot = self._cold
        index_device = self._device if self._transport == "uva" else torch.device("cpu")
        generator = torch.Generator(device=index_device)
        generator.manual_seed(self._base_seed - 1)
        visible_size = min(self._capacity, max(1, self._sample_count))
        indices = self._draw_indices(
            visible_size=visible_size,
            device=index_device,
            generator=generator,
        )
        self._submit_gather(indices, slot=slot)
        self._slot_events[slot].synchronize()
        for packed_slab in _as_slab_list(self._packed_slot(slot)):
            packed_slab.zero_()
        self._gather_in_flight = None
        return self._large_batch_view(slot)

    def start_prepare(
        self,
        tick_id: int,
        sample_count: int,
        min_snapshot_ptr: int | None = None,
    ) -> bool:
        """Schedule a gather for the current cold slot.

        Returns True when this call launches new work. If the same tick is
        already pending or prepared, returns False.
        """
        self._validate_sample_count(sample_count)
        if self._closed:
            raise RuntimeError("Cannot prepare replay batch after pipeline.close()")
        self._refresh_prepare_state()
        with self._prepare_condition:
            if self._prepared_metadata is not None or self._prepare_state not in {
                "idle",
                "ready",
            }:
                prepared_tick = (
                    self._prepared_metadata.tick_id
                    if self._prepared_metadata is not None
                    else self._prepare_tick_id
                )
                if prepared_tick == int(tick_id):
                    return False
                raise RuntimeError(
                    "Cannot prepare a new replay batch before the previous batch is consumed"
                )
            self._prepare_tick_id = int(tick_id)
            self._prepare_required_ptr = (
                int(self._replay_buffer.ptr[0])
                if min_snapshot_ptr is None
                else int(min_snapshot_ptr)
            )
            self._prepared_metadata = None
            self._prepared_indices = None
            self._prepare_error = None
            self._prepare_state = "preparing"
            self._prepare_condition.notify_all()
        return True

    def batch_ready(self, tick_id: int, sample_count: int) -> bool:
        self._validate_sample_count(sample_count)
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                return False
            return True
        self._refresh_prepare_state()
        if self._prepared_metadata is None:
            return False
        if self._prepared_metadata.tick_id != int(tick_id):
            return False
        return self._prepare_state == "ready"

    def wait_until_ready(self, tick_id: int, sample_count: int) -> bool:
        self._validate_sample_count(sample_count)
        metadata = self._prepared_or_wait(tick_id)
        slot = metadata.batch_gpu_slot
        assert slot is not None
        self._slot_events[slot].synchronize()
        self._prepare_state = "ready"
        return True

    def _prepared_or_wait(self, tick_id: int) -> ReplayTickMetadata:
        self._refresh_prepare_state()
        if self._prepared_metadata is None:
            if self._prepare_tick_id is None:
                self.start_prepare(tick_id, self._sample_count)
            with self._prepare_condition:
                while self._prepared_metadata is None and self._prepare_error is None:
                    self._prepare_condition.wait(timeout=0.1)
            if self._prepare_error is not None:
                raise self._prepare_error
            assert self._prepared_metadata is not None
            return self._prepared_metadata
        if self._prepared_metadata.tick_id != int(tick_id):
            raise RuntimeError(
                f"Prepared replay batch tick {self._prepared_metadata.tick_id} "
                f"does not match requested tick {tick_id}"
            )
        return self._prepared_metadata

    def sample_large_batch(self, tick_id: int, sample_count: int) -> Dict[str, torch.Tensor]:
        self._validate_sample_count(sample_count)
        if self._has_hot_batch:
            if self._hot_metadata is not None and self._hot_metadata.tick_id != int(tick_id):
                raise RuntimeError(
                    f"Hot batch tick {self._hot_metadata.tick_id} does not match "
                    f"requested tick {tick_id}"
                )
            assert self._hot_batch is not None
            return self._hot_batch
        if not self.batch_ready(tick_id, sample_count):
            self.wait_until_ready(tick_id, sample_count)
        metadata = self._prepared_or_wait(tick_id)
        slot = metadata.batch_gpu_slot
        assert slot is not None
        torch.cuda.current_stream(self._device).wait_event(self._slot_events[slot])
        with self._prepare_condition:
            if slot != self._cold:
                raise RuntimeError("Prepared replay batch is not in the current cold slot")
            self._hot, self._cold = self._cold, self._hot
            self._has_hot_batch = True
            self._hot_metadata = metadata
            self._hot_indices = self._prepared_indices
            self._prepared_metadata = None
            self._prepared_indices = None
            self._prepare_tick_id = None
            self._prepare_state = "idle"
        batch = self._large_batch_view(self._hot)
        if self._sample_rebuilder is not None:
            assert self._hot_indices is not None
            batch = self._sample_rebuilder(
                batch,
                indices=self._hot_indices,
                device=self._device,
            )
        self._hot_batch = batch
        return batch

    def after_tick(self) -> None:
        self._has_hot_batch = False
        self._hot_batch = None
        self._hot_metadata = None
        self._hot_indices = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._prepare_condition:
            self._prepare_condition.notify_all()
        self._worker.join(timeout=5.0)
        self._copy_pool.shutdown(wait=True, cancel_futures=False)
        for event in self._slot_events:
            try:
                event.synchronize()
            except Exception:
                pass
        self._gpu_packed_slabs.clear()
        self._staging_slabs.clear()
        self._host_slabs.clear()
        self._unregister_all()
