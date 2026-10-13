"""Packed replay storage and bounded collector ingress for off-policy RL."""

import multiprocessing as mp
import time
from typing import Any

import torch

from uni_rl.ipc.row_schema import RowSchema
from uni_rl.ipc.row_schema import field_view as schema_field_view
from uni_rl.ipc.shared_buffer import SharedBufferBase

DEFAULT_REPLAY_INGRESS_DEPTH = 2


def _as_slab_list(packed: "torch.Tensor | list[torch.Tensor]") -> "list[torch.Tensor]":
    """Normalize a packed payload to its per-slab tensor list."""
    return packed if isinstance(packed, list) else [packed]


def _fit_storage_dtype(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast float rows to a replay storage dtype, saturating fp16 overflow.

    Values beyond the fp16 range (and +-Inf) would otherwise be stored as
    +-Inf, and a single non-finite row poisons every categorical TD target
    that samples it. Other dtypes are a plain cast (fp32 rows stay
    bit-identical).
    """
    if dtype is torch.float16 and values.is_floating_point() and values.dtype is not torch.float16:
        limit = torch.finfo(torch.float16).max
        return values.clamp(-limit, limit).to(dtype)
    return values.to(dtype)


def _store_rows(view: torch.Tensor, values: torch.Tensor) -> None:
    """Copy rows into a replay field view through :func:`_fit_storage_dtype`."""
    if view.dtype is torch.float16:
        view.copy_(_fit_storage_dtype(values, view.dtype))
    else:
        view.copy_(values)


class ReplayBuffer(SharedBufferBase):
    """Bounded host ingress for an authoritative device replay ring.

    The collector publishes fixed-depth shared ingress slots. The replay
    pipeline advances ``ptr`` and ``size`` only after the device copy commits.

    Row layout is driven by a :class:`RowSchema`. With the default schema
    (scalar reward, no aux columns, fp32 storage) the layout is exactly the
    historical single fp32 packed tensor.
    """

    DEFAULT_INGRESS_DEPTH = DEFAULT_REPLAY_INGRESS_DEPTH

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        *,
        ingress_slot_rows: int,
        critic_dim: int = 0,
        ingress_depth: int = DEFAULT_INGRESS_DEPTH,
        ingress_device: str | torch.device = "cpu",
        row_schema: RowSchema | None = None,
    ):
        super().__init__(capacity, device, defer_gpu=True)
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._critic_dim = critic_dim
        if row_schema is None:
            row_schema = RowSchema(
                obs_dim=obs_dim,
                action_dim=action_dim,
                critic_dim=critic_dim,
            )
        elif (
            row_schema.obs_dim != obs_dim
            or row_schema.action_dim != action_dim
            or row_schema.critic_dim != critic_dim
        ):
            raise ValueError(
                "row_schema dims do not match the replay buffer dims: "
                f"schema ({row_schema.obs_dim}, {row_schema.action_dim}, "
                f"{row_schema.critic_dim}) vs buffer ({obs_dim}, {action_dim}, {critic_dim})"
            )
        self._schema = row_schema
        self.last_incremental_h2d_time_s = 0.0
        self.trace_recorder: Any | None = None
        self.trace_thread_time = False
        self.trace_cuda_events = True
        self._stop_event: Any | None = None

        self.size = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._init_packed_layout()
        self._ingress_slots: list[torch.Tensor | list[torch.Tensor]] = []
        self._init_bounded_ingress(
            slot_rows=ingress_slot_rows,
            depth=int(ingress_depth),
            ingress_device=ingress_device,
        )

    def _init_packed_layout(self) -> None:
        schema = self._schema
        self._storage_width = schema.storage_width
        # Legacy in-slab column slices (exactly the historical packed offsets
        # for the default single-slab fp32 schema).
        for name, attr in (
            ("obs", "_obs_sl"),
            ("next_obs", "_nobs_sl"),
            ("actions", "_act_sl"),
            ("critic", "_critic_sl"),
            ("next_critic", "_ncritic_sl"),
        ):
            if schema.has_field(name):
                field = schema.field(name)
                setattr(self, attr, slice(field.offset, field.offset + field.width))
        self._rew_col = schema.field("rewards").offset
        self._done_col = schema.field("dones").offset
        self._trunc_col = schema.field("truncated").offset

    def _init_bounded_ingress(
        self,
        *,
        slot_rows: int,
        depth: int,
        ingress_device: str | torch.device = "cpu",
    ) -> None:
        if slot_rows <= 0:
            raise ValueError("ingress_slot_rows must be positive")
        if slot_rows > self.capacity:
            raise ValueError("ingress_slot_rows cannot exceed replay capacity")
        if depth <= 0:
            raise ValueError("ingress_depth must be positive")
        self._ingress_slot_rows = slot_rows
        self._ingress_depth = depth
        self._ingress_device = torch.device(ingress_device)
        if self._ingress_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA replay ingress requested on unavailable {self._ingress_device}"
            )
        slabs = self._schema.slabs
        self._ingress_slabs: list[list[torch.Tensor]] = [
            [
                torch.empty(
                    (slot_rows, slab.width),
                    dtype=slab.dtype,
                    device=self._ingress_device,
                ).share_memory_()
                for slab in slabs
            ]
            for _ in range(depth)
        ]
        # Single-slab schemas (the default) keep the historical bare-tensor
        # ingress slots; multi-slab schemas expose the per-slab list.
        self._ingress_slots = [
            slot_slabs[0] if len(slot_slabs) == 1 else slot_slabs
            for slot_slabs in self._ingress_slabs
        ]
        self._ingress_starts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._ingress_counts = torch.zeros(depth, dtype=torch.int64).share_memory_()
        self._published_ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_publish_seq = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_release_seq_tensor = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_high_water = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_backpressure_waits = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_backpressure_wait_ns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_early_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_dropped_batches = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_closed_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_stop_returns = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._ingress_closed = torch.zeros(1, dtype=torch.bool).share_memory_()
        spawn_context = mp.get_context("spawn")
        self._ingress_free = spawn_context.Semaphore(depth)
        self._ingress_ready = spawn_context.Semaphore(0)
        self._ingress_consume_seq = 0
        self._ingress_release_seq = 0

    @property
    def storage_width(self) -> int:
        return self._storage_width

    @property
    def row_schema(self) -> RowSchema:
        return self._schema

    @property
    def ingress_slot_rows(self) -> int:
        return self._ingress_slot_rows

    @property
    def host_storage_bytes(self) -> int:
        if self._ingress_device.type == "cuda":
            return 0
        return sum(
            slab.numel() * slab.element_size()
            for slot_slabs in self._ingress_slabs
            for slab in slot_slabs
        )

    @property
    def ingress_device(self) -> torch.device:
        return self._ingress_device

    @property
    def published_ptr(self) -> int:
        return int(self._published_ptr[0])

    def _record_ingress_backpressure(self, *, start_ns: int) -> None:
        waited_ns = time.perf_counter_ns() - start_ns
        self._ingress_backpressure_waits[0] += 1
        self._ingress_backpressure_wait_ns[0] += waited_ns

    def _record_ingress_drop(
        self,
        *,
        reason: str,
        wait_start_ns: int,
        waited: bool,
    ) -> None:
        if waited:
            self._record_ingress_backpressure(start_ns=wait_start_ns)
        self._ingress_early_returns[0] += 1
        self._ingress_dropped_batches[0] += 1
        if reason == "closed":
            self._ingress_closed_returns[0] += 1
        elif reason == "stop":
            self._ingress_stop_returns[0] += 1

    def ingress_diagnostics(self) -> dict[str, int | float]:
        """Snapshot bounded-ingress state without touching ingress tensors.

        Every value comes from process-shared host metadata. In particular, a
        CUDA-resident ingress slot is never read, copied to host, or synchronized
        to produce this snapshot.
        Sequence and drop counters describe ingress publication chunks. A
        collector vector larger than ``ingress_slot_rows`` is split into
        multiple independently publishable chunks by ``add_batch()``.
        """
        published_sequence = int(self._ingress_publish_seq[0])
        release_sequence = int(self._ingress_release_seq_tensor[0])
        occupancy = max(0, published_sequence - release_sequence)
        return {
            "ingress_depth": self._ingress_depth,
            "ingress_slot_rows": self._ingress_slot_rows,
            "published_sequence": published_sequence,
            "release_sequence": release_sequence,
            "occupancy": occupancy,
            "high_water_occupancy": int(self._ingress_high_water[0]),
            "backpressure_waits": int(self._ingress_backpressure_waits[0]),
            "backpressure_wait_s": (int(self._ingress_backpressure_wait_ns[0]) / 1_000_000_000),
            "early_returns": int(self._ingress_early_returns[0]),
            "dropped_batches": int(self._ingress_dropped_batches[0]),
            "closed_returns": int(self._ingress_closed_returns[0]),
            "stop_returns": int(self._ingress_stop_returns[0]),
        }

    def attach_stop_event(self, stop_event: Any) -> None:
        self._stop_event = stop_event

    def take_published_ingress(
        self,
    ) -> tuple[int, int, int, torch.Tensor | list[torch.Tensor]] | None:
        if not self._ingress_ready.acquire(block=False):
            return None
        slot = self._ingress_consume_seq % self._ingress_depth
        self._ingress_consume_seq += 1
        start = int(self._ingress_starts[slot])
        count = int(self._ingress_counts[slot])
        slot_slabs = self._ingress_slabs[slot]
        packed: torch.Tensor | list[torch.Tensor]
        if len(slot_slabs) == 1:
            packed = slot_slabs[0][:count]
        else:
            packed = [slab[:count] for slab in slot_slabs]
        return slot, start, count, packed

    def commit_ingress(self, *, slot: int, start: int, count: int) -> None:
        expected_slot = self._ingress_release_seq % self._ingress_depth
        if slot != expected_slot:
            raise RuntimeError(
                f"Ingress slots must commit in publication order: expected {expected_slot}, got {slot}"
            )
        if start != int(self.ptr[0]):
            raise RuntimeError(
                f"Ingress commit is not contiguous: committed ptr {int(self.ptr[0])}, start {start}"
            )
        self.ptr[0] = start + count
        self.size[0] = min(start + count, self.capacity)
        self._ingress_release_seq += 1
        self._ingress_release_seq_tensor[0] = self._ingress_release_seq
        self._ingress_free.release()

    def close(self) -> None:
        if bool(self._ingress_closed[0]):
            return
        self._ingress_closed[0] = True
        for _ in range(self._ingress_depth):
            self._ingress_free.release()
        self._ingress_slots.clear()
        self._ingress_slabs.clear()

    def release_ipc(self) -> None:
        """Release this process's ingress IPC views without closing the buffer."""
        self._ingress_slots.clear()
        self._ingress_slabs.clear()

    def __getstate__(self) -> dict:
        """Custom pickle support.

        The collector subprocess only publishes transitions through
        ``add_batch()``. Trace and stop-event handles are process-local; replay
        storage, ingress metadata, and semaphores remain shared.
        """
        state = self.__dict__.copy()
        state["trace_recorder"] = None
        state["_stop_event"] = None
        return state

    def add(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask=None,
        terminal_next_obs=None,
        critic=None,
        next_critic=None,
        terminal_next_critic=None,
        aux=None,
    ) -> bool:
        """Add batch (called by collector).

        `dones` follows the UniLab env lifecycle contract:
        done = terminated | truncated. `truncated` is the collector-encoded
        bootstrap-eligible flag (truncated & ~terminated, uni_rl #29), so
        learner targets can use clamp(1 - dones + truncated) directly: a
        failure that also timed out never bootstraps, while timeout-only
        rows bootstrap from their patched terminal observations.

        `rewards` is ``(count,)`` for the default single-head schema or
        ``(count, reward_heads)`` for a multi-head schema. `aux` is the
        optional per-row int metadata ``(count, aux_width)``; rows are
        zero-filled when the schema declares aux columns and `aux` is None.
        """
        _trace_ns = time.perf_counter_ns() if self.trace_recorder is not None else 0
        return self._add_to_ingress(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            truncated,
            terminal_mask,
            terminal_next_obs,
            critic,
            next_critic,
            terminal_next_critic,
            aux,
            trace_start_ns=_trace_ns,
        )

    def add_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask=None,
        terminal_next_obs=None,
        critic=None,
        next_critic=None,
        terminal_next_critic=None,
        aux=None,
    ) -> bool:
        """Publish a collector vector in bounded ingress-sized chunks.

        Each chunk is an independently publishable transition batch. Shutdown can
        retain a valid prefix of a vector, but never an invalid transition row.
        """
        count = int(obs.shape[0])
        if count <= self._ingress_slot_rows:
            return self.add(
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                truncated,
                terminal_mask=terminal_mask,
                terminal_next_obs=terminal_next_obs,
                critic=critic,
                next_critic=next_critic,
                terminal_next_critic=terminal_next_critic,
                aux=aux,
            )

        def optional_slice(
            tensor: torch.Tensor | None, start: int, end: int
        ) -> torch.Tensor | None:
            return None if tensor is None else tensor[start:end]

        for start in range(0, count, self._ingress_slot_rows):
            end = min(start + self._ingress_slot_rows, count)
            published = self.add(
                obs[start:end],
                actions[start:end],
                rewards[start:end],
                next_obs[start:end],
                dones[start:end],
                truncated[start:end],
                terminal_mask=optional_slice(terminal_mask, start, end),
                terminal_next_obs=optional_slice(terminal_next_obs, start, end),
                critic=optional_slice(critic, start, end),
                next_critic=optional_slice(next_critic, start, end),
                terminal_next_critic=optional_slice(terminal_next_critic, start, end),
                aux=optional_slice(aux, start, end),
            )
            if not published:
                return False
        return True

    def _add_to_ingress(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        terminal_mask,
        terminal_next_obs,
        critic,
        next_critic,
        terminal_next_critic,
        aux,
        *,
        trace_start_ns: int,
    ) -> bool:
        count = int(obs.shape[0])
        if count > self._ingress_slot_rows:
            raise ValueError(
                f"Transition batch has {count} rows but bounded ingress slots hold "
                f"{self._ingress_slot_rows}"
            )
        has_critic = self._critic_dim > 0 and critic is not None
        if self._critic_dim > 0 and (critic is None or next_critic is None):
            raise ValueError("ReplayBuffer with critic_dim > 0 requires critic and next_critic")
        rewards = self._validate_transition_shapes(
            count,
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            truncated,
            critic,
            next_critic,
            aux,
            has_critic=has_critic,
        )

        wait_start_ns = time.perf_counter_ns()
        waited_for_free_slot = False
        while not self._ingress_free.acquire(timeout=0.05):
            waited_for_free_slot = True
            if bool(self._ingress_closed[0]):
                self._record_ingress_drop(
                    reason="closed",
                    wait_start_ns=wait_start_ns,
                    waited=waited_for_free_slot,
                )
                return False
            if self._stop_event is not None and self._stop_event.is_set():
                self._record_ingress_drop(
                    reason="stop",
                    wait_start_ns=wait_start_ns,
                    waited=waited_for_free_slot,
                )
                return False
        wait_end_ns = time.perf_counter_ns()
        if bool(self._ingress_closed[0]):
            self._record_ingress_drop(
                reason="closed",
                wait_start_ns=wait_start_ns,
                waited=waited_for_free_slot,
            )
            return False
        if waited_for_free_slot:
            self._record_ingress_backpressure(start_ns=wait_start_ns)

        sequence = int(self._ingress_publish_seq[0])
        slot = sequence % self._ingress_depth
        target = self._ingress_slabs[slot]
        try:
            self._write_transition_rows(
                [slab[:count] for slab in target],
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                truncated,
                critic,
                next_critic,
                aux,
                has_critic=has_critic,
            )
            nobs_field = self._schema.field("next_obs")
            self._patch_terminal_next_observations(
                target[nobs_field.slab][:count, nobs_field.offset : nobs_field.end],
                terminal_mask,
                terminal_next_obs,
                self._field_write_view(target, "next_critic", count) if has_critic else None,
                terminal_next_critic,
            )
            if self._ingress_device.type == "cuda":
                # The CPU semaphore cannot order producer CUDA writes against
                # the learner's ingress copy. Publish only after the producer
                # stream that wrote this slot has completed; retain this
                # conservative barrier rather than using a cross-process event.
                torch.cuda.current_stream(self._ingress_device).synchronize()
            start = int(self._published_ptr[0])
            self._ingress_starts[slot] = start
            self._ingress_counts[slot] = count
            self._published_ptr[0] = start + count
            self._ingress_publish_seq[0] = sequence + 1
            published_sequence = sequence + 1
            released_sequence = int(self._ingress_release_seq_tensor[0])
            occupancy = max(0, published_sequence - released_sequence)
            if occupancy > int(self._ingress_high_water[0]):
                self._ingress_high_water[0] = occupancy
            self._ingress_ready.release()
        except BaseException:
            self._ingress_free.release()
            raise

        if self.trace_recorder is not None:
            end_ns = time.perf_counter_ns()
            self.trace_recorder.add_slice(
                "replay/ingress_backpressure",
                category="replay",
                start_ns=wait_start_ns,
                end_ns=wait_end_ns,
                args={"batch_size": count, "slot": slot, "depth": self._ingress_depth},
            )
            self.trace_recorder.add_slice(
                "replay/add",
                category="replay",
                start_ns=trace_start_ns,
                end_ns=end_ns,
                args={
                    "batch_size": count,
                    "device": self.device,
                    "ingress_slot": slot,
                    "published_ptr": start + count,
                },
            )
        return True

    def _field_write_view(
        self,
        targets: list[torch.Tensor],
        field_name: str,
        count: int,
    ) -> torch.Tensor:
        field = self._schema.field(field_name)
        return targets[field.slab][:count, field.offset : field.end]

    def _validate_transition_shapes(
        self,
        count: int,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        critic,
        next_critic,
        aux,
        *,
        has_critic: bool,
    ):
        """Fail closed on transition rows that do not match the row schema."""
        schema = self._schema
        if tuple(obs.shape) != (count, schema.obs_dim):
            raise ValueError(f"obs shape {tuple(obs.shape)} != ({count}, {schema.obs_dim})")
        if tuple(next_obs.shape) != (count, schema.obs_dim):
            raise ValueError(
                f"next_obs shape {tuple(next_obs.shape)} != ({count}, {schema.obs_dim})"
            )
        if tuple(actions.shape) != (count, schema.action_dim):
            raise ValueError(
                f"actions shape {tuple(actions.shape)} != ({count}, {schema.action_dim})"
            )
        if tuple(dones.shape) != (count,):
            raise ValueError(f"dones shape {tuple(dones.shape)} != ({count},)")
        if tuple(truncated.shape) != (count,):
            raise ValueError(f"truncated shape {tuple(truncated.shape)} != ({count},)")
        if rewards.ndim == 1:
            if schema.reward_heads != 1:
                raise ValueError(
                    f"flat (count,) rewards require reward_heads=1, "
                    f"but the row schema declares reward_heads={schema.reward_heads}"
                )
            if rewards.shape[0] != count:
                raise ValueError(f"rewards shape {tuple(rewards.shape)} != ({count},)")
            rewards = rewards.unsqueeze(1)
        elif rewards.ndim != 2 or tuple(rewards.shape) != (count, schema.reward_heads):
            raise ValueError(
                f"rewards shape {tuple(rewards.shape)} != ({count}, {schema.reward_heads})"
            )
        if has_critic:
            assert critic is not None
            assert next_critic is not None
            if tuple(critic.shape) != (count, schema.critic_dim):
                raise ValueError(
                    f"critic shape {tuple(critic.shape)} != ({count}, {schema.critic_dim})"
                )
            if tuple(next_critic.shape) != (count, schema.critic_dim):
                raise ValueError(
                    f"next_critic shape {tuple(next_critic.shape)} != "
                    f"({count}, {schema.critic_dim})"
                )
        if schema.aux_width > 0:
            if aux is not None:
                if tuple(aux.shape) != (count, schema.aux_width):
                    raise ValueError(
                        f"aux shape {tuple(aux.shape)} != ({count}, {schema.aux_width})"
                    )
                if aux.is_floating_point() or aux.is_complex():
                    raise ValueError(f"aux must have an integer dtype, got {aux.dtype}")
        elif aux is not None:
            raise ValueError("transitions carry aux data but the row schema declares aux_width=0")
        return rewards

    def _write_transition_rows(
        self,
        targets: list[torch.Tensor],
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        critic,
        next_critic,
        aux,
        *,
        has_critic: bool,
    ) -> None:
        _store_rows(self._field_write_view(targets, "obs", obs.shape[0]), obs)
        _store_rows(self._field_write_view(targets, "next_obs", obs.shape[0]), next_obs)
        _store_rows(self._field_write_view(targets, "actions", obs.shape[0]), actions)
        self._field_write_view(targets, "rewards", obs.shape[0]).copy_(rewards)
        self._field_write_view(targets, "dones", obs.shape[0]).copy_(
            dones.unsqueeze(1),
        )
        self._field_write_view(targets, "truncated", obs.shape[0]).copy_(
            truncated.unsqueeze(1),
        )
        if has_critic:
            assert critic is not None
            assert next_critic is not None
            _store_rows(self._field_write_view(targets, "critic", obs.shape[0]), critic)
            _store_rows(self._field_write_view(targets, "next_critic", obs.shape[0]), next_critic)
        if self._schema.aux_width > 0:
            aux_view = self._field_write_view(targets, "aux", obs.shape[0])
            if aux is None:
                aux_view.zero_()
            else:
                aux_view.copy_(aux)

    @staticmethod
    def _patch_terminal_next_observations(
        target_next_obs,
        terminal_mask,
        terminal_next_obs,
        target_next_critic=None,
        terminal_next_critic=None,
    ) -> None:
        if terminal_mask is None or terminal_next_obs is None:
            return
        if terminal_mask.ndim != 1 or terminal_mask.shape[0] != target_next_obs.shape[0]:
            return
        if not torch.any(terminal_mask):
            return

        target_next_obs[terminal_mask] = _fit_storage_dtype(
            terminal_next_obs[terminal_mask], target_next_obs.dtype
        )

        if target_next_critic is not None and terminal_next_critic is not None:
            target_next_critic[terminal_mask] = _fit_storage_dtype(
                terminal_next_critic[terminal_mask], target_next_critic.dtype
            )

    def field_view(
        self,
        packed: torch.Tensor | list[torch.Tensor],
        field_name: str,
    ) -> torch.Tensor:
        """View one row-schema field of a packed payload in storage dtype."""
        return schema_field_view(packed, self._schema, field_name)
