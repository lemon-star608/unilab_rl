"""Memory budget estimation for async RL training buffers.

Pure functions that estimate memory usage and warn if the system
is likely to OOM before allocating large shared buffers.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from typing import TYPE_CHECKING

from uni_rl.ipc.inference_ring import estimate_inference_ring_bytes

if TYPE_CHECKING:
    from uni_rl.ipc.row_schema import RowSchema

# CUDA IPC event objects are small in the CUDA API, but each handle has driver
# bookkeeping and allocator slack.  Reserving a page-sized amount per event is
# deliberately conservative: this budget is a fail-closed guard, not an exact
# reproduction of driver allocation.
CUDA_INFERENCE_EVENT_RESERVE_BYTES = 64 * 1024
CUDA_INFERENCE_OVERHEAD_FRACTION = 0.10
CUDA_INFERENCE_MIN_OVERHEAD_BYTES = 32 * 1024 * 1024
CUDA_INFERENCE_TIMING_EVENT_COUNT = 2
CUDA_TENSOR_RUNTIME_OVERHEAD_FRACTION = 0.10
CUDA_TENSOR_RUNTIME_MIN_OVERHEAD_BYTES = 32 * 1024 * 1024


def estimate_cuda_inference_ipc_bytes(
    num_envs: int,
    inference_obs_dim: int,
    action_dim: int,
    *,
    capacity: int,
    ring_on_device: bool,
    persistent_exploration_scratch: int = 0,
) -> dict[str, int | str]:
    """Return a conservative pre-flight budget for CUDA learner inference IPC.

    Exact CUDA allocator and driver overhead is not observable across all
    supported Torch/CUDA combinations.  The estimate therefore separates exact
    tensor bytes from event reserves and a conservative workspace/driver slack.
    It is intended to fail before any inference ring, event, or scratch tensor
    is allocated, not to predict ``torch.cuda.memory_allocated()`` exactly.
    """
    values = (num_envs, inference_obs_dim, action_dim, capacity)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("CUDA inference IPC byte dimensions must be integers")
    if min(values) <= 0:
        raise ValueError("CUDA inference IPC byte dimensions and capacity must be positive")
    if not isinstance(ring_on_device, bool):
        raise TypeError("ring_on_device must be boolean")
    if (
        isinstance(persistent_exploration_scratch, bool)
        or not isinstance(persistent_exploration_scratch, int)
        or persistent_exploration_scratch < 0
    ):
        raise ValueError("persistent exploration scratch bytes must be a non-negative integer")

    ring_storage = (
        estimate_inference_ring_bytes(num_envs, inference_obs_dim, action_dim, capacity=capacity)
        if ring_on_device
        else 0
    )
    observation_scratch = num_envs * inference_obs_dim * 4
    done_scratch = num_envs * 4
    action_output = num_envs * action_dim * 4
    ipc_event_count = capacity * 3 if ring_on_device else 0
    timing_event_count = CUDA_INFERENCE_TIMING_EVENT_COUNT
    cuda_event_count = ipc_event_count + timing_event_count
    cuda_events = cuda_event_count * CUDA_INFERENCE_EVENT_RESERVE_BYTES
    exact = (
        ring_storage
        + observation_scratch
        + done_scratch
        + action_output
        + persistent_exploration_scratch
    )
    overhead = max(
        int(exact * CUDA_INFERENCE_OVERHEAD_FRACTION),
        CUDA_INFERENCE_MIN_OVERHEAD_BYTES,
    )
    total = exact + cuda_events + overhead
    lines = (
        f"  Inference ring storage: {ring_storage / 1024**2:.1f} MB\n"
        f"  Inference obs/done scratch: "
        f"{(observation_scratch + done_scratch) / 1024**2:.1f} MB\n"
        f"  Inference action output: {action_output / 1024**2:.1f} MB\n"
        "  Persistent exploration scratch: "
        f"{persistent_exploration_scratch / 1024**2:.1f} MB\n"
        f"  CUDA IPC event reserve: {cuda_events / 1024**2:.1f} MB "
        f"({ipc_event_count} IPC + {timing_event_count} timing events)\n"
        f"  Conservative workspace/driver/allocator slack: "
        f"{overhead / 1024**2:.1f} MB\n"
    )
    return {
        "ring_storage": ring_storage,
        "observation_scratch": observation_scratch,
        "done_scratch": done_scratch,
        "action_output": action_output,
        "persistent_exploration_scratch": persistent_exploration_scratch,
        "cuda_events": cuda_events,
        "cuda_event_count": cuda_event_count,
        "ipc_event_count": ipc_event_count,
        "observation_event_count": capacity if ring_on_device else 0,
        "action_event_count": capacity if ring_on_device else 0,
        "consumer_done_event_count": capacity if ring_on_device else 0,
        "timing_event_count": timing_event_count,
        "conservative_overhead": overhead,
        "total": total,
        "breakdown": (
            "CUDA inference IPC budget (conservative pre-flight):\n"
            + lines
            + "  The reserve intentionally over-accounts rather than treating "
            "Torch/CUDA byte accounting as exact."
        ),
    }


def raise_if_cuda_memory_over_budget(
    estimated: dict[str, int | str],
    *,
    label: str,
    available_bytes: int,
    threshold: float = 0.8,
    user_knob: str = "training.inference_slot_capacity",
    budget_kind: str = "CUDA inference IPC",
) -> None:
    """Fail closed before CUDA inference IPC resources are materialized."""
    if isinstance(available_bytes, bool) or not isinstance(available_bytes, int):
        raise TypeError("available_bytes must be an integer")
    if available_bytes < 0:
        raise ValueError("available_bytes must be non-negative")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("memory budget threshold must be a number")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("memory budget threshold must be between zero and one")

    total = int(estimated["total"])
    allowed = int(available_bytes * float(threshold))
    if total <= allowed:
        return
    ratio = total / max(available_bytes, 1)
    raise MemoryError(
        f"{label}: conservative {budget_kind} budget "
        f"{total / 1024**2:.1f} MB exceeds the {float(threshold):.0%} limit "
        f"of {allowed / 1024**2:.1f} MB (device free {available_bytes / 1024**2:.1f} MB, "
        f"{ratio:.0%} requested). Reduce {user_knob}, algo.num_envs, or model/input "
        f"dimensions before startup. Budget breakdown:\n{estimated['breakdown']}"
    )


def estimate_offpolicy_bytes(
    num_envs: int,
    replay_buffer_n: int,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    ingress_depth: int = 2,
    ingress_slot_rows: int | None = None,
    inference_ring_bytes: int = 0,
    ingress_on_device: bool = False,
    row_schema: "RowSchema | None" = None,
    replay_storage_owner: str = "gpu",
    reconstruct_next_obs: bool = False,
) -> dict[str, int | str]:
    """Estimate host shared memory for bounded off-policy replay ingress.

    With a CPU-pinned (layered) replay storage owner the authoritative ring is
    accounted as host pinned bytes; the device-resident owner keeps the
    historical zero host share.
    """
    if replay_storage_owner not in ("gpu", "cpu"):
        raise ValueError(
            f"replay_storage_owner must be 'gpu' or 'cpu', got {replay_storage_owner!r}"
        )
    if not isinstance(ingress_on_device, bool):
        raise TypeError("ingress_on_device must be boolean")
    integer_values = (num_envs, replay_buffer_n, obs_dim, action_dim, critic_dim, ingress_depth)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise TypeError("off-policy host memory dimensions must be integers")
    # critic_dim == 0 is the supported shared-critic layout (the critic
    # consumes the actor observation and replay rows omit the duplicated group).
    if min(num_envs, replay_buffer_n, obs_dim, action_dim, ingress_depth) <= 0 or critic_dim < 0:
        raise ValueError("off-policy host memory dimensions must be positive")
    if ingress_slot_rows is None:
        effective_ingress_slot_rows = num_envs
    elif (
        isinstance(ingress_slot_rows, bool)
        or not isinstance(ingress_slot_rows, int)
        or ingress_slot_rows <= 0
        or ingress_slot_rows > num_envs
    ):
        raise ValueError(
            "ingress_slot_rows must be a positive integer no greater than num_envs, "
            f"got {ingress_slot_rows!r} (num_envs={num_envs})"
        )
    else:
        effective_ingress_slot_rows = ingress_slot_rows
    if (
        isinstance(inference_ring_bytes, bool)
        or not isinstance(inference_ring_bytes, int)
        or inference_ring_bytes < 0
    ):
        raise ValueError("inference_ring_bytes must be a non-negative integer")
    row_width = 2 * obs_dim + action_dim + 3 + 2 * critic_dim
    row_bytes = row_width * 4 if row_schema is None else row_schema.storage_row_bytes
    ring_row_bytes = row_bytes
    if reconstruct_next_obs:
        # The ring drops the next_obs/next_critic groups (index reconstruction
        # aliases them from obs/critic rows); ingress slots stay dense.
        element_size = 4 if row_schema is None else int(row_schema.float_dtype.itemsize)
        ring_row_bytes = row_bytes - (obs_dim + critic_dim) * element_size
    capacity = replay_buffer_n * num_envs
    ingress_bytes = (
        0 if ingress_on_device else ingress_depth * effective_ingress_slot_rows * row_bytes
    )
    pinned_replay_bytes = capacity * ring_row_bytes if replay_storage_owner == "cpu" else 0
    inference_ring_line = (
        f"  Inference ring: {inference_ring_bytes / 1024**2:.0f} MB\n"
        if inference_ring_bytes
        else ""
    )
    replay_line = (
        f"Replay: {pinned_replay_bytes / 1024**2:.0f} MB pinned host storage "
        f"({capacity} rows × {ring_row_bytes} B, CPU-layered owner)\n"
        if replay_storage_owner == "cpu"
        else (
            "Replay: 0 MB shared host memory "
            f"({capacity} rows remain authoritative on the learner device)\n"
        )
    )
    return {
        "replay_buffer": pinned_replay_bytes,
        "bounded_ingress_slots": ingress_bytes,
        "ingress_slot_rows": effective_ingress_slot_rows,
        "row_width": row_width,
        "inference_ring": int(inference_ring_bytes),
        "total": pinned_replay_bytes + ingress_bytes + int(inference_ring_bytes),
        "breakdown": (
            replay_line
            + "  Bounded ingress: "
            + (
                "0 MB (device-resident)\n"
                if ingress_on_device
                else (
                    f"{ingress_bytes / 1024**2:.0f} MB "
                    f"({int(ingress_depth)} slots × {effective_ingress_slot_rows} rows × "
                    f"{row_bytes} B)\n"
                )
            )
            + inference_ring_line
            + "  Excludes MuJoCo BatchEnvPool/native allocations, CUDA pinned/shared "
            "registration, and driver memory."
        ),
    }


def estimate_cuda_tensor_runtime_bytes(
    *,
    num_envs: int,
    replay_buffer_n: int,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    inference_obs_dim: int,
    sample_count: int,
    inference_slot_capacity: int,
    replay_ingress_depth: int,
    replay_ingress_slot_rows: int,
    collector_tensor_native: bool,
    persistent_exploration_scratch: int = 0,
    row_schema: "RowSchema | None" = None,
    replay_storage_owner: str = "gpu",
) -> dict[str, int | str]:
    """Conservatively pre-flight all startup CUDA tensor-runtime allocations.

    The replay pipeline retains a second allocation guard after construction. This
    estimate deliberately runs before ingress, ring, replay storage, and batch
    tensors exist so an impossible combination cannot strand CUDA IPC resources.
    With a CPU-pinned (layered) replay owner the authoritative ring is accounted
    on the host, not here; learner batch slots stay on device.
    """
    if replay_storage_owner not in ("gpu", "cpu"):
        raise ValueError(
            f"replay_storage_owner must be 'gpu' or 'cpu', got {replay_storage_owner!r}"
        )
    replay_values = (
        num_envs,
        replay_buffer_n,
        obs_dim,
        action_dim,
        critic_dim,
        sample_count,
        replay_ingress_depth,
        replay_ingress_slot_rows,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in replay_values):
        raise TypeError("CUDA tensor-runtime dimensions must be integers")
    # critic_dim == 0 is the supported shared-critic layout.
    if (
        min(
            num_envs,
            replay_buffer_n,
            obs_dim,
            action_dim,
            sample_count,
            replay_ingress_depth,
            replay_ingress_slot_rows,
        )
        <= 0
        or critic_dim < 0
    ):
        raise ValueError("CUDA tensor-runtime dimensions must be positive")
    if replay_ingress_slot_rows > num_envs:
        raise ValueError(
            "replay_ingress_slot_rows must be no greater than num_envs "
            f"({replay_ingress_slot_rows} > {num_envs})"
        )
    if not isinstance(collector_tensor_native, bool):
        raise TypeError("collector_tensor_native must be boolean")

    inference = estimate_cuda_inference_ipc_bytes(
        num_envs,
        inference_obs_dim,
        action_dim,
        capacity=inference_slot_capacity,
        ring_on_device=collector_tensor_native,
        persistent_exploration_scratch=persistent_exploration_scratch,
    )
    row_width = 2 * obs_dim + action_dim + 3 + 2 * critic_dim
    row_bytes = row_width * 4 if row_schema is None else row_schema.storage_row_bytes
    replay_capacity = replay_buffer_n * num_envs
    replay_storage = replay_capacity * row_bytes if replay_storage_owner == "gpu" else 0
    learner_batch_slots = 2 * sample_count * row_bytes
    replay_ingress = (
        replay_ingress_depth * replay_ingress_slot_rows * row_bytes
        if collector_tensor_native
        else 0
    )
    replay_exact = replay_storage + learner_batch_slots + replay_ingress
    replay_overhead = max(
        int(replay_exact * CUDA_TENSOR_RUNTIME_OVERHEAD_FRACTION),
        CUDA_TENSOR_RUNTIME_MIN_OVERHEAD_BYTES,
    )
    total = int(inference["total"]) + replay_exact + replay_overhead
    storage_line = (
        f"  Replay storage: {replay_storage / 1024**2:.1f} MB "
        f"({replay_capacity} rows × {row_bytes} B)\n"
        if replay_storage_owner == "gpu"
        else (
            "  Replay storage: 0.0 MB on device "
            f"({replay_capacity} rows × {row_bytes} B pinned on host, CPU-layered owner)\n"
        )
    )
    lines = (
        "CUDA tensor-runtime budget (conservative pre-flight):\n"
        + storage_line
        + f"  Learner double-buffer batches: {learner_batch_slots / 1024**2:.1f} MB "
        f"(2 slots × {sample_count} rows × {row_bytes} B)\n"
        f"  CUDA replay ingress: {replay_ingress / 1024**2:.1f} MB "
        f"({replay_ingress_depth} × {replay_ingress_slot_rows} rows; "
        f"collector_tensor_native={collector_tensor_native})\n"
        f"  Inference/IPC subtotal: {int(inference['total']) / 1024**2:.1f} MB\n"
        f"  Conservative replay workspace/allocator slack: "
        f"{replay_overhead / 1024**2:.1f} MB\n"
    )
    return {
        "replay_storage": replay_storage,
        "learner_batch_slots": learner_batch_slots,
        "replay_ingress": replay_ingress,
        "replay_ingress_depth": replay_ingress_depth,
        "replay_ingress_slot_rows": replay_ingress_slot_rows,
        "replay_row_width": row_width,
        "inference_total": int(inference["total"]),
        "conservative_replay_overhead": replay_overhead,
        "total": total,
        "breakdown": (
            lines
            + str(inference["breakdown"])
            + "\nThe estimate intentionally runs before any tensor-runtime allocation."
        ),
    }


def estimate_appo_bytes(
    num_envs: int,
    steps_per_env: int,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    num_slots: int = 4,
) -> dict[str, int | str]:
    """Estimate memory for APPO rollout ring buffer."""
    per_step = obs_dim + action_dim + 1 + 1 + 1 + 1 + critic_dim
    per_slot = num_envs * steps_per_env * per_step * 4
    last_obs_per_slot = num_envs * (obs_dim + critic_dim) * 4
    total_per_slot = per_slot + last_obs_per_slot
    total = total_per_slot * num_slots

    return {
        "ring_buffer": total,
        "total": total,
        "breakdown": (
            f"Ring buffer: {total / 1024**2:.0f} MB "
            f"({num_slots} slots × {num_envs} envs × {steps_per_env} steps × "
            f"{per_step} cols × 4B)"
        ),
    }


def get_available_memory_bytes() -> int | None:
    """Best-effort available memory detection."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    return None


def get_shared_memory_available_bytes(path: str = "/dev/shm") -> int | None:
    """Best-effort available shared-memory space detection."""
    try:
        return int(shutil.disk_usage(path).free)
    except (OSError, ValueError):
        return None


def raise_if_shared_memory_over_budget(
    estimated: dict[str, int | str],
    label: str,
    threshold: float = 0.8,
    path: str = "/dev/shm",
) -> None:
    """Fail before allocating shared buffers that exceed shared-memory capacity."""
    available = get_shared_memory_available_bytes(path)
    if available is None:
        return

    total = int(estimated["total"])
    ratio = total / max(available, 1)
    if total <= available * threshold:
        return

    est_gb = total / 1024**3
    avail_gb = available / 1024**3
    raise MemoryError(
        f"{label}: estimated shared-memory allocation {est_gb:.1f} GB exceeds "
        f"{path} available {avail_gb:.1f} GB ({ratio:.0%} usage). "
        "Reduce algo.num_envs or algo.replay_buffer_n, or increase /dev/shm."
    )


def raise_if_host_memory_over_budget(
    estimated: dict[str, int | str],
    label: str,
    threshold: float = 0.8,
) -> None:
    """Fail closed before pinning host memory beyond available RAM.

    Pinned (page-locked) pages are non-reclaimable, so overcommitting them can
    stall the whole host — the same severity class as VRAM overcommit, which
    already hard-fails. Unlike the pageable host estimate (warn-only), the
    CPU-layered replay pool is therefore gated against MemAvailable. Platforms
    without a readable /proc/meminfo skip the check; cudaHostRegister remains
    the fail-closed gate of last resort there.
    """
    from uni_rl.ipc.replay_pipelines.layered import host_available_memory_bytes

    available = host_available_memory_bytes()
    if available is None:
        warnings.warn(
            f"{label}: host RAM budget check skipped (/proc/meminfo unreadable); "
            "cudaHostRegister remains the fail-closed gate.",
            stacklevel=2,
        )
        return

    total = int(estimated["total"])
    ratio = total / max(available, 1)
    if total <= available * threshold:
        return

    est_gb = total / 1024**3
    avail_gb = available / 1024**3
    raise MemoryError(
        f"{label}: estimated host RAM allocation {est_gb:.1f} GB "
        f"(pinned replay + bounded ingress + inference ring) exceeds "
        f"{threshold:.0%} of MemAvailable {avail_gb:.1f} GB ({ratio:.0%} usage). "
        "Reduce algo.num_envs or algo.replay_buffer_n, or shrink the replay row "
        "(fp16 storage, critic:0, next_obs reconstruction)."
    )


def warn_if_over_budget(
    estimated: dict[str, int | str],
    label: str,
    threshold: float = 0.8,
) -> None:
    """Print a warning if estimated memory exceeds threshold of available."""
    if os.environ.get("UNILAB_SKIP_MEMORY_CHECK"):
        return

    available = get_available_memory_bytes()
    if available is None:
        return

    total = int(estimated["total"])
    ratio = total / available

    if ratio > threshold:
        est_gb = total / 1024**3
        avail_gb = available / 1024**3
        breakdown = estimated.get("breakdown", "")
        print(
            f"\n[Memory Warning] {label}: estimated {est_gb:.1f} GB, "
            f"available {avail_gb:.1f} GB ({ratio:.0%} usage).\n"
            f"  {breakdown}\n"
            f"  Consider reducing algo.num_envs or algo.replay_buffer_n.\n"
            f"  Native backend and driver memory may push actual usage higher.\n"
            f"  Suppress: export UNILAB_SKIP_MEMORY_CHECK=1\n",
            file=sys.stderr,
            flush=True,
        )
