"""Schema-driven packed replay row layouts.

A ``RowSchema`` declares the column groups of one packed replay transition
row and the storage dtype of each group. Groups are laid out in the fixed
legacy order ``obs | next_obs | actions | rewards | dones | truncated |
[critic | next_critic] | [aux]``; maximal runs of same-dtype groups share one
storage slab (one tensor per slab). The default schema (``reward_heads=1``,
``aux_width=0``, ``float_dtype=torch.float32``) therefore produces exactly
the historical single fp32 slab with the historical column offsets.

- ``obs`` / ``next_obs`` / ``critic`` / ``next_critic`` store in the
  configurable ``float_dtype`` (fp32 default, fp16 for compressed runs);
  ``actions`` can optionally retain fp32 with ``action_float_dtype``. All
  fields are read back as fp32.
- ``rewards`` stores ``reward_heads`` fp32 columns (per-head rewards; H=1 is
  the historical scalar column).
- ``dones`` / ``truncated`` stay the two historical fp32 flag columns.
- ``aux`` stores ``aux_width`` int32 columns of env-provided per-row metadata
  (e.g. ``(motion_id, motion_step)`` for reconstruction hooks).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FLOAT_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

_FIELD_ORDER = (
    "obs",
    "next_obs",
    "actions",
    "rewards",
    "dones",
    "truncated",
    "critic",
    "next_critic",
    "aux",
)


def resolve_float_dtype(value: str | torch.dtype) -> torch.dtype:
    """Resolve a storage float dtype from a config string, failing closed."""
    if isinstance(value, torch.dtype):
        if value in FLOAT_DTYPES.values():
            return value
        raise ValueError(
            f"replay float storage dtype must be one of {tuple(FLOAT_DTYPES)}, got {value}"
        )
    resolved = FLOAT_DTYPES.get(str(value))
    if resolved is None:
        raise ValueError(
            f"replay float storage dtype must be one of {tuple(FLOAT_DTYPES)}, got {value!r}"
        )
    return resolved


@dataclass(frozen=True)
class RowField:
    """One column group: semantics, storage dtype, and slab position."""

    name: str
    width: int
    dtype: torch.dtype
    slab: int
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.width


@dataclass(frozen=True)
class RowSlab:
    """One storage tensor: a maximal run of same-dtype column groups."""

    index: int
    dtype: torch.dtype
    width: int
    fields: tuple[str, ...]


class RowSchema:
    """Packed replay row layout generated from column-group declarations."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        critic_dim: int = 0,
        reward_heads: int = 1,
        aux_width: int = 0,
        float_dtype: str | torch.dtype = torch.float32,
        action_float_dtype: str | torch.dtype | None = None,
        store_next_obs: bool = True,
    ) -> None:
        if not isinstance(store_next_obs, bool):
            raise ValueError(f"store_next_obs must be boolean, got {store_next_obs!r}")
        for name, value in (
            ("obs_dim", obs_dim),
            ("action_dim", action_dim),
            ("critic_dim", critic_dim),
            ("reward_heads", reward_heads),
            ("aux_width", aux_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if critic_dim < 0:
            raise ValueError(f"critic_dim must be non-negative, got {critic_dim}")
        if reward_heads < 1:
            raise ValueError(f"reward_heads must be >= 1, got {reward_heads}")
        if aux_width < 0:
            raise ValueError(f"aux_width must be non-negative, got {aux_width}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.reward_heads = reward_heads
        self.aux_width = aux_width
        self.float_dtype = resolve_float_dtype(float_dtype)
        self.action_float_dtype = (
            self.float_dtype
            if action_float_dtype is None
            else resolve_float_dtype(action_float_dtype)
        )
        self.store_next_obs = store_next_obs

        widths: dict[str, int] = {
            "obs": obs_dim,
            "next_obs": obs_dim if store_next_obs else 0,
            "actions": action_dim,
            "rewards": reward_heads,
            "dones": 1,
            "truncated": 1,
            "aux": aux_width,
        }
        if critic_dim > 0:
            widths["critic"] = critic_dim
            widths["next_critic"] = critic_dim if store_next_obs else 0
        else:
            widths["critic"] = 0
            widths["next_critic"] = 0

        dtypes: dict[str, torch.dtype] = {
            name: self.float_dtype for name in ("obs", "next_obs", "critic", "next_critic")
        }
        dtypes["actions"] = self.action_float_dtype
        dtypes["rewards"] = torch.float32
        dtypes["dones"] = torch.float32
        dtypes["truncated"] = torch.float32
        dtypes["aux"] = torch.int32

        fields: dict[str, RowField] = {}
        slabs: list[RowSlab] = []
        slab_width = 0
        slab_dtype: torch.dtype | None = None
        slab_fields: list[str] = []
        for name in _FIELD_ORDER:
            width = widths[name]
            if width == 0:
                continue
            dtype = dtypes[name]
            if slab_dtype is None:
                slab_dtype = dtype
            if dtype != slab_dtype:
                slabs.append(
                    RowSlab(
                        index=len(slabs),
                        dtype=slab_dtype,
                        width=slab_width,
                        fields=tuple(slab_fields),
                    )
                )
                slab_width = 0
                slab_dtype = dtype
                slab_fields = []
            fields[name] = RowField(
                name=name,
                width=width,
                dtype=dtype,
                slab=len(slabs),
                offset=slab_width,
            )
            slab_width += width
            slab_fields.append(name)
        if slab_dtype is not None:
            slabs.append(
                RowSlab(
                    index=len(slabs),
                    dtype=slab_dtype,
                    width=slab_width,
                    fields=tuple(slab_fields),
                )
            )

        self._fields = fields
        self._slabs = tuple(slabs)
        self._storage_width = sum(field.width for field in fields.values())
        self._storage_row_bytes = sum(slab.width * int(slab.dtype.itemsize) for slab in self._slabs)

    @property
    def slabs(self) -> tuple[RowSlab, ...]:
        return self._slabs

    @property
    def storage_width(self) -> int:
        """Total column count across all slabs (not byte-weighted)."""
        return self._storage_width

    @property
    def storage_row_bytes(self) -> int:
        """Exact per-row storage bytes across all slabs."""
        return self._storage_row_bytes

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def field(self, name: str) -> RowField:
        field = self._fields.get(name)
        if field is None:
            raise KeyError(f"Row field {name!r} is unavailable in this schema")
        return field

    def without_next_obs(self) -> "RowSchema":
        """Return the ring layout with the next_obs/next_critic groups dropped.

        Used by the layered pipeline when next_obs is reconstructed by index:
        the ring then stores each observation once while ingress and sampled
        batches keep the full (dense) layout.
        """
        return RowSchema(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            critic_dim=self.critic_dim,
            reward_heads=self.reward_heads,
            aux_width=self.aux_width,
            float_dtype=self.float_dtype,
            action_float_dtype=self.action_float_dtype,
            store_next_obs=False,
        )

    def summary(self) -> dict[str, object]:
        """Config-serializable description persisted with run manifests."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "critic_dim": self.critic_dim,
            "reward_heads": self.reward_heads,
            "aux_width": self.aux_width,
            "float_dtype": str(self.float_dtype).removeprefix("torch."),
            "action_float_dtype": str(self.action_float_dtype).removeprefix("torch."),
            "store_next_obs": self.store_next_obs,
            "storage_width": self._storage_width,
            "storage_row_bytes": self._storage_row_bytes,
            "slabs": [
                {
                    "dtype": str(slab.dtype).removeprefix("torch."),
                    "width": slab.width,
                    "fields": list(slab.fields),
                }
                for slab in self._slabs
            ],
        }


def field_view(
    packed: "torch.Tensor | list[torch.Tensor]",
    schema: RowSchema,
    field_name: str,
) -> torch.Tensor:
    """View one schema field of a packed payload in storage dtype.

    ``rewards`` / ``dones`` / ``truncated`` are squeezed to 1-D when their
    group holds a single column, preserving the historical view shapes.
    """
    field = schema.field(field_name)
    view = field_columns(packed, schema, field_name)
    if field.width == 1:
        return view[:, 0]
    return view


def field_columns(
    packed: "torch.Tensor | list[torch.Tensor]",
    schema: RowSchema,
    field_name: str,
) -> torch.Tensor:
    """Two-dimensional ``(rows, width)`` view of one schema field."""
    field = schema.field(field_name)
    slab = packed[field.slab] if isinstance(packed, list) else packed
    return slab[:, field.offset : field.end]
