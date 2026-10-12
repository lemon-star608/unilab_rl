"""Row-schema layout, validation, and replay-buffer integration tests."""

from __future__ import annotations

import pytest
import torch

from uni_rl.ipc.replay_buffer import ReplayBuffer
from uni_rl.ipc.row_schema import RowSchema, resolve_float_dtype

_OBS_DIM = 8
_ACTION_DIM = 3
_CRITIC_DIM = 3


def test_default_schema_matches_legacy_packed_layout() -> None:
    schema = RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, critic_dim=_CRITIC_DIM)

    assert len(schema.slabs) == 1
    slab = schema.slabs[0]
    assert slab.dtype is torch.float32
    # Legacy formula: 2*obs + action + 3 + 2*critic, all fp32.
    assert schema.storage_width == 2 * _OBS_DIM + _ACTION_DIM + 3 + 2 * _CRITIC_DIM
    assert slab.width == schema.storage_width
    assert schema.storage_row_bytes == schema.storage_width * 4

    assert schema.field("obs").offset == 0
    assert schema.field("next_obs").offset == _OBS_DIM
    assert schema.field("actions").offset == 2 * _OBS_DIM
    assert schema.field("rewards").offset == 2 * _OBS_DIM + _ACTION_DIM
    assert schema.field("rewards").width == 1
    assert schema.field("dones").offset == 2 * _OBS_DIM + _ACTION_DIM + 1
    assert schema.field("truncated").offset == 2 * _OBS_DIM + _ACTION_DIM + 2
    assert schema.field("critic").offset == 2 * _OBS_DIM + _ACTION_DIM + 3
    assert schema.field("next_critic").offset == 2 * _OBS_DIM + _ACTION_DIM + 3 + _CRITIC_DIM
    assert not schema.has_field("aux")


def test_default_schema_without_critic_matches_legacy_layout() -> None:
    schema = RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM)

    assert len(schema.slabs) == 1
    assert schema.storage_width == 2 * _OBS_DIM + _ACTION_DIM + 3
    assert not schema.has_field("critic")
    assert not schema.has_field("next_critic")
    with pytest.raises(KeyError):
        schema.field("critic")


def test_multi_head_aux_schema_splits_an_int32_slab() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
    )

    assert [slab.dtype for slab in schema.slabs] == [torch.float32, torch.int32]
    float_slab, aux_slab = schema.slabs
    assert float_slab.width == 2 * _OBS_DIM + _ACTION_DIM + 2 + 2 + 2 * _CRITIC_DIM
    assert aux_slab.width == 2
    assert schema.field("rewards").width == 2
    assert schema.field("rewards").slab == 0
    assert schema.field("aux").slab == 1
    assert schema.field("aux").offset == 0
    assert schema.storage_row_bytes == float_slab.width * 4 + 2 * 4


def test_fp16_schema_stratifies_float_slabs() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
        float_dtype="fp16",
    )

    # obs/next_obs/actions | rewards+flags | critic/next_critic | aux
    assert [slab.dtype for slab in schema.slabs] == [
        torch.float16,
        torch.float32,
        torch.float16,
        torch.int32,
    ]
    assert schema.field("obs").slab == 0
    assert schema.field("rewards").slab == 1
    assert schema.field("rewards").offset == 0
    assert schema.field("critic").slab == 2
    assert schema.field("critic").offset == 0
    assert (
        schema.storage_row_bytes
        == (2 * _OBS_DIM + _ACTION_DIM) * 2 + 4 * 4 + 2 * _CRITIC_DIM * 2 + 2 * 4
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"obs_dim": 0, "action_dim": 3}, "obs_dim must be positive"),
        ({"obs_dim": 8, "action_dim": 0}, "action_dim must be positive"),
        ({"obs_dim": 8, "action_dim": 3, "critic_dim": -1}, "critic_dim must be non-negative"),
        ({"obs_dim": 8, "action_dim": 3, "reward_heads": 0}, "reward_heads must be >= 1"),
        ({"obs_dim": 8, "action_dim": 3, "aux_width": -1}, "aux_width must be non-negative"),
        ({"obs_dim": 8.0, "action_dim": 3}, "obs_dim must be an integer"),
        ({"obs_dim": 8, "action_dim": 3, "float_dtype": "bf16"}, "fp32"),
        ({"obs_dim": 8, "action_dim": 3, "float_dtype": torch.bfloat16}, "fp32"),
    ],
)
def test_schema_rejects_invalid_declarations(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RowSchema(**kwargs)


def test_resolve_float_dtype_accepts_torch_dtype() -> None:
    assert resolve_float_dtype(torch.float16) is torch.float16
    assert resolve_float_dtype("fp32") is torch.float32


def _schema_buffer(schema: RowSchema, *, capacity: int = 16, slot_rows: int = 4) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        obs_dim=schema.obs_dim,
        action_dim=schema.action_dim,
        device="cpu",
        critic_dim=schema.critic_dim,
        ingress_slot_rows=slot_rows,
        row_schema=schema,
    )


def _pattern_batch(n: int, heads: int):
    rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    return {
        "obs": rows * 10 + torch.arange(_OBS_DIM, dtype=torch.float32),
        "actions": rows * 100 + torch.arange(_ACTION_DIM, dtype=torch.float32),
        "rewards": rows * 7 + torch.arange(heads, dtype=torch.float32),
        "next_obs": rows * 1000 + torch.arange(_OBS_DIM, dtype=torch.float32),
        "dones": torch.zeros(n),
        "truncated": torch.ones(n),
        "critic": rows * 5 + torch.arange(_CRITIC_DIM, dtype=torch.float32),
        "next_critic": rows * 6 + torch.arange(_CRITIC_DIM, dtype=torch.float32),
    }


def _take(buf: ReplayBuffer):
    ingress = buf.take_published_ingress()
    assert ingress is not None
    return ingress


def test_default_buffer_keeps_bare_fp32_slots_and_legacy_views() -> None:
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        critic_dim=_CRITIC_DIM,
        ingress_slot_rows=4,
    )
    assert isinstance(buf._ingress_slots[0], torch.Tensor)
    assert buf._ingress_slots[0].dtype is torch.float32
    assert buf._ingress_slots[0].shape == (4, buf.storage_width)

    n = 4
    batch = _pattern_batch(n, 1)
    rewards = batch["rewards"].squeeze(1).contiguous()
    assert buf.add(
        batch["obs"],
        batch["actions"],
        rewards,
        batch["next_obs"],
        batch["dones"],
        batch["truncated"],
        critic=batch["critic"],
        next_critic=batch["next_critic"],
    )
    slot, start, count, packed = _take(buf)
    assert isinstance(packed, torch.Tensor)
    torch.testing.assert_close(buf.field_view(packed, "rewards"), rewards)
    torch.testing.assert_close(buf.field_view(packed, "obs"), batch["obs"])
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


def test_schema_buffer_roundtrip_multi_head_aux() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
    )
    buf = _schema_buffer(schema)
    assert isinstance(buf._ingress_slots[0], list)
    assert [slab.dtype for slab in buf._ingress_slots[0]] == [torch.float32, torch.int32]

    n = 4
    batch = _pattern_batch(n, 2)
    aux = torch.arange(n * 2, dtype=torch.int64).reshape(n, 2) + 11
    assert buf.add(
        batch["obs"],
        batch["actions"],
        batch["rewards"],
        batch["next_obs"],
        batch["dones"],
        batch["truncated"],
        critic=batch["critic"],
        next_critic=batch["next_critic"],
        aux=aux,
    )
    slot, start, count, packed = _take(buf)
    assert isinstance(packed, list) and len(packed) == 2
    assert buf.field_view(packed, "rewards").shape == (n, 2)
    assert torch.equal(buf.field_view(packed, "rewards"), batch["rewards"])
    assert torch.equal(buf.field_view(packed, "aux"), aux.to(torch.int32))
    # Scalar flag groups stay 1-D like the legacy views.
    assert buf.field_view(packed, "dones").shape == (n,)
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


def test_schema_buffer_zero_fills_missing_aux() -> None:
    buf = _schema_buffer(RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, aux_width=2))
    n = 3
    batch = _pattern_batch(n, 1)
    assert buf.add(
        batch["obs"],
        batch["actions"],
        batch["rewards"].squeeze(1),
        batch["next_obs"],
        batch["dones"],
        batch["truncated"],
    )
    slot, start, count, packed = _take(buf)
    assert torch.equal(buf.field_view(packed, "aux"), torch.zeros(n, 2, dtype=torch.int32))
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


def test_fp16_buffer_casts_float_groups_and_keeps_rewards_fp32() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
        float_dtype="fp16",
    )
    buf = _schema_buffer(schema)
    n = 4
    batch = _pattern_batch(n, 2)
    aux = torch.ones(n, 2, dtype=torch.int32)
    assert buf.add(
        batch["obs"],
        batch["actions"],
        batch["rewards"],
        batch["next_obs"],
        batch["dones"],
        batch["truncated"],
        critic=batch["critic"],
        next_critic=batch["next_critic"],
        aux=aux,
    )
    slot, start, count, packed = _take(buf)
    assert [slab.dtype for slab in packed] == [
        torch.float16,
        torch.float32,
        torch.float16,
        torch.int32,
    ]
    assert torch.equal(buf.field_view(packed, "obs"), batch["obs"].to(torch.float16))
    assert torch.equal(buf.field_view(packed, "actions"), batch["actions"].to(torch.float16))
    assert torch.equal(buf.field_view(packed, "rewards"), batch["rewards"])
    assert torch.equal(buf.field_view(packed, "critic"), batch["critic"].to(torch.float16))
    buf.commit_ingress(slot=slot, start=start, count=count)
    buf.close()


def test_buffer_rejects_schema_dim_mismatch() -> None:
    schema = RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM)
    with pytest.raises(ValueError, match="do not match"):
        ReplayBuffer(
            capacity=16,
            obs_dim=_OBS_DIM + 1,
            action_dim=_ACTION_DIM,
            device="cpu",
            ingress_slot_rows=4,
            row_schema=schema,
        )


def test_buffer_fail_closed_on_shape_and_dtype_mismatches() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
    )
    buf = _schema_buffer(schema)
    n = 4
    batch = _pattern_batch(n, 2)
    aux = torch.ones(n, 2, dtype=torch.int32)

    def add(**overrides):
        kwargs = {
            "obs": batch["obs"],
            "actions": batch["actions"],
            "rewards": batch["rewards"],
            "next_obs": batch["next_obs"],
            "dones": batch["dones"],
            "truncated": batch["truncated"],
            "critic": batch["critic"],
            "next_critic": batch["next_critic"],
            "aux": aux,
        }
        kwargs.update(overrides)
        positional = (
            kwargs.pop("obs"),
            kwargs.pop("actions"),
            kwargs.pop("rewards"),
            kwargs.pop("next_obs"),
            kwargs.pop("dones"),
            kwargs.pop("truncated"),
        )
        return buf.add(*positional, **kwargs)

    with pytest.raises(ValueError, match="reward_heads=2"):
        add(rewards=torch.randn(n))
    with pytest.raises(ValueError, match="rewards shape"):
        add(rewards=torch.randn(n, 3))
    with pytest.raises(ValueError, match="aux shape"):
        add(aux=torch.ones(n, 1, dtype=torch.int32))
    with pytest.raises(ValueError, match="integer dtype"):
        add(aux=torch.ones(n, 2))
    with pytest.raises(ValueError, match="obs shape"):
        add(obs=torch.randn(n, _OBS_DIM + 1))
    with pytest.raises(ValueError, match="dones shape"):
        add(dones=torch.zeros(n, 1))
    buf.close()


def test_default_schema_buffer_rejects_aux_and_multi_head_rewards() -> None:
    buf = ReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=4,
    )
    n = 4
    batch = _pattern_batch(n, 2)
    with pytest.raises(ValueError, match="rewards shape"):
        buf.add(
            batch["obs"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
            batch["truncated"],
        )
    with pytest.raises(ValueError, match="aux_width=0"):
        buf.add(
            batch["obs"],
            batch["actions"],
            batch["rewards"][:, 0].contiguous(),
            batch["next_obs"],
            batch["dones"],
            batch["truncated"],
            aux=torch.ones(n, 2, dtype=torch.int32),
        )
    buf.close()


def test_without_next_obs_drops_only_the_next_groups() -> None:
    schema = RowSchema(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        reward_heads=2,
        aux_width=2,
        float_dtype="fp16",
    )
    ring = schema.without_next_obs()

    assert ring.store_next_obs is False
    assert not ring.has_field("next_obs")
    assert not ring.has_field("next_critic")
    for name in ("obs", "actions", "rewards", "dones", "truncated", "critic", "aux"):
        assert ring.has_field(name)
    assert ring.field("obs").offset == 0
    # actions shift into the columns next_obs occupied before.
    assert ring.field("actions").offset == _OBS_DIM
    assert ring.field("actions").width == _ACTION_DIM
    assert ring.field("critic").offset == 0
    assert ring.field("critic").slab == 2
    assert ring.storage_row_bytes == schema.storage_row_bytes - (_OBS_DIM + _CRITIC_DIM) * 2
    # The source schema is unchanged.
    assert schema.has_field("next_obs")
    assert schema.store_next_obs is True


def test_store_next_obs_flag_is_validated() -> None:
    with pytest.raises(ValueError, match="store_next_obs"):
        RowSchema(obs_dim=_OBS_DIM, action_dim=_ACTION_DIM, store_next_obs=1)  # type: ignore[arg-type]
