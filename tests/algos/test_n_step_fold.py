"""N-step fold semantics, raw stats ring, and stats drain tests."""

from __future__ import annotations

import threading

import pytest
import torch

from uni_rl.ipc.reward_stats_ring import RawRewardStatsRing
from uni_rl.offpolicy.n_step import NStepRewardFold

_M = 3
_OBS_DIM = 4
_CRITIC_DIM = 2


def _step_tensors(
    step: int,
    *,
    heads: int,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor,
    final_obs: torch.Tensor | None,
) -> dict:
    obs = torch.full((_M, _OBS_DIM), float(step))
    next_obs = torch.full((_M, _OBS_DIM), float(step) + 1000.0)
    critic = torch.full((_M, _CRITIC_DIM), float(step) + 0.1)
    next_critic = torch.full((_M, _CRITIC_DIM), float(step) + 1000.1)
    aux = torch.full((_M, 2), step, dtype=torch.int32)
    return {
        "obs": obs,
        "actions": torch.full((_M, 2), float(step) + 0.5),
        "rewards": rewards,
        "next_obs": next_obs,
        "dones": dones,
        "truncated": truncated,
        "terminal_mask": dones > 0.5,
        "terminal_next_obs": final_obs,
        "critic": critic,
        "next_critic": next_critic,
        "terminal_next_critic": final_obs[..., :_CRITIC_DIM] if final_obs is not None else None,
        "aux": aux,
    }


def _script(T: int, *, heads: int) -> list[dict]:
    """Scripted env steps with terminated, timeout, and back-to-back episodes."""
    torch.manual_seed(0)
    steps = []
    for t in range(T):
        rewards = torch.randn(_M, heads) if heads > 1 else torch.randn(_M)
        dones = torch.zeros(_M)
        truncated = torch.zeros(_M)
        if t == 4:
            dones[0] = 1.0  # terminated mid-span
        if t == 5:
            dones[1] = 1.0  # back-to-back: env 1 resets right after
        if t == 7:
            dones[2] = 1.0
            truncated[2] = 1.0  # timeout mid-span
        if t == 11:
            dones[0] = 1.0
            dones[1] = 1.0
        final_obs = None
        if dones.any():
            final_obs = torch.full((_M, _OBS_DIM), float(t) + 5000.0)
        steps.append(
            _step_tensors(
                t,
                heads=heads,
                rewards=rewards,
                dones=dones,
                truncated=truncated,
                final_obs=final_obs,
            )
        )
    return steps


def _reference_fold(steps: list[dict], n: int, gamma: float) -> list[dict]:
    """Per-env loop mirror of the original n-step fold (same accumulation order)."""
    out = []
    for emit_t in range(n - 1, len(steps)):
        start = emit_t - n + 1
        row = {"rewards": steps[emit_t]["rewards"].clone()}
        span = {
            name: steps[emit_t][name].clone()
            for name in ("dones", "truncated", "next_obs", "next_critic", "terminal_mask")
        }
        terminal_next_obs = steps[emit_t]["terminal_next_obs"]
        for index in reversed(range(start, emit_t)):
            done = steps[index]["dones"]
            if row["rewards"].ndim == 2:
                chain = (1.0 - done).unsqueeze(-1)
            else:
                chain = 1.0 - done
            row["rewards"] = steps[index]["rewards"] + gamma * row["rewards"] * chain
            mask = done > 0.5
            if not bool(mask.any()):
                continue
            for name in span:
                span[name][mask] = steps[index][name][mask]
            if steps[index]["terminal_next_obs"] is not None:
                if terminal_next_obs is None:
                    terminal_next_obs = steps[index]["terminal_next_obs"].clone()
                else:
                    terminal_next_obs[mask] = steps[index]["terminal_next_obs"][mask]
        row.update(span)
        row["terminal_next_obs"] = terminal_next_obs
        row["obs"] = steps[start]["obs"]
        row["actions"] = steps[start]["actions"]
        row["critic"] = steps[start]["critic"]
        row["aux"] = steps[start]["aux"]
        out.append(row)
    return out


def test_fold_preserves_bootstrap_eligible_truncation_encoding() -> None:
    """The fold must not re-derive truncation from the combined done flag.

    Collector input already encodes replay ``truncated`` as bootstrap-eligible
    truncation (``truncated & ~terminated``, uni_rl #29): env0 normal, env1
    failure-only, env2 timeout-only, env3 failure+timeout. The folded row must
    carry that encoding to replay verbatim so FlashSAC's
    ``clamp(1 - dones + truncated)`` never bootstraps the overlap row.
    """
    dones = torch.tensor([0.0, 1.0, 1.0, 1.0])
    truncated = torch.tensor([0.0, 0.0, 1.0, 0.0])
    folder = NStepRewardFold(1, 0.99)
    row = folder.push(
        _step_tensors(
            0,
            heads=1,
            rewards=torch.ones(4),
            dones=dones,
            truncated=truncated,
            final_obs=torch.full((4, _OBS_DIM), 99.0),
        )
    )
    assert row is not None
    assert torch.equal(row["dones"], dones)
    assert torch.equal(row["truncated"], truncated)


@pytest.mark.parametrize("heads", [1, 2])
def test_fold_matches_reference_rollout(heads: int) -> None:
    n, gamma, T = 5, 0.98, 14
    steps = _script(T, heads=heads)
    folder = NStepRewardFold(n, gamma)
    emitted = []
    for transition in steps:
        row = folder.push(transition)
        if row is not None:
            emitted.append(row)

    reference = _reference_fold(steps, n, gamma)
    assert len(emitted) == T - n + 1 == len(reference)
    for got, want in zip(emitted, reference):
        assert torch.equal(got["rewards"], want["rewards"])
        assert torch.equal(got["dones"], want["dones"])
        assert torch.equal(got["truncated"], want["truncated"])
        assert torch.equal(got["next_obs"], want["next_obs"])
        assert torch.equal(got["next_critic"], want["next_critic"])
        assert torch.equal(got["terminal_mask"], want["terminal_mask"])
        assert torch.equal(got["obs"], want["obs"])
        assert torch.equal(got["actions"], want["actions"])
        assert torch.equal(got["critic"], want["critic"])
        assert torch.equal(got["aux"], want["aux"])
        if want["terminal_next_obs"] is None:
            assert got["terminal_next_obs"] is None
        else:
            assert torch.equal(got["terminal_next_obs"], want["terminal_next_obs"])


def test_fold_chain_stops_at_first_done() -> None:
    n, gamma = 3, 0.5
    folder = NStepRewardFold(n, gamma)
    steps = _script(3, heads=1)
    steps[2]["dones"][1] = 1.0
    row = None
    for transition in steps:
        row = folder.push(transition) or row
    assert row is not None
    # Env 1: done at the last span step keeps its reward; accumulation is
    # newest-first, R = r0 + g (r1 + g r2).
    want = steps[0]["rewards"][1] + gamma * (
        steps[1]["rewards"][1] + gamma * steps[2]["rewards"][1]
    )
    assert torch.equal(row["rewards"][1], want)
    assert row["dones"][1] == 1.0
    assert torch.equal(row["next_obs"][1], steps[2]["next_obs"][1])
    # Env 0 (no done in span): same full-window arithmetic, flags stay zero.
    want0 = steps[0]["rewards"][0] + gamma * (
        steps[1]["rewards"][0] + gamma * steps[2]["rewards"][0]
    )
    assert torch.equal(row["rewards"][0], want0)
    assert row["dones"][0] == 0.0


def test_fold_early_done_truncates_tail_rewards() -> None:
    n, gamma = 5, 0.9
    folder = NStepRewardFold(n, gamma)
    steps = _script(6, heads=1)
    steps[1]["dones"][2] = 1.0
    row = None
    for transition in steps[:5]:
        row = folder.push(transition) or row
    assert row is not None
    # Env 2 terminates at t=1: rewards from t>=2 must not leak into the fold.
    want = steps[0]["rewards"][2] + gamma * steps[1]["rewards"][2]
    assert torch.equal(row["rewards"][2], want)
    assert row["dones"][2] == 1.0
    assert torch.equal(row["next_obs"][2], steps[1]["next_obs"][2])


def test_fold_passthrough_fields_match_oldest_transition_and_are_alias_safe() -> None:
    folder = NStepRewardFold(2, 0.99)
    first = _step_tensors(
        0,
        heads=1,
        rewards=torch.ones(_M),
        dones=torch.zeros(_M),
        truncated=torch.zeros(_M),
        final_obs=None,
    )
    second = _step_tensors(
        1,
        heads=1,
        rewards=torch.ones(_M),
        dones=torch.zeros(_M),
        truncated=torch.zeros(_M),
        final_obs=None,
    )
    expected_obs = first["obs"].clone()
    assert folder.push(first) is None
    # Envs may reuse observation buffers; mutating the pushed tensors must not
    # corrupt the window held for the next emit.
    first["obs"].fill_(-999.0)
    first["aux"].fill_(-999)
    row = folder.push(second)
    assert row is not None
    assert torch.equal(row["obs"], expected_obs)
    assert torch.equal(
        row["aux"],
        _step_tensors(
            0,
            heads=1,
            rewards=torch.ones(_M),
            dones=torch.zeros(_M),
            truncated=torch.zeros(_M),
            final_obs=None,
        )["aux"],
    )


def test_fold_n1_is_identity_by_value() -> None:
    folder = NStepRewardFold(1, 0.99)
    transition = _step_tensors(
        0,
        heads=1,
        rewards=torch.randn(_M),
        dones=torch.zeros(_M),
        truncated=torch.zeros(_M),
        final_obs=None,
    )
    row = folder.push(transition)
    assert row is not None
    assert torch.equal(row["rewards"], transition["rewards"])
    assert torch.equal(row["next_obs"], transition["next_obs"])
    assert torch.equal(row["dones"], transition["dones"])


def test_fold_rejects_invalid_n_step() -> None:
    for bad in (0, -1, 1.5, "5", True):
        with pytest.raises(ValueError, match="n_step"):
            NStepRewardFold(bad, 0.99)  # type: ignore[arg-type]


def test_stats_ring_publish_drain_wrap_and_drop() -> None:
    ring = RawRewardStatsRing(num_envs=2, reward_heads=2, depth=4)
    for t in range(3):
        rewards = torch.full((2, 2), float(t))
        assert ring.publish(rewards, torch.zeros(2))
    rewards, dones, read = ring.drain()
    assert read == 3
    assert rewards.shape == (3, 2, 2)
    assert torch.equal(rewards[:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    ring.commit_drain(read)

    # Wrap the ring fully once; order must survive the modulo wrap.
    for t in range(3, 7):
        assert ring.publish(torch.full((2, 2), float(t)), torch.ones(2))
    rewards, dones, read = ring.drain()
    assert torch.equal(rewards[:, 0, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(dones.sum(), torch.tensor(8.0))
    ring.commit_drain(read)

    # Overflow drops without blocking and counts the drop.
    for _ in range(4):
        assert ring.publish(torch.zeros(2, 2), torch.zeros(2))
    ok = ring.publish(torch.zeros(2, 2), torch.zeros(2))
    assert not ok
    assert ring.diagnostics()["dropped"] == 1
    ring.commit_drain(ring.drain()[2])


def test_stats_ring_validates_shapes() -> None:
    ring = RawRewardStatsRing(num_envs=2, reward_heads=2, depth=2)
    with pytest.raises(ValueError, match="reward_heads=1"):
        ring.publish(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="rewards shape"):
        ring.publish(torch.zeros(2, 3), torch.zeros(2))
    with pytest.raises(ValueError, match="dones shape"):
        ring.publish(torch.zeros(2, 2), torch.zeros(2, 1))


def test_drain_raw_reward_stats_feeds_normalizer_and_advances_cursor() -> None:
    from uni_rl.offpolicy.runner import drain_raw_reward_stats

    class _Learner:
        reward_normalizer = object()

        def __init__(self) -> None:
            self.calls = []

        def update_reward_stats(self, rewards, dones) -> None:
            self.calls.append((rewards.clone(), dones.clone()))

    ring = RawRewardStatsRing(num_envs=2, reward_heads=1, depth=4)
    ring.publish(torch.tensor([1.0, 2.0]), torch.zeros(2))
    ring.publish(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 1.0]))

    learner = _Learner()
    read = drain_raw_reward_stats(learner, ring, start_read=0)
    assert read == 2
    rewards, dones = learner.calls[0]
    assert rewards.shape == (2, 2, 1)
    assert torch.equal(rewards[:, :, 0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(dones[1], torch.tensor([0.0, 1.0]))

    # A second drain without new publications is a no-op.
    read = drain_raw_reward_stats(learner, ring, start_read=read)
    assert read == 2
    assert len(learner.calls) == 1

    # Without a normalizer the cursor still advances (rows are discarded).
    class _NoNorm:
        pass

    ring.publish(torch.ones(2), torch.zeros(2))
    read = drain_raw_reward_stats(_NoNorm(), ring, start_read=read)
    assert read == 3


def test_emit_collector_transition_folds_and_publishes_stats() -> None:
    from uni_rl.offpolicy.worker import _emit_collector_transition

    folder = NStepRewardFold(2, 0.5)
    ring = RawRewardStatsRing(num_envs=_M, reward_heads=1, depth=4)
    rows = []
    stop = threading.Event()
    transitions = [
        _step_tensors(
            t,
            heads=1,
            rewards=torch.full((_M,), float(t) + 1.0),
            dones=torch.zeros(_M),
            truncated=torch.zeros(_M),
            final_obs=None,
        )
        for t in range(2)
    ]
    for transition in transitions:
        _emit_collector_transition(
            transition,
            folder=folder,
            reward_stats_ring=ring,
            replay_add=lambda *args, **kwargs: rows.append((args, kwargs)) or True,
            stop_event=stop,
        )
    # The first step fills the window; the second emits the folded row.
    assert len(rows) == 1
    args, kwargs = rows[0]
    assert torch.equal(args[2], transitions[0]["rewards"] + 0.5 * transitions[1]["rewards"])
    assert torch.equal(args[0], transitions[0]["obs"])
    rewards, dones, read = ring.drain()
    assert read == 2
    assert torch.equal(rewards[:, 0, 0], torch.tensor([1.0, 2.0]))
