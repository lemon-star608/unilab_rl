"""Bounded collector-to-learner raw reward/done channel for n-step folding.

With collector-side n-step folding, replay rows carry *folded* rewards, but
the learner-side ``RewardNormalizer`` must keep seeing *raw* per-step rewards
and episode flags (that is what the original implementation feeds its
normalizer). This tiny shared ring carries them: the collector publishes one
``(num_envs[, heads])`` reward/done pair per env step, and the learner drains
it in order once per training iteration.

The channel is single-writer single-reader over process-shared tensors. A
full ring must never stall collection, so overflow drops the oldest
unpublished step and counts the drop in ``dropped`` (statistics skew is
bounded and visible); the depth default covers long learner stalls.
"""

from __future__ import annotations

import torch

DEFAULT_STATS_RING_DEPTH = 1024


class RawRewardStatsRing:
    """Process-shared raw (reward, done) ring for learner reward statistics."""

    def __init__(
        self,
        num_envs: int,
        reward_heads: int = 1,
        depth: int = DEFAULT_STATS_RING_DEPTH,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if reward_heads <= 0:
            raise ValueError(f"reward_heads must be positive, got {reward_heads}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.num_envs = int(num_envs)
        self.reward_heads = int(reward_heads)
        self.depth = int(depth)
        self.rewards = torch.zeros(
            (self.depth, self.num_envs, self.reward_heads), dtype=torch.float32
        ).share_memory_()
        self.dones = torch.zeros((self.depth, self.num_envs), dtype=torch.float32).share_memory_()
        self.write_ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.read_ptr = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.dropped = torch.zeros(1, dtype=torch.int64).share_memory_()

    def publish(self, rewards: torch.Tensor, dones: torch.Tensor) -> bool:
        """Append one env step of raw rewards; False (counted) when full."""
        if rewards.ndim == 1:
            if self.reward_heads != 1:
                raise ValueError(f"flat rewards require reward_heads=1, got {self.reward_heads}")
            rewards = rewards.unsqueeze(1)
        expected = (self.num_envs, self.reward_heads)
        if tuple(rewards.shape) != expected:
            raise ValueError(f"rewards shape {tuple(rewards.shape)} != {expected}")
        if tuple(dones.shape) != (self.num_envs,):
            raise ValueError(f"dones shape {tuple(dones.shape)} != ({self.num_envs},)")
        write = int(self.write_ptr[0])
        if write - int(self.read_ptr[0]) >= self.depth:
            self.dropped[0] += 1
            return False
        index = write % self.depth
        self.rewards[index].copy_(rewards)
        self.dones[index].copy_(dones)
        # The data copy commits before the pointer advance; the reader only
        # reads rows below the pointer it observed.
        self.write_ptr[0] = write + 1
        return True

    def drain(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Snapshot (rewards, dones, new read ptr) of every published step."""
        read = int(self.read_ptr[0])
        write = int(self.write_ptr[0])
        count = write - read
        if count <= 0:
            return (
                self.rewards[:0].clone(),
                self.dones[:0].clone(),
                read,
            )
        indices = torch.arange(read, write) % self.depth
        rewards = self.rewards[indices].clone()
        dones = self.dones[indices].clone()
        return rewards, dones, write

    def commit_drain(self, new_read_ptr: int) -> None:
        self.read_ptr[0] = int(new_read_ptr)

    def diagnostics(self) -> dict[str, int]:
        return {
            "depth": self.depth,
            "occupancy": max(0, int(self.write_ptr[0]) - int(self.read_ptr[0])),
            "dropped": int(self.dropped[0]),
        }
