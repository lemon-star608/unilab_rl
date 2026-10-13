"""Collector-side n-step return folding (FlashDexRetarget semantics).

One env-vector transition is pushed per collector step; once the window holds
``n_step`` transitions, the oldest one is emitted folded:

- ``rewards``: ``R_t = sum_i gamma**i * r_{t+i} * prod_{j<i} (1 - done_{t+j})``
  — the discounted chain stops after the first done (terminated or
  truncated) in the span; the done step's own reward is included. No rows are
  dropped and no deque clearing happens on episode boundaries (matching the
  original ``_get_n_step_prev_transition``).
- ``dones`` / ``truncated`` / ``next_obs`` / ``next_critic`` / terminal
  fields: taken from the first done transition in the span (the span end), or
  from the newest transition when the span is done-free.
  ``dones`` is the combined episode flag (terminated | truncated); the
  collector-encoded ``truncated`` is bootstrap-eligible truncation only
  (truncated & ~terminated), so a folded failure+timeout span end also keeps
  ``truncated == 0``.
- ``obs`` / ``actions`` / ``critic`` / ``aux``: passed through from the
  oldest transition (the row's own step).

  Width-4 replay aux ``(motion_id, motion_step, next_motion_id,
  next_motion_step)`` is the one exception: its first two columns describe the
  row's own ``obs`` (oldest), while the next-context columns must follow the
  span end exactly like ``next_obs``. Passing the oldest next-context through
  unchanged would rebuild ``next_obs`` from a frame that is ``n_step - 1``
  steps stale for every folded row.

Multi-head rewards ``(m, H)`` fold head-wise with the episode mask broadcast
over heads. The emitted terminal fields ride with the span end, so the
buffer's ``_patch_terminal_next_observations`` keeps storing the true final
observation for timeout rows exactly as in the 1-step path.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch


class NStepRewardFold:
    """Fold one n-step window of env-vector transitions per collector step."""

    _SPAN_END_FIELDS = ("dones", "truncated", "next_obs", "next_critic", "terminal_mask")
    _AUX_WIDTH = 4

    def __init__(self, n_step: int, gamma: float) -> None:
        if isinstance(n_step, bool) or not isinstance(n_step, int) or n_step < 1:
            raise ValueError(f"n_step must be a positive integer, got {n_step!r}")
        self.n_step = n_step
        self.gamma = float(gamma)
        self._transitions: deque[dict[str, Any]] = deque(maxlen=n_step)

    def push(self, transition: dict[str, Any]) -> dict[str, Any] | None:
        """Append one env-step transition; emit the folded oldest row when full.

        The transition dict carries tensors keyed ``obs``, ``actions``,
        ``rewards``, ``next_obs``, ``dones``, ``truncated``, ``terminal_mask``,
        ``terminal_next_obs``, ``critic``, ``next_critic``,
        ``terminal_next_critic``, ``aux`` (optional fields may be None). Every
        tensor is cloned on push: env implementations may produce observations
        in reused buffers, and the window holds fields for ``n_step - 1``
        steps before they are emitted.
        """
        self._transitions.append(
            {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in transition.items()
            }
        )
        if len(self._transitions) < self.n_step:
            return None
        return self._fold_oldest()

    def _fold_oldest(self) -> dict[str, Any]:
        newest = self._transitions[-1]
        rewards = newest["rewards"].clone()
        span_end: dict[str, Any] = {
            name: newest[name].clone() if newest[name] is not None else None
            for name in self._SPAN_END_FIELDS
        }
        # Clone: the masked writes below must not edit the window's own
        # entry, which later spans read as an older transition.
        terminal_next_obs = (
            newest["terminal_next_obs"].clone() if newest["terminal_next_obs"] is not None else None
        )
        terminal_next_critic = (
            newest["terminal_next_critic"].clone()
            if newest["terminal_next_critic"] is not None
            else None
        )
        next_aux = self._span_end_next_aux(newest)

        for index in reversed(range(self.n_step - 1)):
            transition = self._transitions[index]
            done = transition["dones"]
            if rewards.ndim == 2:
                chain_mask = (1.0 - done).unsqueeze(-1)
            else:
                chain_mask = 1.0 - done
            rewards = transition["rewards"] + self.gamma * rewards * chain_mask

            mask = done > 0.5
            if not bool(mask.any()):
                continue
            for name in self._SPAN_END_FIELDS:
                if span_end[name] is not None:
                    span_end[name][mask] = transition[name][mask]
            # The span end's own terminal observation rides with its mask;
            # rows outside the mask are never read by the terminal patch.
            if transition["terminal_next_obs"] is not None:
                if terminal_next_obs is None:
                    terminal_next_obs = transition["terminal_next_obs"].clone()
                else:
                    terminal_next_obs[mask] = transition["terminal_next_obs"][mask]
            if transition["terminal_next_critic"] is not None:
                if terminal_next_critic is None:
                    terminal_next_critic = transition["terminal_next_critic"].clone()
                else:
                    terminal_next_critic[mask] = transition["terminal_next_critic"][mask]
            if next_aux is not None:
                next_aux[mask] = transition["aux"][mask, 2:4]

        oldest = self._transitions[0]
        aux = oldest["aux"]
        if next_aux is not None:
            aux = torch.cat((oldest["aux"][:, 0:2], next_aux), dim=1)
        return {
            "obs": oldest["obs"],
            "actions": oldest["actions"],
            "rewards": rewards,
            "next_obs": span_end["next_obs"],
            "dones": span_end["dones"],
            "truncated": span_end["truncated"],
            "terminal_mask": span_end["terminal_mask"],
            "terminal_next_obs": terminal_next_obs,
            "critic": oldest["critic"],
            "next_critic": span_end["next_critic"],
            "terminal_next_critic": terminal_next_critic,
            "aux": aux,
        }

    @classmethod
    def _span_end_next_aux(cls, newest: dict[str, Any]) -> torch.Tensor | None:
        """Next-context aux columns for the span end; ``None`` for other widths.

        Only the exact width-4 layout is folded. Width 2 keeps the
        ``step + n_step`` fallback in the rebuilder, and other widths have no
        next-context contract, so they pass through unchanged.
        """
        aux = newest["aux"]
        if not isinstance(aux, torch.Tensor) or aux.ndim != 2 or aux.shape[1] != cls._AUX_WIDTH:
            return None
        return aux[:, 2:4].clone()
