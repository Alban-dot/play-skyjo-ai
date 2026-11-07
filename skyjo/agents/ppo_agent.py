"""Agent wrapping a trained MaskablePPO model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skyjo.agents.base import Agent
from skyjo.env.actions import DRAW, FLIP, PLACE
from skyjo.env.observation import encode
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState
from skyjo.game.view import PlayerView

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO


class PPOAgent(Agent):
    """``model``: a MaskablePPO model, or anything with the same ``predict`` (e.g.
    skyjo.agents.numpy_policy.NumpyPolicy). Not deterministic: actions drawn by
    the model's own sampling (torch for MaskablePPO), or with ``numpy_rng`` from
    the policy's probabilities."""

    name = "ppo"

    def __init__(self, model: MaskablePPO, deterministic: bool = True,
                 numpy_rng: np.random.Generator | None = None) -> None:
        self.model = model
        self.deterministic = deterministic
        self.numpy_rng = numpy_rng

    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> PPOAgent:
        from sb3_contrib import MaskablePPO  # torch only needed to load a model

        return cls(MaskablePPO.load(str(path), device="cpu"), deterministic)

    def probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Probabilities of the actions under the policy (masked actions: 0)."""
        if hasattr(self.model, "probabilities"):
            return self.model.probabilities(obs, action_mask)
        import torch as th

        with th.no_grad():
            dist = self.model.policy.get_distribution(th.as_tensor(obs[None], dtype=th.float32),
                                                      action_masks=action_mask[None])
            return dist.distribution.probs[0].numpy()

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        obs = encode(view)
        if not self.deterministic and self.numpy_rng is not None:
            p = self.probabilities(obs, action_mask).astype(np.float64)
            return int(self.numpy_rng.choice(len(p), p=p / p.sum()))
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=self.deterministic)
        return int(action)


class LoopGuardPPOAgent(PPOAgent):
    """PPO agent that breaks the loops of deterministic play (two agents passing
    the same card back and forth through the discard, neither ending the round).

    At each of its turns, the public state (grids, discard pile, draw pile size,
    scores) is recorded. When the same state comes back ``repeats`` times in the
    round, the agent draws from the deck instead of choosing: the draw pile
    shrinks, so that exact state cannot come back. From the ``finish_after``-th
    such intervention in the round, once it has drawn with a single hidden card
    left, it must end the round: the network chooses between placing the card on
    that hidden card and discarding it to reveal that card. Otherwise it plays as
    the plain agent.
    """

    name = "ppo-guard"

    def __init__(self, model: MaskablePPO, deterministic: bool = True,
                 repeats: int = 3, finish_after: int = 2, numpy_rng: np.random.Generator | None = None) -> None:
        super().__init__(model, deterministic)
        self.numpy_rng = numpy_rng
        self.repeats = repeats
        self.finish_after = finish_after
        self.forced_draws = 0  # cumulative, for reports
        self.forced_finishes = 0
        self.reset()

    def reset(self) -> None:
        self._round = None
        self._seen: Counter = Counter()
        self._interventions = 0

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        if view.round_number != self._round:
            self._round = view.round_number
            self._seen.clear()
            self._interventions = 0
        if view.phase == Phase.CHOOSE_SOURCE:
            key = (view.values.tobytes(), view.states.tobytes(), view.discard_pile,
                   view.draw_pile_size, view.total_scores)
            self._seen[key] += 1
            if self._seen[key] >= self.repeats:
                self._interventions += 1
                self.forced_draws += 1
                return DRAW
        elif view.phase == Phase.PLACE_DRAWN and self._interventions >= self.finish_after:
            hidden = np.flatnonzero(view.my_states == CardState.HIDDEN)
            if len(hidden) == 1:
                finish = np.zeros_like(action_mask)
                finish[[PLACE + hidden[0], FLIP + hidden[0]]] = True
                self.forced_finishes += 1
                finish &= action_mask
                # under the column rule, the mask may hold only a cell other than the hidden one
                return super().act(view, finish if finish.any() else action_mask)
        return super().act(view, action_mask)
