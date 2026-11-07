"""Agent playing uniformly at random among legal actions."""

from __future__ import annotations

import numpy as np

from skyjo.agents.base import Agent
from skyjo.game.view import PlayerView


class RandomAgent(Agent):
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(action_mask)))
