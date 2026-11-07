"""Common agent interface and a helper to play a full game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from skyjo.env.actions import action_mask, apply_action
from skyjo.game.engine import SkyjoGame
from skyjo.game.view import PlayerView, make_view


class GameBlocked(RuntimeError):
    """A game did not finish within the allowed number of actions."""


class Agent(ABC):
    """Anything that can choose a legal action from a player's view."""

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of each game."""

    def reseed(self, rng: np.random.Generator) -> None:
        """Replace the agent's random generator (paired evaluations); agents
        without randomness ignore it."""
        if hasattr(self, "rng"):
            self.rng = rng

    @abstractmethod
    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        """Return an action index for which ``action_mask`` is True."""


def play_game(
    agents: Sequence[Agent],
    rng: np.random.Generator,
    on_step: Callable[[SkyjoGame], None] | None = None,
    max_steps: int = 100_000,
    round_rng: Callable[[int], np.random.Generator] | None = None,
) -> SkyjoGame:
    """Play one full game, agent i controlling player i. Returns the finished game.
    ``round_rng``: see SkyjoGame."""
    game = SkyjoGame(len(agents), rng, round_rng)
    for agent in agents:
        agent.reset()
    for _ in range(max_steps):
        if game.is_over:
            return game
        player = game.current_player
        action = agents[player].act(make_view(game, player), action_mask(game))
        apply_action(game, action)
        if on_step is not None:
            on_step(game)
    if not game.is_over:
        raise GameBlocked(f"Game not finished after {max_steps} steps.")
    return game
