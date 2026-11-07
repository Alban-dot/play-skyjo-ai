"""What one player is allowed to see of the game (no hidden information)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState


@dataclass(frozen=True)
class PlayerView:
    """Public information from the point of view of ``player``.

    ``values`` and ``states`` have shape (n_players, 12) and use absolute
    player indices. Hidden and removed cards have value 0 in ``values``.
    """

    player: int
    n_players: int
    current_player: int
    phase: Phase
    values: np.ndarray
    states: np.ndarray
    hand: int | None  # card in hand, only if ``player`` holds it
    top_discard: int | None
    discard_pile: tuple[int, ...]  # public: every card since the last reshuffle
    draw_pile_size: int
    total_scores: tuple[int, ...]
    round_number: int
    finisher: int | None

    @property
    def my_values(self) -> np.ndarray:
        return self.values[self.player]

    @property
    def my_states(self) -> np.ndarray:
        return self.states[self.player]


def make_view(game: SkyjoGame, player: int) -> PlayerView:
    states = np.stack([p.states for p in game.players]).copy()
    values = np.stack([p.values for p in game.players]).astype(np.int8)
    values[states != CardState.VISIBLE] = 0
    hand = game.hand if game.current_player == player else None
    return PlayerView(
        player=player,
        n_players=game.n_players,
        current_player=game.current_player,
        phase=game.phase,
        values=values,
        states=states,
        hand=hand,
        top_discard=game.deck.top_discard,
        discard_pile=tuple(game.deck.discard_pile),
        draw_pile_size=len(game.deck.draw_pile),
        total_scores=tuple(game.total_scores),
        round_number=game.round_number,
        finisher=game.finisher,
    )
