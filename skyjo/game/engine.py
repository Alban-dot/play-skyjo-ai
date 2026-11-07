"""Game engine: turn flow, rounds and game end, as a state machine."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import numpy as np

from skyjo.game.constants import GRID_SIZE, MAX_PLAYERS, MIN_PLAYERS, N_INITIAL_FLIPS
from skyjo.game.deck import Deck
from skyjo.game.player_state import PlayerState
from skyjo.game.rules import (
    finalize_grid,
    first_player,
    is_game_over,
    remove_completed_columns,
    round_scores,
    winners,
)


class Phase(IntEnum):
    SETUP_FLIP = 0      # start of round: each player flips 2 cards
    CHOOSE_SOURCE = 1   # draw from the deck or take the discard
    PLACE_DRAWN = 2     # drawn card: place it, or discard it and flip a card
    PLACE_DISCARD = 3   # card taken from the discard: must be placed
    GAME_OVER = 4


class SkyjoGame:
    """Full Skyjo game for ``n_players``.

    Each public action method checks that it is legal in the current phase
    and raises ``ValueError`` otherwise. Rounds chain automatically; the
    previous round's result is kept in ``last_round_scores`` and
    ``last_round_players``.
    """

    def __init__(
        self,
        n_players: int,
        rng: np.random.Generator,
        round_rng: Callable[[int], np.random.Generator] | None = None,
    ) -> None:
        """With ``round_rng``, each round draws its randomness (deal, reshuffles,
        first player) from ``round_rng(round_number)`` instead of ``rng``: the
        deal of round r no longer depends on how the previous rounds were played."""
        if not MIN_PLAYERS <= n_players <= MAX_PLAYERS:
            raise ValueError(f"n_players must be in [{MIN_PLAYERS}, {MAX_PLAYERS}].")
        self.n_players = n_players
        self.rng = rng
        self.round_rng = round_rng
        self.deck = Deck(rng)
        self.total_scores = [0] * n_players
        self.round_number = 0
        self.last_round_scores: list[int] | None = None
        self.last_round_players: list[PlayerState] | None = None
        self.last_round_finisher: int | None = None
        self._start_round()

    # --- round lifecycle -------------------------------------------------

    def _start_round(self) -> None:
        if self.round_rng is not None:
            self.rng = self.deck.rng = self.round_rng(self.round_number + 1)
        self.deck.reset()
        self.players = [
            PlayerState([self.deck.draw() for _ in range(GRID_SIZE)])
            for _ in range(self.n_players)
        ]
        self.deck.start_discard()
        self.round_number += 1
        self.phase = Phase.SETUP_FLIP
        self.current_player = 0
        self.hand: int | None = None  # card drawn or taken, not yet placed
        self.finisher: int | None = None  # player who revealed all cards first
        self.turns_left: int | None = None  # last turns left once finisher is set
        self.turns_played = 0  # turns completed in this round, all players
        self._setup_flips = 0

    def _end_turn(self) -> None:
        player = self.players[self.current_player]
        remove_completed_columns(player, self.deck)
        self.turns_played += 1

        if self.finisher is None:
            if player.all_visible():
                self.finisher = self.current_player
                self.turns_left = self.n_players - 1
        else:
            self.turns_left -= 1

        if self.turns_left == 0:
            self._end_round()
            return
        self.current_player = (self.current_player + 1) % self.n_players
        self.phase = Phase.CHOOSE_SOURCE

    def _end_round(self) -> None:
        for player in self.players:
            finalize_grid(player, self.deck)
        scores = round_scores(self.players, self.finisher)
        self.total_scores = [t + s for t, s in zip(self.total_scores, scores)]
        self.last_round_scores = scores
        self.last_round_players = self.players
        self.last_round_finisher = self.finisher  # None if ended by force

        if is_game_over(self.total_scores):
            self.phase = Phase.GAME_OVER
        else:
            self._start_round()

    def force_end_round(self) -> None:
        """End the round now, between two turns (not part of the rules: used
        by the env to stop rounds that never end). Every card is revealed and
        counted; nobody finished, so nobody is doubled."""
        self._require(Phase.CHOOSE_SOURCE)
        if self.finisher is not None:
            raise ValueError("The round is already ending normally.")
        self._end_round()

    # --- actions -----------------------------------------------------------

    def setup_flip(self, pos: int) -> int:
        """Flip one of the current player's cards during setup."""
        self._require(Phase.SETUP_FLIP)
        value = self.players[self.current_player].flip(pos)
        self._setup_flips += 1
        if self._setup_flips < N_INITIAL_FLIPS:
            return value

        self._setup_flips = 0
        if self.current_player < self.n_players - 1:
            self.current_player += 1
        else:
            self.current_player = first_player(self.players, self.total_scores, self.rng)
            self.phase = Phase.CHOOSE_SOURCE
        return value

    def draw_from_deck(self) -> int:
        """Draw the top card of the draw pile into the hand."""
        self._require(Phase.CHOOSE_SOURCE)
        self.hand = self.deck.draw()
        self.phase = Phase.PLACE_DRAWN
        return self.hand

    def take_discard(self) -> int:
        """Take the top card of the discard pile into the hand."""
        self._require(Phase.CHOOSE_SOURCE)
        if not self.can_take_discard():
            raise ValueError("Discard pile is empty.")
        self.hand = self.deck.take_discard()
        self.phase = Phase.PLACE_DISCARD
        return self.hand

    def place(self, pos: int) -> None:
        """Put the hand card at ``pos``; the replaced card is discarded."""
        self._require(Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)
        old = self.players[self.current_player].replace(pos, self.hand)
        self.hand = None
        self.deck.discard(old)
        self._end_turn()

    def discard_and_flip(self, pos: int) -> None:
        """Discard the drawn card, then flip the hidden card at ``pos``."""
        self._require(Phase.PLACE_DRAWN)
        if pos not in self.legal_flip_positions():
            raise ValueError(f"Position {pos} is not a hidden card.")
        self.deck.discard(self.hand)
        self.hand = None
        self.players[self.current_player].flip(pos)
        self._end_turn()

    # --- legal moves -------------------------------------------------------

    def legal_setup_positions(self) -> list[int]:
        if self.phase != Phase.SETUP_FLIP:
            return []
        return self.players[self.current_player].hidden_positions()

    def can_take_discard(self) -> bool:
        return self.phase == Phase.CHOOSE_SOURCE and self.deck.top_discard is not None

    def legal_place_positions(self) -> list[int]:
        if self.phase not in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
            return []
        return self.players[self.current_player].active_positions()

    def legal_flip_positions(self) -> list[int]:
        if self.phase != Phase.PLACE_DRAWN:
            return []
        return self.players[self.current_player].hidden_positions()

    # --- info --------------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def winners(self) -> list[int]:
        if not self.is_over:
            raise ValueError("Game is not over.")
        return winners(self.total_scores)

    def _require(self, *phases: Phase) -> None:
        if self.phase not in phases:
            raise ValueError(f"Action not allowed in phase {self.phase.name}.")
