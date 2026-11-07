"""Cautious bot that takes risks on columns early in the round."""

from __future__ import annotations

import numpy as np

from skyjo.agents.heuristic_agent import COMPLETE, CautiousHeuristicAgent
from skyjo.env.actions import FLIP
from skyjo.game.constants import N_COLS
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState, PlayerState
from skyjo.game.view import PlayerView

BET = "bet"
PLACEMENT_PHASES = (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)


def bet_cells(view: PlayerView, card: int) -> list[int]:
    """Hidden cells of the columns whose only visible card equals ``card`` and
    whose two other cells are hidden."""
    cells = []
    for col in range(N_COLS):
        column = PlayerState.column_positions(col)
        visible = [p for p in column if view.my_states[p] == CardState.VISIBLE]
        hidden = [p for p in column if view.my_states[p] == CardState.HIDDEN]
        if len(visible) == 1 and len(hidden) == 2 and view.my_values[visible[0]] == card:
            cells += hidden
    return cells


def same_value_columns(view: PlayerView, card: int) -> frozenset[int]:
    """Cells of the columns holding a visible card equal to ``card``."""
    cells = set()
    for col in range(N_COLS):
        column = PlayerState.column_positions(col)
        if any(view.my_states[p] == CardState.VISIBLE and view.my_values[p] == card for p in column):
            cells.update(column)
    return frozenset(cells)


class ColumnSeekerAgent(CautiousHeuristicAgent):
    """CautiousHeuristicAgent (rules 1-5 unchanged) that seeks columns during
    its first ``seek_until_turn`` turns of each round.

    While seeking, for the card in hand (drawn, or the discard it considers):
      1. complete a column if the card is >= min_column_value;
      2. else bet: place it on a hidden cell of a column whose only visible
         card equals it (other two cells hidden), if
         min_column_value <= card <= max_bet_value;
      3. else play as the cautious bot; a card < min_column_value never goes
         into a column holding a visible card of its value (no completion, no
         pairing, no bet).
    The discard is taken when it allows 1 or 2, otherwise by the cautious rule.
    Rule 5 keeps priority: a refusal places the card on the highest visible
    card, whatever its column.
    From turn seek_until_turn + 1 on, it plays exactly as the cautious bot;
    rule 5 (prudence at the end of the round) applies in every case.

    ``bets`` records the bets played, as (round, column); the caller empties it.
    """

    name = "seeker"

    def __init__(self, rng: np.random.Generator, seek_until_turn: int = 10, min_column_value: int = 1,
                 max_bet_value: int = 6, **kwargs) -> None:
        self.seek_until_turn = seek_until_turn
        self.min_column_value = min_column_value
        self.max_bet_value = max_bet_value
        self.bets: list[tuple[int, int]] = []
        super().__init__(rng, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._seek_round = None
        self._turns = 0  # my turns in the current round, the current one included
        self._kind = None  # kind of the last placement computed

    @property
    def seeking(self) -> bool:
        return self._turns <= self.seek_until_turn

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        if view.round_number != self._seek_round:
            self._seek_round, self._turns = view.round_number, 0
        if view.phase == Phase.CHOOSE_SOURCE:
            self._turns += 1
        self._kind = None
        action = super().act(view, action_mask)
        if (view.phase in PLACEMENT_PHASES and self.seeking and self._kind == BET
                and not self.refused and action < FLIP):
            self.bets.append((view.round_number, action % N_COLS))
        return action

    def _worth_taking(self, kind: str, card: int) -> bool:
        return kind == BET or super()._worth_taking(kind, card)  # a bet only exists while seeking

    def _placement(self, view: PlayerView, card: int) -> tuple[str, int, list[int]] | None:
        if not self.seeking:
            placement = super()._placement(view, card)
        elif card < self.min_column_value:
            placement = self._grid_placement(view, card, same_value_columns(view, card))
        else:
            column = self._column_placement(view, card)
            bets = bet_cells(view, card) if card <= self.max_bet_value else []
            if column is not None and column[0] == COMPLETE:
                placement = column
            elif bets:
                placement = BET, int(self.rng.choice(bets)), bets
            else:
                placement = column or self._grid_placement(view, card)
        self._kind = None if placement is None else placement[0]
        return placement
