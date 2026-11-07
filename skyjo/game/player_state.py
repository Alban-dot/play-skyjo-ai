"""State of one player's 3x4 grid."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from skyjo.game.constants import GRID_SIZE, N_COLS, N_ROWS


class CardState(IntEnum):
    HIDDEN = 0
    VISIBLE = 1
    REMOVED = 2


class PlayerState:
    """One player's grid.

    Positions are flat indices 0..11 (``pos = row * N_COLS + col``), so the
    grid always has 12 slots: removed columns are marked REMOVED, not deleted.
    ``values`` holds the true card values, including hidden ones; hiding them
    from the agent is the job of the observation code.
    """

    def __init__(self, cards: list[int]) -> None:
        if len(cards) != GRID_SIZE:
            raise ValueError(f"Expected {GRID_SIZE} cards, got {len(cards)}.")
        self.values = np.array(cards, dtype=np.int8)
        self.states = np.full(GRID_SIZE, CardState.HIDDEN, dtype=np.int8)

    # --- actions on the grid ---------------------------------------------

    def flip(self, pos: int) -> int:
        """Reveal a hidden card and return its value."""
        if self.states[pos] != CardState.HIDDEN:
            raise ValueError(f"Position {pos} is not hidden.")
        self.states[pos] = CardState.VISIBLE
        return int(self.values[pos])

    def replace(self, pos: int, card: int) -> int:
        """Put ``card`` face up at ``pos`` and return the replaced card."""
        if self.states[pos] == CardState.REMOVED:
            raise ValueError(f"Position {pos} has been removed.")
        old = int(self.values[pos])
        self.values[pos] = card
        self.states[pos] = CardState.VISIBLE
        return old

    def remove_column(self, col: int) -> list[int]:
        """Mark a column as removed and return its card values (top to bottom)."""
        positions = self.column_positions(col)
        if any(self.states[p] == CardState.REMOVED for p in positions):
            raise ValueError(f"Column {col} is already removed.")
        removed = [int(self.values[p]) for p in positions]
        self.states[positions] = CardState.REMOVED
        self.values[positions] = 0
        return removed

    def reveal_all(self) -> None:
        """Reveal every hidden card (end of round)."""
        self.states[self.states == CardState.HIDDEN] = CardState.VISIBLE

    # --- queries -----------------------------------------------------------

    @staticmethod
    def column_positions(col: int) -> list[int]:
        return [row * N_COLS + col for row in range(N_ROWS)]

    def hidden_positions(self) -> list[int]:
        return np.flatnonzero(self.states == CardState.HIDDEN).tolist()

    def visible_positions(self) -> list[int]:
        return np.flatnonzero(self.states == CardState.VISIBLE).tolist()

    def active_positions(self) -> list[int]:
        """Positions still in play (hidden or visible)."""
        return np.flatnonzero(self.states != CardState.REMOVED).tolist()

    def all_visible(self) -> bool:
        """True when no hidden card remains (triggers the end of the round)."""
        return not np.any(self.states == CardState.HIDDEN)

    def visible_score(self) -> int:
        """Sum of the visible cards only (what other players can see)."""
        return int(self.values[self.states == CardState.VISIBLE].sum())

    def score(self) -> int:
        """Sum of all cards still in play, hidden included (end-of-round score)."""
        return int(self.values[self.states != CardState.REMOVED].sum())

    def __repr__(self) -> str:
        cells = []
        for pos in range(GRID_SIZE):
            state = self.states[pos]
            cells.append("  ." if state == CardState.REMOVED
                         else "  ?" if state == CardState.HIDDEN
                         else f"{self.values[pos]:3d}")
        rows = [" ".join(cells[r * N_COLS:(r + 1) * N_COLS]) for r in range(N_ROWS)]
        return "\n".join(rows)
