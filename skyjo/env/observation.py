"""Fixed-size observation vector built from a PlayerView (no hidden information).

Everything is seen from the acting player: "me" first, then the opponents in
the order they play after me. Opponent slots up to MAX_PLAYERS - 1 are padded
with zeros. Layout (see LAYOUT for exact slices):

    phase          one-hot over the 4 playing phases                     4
    hand           one-hot card value + "has card" flag                 16
    top_discard    one-hot card value + "not empty" flag                16
    my_grid        12 x (state one-hot 3 + value one-hot 15)           216
    opponents      7 x (present flag + 12 x (state one-hot 3 + value/12))  343
    unknown_probs  probability that an unknown card has each value      15
    total_scores   me + 7 opponent slots, / 100                          8
    visible_sums   me + 7 opponent slots, / 50                           8
    last_turn      1 if someone already revealed all cards               1
    draw_pile      draw pile size / 150                                  1
"""

from __future__ import annotations

import numpy as np

from skyjo.game.constants import (
    CARD_DISTRIBUTION,
    DECK_SIZE,
    END_GAME_SCORE,
    GRID_SIZE,
    MAX_CARD_VALUE,
    MAX_PLAYERS,
    MIN_CARD_VALUE,
)
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState
from skyjo.game.view import PlayerView

N_VALUES = MAX_CARD_VALUE - MIN_CARD_VALUE + 1  # 15
N_STATES = len(CardState)  # 3
PLAYING_PHASES = (Phase.SETUP_FLIP, Phase.CHOOSE_SOURCE, Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)
OPP_SLOT = 1 + GRID_SIZE * (N_STATES + 1)  # 49

_SIZES = {
    "phase": len(PLAYING_PHASES),
    "hand": N_VALUES + 1,
    "top_discard": N_VALUES + 1,
    "my_grid": GRID_SIZE * (N_STATES + N_VALUES),
    "opponents": (MAX_PLAYERS - 1) * OPP_SLOT,
    "unknown_probs": N_VALUES,
    "total_scores": MAX_PLAYERS,
    "visible_sums": MAX_PLAYERS,
    "last_turn": 1,
    "draw_pile": 1,
}
LAYOUT: dict[str, slice] = {}
_start = 0
for _name, _size in _SIZES.items():
    LAYOUT[_name] = slice(_start, _start + _size)
    _start += _size
OBS_SIZE = _start  # 628

VISIBLE_SUM_SCALE = 50.0
_INITIAL_COUNTS = np.array(
    [CARD_DISTRIBUTION[v] for v in range(MIN_CARD_VALUE, MAX_CARD_VALUE + 1)], dtype=float
)


def _value_index(value: int) -> int:
    return value - MIN_CARD_VALUE


def _card_one_hot(card: int | None) -> np.ndarray:
    """One-hot over card values + presence flag (all zeros if no card)."""
    out = np.zeros(N_VALUES + 1, dtype=np.float32)
    if card is not None:
        out[_value_index(card)] = 1.0
        out[N_VALUES] = 1.0
    return out


def seat_order(view: PlayerView) -> list[int]:
    """Absolute player indices: me first, then opponents in playing order."""
    return [(view.player + k) % view.n_players for k in range(view.n_players)]


def unknown_card_probs(view: PlayerView) -> np.ndarray:
    """Probability that a card whose location I don't know has each value.

    Known cards: visible cards on every grid, the whole discard pile (buried
    cards and removed columns included) and the card in my hand. Everything
    is recomputed from the current state, so it restarts each round and
    after a reshuffle of the discard pile into the draw pile.
    """
    counts = _INITIAL_COUNTS.copy()
    visible = view.values[view.states == CardState.VISIBLE]
    known = list(visible.tolist()) + list(view.discard_pile)
    if view.hand is not None:
        known.append(view.hand)
    for card in known:
        counts[_value_index(int(card))] -= 1
    total = counts.sum()
    return (counts / total if total > 0 else counts).astype(np.float32)


def encode(view: PlayerView) -> np.ndarray:
    """Observation vector of length OBS_SIZE (float32)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    if view.phase in PLAYING_PHASES:
        obs[LAYOUT["phase"].start + PLAYING_PHASES.index(view.phase)] = 1.0
    obs[LAYOUT["hand"]] = _card_one_hot(view.hand)
    obs[LAYOUT["top_discard"]] = _card_one_hot(view.top_discard)

    # My grid: state one-hot + value one-hot (value only if visible).
    my_grid = np.zeros((GRID_SIZE, N_STATES + N_VALUES), dtype=np.float32)
    for pos in range(GRID_SIZE):
        state = int(view.my_states[pos])
        my_grid[pos, state] = 1.0
        if state == CardState.VISIBLE:
            my_grid[pos, N_STATES + _value_index(int(view.my_values[pos]))] = 1.0
    obs[LAYOUT["my_grid"]] = my_grid.ravel()

    # Opponents: present flag + state one-hot + scaled value, padded to 7 slots.
    seats = seat_order(view)
    opp_start = LAYOUT["opponents"].start
    for slot, player in enumerate(seats[1:]):
        block = np.zeros((GRID_SIZE, N_STATES + 1), dtype=np.float32)
        states = view.states[player]
        block[np.arange(GRID_SIZE), states] = 1.0
        block[:, N_STATES] = view.values[player] / MAX_CARD_VALUE
        start = opp_start + slot * OPP_SLOT
        obs[start] = 1.0
        obs[start + 1:start + OPP_SLOT] = block.ravel()

    obs[LAYOUT["unknown_probs"]] = unknown_card_probs(view)

    for slot, player in enumerate(seats):
        obs[LAYOUT["total_scores"].start + slot] = view.total_scores[player] / END_GAME_SCORE
        obs[LAYOUT["visible_sums"].start + slot] = view.values[player].sum() / VISIBLE_SUM_SCALE

    obs[LAYOUT["last_turn"]] = float(view.finisher is not None)
    obs[LAYOUT["draw_pile"]] = view.draw_pile_size / DECK_SIZE
    return obs
