"""Pure rule functions (no turn flow; that lives in engine.py)."""

from __future__ import annotations

import numpy as np

from skyjo.game.constants import END_GAME_SCORE, N_COLS
from skyjo.game.deck import Deck
from skyjo.game.player_state import CardState, PlayerState


# --- columns -----------------------------------------------------------------

def completed_columns(player: PlayerState) -> list[int]:
    """Columns whose 3 cards are all visible and of the same value."""
    cols = []
    for col in range(N_COLS):
        positions = player.column_positions(col)
        if not all(player.states[p] == CardState.VISIBLE for p in positions):
            continue
        if len({int(player.values[p]) for p in positions}) == 1:
            cols.append(col)
    return cols


def remove_completed_columns(player: PlayerState, deck: Deck) -> list[int]:
    """Remove every completed column and put its cards on the discard pile.

    The caller must discard the replaced card *before* calling this, so that
    the top of the discard pile ends up being a card from the removed column.
    """
    cols = completed_columns(player)
    for col in cols:
        for card in player.remove_column(col):
            deck.discard(card)
    return cols


def finalize_grid(player: PlayerState, deck: Deck) -> list[int]:
    """End of round: reveal all hidden cards, then remove completed columns."""
    player.reveal_all()
    return remove_completed_columns(player, deck)


# --- start of round ----------------------------------------------------------

def first_player(
    players: list[PlayerState],
    total_scores: list[int],
    rng: np.random.Generator,
) -> int:
    """Index of the player who starts the round.

    Highest sum of visible cards (the 2 initial flips); ties broken by the
    highest cumulative total score; remaining ties broken at random.
    """
    keys = [(p.visible_score(), total) for p, total in zip(players, total_scores)]
    best = max(keys)
    candidates = [i for i, key in enumerate(keys) if key == best]
    return int(rng.choice(candidates))


# --- end of round ------------------------------------------------------------

def round_scores(players: list[PlayerState], finisher: int | None) -> list[int]:
    """Round score of each player, with the finisher's penalty applied.

    The finisher's score is doubled if it is positive and not strictly the
    lowest (a tie for the lowest score also triggers the penalty). Without a
    finisher (round ended by force), nobody is doubled.
    """
    scores = [p.score() for p in players]
    if finisher is None:
        return scores
    own = scores[finisher]
    others = [s for i, s in enumerate(scores) if i != finisher]
    if own > 0 and own >= min(others):
        scores[finisher] = own * 2
    return scores


# --- end of game -------------------------------------------------------------

def is_game_over(total_scores: list[int]) -> bool:
    return max(total_scores) >= END_GAME_SCORE


def winners(total_scores: list[int]) -> list[int]:
    """Indices of the players with the lowest total (several if tied)."""
    best = min(total_scores)
    return [i for i, s in enumerate(total_scores) if s == best]
