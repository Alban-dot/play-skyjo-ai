"""Integer action space shared by the RL env and all agents.

Each index always has the same meaning, whatever the phase:

    0..11   place the hand card at position a
    12..23  flip the hidden card at position a - 12
            (setup flip, or discard the drawn card then flip)
    24      draw from the deck
    25      take the top of the discard pile

Phase -> legal actions:
    SETUP_FLIP     12..23 (hidden cards)
    CHOOSE_SOURCE  24, 25 (if the discard is not empty)
    PLACE_DRAWN    0..11 (active cells), 12..23 (hidden cards)
    PLACE_DISCARD  0..11 (active cells)
"""

from __future__ import annotations

import numpy as np

from skyjo.game.constants import GRID_SIZE
from skyjo.game.engine import Phase, SkyjoGame

PLACE = 0
FLIP = GRID_SIZE  # 12
DRAW = 2 * GRID_SIZE  # 24
TAKE_DISCARD = DRAW + 1  # 25
N_ACTIONS = TAKE_DISCARD + 1  # 26


def action_mask(game: SkyjoGame) -> np.ndarray:
    """Boolean array of length N_ACTIONS, True for legal actions."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if game.phase == Phase.SETUP_FLIP:
        for pos in game.legal_setup_positions():
            mask[FLIP + pos] = True
    elif game.phase == Phase.CHOOSE_SOURCE:
        mask[DRAW] = True
        mask[TAKE_DISCARD] = game.can_take_discard()
    elif game.phase in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
        mask[game.legal_place_positions()] = True
        for pos in game.legal_flip_positions():
            mask[FLIP + pos] = True
    return mask


def apply_action(game: SkyjoGame, action: int) -> None:
    """Play ``action`` on the game; raise ValueError if it is illegal."""
    if not 0 <= action < N_ACTIONS or not action_mask(game)[action]:
        raise ValueError(f"Illegal action {action} in phase {game.phase.name}.")
    if action == DRAW:
        game.draw_from_deck()
    elif action == TAKE_DISCARD:
        game.take_discard()
    elif action >= FLIP:
        if game.phase == Phase.SETUP_FLIP:
            game.setup_flip(action - FLIP)
        else:
            game.discard_and_flip(action - FLIP)
    else:
        game.place(action)
