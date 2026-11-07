"""Shared test helpers."""

from collections import Counter

import numpy as np

from skyjo.agents.base import Agent
from skyjo.agents.heuristic_agent import HeuristicAgent
from skyjo.env.actions import TAKE_DISCARD
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState, PlayerState


def all_cards(game: SkyjoGame) -> Counter:
    """Every card currently in the game (piles, grids, hand)."""
    cards = list(game.deck.draw_pile) + list(game.deck.discard_pile)
    for p in game.players:
        cards += [int(p.values[pos]) for pos in p.active_positions()]
    if game.hand is not None:
        cards.append(game.hand)
    return Counter(cards)


def rig(game, grids, visible, draw=None, discard=None, current=0):
    """Replace the game state with hand-made grids, ready for a normal turn."""
    game.players = [PlayerState([v for row in rows for v in row]) for rows in grids]
    for player, positions in zip(game.players, visible):
        for pos in positions:
            player.flip(pos)
    game.phase = Phase.CHOOSE_SOURCE
    game.current_player = current
    if draw is not None:
        game.deck.draw_pile = list(draw)  # last element = top
    if discard is not None:
        game.deck.discard_pile = list(discard)


class StallingAgent(Agent):
    """Never ends a round: always swaps the discard onto a visible card."""

    def __init__(self, rng):
        self.inner = HeuristicAgent(rng)

    def act(self, view, mask):
        if view.phase == Phase.CHOOSE_SOURCE and mask[TAKE_DISCARD]:
            return TAKE_DISCARD
        if view.phase == Phase.PLACE_DISCARD:
            return int(np.flatnonzero(view.my_states == CardState.VISIBLE)[0])
        return self.inner.act(view, mask)
