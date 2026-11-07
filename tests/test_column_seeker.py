import copy

import numpy as np
import pytest

from skyjo.agents.base import Agent
from skyjo.agents.column_rule import opportunities
from skyjo.agents.column_seeker import BET, ColumnSeekerAgent, bet_cells
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent
from skyjo.agents.registry import make_variant
from skyjo.env.actions import DRAW, FLIP, TAKE_DISCARD, action_mask
from skyjo.game.constants import N_COLS
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState, PlayerState
from skyjo.game.view import make_view
from skyjo.simulation import simulate
from tests.helpers import rig

# Column 0 = positions 0, 4, 8; column 1 = 1, 5, 9; etc.
OTHER = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
PLACEMENT = (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)


def setup(grid, visible, discard):
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [grid, OTHER], [visible, []], draw=[11, 11, 11], discard=discard)
    return game


def act(game, agent):
    return agent.act(make_view(game, game.current_player), action_mask(game))


class Watcher(Agent):
    """Plays as a ColumnSeekerAgent and checks each of its decisions with ``check``."""

    def __init__(self, seeker, check):
        self.seeker, self.check = seeker, check
        self.checked = 0

    def reset(self):
        self.seeker.reset()

    def reseed(self, rng):
        self.seeker.reseed(rng)

    def act(self, view, mask):
        before = (copy.deepcopy(self.seeker.rng), self.seeker._round, self.seeker._refusals)
        action = self.seeker.act(view, mask)
        self.checked += self.check(self.seeker, view, mask, action, before)
        return action


def play(watcher, n_players=3, n_games=40):
    bots = [CautiousHeuristicAgent(np.random.default_rng(i + 1)) for i in range(n_players - 1)]
    simulate([watcher, *bots], n_games, np.random.default_rng(0), deal_seed=7, max_steps=5000)
    return watcher.checked


# --- min_column_value -----------------------------------------------------------

def test_never_completes_a_column_below_min_column_value():
    grid = [[0, 12, 2, 3], [0, 2, 3, 4], [5, 3, 4, 6]]
    for cls, expected in ((CautiousHeuristicAgent, 8), (ColumnSeekerAgent, 1)):
        game = setup(grid, [0, 4, 1], discard=[0])
        agent = cls(np.random.default_rng(0))
        assert act(game, agent) == TAKE_DISCARD
        game.take_discard()
        # the cautious bot completes the column of 0s; the seeker replaces the 12 instead
        assert act(game, agent) == expected


@pytest.mark.parametrize("min_value", [1, 3])
def test_never_completes_nor_prepares_a_column_below_min_column_value_while_seeking(min_value):
    def check(seeker, view, mask, action, before):
        if (view.phase not in PLACEMENT or not seeker.seeking or view.hand >= min_value or action >= FLIP
                or seeker.refused):  # rule 5 keeps priority
            return 0
        safe, _ = opportunities(view)
        column = PlayerState.column_positions(action % N_COLS)
        equal = [p for p in column if p != action and view.my_states[p] == CardState.VISIBLE
                 and view.my_values[p] == view.hand]
        assert action not in safe and not equal  # no completion, no pairing, no bet
        return 1

    seeker = ColumnSeekerAgent(np.random.default_rng(0), seek_until_turn=30, min_column_value=min_value)
    assert play(Watcher(seeker, check)) > 50


# --- max_bet_value --------------------------------------------------------------

@pytest.mark.parametrize("card, bets", [(6, True), (7, False)])
def test_bets_up_to_max_bet_value(card, bets):
    grid = [[card, 1, 2, 3], [9, 2, 3, 4], [9, 3, 4, 6]]
    game = setup(grid, [0, 1], discard=[card])
    seeker = ColumnSeekerAgent(np.random.default_rng(0), max_bet_value=6)
    assert act(game, CautiousHeuristicAgent(np.random.default_rng(0))) == DRAW
    if not bets:
        assert act(game, seeker) == DRAW
        return
    assert act(game, seeker) == TAKE_DISCARD
    game.take_discard()
    assert act(game, seeker) in (4, 8)  # a hidden cell of the column of the visible 6
    assert seeker.bets == [(game.round_number, 0)]


def test_bets_respect_max_bet_value_in_play():
    def check(seeker, view, mask, action, before):
        if view.phase in PLACEMENT and seeker._kind == BET:
            assert 1 <= view.hand <= 4 and action in bet_cells(view, view.hand)
            return 1
        return 0

    seeker = ColumnSeekerAgent(np.random.default_rng(0), max_bet_value=4)
    assert play(Watcher(seeker, check)) > 20


# --- switch at seek_until_turn ------------------------------------------------------

def test_switches_to_cautious_after_seek_until_turn():
    grid = [[6, 1, 2, 3], [9, 2, 3, 4], [9, 3, 4, 6]]
    game = setup(grid, [0, 1], discard=[6])
    seeker = ColumnSeekerAgent(np.random.default_rng(0), seek_until_turn=2)
    assert [act(game, seeker) for _ in range(3)] == [TAKE_DISCARD, TAKE_DISCARD, DRAW]
    game.round_number += 1  # a new round counts its turns from 1 again
    assert act(game, seeker) == TAKE_DISCARD


def test_plays_exactly_as_cautious_after_the_switch():
    shadow = CautiousHeuristicAgent(np.random.default_rng(0))

    def check(seeker, view, mask, action, before):
        if seeker.seeking:
            return 0
        shadow.rng, shadow._round, shadow._refusals = before
        assert shadow.act(view, mask) == action
        return 1

    seeker = ColumnSeekerAgent(np.random.default_rng(0), seek_until_turn=4)
    assert play(Watcher(seeker, check)) > 500


# --- bets disabled -----------------------------------------------------------------

@pytest.mark.parametrize("n_players", [2, 4])
def test_plays_exactly_as_cautious_when_bets_are_disabled(n_players):
    def results(subject):
        bots = [CautiousHeuristicAgent(np.random.default_rng(i + 1)) for i in range(n_players - 1)]
        r = simulate([subject, *bots], 40, np.random.default_rng(0), deal_seed=3, max_steps=5000)
        return r.scores, r.rounds, r.finished_rounds

    disabled = ColumnSeekerAgent(np.random.default_rng(0), min_column_value=-2, max_bet_value=-3)
    assert results(disabled) == results(CautiousHeuristicAgent(np.random.default_rng(0)))
    assert disabled.bets == []


def test_registry_builds_a_seeker_with_parameters():
    bot = make_variant("seeker:seek_until_turn=6,max_bet_value=8", np.random.default_rng(0))
    assert isinstance(bot, ColumnSeekerAgent)
    assert (bot.seek_until_turn, bot.min_column_value, bot.max_bet_value) == (6, 1, 8)
    default = make_variant("seeker", np.random.default_rng(0))
    assert (default.seek_until_turn, default.min_column_value, default.max_bet_value) == (10, 1, 6)
