import numpy as np
import pytest

from skyjo.agents.heuristic_agent import (
    CautiousHeuristicAgent,
    HeuristicAgent,
    estimated_score,
)
from skyjo.env.actions import FLIP, DRAW, TAKE_DISCARD, action_mask
from skyjo.game.constants import MEAN_CARD_VALUE
from skyjo.game.engine import SkyjoGame
from skyjo.game.player_state import CardState, PlayerState
from skyjo.game.view import make_view
from tests.helpers import rig

# Column 0 = positions 0, 4, 8; column 1 = 1, 5, 9; etc.
GRID = [[10, 1, 2, 3], [7, 2, 3, 4], [9, 3, 4, 6]]
OTHER = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]


def setup(visible, draw=None, discard=None, grid=GRID, **kwargs):
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [grid, OTHER], [visible, []], draw=draw, discard=discard)
    agent = HeuristicAgent(np.random.default_rng(0), **kwargs)
    return game, agent


def act(game, agent):
    return agent.act(make_view(game, game.current_player), action_mask(game))


# --- source choice (rule 1) --------------------------------------------------

def test_takes_low_discard():
    game, agent = setup([0], discard=[3])
    assert act(game, agent) == TAKE_DISCARD


def test_draws_when_discard_high_and_useless():
    game, agent = setup([1, 5], discard=[9])
    assert act(game, agent) == DRAW


def test_discard_threshold_is_a_parameter():
    # A 5 can go on a hidden card (hidden_threshold=5); only the threshold decides.
    game, agent = setup([1, 5], discard=[5], hidden_threshold=5)
    assert act(game, agent) == DRAW
    game, agent = setup([1, 5], discard=[5], hidden_threshold=5, discard_threshold=5)
    assert act(game, agent) == TAKE_DISCARD


def test_takes_discard_to_complete_column_even_if_high():
    grid = [[9, 1, 2, 3], [9, 2, 3, 4], [12, 3, 4, 6]]
    game, agent = setup([0, 4, 8], discard=[9], grid=grid)
    assert act(game, agent) == TAKE_DISCARD
    game.take_discard()
    assert act(game, agent) == 8  # replaces the 12 -> column completed


def test_takes_discard_to_pair_when_replacing_higher_card():
    # Column 0 = visible 10, 7, 9. Discard 7 pairs with the 7, replacing the 10.
    game, agent = setup([0, 4, 8], discard=[7])
    assert act(game, agent) == TAKE_DISCARD
    game.take_discard()
    assert act(game, agent) == 0


def test_no_pair_with_high_card_on_hidden_cell():
    # A visible 12 with two hidden cells: pairing a 12 would cost ~7 points.
    grid = [[12, 1, 2, 3], [5, 2, 3, 4], [5, 3, 4, 6]]
    game, agent = setup([0, 1], discard=[12], grid=grid)
    assert act(game, agent) == DRAW


# --- placement (rules 2, 4) and discard-and-flip (rule 3) --------------------

def test_replaces_highest_visible_card_with_enough_gain():
    game, agent = setup([0, 1], draw=[6])  # visible 10 and 1
    game.draw_from_deck()
    assert act(game, agent) == 0


def test_min_gain_blocks_small_improvement():
    game, agent = setup([0, 1], draw=[9])  # 10 -> 9 is a gain of only 1
    game.draw_from_deck()
    action = act(game, agent)
    assert FLIP <= action < DRAW  # rule 3: discard and flip
    assert action - FLIP in game.players[0].hidden_positions()


def test_min_gain_is_a_parameter():
    game, agent = setup([0, 1], draw=[9], min_gain=1)
    game.draw_from_deck()
    assert act(game, agent) == 0


def test_low_card_goes_on_hidden_card():
    game, agent = setup([1, 5], draw=[4])  # visible 1 and 2: no visible gain
    game.draw_from_deck()
    action = act(game, agent)
    assert action in game.players[0].hidden_positions()


def test_hidden_threshold_is_a_parameter():
    game, agent = setup([1, 5], draw=[4], hidden_threshold=3)
    game.draw_from_deck()
    assert act(game, agent) >= FLIP


def test_completes_column_with_drawn_card():
    grid = [[1, 2, 3, 4], [8, 2, 3, 5], [8, 3, 4, 6]]
    game, agent = setup([4, 8], draw=[8], grid=grid)
    game.draw_from_deck()
    assert act(game, agent) == 0  # hidden third cell of column 0


def test_forced_placement_when_no_hidden_card():
    game, agent = setup(list(range(12)), draw=[11], grid=OTHER)
    game.draw_from_deck()
    assert act(game, agent) == 11  # the highest card (6) is replaced


# --- variant: take the discard for a replacement gain -------------------------

def test_variant_takes_discard_for_replacement_gain():
    # Discard 5, visible 10: gain 5 >= min_gain, but 5 > discard_threshold.
    game, agent = setup([0, 1], discard=[5])
    assert act(game, agent) == DRAW  # base agent unchanged
    game, agent = setup([0, 1], discard=[5], take_discard_for_gain=True)
    assert act(game, agent) == TAKE_DISCARD
    game.take_discard()
    assert act(game, agent) == 0


# --- cautious agent (rule 5) ----------------------------------------------

# My grid: everything visible except position 11 -> estimated 36 + hidden card.
MINE = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
LOW_OPP = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]  # ~5 + 5.07 with 1 hidden
HIGH_OPP = [[5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]]  # ~80 + 5.07 with 1 hidden


def cautious_setup(opp, draw, **kwargs):
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [MINE, opp], [list(range(11)), list(range(11))], draw=draw, discard=[0])
    game.draw_from_deck()
    return game, CautiousHeuristicAgent(np.random.default_rng(0), **kwargs)


def test_base_agent_would_finish_the_round():
    game, _ = cautious_setup(LOW_OPP, draw=[11])
    base = HeuristicAgent(np.random.default_rng(0))
    assert act(game, base) == FLIP + 11  # flips the last hidden card


def test_cautious_refuses_to_finish_when_not_lowest():
    game, agent = cautious_setup(LOW_OPP, draw=[11])
    assert act(game, agent) in (7, 10)  # replaces a visible 5 instead


def test_cautious_finishes_when_strictly_lowest():
    game, agent = cautious_setup(HIGH_OPP, draw=[11])
    assert act(game, agent) == FLIP + 11


def test_cautious_ignores_rule_when_round_already_ending():
    game, agent = cautious_setup(LOW_OPP, draw=[11])
    game.finisher = 1
    assert act(game, agent) == FLIP + 11


def test_cautious_gives_up_after_max_refusals():
    game, agent = cautious_setup(LOW_OPP, draw=[11], max_refusals=0)
    assert act(game, agent) == FLIP + 11


def test_estimated_score_counts_hidden_and_complete_columns():
    p = PlayerState([v for row in [[7, 1, 2, 3], [7, 2, 3, 4], [7, 3, 4, 6]] for v in row])
    p.reveal_all()
    p.states[11] = CardState.HIDDEN
    expected = (1 + 2 + 3) + (2 + 3 + 4) + (3 + 4) + MEAN_CARD_VALUE  # column of 7s removed
    assert estimated_score(p.values, p.states) == pytest.approx(expected)
