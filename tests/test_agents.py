from collections import Counter

import numpy as np
import pytest

from skyjo.agents.base import play_game
from skyjo.agents.random_agent import RandomAgent
from skyjo.env.actions import (
    FLIP,
    DRAW,
    N_ACTIONS,
    TAKE_DISCARD,
    action_mask,
    apply_action,
)
from skyjo.game.deck import full_deck
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState
from skyjo.game.view import make_view
from tests.helpers import all_cards


def new_game(n_players: int = 2, seed: int = 0) -> SkyjoGame:
    return SkyjoGame(n_players, np.random.default_rng(seed))


def finish_setup(game: SkyjoGame) -> None:
    while game.phase == Phase.SETUP_FLIP:
        game.setup_flip(game.legal_setup_positions()[0])


# --- view --------------------------------------------------------------------

def test_view_never_contains_hidden_values():
    game = new_game(3)
    finish_setup(game)
    for p in range(3):
        view = make_view(game, p)
        assert np.all(view.values[view.states != CardState.VISIBLE] == 0)
        assert np.count_nonzero(view.states == CardState.VISIBLE) == 6


def test_hand_visible_only_to_current_player():
    game = new_game(2)
    finish_setup(game)
    game.draw_from_deck()
    cp = game.current_player
    assert make_view(game, cp).hand == game.hand
    assert make_view(game, 1 - cp).hand is None


def test_view_is_a_copy():
    game = new_game(2)
    view = make_view(game, 0)
    game.setup_flip(0)
    assert view.states[0, 0] == CardState.HIDDEN


# --- action mask -------------------------------------------------------------

def test_mask_setup_phase():
    mask = action_mask(new_game())
    assert mask.shape == (N_ACTIONS,)
    assert mask[FLIP:DRAW].all() and mask.sum() == 12  # setup flips use 12..23


def test_mask_choose_source():
    game = new_game()
    finish_setup(game)
    mask = action_mask(game)
    assert mask[DRAW] and mask[TAKE_DISCARD]
    assert mask.sum() == 2


def test_mask_place_drawn_vs_discard():
    game = new_game()
    finish_setup(game)
    hidden = game.players[game.current_player].hidden_positions()

    game.draw_from_deck()
    mask = action_mask(game)
    assert mask[:12].all()
    assert set(np.flatnonzero(mask[FLIP:DRAW])) == set(hidden)
    assert not mask[DRAW] and not mask[TAKE_DISCARD]

    game.place(0)
    game.take_discard()
    mask = action_mask(game)
    assert mask[:12].all() and not mask[12:].any()


def test_mask_excludes_removed_column():
    game = new_game()
    finish_setup(game)
    game.players[game.current_player].remove_column(2)
    game.draw_from_deck()
    mask = action_mask(game)
    for pos in (2, 6, 10):
        assert not mask[pos] and not mask[FLIP + pos]


def test_apply_action_rejects_illegal():
    game = new_game()
    with pytest.raises(ValueError):
        apply_action(game, DRAW)  # still in setup
    with pytest.raises(ValueError):
        apply_action(game, N_ACTIONS)


def test_apply_action_maps_to_engine():
    game = new_game()
    finish_setup(game)
    cp = game.current_player
    apply_action(game, DRAW)
    assert game.phase == Phase.PLACE_DRAWN
    hidden = game.players[cp].hidden_positions()[0]
    apply_action(game, FLIP + hidden)
    assert game.players[cp].states[hidden] == CardState.VISIBLE
    assert game.current_player != cp


# --- random agent ------------------------------------------------------------

@pytest.mark.parametrize("n_players", [2, 5, 8])
def test_random_agents_full_games_conserve_cards(n_players):
    def check(game):
        assert all_cards(game) == Counter(full_deck())

    rng = np.random.default_rng(0)
    agents = [RandomAgent(np.random.default_rng(i)) for i in range(n_players)]
    for _ in range(3):
        game = play_game(agents, rng, on_step=check)
        assert game.is_over


def test_make_variant_sets_bot_parameters():
    from skyjo.agents.heuristic_agent import CautiousHeuristicAgent, HeuristicAgent
    from skyjo.agents.registry import make_variant

    rng = np.random.default_rng(0)
    bot = make_variant("heuristic:discard_threshold=2,min_gain=3", rng)
    assert type(bot) is HeuristicAgent
    assert (bot.discard_threshold, bot.min_gain, bot.take_discard_for_gain) == (2, 3, False)
    bot = make_variant("cautious:max_refusals=3,take_discard_for_gain=true", rng)
    assert isinstance(bot, CautiousHeuristicAgent)
    assert bot.max_refusals == 3 and bot.take_discard_for_gain
    assert type(make_variant("heuristic", rng)) is HeuristicAgent
    with pytest.raises(ValueError):
        make_variant("random:min_gain=1", rng)


# --- loop guard ------------------------------------------------------------------

from skyjo.agents.base import GameBlocked, play_game  # noqa: E402
from skyjo.agents.ppo_agent import LoopGuardPPOAgent, PPOAgent  # noqa: E402
from skyjo.env.actions import DRAW, FLIP, TAKE_DISCARD  # noqa: E402
from skyjo.game.engine import Phase  # noqa: E402
from skyjo.game.player_state import CardState  # noqa: E402


class LoopingStub(PPOAgent):
    """Stands for a deterministic network that never ends a round: it always takes
    the discard and lays it on a visible card; after a draw it reveals a hidden
    card, except the last one. Under a restricted mask it plays the first legal action."""

    def __init__(self, model=None, deterministic=True):
        super().__init__(model, deterministic)

    def act(self, view, mask):
        legal = np.flatnonzero(mask)
        visible = [p for p in np.flatnonzero(view.my_states == CardState.VISIBLE) if mask[p]]
        hidden = np.flatnonzero(view.my_states == CardState.HIDDEN)
        if view.phase == Phase.CHOOSE_SOURCE:
            return TAKE_DISCARD if mask[TAKE_DISCARD] else DRAW
        if view.phase == Phase.PLACE_DRAWN and len(legal) <= 2:
            return int(legal[0])  # restricted mask: forced to end the round
        if view.phase == Phase.PLACE_DRAWN and len(hidden) > 1:
            return FLIP + int(hidden[0])
        return int(visible[0]) if visible else int(legal[0])


class GuardedStub(LoopGuardPPOAgent, LoopingStub):
    def __init__(self):
        LoopGuardPPOAgent.__init__(self, model=None)


def test_looping_agents_block_without_the_guard():
    with pytest.raises(GameBlocked):
        play_game([LoopingStub(), LoopingStub()], np.random.default_rng(0), max_steps=3000)


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_loop_guard_ends_the_loops(n_players):
    for seed in range(5):
        agents = [GuardedStub() for _ in range(n_players)]
        game = play_game(agents, np.random.default_rng(seed), max_steps=20_000)
        assert game.is_over
        assert sum(a.forced_draws for a in agents) > 0
        assert sum(a.forced_finishes for a in agents) > 0


def test_loop_guard_plays_like_the_network_when_no_state_repeats():
    from skyjo.agents.random_agent import RandomAgent

    def run(agent):
        game = play_game([agent, RandomAgent(np.random.default_rng(3))], np.random.default_rng(4))
        return game.total_scores

    guard = GuardedStub()
    assert run(guard) == run(LoopingStub())
    assert guard.forced_draws == guard.forced_finishes == 0


def test_loop_guard_finish_falls_back_when_the_rule_mask_excludes_the_hidden_card():
    import dataclasses

    from skyjo.game.engine import SkyjoGame
    from skyjo.game.view import make_view
    from tests.helpers import rig

    game = SkyjoGame(2, np.random.default_rng(0))
    grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    rig(game, [grid, grid], [list(range(11)), []])  # position 11 is the last hidden card
    view = dataclasses.replace(make_view(game, 0), phase=Phase.PLACE_DRAWN, hand=3)
    agent = GuardedStub()
    agent._round, agent._interventions = view.round_number, 2
    only = np.zeros(26, dtype=bool)
    only[2] = True  # a column rule mask: only position 2
    assert agent.act(view, only) == 2
