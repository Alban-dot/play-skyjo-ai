import copy

import numpy as np
import pytest

from skyjo.agents.random_agent import RandomAgent
from skyjo.env.actions import DRAW, FLIP, N_ACTIONS, PLACE, TAKE_DISCARD, action_mask, apply_action
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.view import make_view

ALLOWED_RANGES = {
    Phase.SETUP_FLIP: set(range(FLIP, DRAW)),
    Phase.CHOOSE_SOURCE: {DRAW, TAKE_DISCARD},
    Phase.PLACE_DRAWN: set(range(PLACE, DRAW)),
    Phase.PLACE_DISCARD: set(range(PLACE, FLIP)),
}


def engine_accepts(game: SkyjoGame, action: int) -> bool:
    """Ask the engine directly (no mask) whether ``action`` is legal."""
    g = copy.deepcopy(game)
    try:
        if action == DRAW:
            g.draw_from_deck()
        elif action == TAKE_DISCARD:
            g.take_discard()
        elif action >= FLIP:
            if g.phase == Phase.SETUP_FLIP:
                g.setup_flip(action - FLIP)
            else:
                g.discard_and_flip(action - FLIP)
        else:
            g.place(action)
    except ValueError:
        return False
    return True


def random_game_states(n_players: int, seed: int):
    """Yield every state of a random game (before each move)."""
    game = SkyjoGame(n_players, np.random.default_rng(seed))
    agent = RandomAgent(np.random.default_rng(seed + 100))
    while not game.is_over:
        yield game
        p = game.current_player
        apply_action(game, agent.act(make_view(game, p), action_mask(game)))
    yield game


@pytest.mark.parametrize("n_players", [2, 4])
def test_mask_matches_engine_in_every_state(n_players):
    for game in random_game_states(n_players, seed=n_players):
        mask = action_mask(game)
        for action in range(N_ACTIONS):
            assert mask[action] == engine_accepts(game, action), (game.phase, action)


@pytest.mark.parametrize("n_players", [2, 4])
def test_mask_uses_only_the_phase_actions(n_players):
    seen: dict[Phase, set[int]] = {phase: set() for phase in ALLOWED_RANGES}
    for game in random_game_states(n_players, seed=10 + n_players):
        legal = set(np.flatnonzero(action_mask(game)).tolist())
        if game.is_over:
            assert legal == set()
            continue
        assert legal, "a playing phase always has at least one legal action"
        assert legal <= ALLOWED_RANGES[game.phase]
        seen[game.phase] |= legal
    # The 4 phases together use all 26 actions.
    assert set().union(*seen.values()) == set(range(N_ACTIONS))


def test_setup_flip_uses_flip_actions():
    game = SkyjoGame(2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        apply_action(game, 3)  # placing is illegal during setup
    apply_action(game, FLIP + 3)
    assert 3 in game.players[0].visible_positions()
