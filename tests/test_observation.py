import copy

import numpy as np
import pytest

from skyjo.env.observation import (
    LAYOUT,
    N_STATES,
    N_VALUES,
    OBS_SIZE,
    OPP_SLOT,
    encode,
    unknown_card_probs,
)
from skyjo.game.constants import CARD_DISTRIBUTION, MIN_CARD_VALUE
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState
from skyjo.game.rules import remove_completed_columns
from skyjo.game.view import make_view
from tests.helpers import rig
from tests.test_actions import random_game_states

INITIAL = np.array([CARD_DISTRIBUTION[v] for v in range(MIN_CARD_VALUE, 13)], dtype=float)
GRID = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]


def obs(game, player):
    return encode(make_view(game, player))


def idx(value: int) -> int:
    return value - MIN_CARD_VALUE


# --- no information leak -----------------------------------------------------

def scramble_hidden(game: SkyjoGame, viewer: int, rng) -> SkyjoGame:
    """Copy of the game where everything ``viewer`` cannot see is changed."""
    g = copy.deepcopy(game)
    for p in g.players:
        for pos in p.hidden_positions():
            p.values[pos] = rng.integers(-2, 13)
    g.deck.draw_pile = [int(v) for v in rng.integers(-2, 13, size=len(g.deck.draw_pile))]
    if g.hand is not None and g.current_player != viewer:
        g.hand = int(rng.integers(-2, 13))
    return g


@pytest.mark.parametrize("n_players", [2, 4, 6, 8])
def test_hidden_cards_never_change_the_observation(n_players):
    rng = np.random.default_rng(0)
    n_checked = 0
    for game in random_game_states(n_players, seed=n_players):
        for viewer in range(n_players):
            twin = scramble_hidden(game, viewer, rng)
            assert np.array_equal(obs(game, viewer), obs(twin, viewer))
            n_checked += 1
    assert n_checked > 100


def test_visible_card_does_change_the_observation():
    # Sanity check: the leak test above is not trivially true.
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[0], [0]])
    before = obs(game, 0)
    game.players[1].values[0] = 9  # visible card of the opponent
    assert not np.array_equal(before, obs(game, 0))


# --- fixed size and padding --------------------------------------------------

def test_same_size_for_2_to_8_players():
    for n in range(2, 9):
        game = SkyjoGame(n, np.random.default_rng(0))
        assert obs(game, 0).shape == (OBS_SIZE,)
    assert OBS_SIZE == 628
    assert LAYOUT["draw_pile"].stop == OBS_SIZE


@pytest.mark.parametrize("n_players", [2, 4, 5, 6, 8])
def test_empty_opponent_slots_are_zero(n_players):
    game = SkyjoGame(n_players, np.random.default_rng(0))
    game.total_scores = [10 * (i + 1) for i in range(n_players)]
    while game.phase == Phase.SETUP_FLIP:
        game.setup_flip(game.legal_setup_positions()[0])
    o = obs(game, 0)

    opp = o[LAYOUT["opponents"]].reshape(7, OPP_SLOT)
    assert np.all(opp[:n_players - 1, 0] == 1.0)  # present flags
    assert np.all(opp[n_players - 1:] == 0.0)  # absent slots fully zero
    assert np.all(o[LAYOUT["total_scores"]][n_players:] == 0.0)
    assert np.all(o[LAYOUT["visible_sums"]][n_players:] == 0.0)


def test_seat_order_is_relative_to_viewer():
    game = SkyjoGame(3, np.random.default_rng(0))
    game.total_scores = [10, 20, 30]
    scores = obs(game, 1)[LAYOUT["total_scores"]][:3]
    assert scores.tolist() == pytest.approx([0.2, 0.3, 0.1])  # me (1), then 2, then 0


# --- field encodings ---------------------------------------------------------

def test_my_grid_encoding():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[5], []])
    game.players[0].remove_column(3)
    grid = obs(game, 0)[LAYOUT["my_grid"]].reshape(12, N_STATES + N_VALUES)

    assert grid[5, CardState.VISIBLE] == 1 and grid[5, N_STATES + idx(3)] == 1
    assert grid[0, CardState.HIDDEN] == 1 and grid[0, N_STATES:].sum() == 0
    assert grid[3, CardState.REMOVED] == 1 and grid[3, N_STATES:].sum() == 0
    assert np.all(grid[:, :N_STATES].sum(axis=1) == 1)


def test_phase_hand_and_discard_encoding():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[], []], draw=[7], discard=[-2])
    o = obs(game, 0)
    assert o[LAYOUT["phase"]].tolist() == [0, 1, 0, 0]
    assert o[LAYOUT["hand"]].sum() == 0
    assert o[LAYOUT["top_discard"]][idx(-2)] == 1 and o[LAYOUT["top_discard"]][-1] == 1

    game.draw_from_deck()
    assert obs(game, 0)[LAYOUT["hand"]][idx(7)] == 1
    assert obs(game, 1)[LAYOUT["hand"]].sum() == 0  # opponent doesn't see my card
    assert obs(game, 0)[LAYOUT["phase"]].tolist() == [0, 0, 1, 0]


def test_last_turn_flag():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[], []])
    assert obs(game, 1)[LAYOUT["last_turn"]] == 0
    game.finisher = 0
    assert obs(game, 1)[LAYOUT["last_turn"]] == 1


# --- card counting -----------------------------------------------------------

@pytest.mark.parametrize("n_players", [2, 4])
def test_unknown_probs_sum_to_one_in_every_state(n_players):
    for game in random_game_states(n_players, seed=20 + n_players):
        for viewer in range(n_players):
            assert unknown_card_probs(make_view(game, viewer)).sum() == pytest.approx(1.0)


def expected_probs(known: list[int]) -> np.ndarray:
    counts = INITIAL.copy()
    for card in known:
        counts[idx(card)] -= 1
    return counts / counts.sum()


def test_counting_includes_buried_discard_and_hand():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[0, 1], [5]], draw=[8], discard=[12, 12, 7])
    known = [1, 2] + [3] + [12, 12, 7]
    assert unknown_card_probs(make_view(game, 0)) == pytest.approx(expected_probs(known))

    game.draw_from_deck()  # I now hold an 8; the opponent does not see it
    assert unknown_card_probs(make_view(game, 0)) == pytest.approx(expected_probs(known + [8]))
    assert unknown_card_probs(make_view(game, 1)) == pytest.approx(expected_probs(known))


def test_counting_includes_removed_columns():
    grid = [[9, 1, 2, 3], [9, 2, 3, 4], [9, 3, 4, 6]]
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [grid, GRID], [[0, 4, 8], []], discard=[])
    before = unknown_card_probs(make_view(game, 1))
    remove_completed_columns(game.players[0], game.deck)  # 9s go to the discard
    after = unknown_card_probs(make_view(game, 1))
    assert after == pytest.approx(before)  # still known, now via the discard
    assert after == pytest.approx(expected_probs([9, 9, 9]))


def test_counting_after_reshuffle():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [GRID, GRID], [[], []], draw=[], discard=[12, 12, 12, -2])
    before = unknown_card_probs(make_view(game, 0))
    assert before == pytest.approx(expected_probs([12, 12, 12, -2]))
    game.deck.reshuffle_discard()  # the three 12s go back into the draw pile
    after = unknown_card_probs(make_view(game, 0))
    assert after == pytest.approx(expected_probs([-2]))


def test_counting_restarts_each_round():
    game = SkyjoGame(2, np.random.default_rng(0))
    view = make_view(game, 0)  # start of round: only the first discard is known
    assert len(view.discard_pile) == 1
    assert unknown_card_probs(view) == pytest.approx(expected_probs(list(view.discard_pile)))


# --- no information leak: forced edge states ---------------------------------

def assert_no_leak(game: SkyjoGame, rng) -> None:
    for viewer in range(game.n_players):
        twin = scramble_hidden(game, viewer, rng)
        assert np.array_equal(obs(game, viewer), obs(twin, viewer))


def game_after_setup(n_players: int, seed: int) -> SkyjoGame:
    game = SkyjoGame(n_players, np.random.default_rng(seed))
    while game.phase == Phase.SETUP_FLIP:
        game.setup_flip(game.legal_setup_positions()[0])
    return game


def force_empty_draw_pile(game: SkyjoGame) -> None:
    """Move the whole draw pile under the discard (top card unchanged)."""
    game.deck.discard_pile = game.deck.draw_pile + game.deck.discard_pile
    game.deck.draw_pile = []


@pytest.mark.parametrize("n_players", [2, 4])
def test_no_leak_right_after_reshuffle(n_players):
    rng = np.random.default_rng(1)

    # State right after the reshuffle, before anyone draws.
    game = game_after_setup(n_players, seed=3)
    force_empty_draw_pile(game)
    top = game.deck.top_discard
    game.deck.reshuffle_discard()
    assert game.deck.discard_pile == [top] and len(game.deck.draw_pile) > 0
    assert_no_leak(game, rng)

    # State right after a draw that triggered the reshuffle (card in hand).
    game = game_after_setup(n_players, seed=4)
    force_empty_draw_pile(game)
    game.draw_from_deck()
    assert len(game.deck.discard_pile) == 1 and game.hand is not None
    assert_no_leak(game, rng)


@pytest.mark.parametrize("n_players", [2, 4])
def test_no_leak_during_last_turns(n_players):
    rng = np.random.default_rng(2)
    n_last_turn_states = 0
    for game in random_game_states(n_players, seed=30 + n_players):
        if game.is_over or game.finisher is None:
            continue
        n_last_turn_states += 1
        assert_no_leak(game, rng)
        for viewer in range(n_players):
            assert obs(game, viewer)[LAYOUT["last_turn"]] == 1
    assert n_last_turn_states > 0


# --- rule 5 of the cautious bot is computable from the observation -------------

def decode_grids(o: np.ndarray, n_players: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(values, states) of every grid, me first, read back from the observation."""
    mine = o[LAYOUT["my_grid"]].reshape(12, N_STATES + N_VALUES)
    my_states = mine[:, :N_STATES].argmax(axis=1)
    my_values = np.where(my_states == CardState.VISIBLE,
                         mine[:, N_STATES:].argmax(axis=1) + MIN_CARD_VALUE, 0)
    grids = [(my_values, my_states)]
    for slot in range(n_players - 1):
        start = LAYOUT["opponents"].start + slot * OPP_SLOT
        block = o[start + 1:start + OPP_SLOT].reshape(12, N_STATES + 1)
        grids.append((np.rint(block[:, N_STATES] * 12).astype(int), block[:, :N_STATES].argmax(axis=1)))
    return grids


@pytest.mark.parametrize("n_players", [2, 4])
def test_cautious_score_estimates_are_readable_from_observation(n_players):
    from skyjo.agents.heuristic_agent import estimated_score
    from skyjo.env.observation import seat_order

    for game in random_game_states(n_players, seed=10 + n_players):
        view = make_view(game, game.current_player)
        decoded = decode_grids(encode(view), n_players)
        for (values, states), player in zip(decoded, seat_order(view)):
            assert estimated_score(values, states) == pytest.approx(
                estimated_score(view.values[player], view.states[player]))
