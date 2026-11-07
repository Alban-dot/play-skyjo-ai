from collections import Counter

import numpy as np
import pytest

from skyjo.game.deck import full_deck
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import PlayerState
from tests.helpers import all_cards, rig

# Grids with no two equal cards in a column (no accidental column removal).
DISTINCT = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # sum 42
LOW = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]  # sum 6
HIGH = [[5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]]  # sum 90
ALL_BUT_LAST = list(range(11))


def new_game(n_players: int = 2, seed: int = 0) -> SkyjoGame:
    return SkyjoGame(n_players, np.random.default_rng(seed))


def random_move(game, rng) -> None:
    if game.phase == Phase.SETUP_FLIP:
        game.setup_flip(int(rng.choice(game.legal_setup_positions())))
    elif game.phase == Phase.CHOOSE_SOURCE:
        if game.can_take_discard() and rng.random() < 0.5:
            game.take_discard()
        else:
            game.draw_from_deck()
    elif game.phase == Phase.PLACE_DRAWN and game.legal_flip_positions() and rng.random() < 0.5:
        game.discard_and_flip(int(rng.choice(game.legal_flip_positions())))
    else:
        game.place(int(rng.choice(game.legal_place_positions())))


def play_random_game(n_players: int, seed: int, check=None) -> SkyjoGame:
    game = new_game(n_players, seed)
    rng = np.random.default_rng(seed + 1000)
    for _ in range(100_000):
        if game.is_over:
            return game
        random_move(game, rng)
        if check:
            check(game)
    raise AssertionError("Game did not end.")


# --- setup -------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 9])
def test_invalid_player_count(n):
    with pytest.raises(ValueError):
        new_game(n)


def test_new_game_initial_state():
    game = new_game(4)
    assert game.round_number == 1
    assert game.phase == Phase.SETUP_FLIP
    assert game.current_player == 0
    assert all(len(p.hidden_positions()) == 12 for p in game.players)
    assert len(game.deck.discard_pile) == 1
    assert all_cards(game) == Counter(full_deck())


def test_setup_flips_then_first_player():
    game = new_game(3)
    game.players = [PlayerState([v for row in g for v in row]) for g in (DISTINCT, HIGH, LOW)]
    order = []
    for _ in range(6):
        order.append(game.current_player)
        game.setup_flip(game.legal_setup_positions()[0])
    assert order == [0, 0, 1, 1, 2, 2]
    # Flipped sums: P0 = 1+2, P1 = 5+6, P2 = 0+1 -> P1 starts.
    assert game.phase == Phase.CHOOSE_SOURCE
    assert game.current_player == 1


def test_actions_in_wrong_phase_raise():
    game = new_game()
    with pytest.raises(ValueError):
        game.draw_from_deck()  # still in setup
    rig(game, [DISTINCT, LOW], [[], []], discard=[3])
    with pytest.raises(ValueError):
        game.place(0)  # nothing in hand
    game.take_discard()
    with pytest.raises(ValueError):
        game.discard_and_flip(0)  # discard card must be placed
    with pytest.raises(ValueError):
        game.setup_flip(0)


# --- a normal turn -----------------------------------------------------------

def test_draw_then_place():
    game = new_game()
    rig(game, [DISTINCT, LOW], [[], []], draw=[7, 9], discard=[2])
    assert game.draw_from_deck() == 9
    assert game.phase == Phase.PLACE_DRAWN
    game.place(0)  # replaces a hidden 1
    p0 = game.players[0]
    assert p0.values[0] == 9 and 0 in p0.visible_positions()
    assert game.deck.top_discard == 1
    assert game.current_player == 1
    assert game.phase == Phase.CHOOSE_SOURCE


def test_take_discard_then_place():
    game = new_game()
    rig(game, [DISTINCT, LOW], [[], []], discard=[4, -2])
    assert game.take_discard() == -2
    assert game.phase == Phase.PLACE_DISCARD
    game.place(5)  # replaces a hidden 3
    assert game.players[0].values[5] == -2
    assert game.deck.discard_pile == [4, 3]


def test_discard_and_flip():
    game = new_game()
    rig(game, [DISTINCT, LOW], [[], []], draw=[11], discard=[0])
    game.draw_from_deck()
    game.discard_and_flip(6)
    assert game.deck.top_discard == 11
    assert game.players[0].visible_positions() == [6]
    assert game.current_player == 1


def test_illegal_flip_leaves_state_unchanged():
    game = new_game()
    rig(game, [DISTINCT, LOW], [[2], []], draw=[11], discard=[0])
    game.draw_from_deck()
    with pytest.raises(ValueError):
        game.discard_and_flip(2)  # already visible
    assert game.phase == Phase.PLACE_DRAWN
    assert game.hand == 11
    assert game.deck.discard_pile == [0]


def test_place_on_removed_column_raises():
    game = new_game()
    rig(game, [DISTINCT, LOW], [[], []], draw=[4])
    game.players[0].remove_column(0)
    game.draw_from_deck()
    with pytest.raises(ValueError):
        game.place(4)
    assert 4 not in game.legal_place_positions()


def test_column_removed_during_turn_discard_order():
    grid = [[5, 1, 2, 3], [5, 2, 3, 4], [8, 3, 4, 6]]
    game = new_game()
    rig(game, [grid, LOW], [[0, 4], []], draw=[5], discard=[12])
    game.draw_from_deck()
    game.place(8)
    assert game.deck.discard_pile == [12, 8, 5, 5, 5]
    assert game.players[0].active_positions() == [1, 2, 3, 5, 6, 7, 9, 10, 11]


# --- end of round / game -----------------------------------------------------

def test_end_of_round_last_turns_and_penalty():
    game = new_game(3)
    # P0 has one hidden card left; P1 and P2 fully hidden.
    rig(game, [DISTINCT, LOW, HIGH], [ALL_BUT_LAST, [], []],
        draw=[3, 0, 12], discard=[4])

    game.draw_from_deck()        # P0 draws 12
    game.discard_and_flip(11)    # reveals last card -> P0 is the finisher
    assert game.finisher == 0 and game.turns_left == 2

    game.draw_from_deck()        # P1 draws 0
    game.place(0)                # replaces a 0: P1 still sums to 6
    assert game.round_number == 1 and game.current_player == 2

    game.take_discard()          # P2 takes the 0
    game.place(0)                # replaces a 5: P2 sums to 85

    # P0 (42) is not the lowest -> doubled. Next round starts.
    assert game.last_round_scores == [84, 6, 85]
    assert game.total_scores == [84, 6, 85]
    assert game.round_number == 2
    assert game.phase == Phase.SETUP_FLIP
    assert all(p.all_visible() for p in game.last_round_players)


def test_final_reveal_removes_hidden_column_then_game_over():
    p1 = [[9, 1, 2, 3], [9, 2, 3, 4], [9, 0, 1, 0]]  # hidden column of 9s
    game = new_game(2)
    rig(game, [DISTINCT, p1], [ALL_BUT_LAST, [1]], draw=[3, 1, 12], discard=[4])
    game.total_scores = [20, 0]

    game.draw_from_deck()
    game.discard_and_flip(11)    # P0 finishes with 42
    game.draw_from_deck()        # P1 draws 1
    game.place(1)                # replaces a visible 1

    # P1: 43 - 27 (removed column of 9s) = 16. P0 doubled: 84.
    assert game.last_round_scores == [84, 16]
    assert game.total_scores == [104, 16]
    assert game.deck.discard_pile[-3:] == [9, 9, 9]
    assert game.is_over
    assert game.winners() == [1]
    with pytest.raises(ValueError):
        game.draw_from_deck()


def test_winners_before_end_raises():
    with pytest.raises(ValueError):
        new_game().winners()


# --- full random games -------------------------------------------------------

@pytest.mark.parametrize("n_players", range(2, 9))
def test_random_games_terminate_and_conserve_cards(n_players):
    def check(game):
        assert all_cards(game) == Counter(full_deck())

    for seed in range(3):
        game = play_random_game(n_players, seed, check)
        assert max(game.total_scores) >= 100
        assert game.winners()


def test_same_seed_same_game():
    a = play_random_game(4, seed=7)
    b = play_random_game(4, seed=7)
    assert a.total_scores == b.total_scores
    assert a.round_number == b.round_number


def test_force_end_round_counts_every_card_without_doubling():
    game = SkyjoGame(2, np.random.default_rng(0))
    me = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # sum 42, 11 hidden cards
    opp = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]  # sum 6, 1 hidden card
    rig(game, [me, opp], visible=[[0], list(range(11))])
    game.force_end_round()
    assert game.last_round_scores == [42, 6]  # hidden cards counted, nobody doubled
    assert game.total_scores == [42, 6]
    assert all(p.all_visible() for p in game.last_round_players)
    assert game.round_number == 2 and game.phase == Phase.SETUP_FLIP and game.turns_played == 0


def test_force_end_round_only_between_turns_of_an_unfinished_round():
    game = SkyjoGame(2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        game.force_end_round()  # setup phase
    rig(game, [[[1] * 4] * 3, [[2] * 4] * 3], visible=[[0], [0]])
    game.draw_from_deck()
    with pytest.raises(ValueError):
        game.force_end_round()  # a card is in hand
    game.discard_and_flip(1)
    game.finisher = 0  # the round is ending normally (last turns)
    with pytest.raises(ValueError):
        game.force_end_round()


def test_turns_played_counts_completed_turns_of_the_round():
    game = SkyjoGame(2, np.random.default_rng(0))
    rig(game, [[[1] * 4] * 3, [[2] * 4] * 3], visible=[[0], [0]])
    game.turns_played = 0
    for expected in (1, 2, 3):
        game.draw_from_deck()
        game.discard_and_flip(game.legal_flip_positions()[0])
        assert game.turns_played == expected
