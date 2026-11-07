import numpy as np

from skyjo.game.deck import Deck
from skyjo.game.player_state import PlayerState
from skyjo.game.rules import (
    completed_columns,
    finalize_grid,
    first_player,
    is_game_over,
    remove_completed_columns,
    round_scores,
    winners,
)

# Grid layout (flat positions):
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# Column 0 = positions 0, 4, 8.


def grid(rows: list[list[int]]) -> PlayerState:
    return PlayerState([v for row in rows for v in row])


def show(player: PlayerState, positions: list[int]) -> PlayerState:
    for pos in positions:
        player.flip(pos)
    return player


def empty_deck() -> Deck:
    deck = Deck(np.random.default_rng(0))
    deck.discard_pile = []
    return deck


DIFFERENT = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


# --- columns -----------------------------------------------------------------

def test_three_equal_visible_cards_complete_column():
    p = show(grid([[5, 1, 2, 3], [5, 4, 6, 7], [5, 8, 9, 10]]), [0, 4, 8])
    assert completed_columns(p) == [0]


def test_hidden_card_blocks_column():
    p = show(grid([[5, 1, 2, 3], [5, 4, 6, 7], [5, 8, 9, 10]]), [0, 4])
    assert completed_columns(p) == []


def test_different_visible_cards_do_not_complete():
    p = grid(DIFFERENT)
    p.reveal_all()
    assert completed_columns(p) == []


def test_removed_column_not_detected_again():
    p = show(grid([[5, 1, 2, 3], [5, 4, 6, 7], [5, 8, 9, 10]]), [0, 4, 8])
    deck = empty_deck()
    assert remove_completed_columns(p, deck) == [0]
    assert completed_columns(p) == []
    assert remove_completed_columns(p, deck) == []


def test_columns_of_zeros_and_minus_two_are_removed():
    p = grid([[0, -2, 1, 2], [0, -2, 3, 4], [0, -2, 5, 6]])
    show(p, [0, 4, 8, 1, 5, 9])
    deck = empty_deck()
    assert remove_completed_columns(p, deck) == [0, 1]
    assert p.score() == 1 + 2 + 3 + 4 + 5 + 6


def test_flip_can_complete_column():
    p = show(grid([[7, 1, 2, 3], [7, 4, 6, 8], [7, 9, 10, 11]]), [0, 4])
    p.flip(8)
    assert completed_columns(p) == [0]


def test_two_columns_completed_at_once():
    p = grid([[3, 1, 9, 2], [3, 4, 9, 6], [3, 8, 9, 10]])
    p.reveal_all()
    deck = empty_deck()
    assert remove_completed_columns(p, deck) == [0, 2]
    assert deck.discard_pile == [3, 3, 3, 9, 9, 9]


def test_discard_order_replaced_card_first_then_column():
    # Column 0 = 5 / 5 / 8 (8 hidden). Player places a drawn 5 on the 8.
    p = show(grid([[5, 1, 2, 3], [5, 4, 6, 7], [8, 9, 10, 11]]), [0, 4])
    deck = empty_deck()
    deck.discard(12)  # some earlier discard

    old = p.replace(8, 5)
    deck.discard(old)
    remove_completed_columns(p, deck)

    assert deck.discard_pile == [12, 8, 5, 5, 5]
    assert deck.top_discard == 5


def test_all_columns_removed_counts_as_all_visible():
    p = grid([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    p.reveal_all()
    remove_completed_columns(p, empty_deck())
    assert p.active_positions() == []
    assert p.all_visible()
    assert p.score() == 0


def test_final_reveal_removes_hidden_identical_column_before_scoring():
    # Column 0 = three hidden 7s; column 1 = two visible 2s + a hidden 3.
    p = show(grid([[7, 2, 1, 4], [7, 2, 5, 6], [7, 3, 0, 1]]), [1, 5])
    deck = empty_deck()
    assert finalize_grid(p, deck) == [0]
    assert p.all_visible()
    assert p.score() == (2 + 2 + 3) + (1 + 5 + 0) + (4 + 6 + 1)  # 24 instead of 45
    assert deck.discard_pile == [7, 7, 7]


# --- first player ------------------------------------------------------------

def flipped(a: int, b: int) -> PlayerState:
    """Grid whose 2 revealed cards are a and b."""
    return show(grid([[a, b, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), [0, 1])


def test_first_player_highest_sum():
    players = [flipped(1, 2), flipped(10, 3), flipped(5, 5)]
    assert first_player(players, [0, 0, 0], np.random.default_rng(0)) == 1


def test_first_player_tie_broken_by_highest_total():
    players = [flipped(6, 6), flipped(10, 2), flipped(1, 1)]
    assert first_player(players, [20, 35, 90], np.random.default_rng(0)) == 1


def test_first_player_full_tie_random_among_tied_only():
    players = [flipped(6, 6), flipped(1, 1), flipped(10, 2), flipped(12, 0)]
    totals = [30, 99, 30, 10]
    chosen = {first_player(players, totals, np.random.default_rng(s)) for s in range(50)}
    assert chosen == {0, 2}


def test_first_player_random_is_reproducible():
    players = [flipped(4, 4), flipped(4, 4)]
    a = [first_player(players, [0, 0], np.random.default_rng(s)) for s in range(20)]
    b = [first_player(players, [0, 0], np.random.default_rng(s)) for s in range(20)]
    assert a == b


# --- round scores and penalty -----------------------------------------------

def scored(total: int) -> PlayerState:
    """Grid with all cards hidden summing to ``total``."""
    return grid([[total, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])


def test_finisher_strictly_lowest_not_doubled():
    assert round_scores([scored(5), scored(10), scored(8)], finisher=0) == [5, 10, 8]


def test_finisher_tied_for_lowest_is_doubled():
    assert round_scores([scored(5), scored(5), scored(8)], finisher=0) == [10, 5, 8]


def test_finisher_not_lowest_is_doubled():
    assert round_scores([scored(4), scored(9)], finisher=1) == [4, 18]


def test_finisher_zero_or_negative_not_doubled():
    assert round_scores([scored(-3), scored(0)], finisher=1) == [-3, 0]
    assert round_scores([scored(-5), scored(-1)], finisher=1) == [-5, -1]


def test_score_includes_hidden_cards_and_ignores_removed_columns():
    p = grid([[4, 1, 2, 3], [4, 1, 2, 3], [4, 1, 2, 3]])
    show(p, [0, 4, 8, 1])
    remove_completed_columns(p, empty_deck())
    assert p.score() == 3 * (1 + 2 + 3)


# --- end of game -------------------------------------------------------------

def test_game_over_threshold():
    assert not is_game_over([99, 50])
    assert is_game_over([100, 50])
    assert is_game_over([30, 140, 12])


def test_winners_lowest_total_and_ties():
    assert winners([120, 45, 80]) == [1]
    assert winners([45, 100, 45]) == [0, 2]
