import pytest

from skyjo.game.player_state import CardState, PlayerState

CARDS = list(range(12))  # value == position, easy to check


def test_new_grid_is_hidden():
    p = PlayerState(CARDS)
    assert p.hidden_positions() == list(range(12))
    assert p.visible_positions() == []
    assert not p.all_visible()


def test_wrong_size_raises():
    with pytest.raises(ValueError):
        PlayerState([1, 2, 3])


def test_flip():
    p = PlayerState(CARDS)
    assert p.flip(5) == 5
    assert p.visible_positions() == [5]
    with pytest.raises(ValueError):
        p.flip(5)


def test_replace_hidden_and_visible():
    p = PlayerState(CARDS)
    assert p.replace(3, -2) == 3
    assert p.states[3] == CardState.VISIBLE
    assert p.replace(3, 7) == -2
    assert p.visible_score() == 7


def test_remove_column():
    p = PlayerState(CARDS)
    assert p.remove_column(1) == [1, 5, 9]
    assert p.active_positions() == [0, 2, 3, 4, 6, 7, 8, 10, 11]
    with pytest.raises(ValueError):
        p.replace(5, 0)
    with pytest.raises(ValueError):
        p.remove_column(1)


def test_reveal_all_and_score():
    p = PlayerState(CARDS)
    p.remove_column(0)  # removes 0, 4, 8
    p.reveal_all()
    assert p.all_visible()
    assert p.score() == sum(CARDS) - (0 + 4 + 8)
