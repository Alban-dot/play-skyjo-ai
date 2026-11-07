from collections import Counter

import numpy as np
import pytest

from skyjo.game.constants import CARD_DISTRIBUTION, DECK_SIZE
from skyjo.game.deck import Deck, full_deck


def make_deck(seed: int = 0) -> Deck:
    return Deck(np.random.default_rng(seed))


def test_full_deck_has_150_cards():
    assert len(full_deck()) == DECK_SIZE == 150


def test_distribution():
    counts = Counter(full_deck())
    assert counts[-2] == 5
    assert counts[-1] == 10
    assert counts[0] == 15
    for value in range(1, 13):
        assert counts[value] == 10
    assert set(counts) == set(CARD_DISTRIBUTION)


def test_new_deck_is_shuffled_and_complete():
    deck = make_deck()
    assert len(deck) == 150
    assert deck.discard_pile == []
    assert Counter(deck.draw_pile) == Counter(full_deck())
    assert deck.draw_pile != full_deck()


def test_same_seed_same_order():
    assert make_deck(42).draw_pile == make_deck(42).draw_pile


def test_different_seed_different_order():
    assert make_deck(1).draw_pile != make_deck(2).draw_pile


def test_draw_and_discard():
    deck = make_deck()
    deck.start_discard()
    assert len(deck.draw_pile) == 149
    assert deck.top_discard is not None

    card = deck.draw()
    deck.discard(card)
    assert deck.top_discard == card
    assert deck.take_discard() == card
    assert len(deck) == 149  # one card is "in hand" after take_discard


def test_take_from_empty_discard_raises():
    with pytest.raises(RuntimeError):
        make_deck().take_discard()


def test_reshuffle_when_draw_pile_empty():
    deck = make_deck()
    # Move every card to the discard pile.
    while deck.draw_pile:
        deck.discard(deck.draw())
    top = deck.top_discard
    assert len(deck.discard_pile) == 150

    deck.draw()  # triggers the reshuffle
    assert deck.discard_pile == [top]  # top discard card is kept
    assert len(deck.draw_pile) == 148
    assert len(deck) == 149  # no card lost or duplicated (one is in hand)


def test_reshuffle_preserves_cards():
    deck = make_deck()
    while deck.draw_pile:
        deck.discard(deck.draw())
    deck.reshuffle_discard()
    assert Counter(deck.draw_pile + deck.discard_pile) == Counter(full_deck())


def test_reshuffle_impossible_raises():
    deck = make_deck()
    deck.draw_pile = []
    deck.discard_pile = [5]
    with pytest.raises(RuntimeError):
        deck.draw()
