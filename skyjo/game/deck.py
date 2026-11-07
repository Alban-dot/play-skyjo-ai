"""Draw pile and discard pile."""

from __future__ import annotations

import numpy as np

from skyjo.game.constants import CARD_DISTRIBUTION


def full_deck() -> list[int]:
    """Return the 150 card values, unshuffled."""
    return [value for value, count in CARD_DISTRIBUTION.items() for _ in range(count)]


class Deck:
    """Draw pile + discard pile. The end of each list is the top of the pile.

    All randomness goes through the injected ``rng`` for reproducibility.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.draw_pile: list[int] = []
        self.discard_pile: list[int] = []
        self.reset()

    def reset(self) -> None:
        """Rebuild and shuffle the full 150-card deck; empty the discard pile."""
        self.draw_pile = full_deck()
        self.rng.shuffle(self.draw_pile)
        self.discard_pile = []

    # --- draw pile -------------------------------------------------------

    def draw(self) -> int:
        """Take the top card of the draw pile, reshuffling the discard if needed."""
        if not self.draw_pile:
            self.reshuffle_discard()
        return self.draw_pile.pop()

    def reshuffle_discard(self) -> None:
        """Shuffle the discard pile (except its top card) into a new draw pile."""
        if len(self.discard_pile) <= 1:
            raise RuntimeError("No cards left to reshuffle into the draw pile.")
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile
        self.rng.shuffle(self.draw_pile)
        self.discard_pile = [top]

    # --- discard pile ----------------------------------------------------

    @property
    def top_discard(self) -> int | None:
        """Value of the visible discard card, or None if the pile is empty."""
        return self.discard_pile[-1] if self.discard_pile else None

    def discard(self, card: int) -> None:
        """Put a card face up on top of the discard pile."""
        self.discard_pile.append(card)

    def take_discard(self) -> int:
        """Take the top card of the discard pile."""
        if not self.discard_pile:
            raise RuntimeError("Discard pile is empty.")
        return self.discard_pile.pop()

    def start_discard(self) -> None:
        """Flip the top card of the draw pile to start the discard pile."""
        self.discard(self.draw())

    # --- info ------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of cards held by the deck (draw + discard)."""
        return len(self.draw_pile) + len(self.discard_pile)
