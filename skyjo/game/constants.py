"""Game constants for Skyjo (reference rules: docs/RULES.md)."""

# Card value -> number of copies in the deck.
CARD_DISTRIBUTION: dict[int, int] = {
    -2: 5,
    -1: 10,
    0: 15,
    **{value: 10 for value in range(1, 13)},
}
DECK_SIZE: int = sum(CARD_DISTRIBUTION.values())  # 150
MIN_CARD_VALUE: int = min(CARD_DISTRIBUTION)  # -2
MAX_CARD_VALUE: int = max(CARD_DISTRIBUTION)  # 12
MEAN_CARD_VALUE: float = (
    sum(value * count for value, count in CARD_DISTRIBUTION.items()) / DECK_SIZE
)  # ~5.07, expected value of an unknown card

# Player grid.
N_ROWS: int = 3
N_COLS: int = 4
GRID_SIZE: int = N_ROWS * N_COLS  # 12
N_INITIAL_FLIPS: int = 2  # cards each player reveals at the start of a round

# Players.
MIN_PLAYERS: int = 2
MAX_PLAYERS: int = 8  # official maximum, also the observation padding size

# End of game.
END_GAME_SCORE: int = 100  # game ends when a player reaches or exceeds this total

assert DECK_SIZE == 150
