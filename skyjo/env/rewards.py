"""Reward functions (interchangeable: the env config chooses one by name)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from skyjo.game.rules import winners


class RewardFn(Protocol):
    def on_round_end(self, round_scores: list[int], me: int) -> float: ...

    def on_game_end(self, total_scores: list[int], me: int) -> float: ...


class ScoreDiffReward:
    """Round end: (mean opponent round score - my round score) / scale.
    Game end: +win_bonus if I am the only winner, 0 if I share the win,
    -win_bonus otherwise (winners = lowest total of all players).

    Summed over a game, the round rewards equal
    (mean opponent total - my total) / scale.
    """

    def __init__(self, scale: float = 20.0, win_bonus: float = 1.0) -> None:
        self.scale = scale
        self.win_bonus = win_bonus

    def on_round_end(self, round_scores: list[int], me: int) -> float:
        others = [s for i, s in enumerate(round_scores) if i != me]
        return (float(np.mean(others)) - round_scores[me]) / self.scale

    def on_game_end(self, total_scores: list[int], me: int) -> float:
        win = winners(total_scores)
        if me not in win:
            return -self.win_bonus
        return self.win_bonus if len(win) == 1 else 0.0
