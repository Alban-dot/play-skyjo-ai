"""Play many games between agents and collect statistics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from skyjo.agents.base import Agent, GameBlocked, play_game
from skyjo.game.engine import SkyjoGame


@dataclass
class SimulationResult:
    n_games: int
    wins: list[int]  # per agent; a shared win (tie) counts for each winner
    ties: int  # games with several winners
    scores: list[list[int]]  # per agent, final total score of each game
    rounds: list[int]  # number of rounds of each game
    blocked: int = 0  # games stopped at max_steps, excluded from all the above
    game_ids: list[int] = field(default_factory=list)  # index of each finished game
    blocked_ids: list[int] = field(default_factory=list)
    finished_rounds: list[list[int]] = field(default_factory=list)  # per agent, rounds it ended, per game

    def finish_share(self, agent: int) -> float:
        """Share of all rounds that ``agent`` ended by revealing its last card."""
        return sum(self.finished_rounds[agent]) / sum(self.rounds)

    def winrate(self, agent: int) -> float:
        return self.wins[agent] / self.n_games

    def mean_score(self, agent: int) -> float:
        return float(np.mean(self.scores[agent]))

    def score_diff(self, a: int, b: int, z: float = 1.96) -> tuple[float, float, float]:
        """Mean of (score a - score b) per game, with its confidence interval."""
        diff = np.array(self.scores[a], dtype=float) - np.array(self.scores[b], dtype=float)
        mean = float(diff.mean())
        half = z * float(diff.std(ddof=1)) / math.sqrt(len(diff)) if len(diff) > 1 else math.inf
        return mean, mean - half, mean + half


    def score_gap(self, agent: int, z: float = 1.96) -> tuple[float, float, float]:
        """Mean of (score of ``agent`` - mean score of the others) per game,
        with its confidence interval; equals score_diff(agent, other) at 2 players."""
        scores = np.array(self.scores, dtype=float)
        others = np.delete(scores, agent, axis=0).mean(axis=0)
        gap = scores[agent] - others
        mean = float(gap.mean())
        half = z * float(gap.std(ddof=1)) / math.sqrt(len(gap)) if len(gap) > 1 else math.inf
        return mean, mean - half, mean + half


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def simulate(
    agents: Sequence[Agent],
    n_games: int,
    rng: np.random.Generator,
    on_step: Callable[[SkyjoGame], None] | None = None,
    max_steps: int | None = None,
    deal_seed: int | None = None,
) -> SimulationResult:
    """Play ``n_games`` games; seats rotate each game so no agent is favoured.

    With ``max_steps``, a game not finished after that many actions is counted
    as blocked and excluded from the results (``n_games`` = finished games);
    without it, such a game raises after play_game's default limit.

    With ``deal_seed`` (paired evaluations), ``rng`` is not used: the deal of
    round r of game g comes from its own seed (deal_seed, g, r), and the agent at
    seat s of game g is reseeded from (deal_seed, g, s). Two runs with the same
    deal_seed and number of players then get exactly the same deals, whatever
    the agents and however many rounds the games last.
    """
    n = len(agents)
    wins, ties, blocked = [0] * n, 0, 0
    scores: list[list[int]] = [[] for _ in range(n)]
    finished_rounds: list[list[int]] = [[] for _ in range(n)]
    rounds, game_ids, blocked_ids = [], [], []
    finishers: list[int | None] = []  # seat that ended each round of the current game

    def track(game: SkyjoGame) -> None:
        ended = game.round_number - (0 if game.is_over else 1)
        if len(finishers) < ended:
            finishers.append(game.last_round_finisher)
        if on_step is not None:
            on_step(game)

    for g in range(n_games):
        seats = [(g + s) % n for s in range(n)]  # seat s is played by agent seats[s]
        players = [agents[a] for a in seats]
        round_rng = None
        if deal_seed is not None:
            round_rng = lambda r, g=g: np.random.default_rng([deal_seed, 0, g, r])  # noqa: E731
            for seat, agent in enumerate(players):
                agent.reseed(np.random.default_rng([deal_seed, 1, g, seat]))
        finishers.clear()
        try:
            game = play_game(players, rng, track, max_steps or 100_000, round_rng)
        except GameBlocked:
            if max_steps is None:
                raise
            blocked += 1
            blocked_ids.append(g)
            continue
        winners = game.winners()
        ties += len(winners) > 1
        for seat, a in enumerate(seats):
            scores[a].append(game.total_scores[seat])
            wins[a] += seat in winners
            finished_rounds[a].append(finishers.count(seat))
        rounds.append(game.round_number)
        game_ids.append(g)
    return SimulationResult(n_games - blocked, wins, ties, scores, rounds, blocked,
                            game_ids, blocked_ids, finished_rounds)
