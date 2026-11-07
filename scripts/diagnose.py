"""Per-round diagnostics of agents playing against a base agent.

For each agent: turns per round, how often it ends the round, how often its
score is doubled, and its mean round score. With --n-players N, each game is
the tested agent against N-1 base agents; the base agents' figures are per
player (averaged over the N-1 seats).

Usage:
    python scripts/diagnose.py [--n-games 1000] [--seed 0]
    python scripts/diagnose.py --model models/best/<run>/model.zip --base heuristic cautious
    python scripts/diagnose.py --model models/frozen/<model>.zip --base cautious --n-players 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.agents.base import Agent, play_game  # noqa: E402
from skyjo.agents.registry import AGENTS  # noqa: E402
from skyjo.game.engine import Phase, SkyjoGame  # noqa: E402

TESTED = ["heuristic", "heuristic-gain", "cautious"]
BASE = "heuristic"


@dataclass
class Stats:
    rounds: int = 0
    turns: int = 0
    ended: int = 0  # rounds where this agent revealed all its cards first
    doubled: int = 0
    doubled_when_ended: int = 0
    round_scores: list[int] = field(default_factory=list)

    def row(self, label: str) -> str:
        return (f"{label:<26}{self.turns / self.rounds:>12.1f}{self.ended / self.rounds:>12.1%}"
                f"{self.doubled / self.rounds:>11.1%}"
                f"{self.doubled_when_ended / max(self.ended, 1):>16.1%}"
                f"{np.mean(self.round_scores):>14.1f}")


class Recorder:
    """on_step hook: counts turns and records each finished round."""

    def __init__(self, seat_stats: list[Stats]) -> None:
        self.seat_stats = seat_stats
        self.round = 1
        self.finisher: int | None = None

    def __call__(self, game: SkyjoGame) -> None:
        if game.phase in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
            self.seat_stats[game.current_player].turns += 1  # source just chosen
        if game.is_over or game.round_number != self.round:
            self._record_round(game, self.finisher)
            self.round = game.round_number
        self.finisher = game.finisher

    def _record_round(self, game: SkyjoGame, finisher: int) -> None:
        for seat, stats in enumerate(self.seat_stats):
            score = game.last_round_scores[seat]
            doubled = score != game.last_round_players[seat].score()
            stats.rounds += 1
            stats.round_scores.append(score)
            stats.doubled += doubled
            if seat == finisher:
                stats.ended += 1
                stats.doubled_when_ended += doubled


def make_agent(name: str, rng: np.random.Generator, model: str | None) -> Agent:
    if name == "ppo":
        from skyjo.agents.ppo_agent import PPOAgent  # torch only needed here

        return PPOAgent.load(model, deterministic=True)
    return AGENTS[name](rng)


def diagnose(
    name: str, n_games: int, seed: int, base: str = BASE, model: str | None = None,
    n_players: int = 2,
) -> tuple[Stats, Stats]:
    """Play ``name`` against ``n_players - 1`` ``base`` agents; return (stats of
    name, stats of the base agents, all seats pooled)."""
    game_seq, seq_a, *base_seqs = np.random.SeedSequence(seed).spawn(n_players + 1)
    agents = [make_agent(name, np.random.default_rng(seq_a), model)]
    agents += [make_agent(base, np.random.default_rng(seq), model) for seq in base_seqs]
    mine, others = Stats(), Stats()
    stats = [mine] + [others] * (n_players - 1)
    rng = np.random.default_rng(game_seq)
    for g in range(n_games):
        seats = [(g + s) % n_players for s in range(n_players)]  # seat s is played by agent seats[s]
        play_game([agents[a] for a in seats], rng, on_step=Recorder([stats[a] for a in seats]))
    return mine, others


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-round diagnostics vs the base agent.")
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="PPO model .zip: diagnose it instead of the bots")
    parser.add_argument("--base", nargs="+", default=[BASE], choices=list(AGENTS),
                        help="opponent(s) to play against")
    parser.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4],
                        help="tested agent + N-1 base agents")
    args = parser.parse_args(argv)
    tested = ["ppo"] if args.model else TESTED

    print(f"{args.n_games} parties par adversaire, {args.n_players} joueurs, seed {args.seed}"
          + (" (adversaires : moyenne par joueur)" if args.n_players > 2 else "") + "\n")
    print(f"{'agent':<26}{'tours/manche':>12}{'termine':>12}{'doublé':>11}"
          f"{'doublé|termine':>16}{'score/manche':>14}")
    for base_name in args.base:
        for name in tested:
            mine, base = diagnose(name, args.n_games, args.seed, base_name, args.model, args.n_players)
            print(mine.row(f"{name} vs {base_name}"))
            print(base.row(f"  (adversaire {base_name})"))


if __name__ == "__main__":
    main()
