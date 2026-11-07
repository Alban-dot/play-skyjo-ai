"""Play agents against each other over many games (seats rotate each game).

Usage:
    python scripts/simulate.py heuristic random [--n-games 1000] [--seed 0]
    python scripts/simulate.py cautious heuristic-gain
    python scripts/simulate.py heuristic random random random   # 4 players
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.agents.registry import AGENTS  # noqa: E402
from skyjo.simulation import simulate, wilson_interval  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate games between agents.")
    parser.add_argument("agents", nargs="*", default=["heuristic", "random"],
                        choices=list(AGENTS), help="one agent per player (2 to 8)")
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    names = args.agents

    game_seq, *agent_seqs = np.random.SeedSequence(args.seed).spawn(len(names) + 1)
    agents = [AGENTS[name](np.random.default_rng(s)) for name, s in zip(names, agent_seqs)]
    labels = [f"{name} #{i}" if names.count(name) > 1 else name for i, name in enumerate(names)]

    start = time.perf_counter()
    result = simulate(agents, args.n_games, np.random.default_rng(game_seq))
    elapsed = time.perf_counter() - start

    print(f"{args.n_games} parties, {len(names)} joueurs, seed {args.seed} ({elapsed:.1f} s)\n")
    print(f"{'agent':<16}{'victoires':>10}{'winrate':>10}{'IC 95 %':>18}{'score moyen':>14}")
    for i, label in enumerate(labels):
        lo, hi = wilson_interval(result.wins[i], result.n_games)
        print(f"{label:<16}{result.wins[i]:>10}{result.winrate(i):>10.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>18}{result.mean_score(i):>14.1f}")
    print(f"\nÉgalités : {result.ties}   Manches par partie : {np.mean(result.rounds):.2f}")

    if len(names) == 2:
        mean, lo, hi = result.score_diff(0, 1)
        print(f"Écart de score {labels[0]} - {labels[1]} : {mean:+.1f} "
              f"(IC 95 % [{lo:+.1f}, {hi:+.1f}], négatif = {labels[0]} meilleur)")


if __name__ == "__main__":
    main()
