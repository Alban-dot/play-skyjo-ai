"""Check that playing through SkyjoEnv gives the same results as simulate.py.

HeuristicAgent plays inside the env (via the optional PlayerView in info)
against HeuristicAgent opponents; the same match-up is run with simulate().

Usage: python scripts/check_env_integration.py [--n-games 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.agents.heuristic_agent import HeuristicAgent  # noqa: E402
from skyjo.env.skyjo_env import SkyjoEnv  # noqa: E402
from skyjo.simulation import simulate, wilson_interval  # noqa: E402


def run_env(n_games: int, seed: int) -> tuple[int, np.ndarray, np.ndarray]:
    env = SkyjoEnv(opponent_factory=HeuristicAgent, return_view=True)
    agent = HeuristicAgent(np.random.default_rng(seed))
    wins, mine, theirs = 0, [], []
    for g in range(n_games):
        agent.reset()
        _, info = env.reset(seed=seed + g)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(info["view"], env.action_masks())
            _, _, terminated, truncated, info = env.step(action)
        me = info["seat"]
        wins += info["is_winner"]
        mine.append(info["total_scores"][me])
        theirs.append(info["total_scores"][1 - me])
    return wins, np.array(mine), np.array(theirs)


def compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se
    print(f"  {label:<22} env {a.mean():6.1f}   simulate {b.mean():6.1f}   "
          f"écart {a.mean() - b.mean():+5.1f}  ({z:+.1f} erreurs standard)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    n = args.n_games

    start = time.perf_counter()
    wins, mine, theirs = run_env(n, args.seed)
    ref = simulate(
        [HeuristicAgent(np.random.default_rng(args.seed + 1)),
         HeuristicAgent(np.random.default_rng(args.seed + 2))],
        n, np.random.default_rng(args.seed + 3),
    )
    elapsed = time.perf_counter() - start

    print(f"Heuristique contre heuristique, {n} parties par méthode ({elapsed:.0f} s)\n")
    for label, w in (("env", wins), ("simulate", ref.wins[0])):
        lo, hi = wilson_interval(w, n)
        print(f"  winrate {label:<9} {w / n:6.1%}   IC 95 % [{lo:.1%}, {hi:.1%}]")
    print()
    compare("score de l'agent", mine, np.array(ref.scores[0]))
    compare("score de l'adversaire", theirs, np.array(ref.scores[1]))
    print("\nAttendu : winrates proches de 50 %, écarts de score sous ~3 erreurs standard.")


if __name__ == "__main__":
    main()
