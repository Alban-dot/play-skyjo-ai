"""Collect demonstrations of the cautious bot in the training env, print stats.

Usage:
    python scripts/collect_demos.py --config configs/two_players_b.yaml --n-games 3000
        [--seed 0] [--out data/demos_cautious.npz]
    python scripts/collect_demos.py --config configs/two_players_b.yaml --n-games 7500
        --end-only --seed 2000 --out data/demos_end_2000.npz   # end-of-round decisions only
    python scripts/collect_demos.py --config configs/two_players_b.yaml --n-games 800
        --model models/copy/copy_3000_end/model.zip --seed 4000 --out data/copy_games_4000.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.env.actions import DRAW, FLIP, TAKE_DISCARD  # noqa: E402
from skyjo.game.engine import Phase  # noqa: E402
from skyjo.simulation import wilson_interval  # noqa: E402
from skyjo.training.imitation import FINISHED, REFUSED, Demos, collect_demos  # noqa: E402
from skyjo.training.train import load_config  # noqa: E402


def print_end_stats(demos: Demos) -> None:
    n_games = len(demos.won)
    finished, refused = (demos.end_choice == FINISHED).sum(), (demos.end_choice == REFUSED).sum()
    print(f"{n_games:,} parties, {len(demos.actions):,} décisions de fin possible gardées "
          f"(politique seulement : {demos.policy_only.all()})")
    print(f"  le prudent finit : {finished:,}   refuse : {refused:,}   "
          f"part de refus : {refused / (finished + refused):.1%}")
    print(f"  par partie : {finished / n_games:.2f} fins, {refused / n_games:.2f} refus")


def print_stats(demos: Demos) -> None:
    n, n_games = len(demos.actions), len(demos.won)
    per_game = np.bincount(demos.game, minlength=n_games)
    print(f"{n_games} parties, {n:,} décisions")
    print(f"Décisions par partie : moyenne {per_game.mean():.1f}, médiane {np.median(per_game):.0f}, "
          f"min {per_game.min()}, max {per_game.max()}")
    print(f"Mémoire des observations : {demos.obs.nbytes / 1e6:,.0f} Mo\n")

    print(f"{'adversaire':<12}{'parties':>9}{'winrate':>10}{'IC 95 %':>18}{'écart de score':>16}")
    for name in np.unique(demos.opponent):
        sel = demos.opponent == name
        wins = int(demos.won[sel].sum())
        lo, hi = wilson_interval(wins, int(sel.sum()))
        diff = float(np.mean(demos.my_total[sel] - demos.opp_total[sel]))
        print(f"{name:<12}{int(sel.sum()):>9}{wins / sel.sum():>10.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>18}{diff:>+16.1f}")

    print(f"\n{'phase':<16}{'décisions':>11}{'part':>8}   actions")
    a = demos.actions
    for phase in (Phase.SETUP_FLIP, Phase.CHOOSE_SOURCE, Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
        sel = demos.phase == phase
        if phase == Phase.CHOOSE_SOURCE:
            detail = (f"piocher {np.mean(a[sel] == DRAW):.1%}, "
                      f"prendre la défausse {np.mean(a[sel] == TAKE_DISCARD):.1%}")
        elif phase == Phase.PLACE_DRAWN:
            detail = (f"placer {np.mean(a[sel] < FLIP):.1%}, "
                      f"défausser + retourner {np.mean((a[sel] >= FLIP) & (a[sel] < DRAW)):.1%}")
        else:
            detail = ""
        print(f"{phase.name:<16}{int(sel.sum()):>11,}{sel.mean():>8.1%}   {detail}")

    place = np.isin(demos.phase, [Phase.PLACE_DRAWN, Phase.PLACE_DISCARD])
    print(f"\nCompteur de refus à 10 : {demos.capped.sum():,} décisions "
          f"({demos.capped.mean():.2%} de toutes, {demos.capped[place].mean():.2%} des placements)")
    print(f"  dont la limite a changé la décision : {demos.cap_decisive.sum():,} "
          f"({demos.cap_decisive.mean():.3%} de toutes)")
    games_capped = len(np.unique(demos.game[demos.capped]))
    print(f"  parties touchées : {games_capped} ({games_capped / n_games:.1%})")

    returns = np.bincount(demos.game, weights=demos.rewards, minlength=n_games)
    print(f"\nSomme des récompenses par partie : moyenne {returns.mean():+.2f}, "
          f"écart type {returns.std():.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect cautious-bot demonstrations.")
    parser.add_argument("--config", required=True, help="training config (env section is used)")
    parser.add_argument("--n-games", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/demos_cautious.npz")
    parser.add_argument("--end-only", action="store_true",
                        help="keep only the end-of-round decisions (policy only)")
    parser.add_argument("--model", help="PPO model playing instead of the bot (sampled actions)")
    args = parser.parse_args(argv)

    env_cfg = load_config(args.config)["env"]
    agent = None
    if args.model:
        from skyjo.agents.ppo_agent import PPOAgent  # torch only needed here

        agent = PPOAgent.load(args.model, deterministic=False)
    start = time.perf_counter()
    demos = collect_demos(env_cfg, args.n_games, args.seed, end_only=args.end_only, agent=agent)
    elapsed = time.perf_counter() - start
    demos.save(args.out)
    print(f"Collecte : {elapsed:.0f} s ({len(demos.actions) / elapsed:,.0f} décisions/s), "
          f"fichier {args.out} ({Path(args.out).stat().st_size / 1e6:,.0f} Mo)\n")
    print_end_stats(demos) if args.end_only else print_stats(demos)


if __name__ == "__main__":
    main()
