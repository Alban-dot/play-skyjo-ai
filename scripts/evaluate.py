"""Evaluate a trained PPO model against bots (seats rotate each game).

With --n-players N, each game is the model against N-1 copies of the opponent.

Deals are paired by default: the deal of round r of game g comes from its own
seed (seed, g, r) and each bot seat is reseeded per game, so two models
evaluated with the same seed get exactly the same deals (see
scripts/compare_paired.py). --shared-rng restores the former draw (one generator
for the whole series), to reproduce results obtained before.

Usage:
    python scripts/evaluate.py --model models/best/<run>/model.zip
        [--opponents heuristic cautious] [--n-games 1000] [--seed 0] [--n-players 2]
        [--per-game-csv games.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO  # noqa: E402

from skyjo.agents.ppo_agent import LoopGuardPPOAgent, PPOAgent  # noqa: E402
from skyjo.agents.registry import AGENTS, make_variant  # noqa: E402
from skyjo.game.constants import MAX_PLAYERS, MIN_PLAYERS  # noqa: E402
from skyjo.simulation import SimulationResult, simulate, wilson_interval  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO model against bots.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponents", nargs="+", default=["heuristic", "cautious"],
                        help=f"bots ({', '.join(AGENTS)}), variants name:key=value,... "
                             "(e.g. heuristic:min_gain=3) or ppo:<model.zip>; a mixed table "
                             "joins its N-1 opponents with '+' (e.g. cautious+heuristic+ppo:<model.zip>) "
                             "and sets its own number of players")
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int,
                        help="actions per game; longer games are counted as blocked and excluded")
    parser.add_argument("--sample", action="store_true",
                        help="PPO actions sampled from the policy instead of deterministic")
    parser.add_argument("--n-players", type=int, default=2, choices=range(MIN_PLAYERS, MAX_PLAYERS + 1),
                        help="model + N-1 opponents of the same kind (single opponents only); "
                             "the models were trained at 2 to 4 players")
    parser.add_argument("--per-game-csv",
                        help="write one row per game: table, game, blocked, model seat, rounds, "
                             "rounds ended by the model, model score then the opponents' scores")
    parser.add_argument("--shared-rng", action="store_true",
                        help="former draw: one generator for the whole series (unpaired deals)")
    parser.add_argument("--loop-guard", action="store_true",
                        help="every network at the table breaks repeated states "
                             "(skyjo.agents.ppo_agent.LoopGuardPPOAgent)")
    args = parser.parse_args(argv)
    n = args.n_players

    deterministic = not args.sample
    if args.sample:
        import torch as th

        th.manual_seed(args.seed)
    networks: dict[str, MaskablePPO] = {}

    def network_agent(path: str) -> PPOAgent:
        """One agent per seat (the loop guard keeps a per-agent history), one network per file."""
        if path not in networks:
            networks[path] = PPOAgent.load(path).model
        if args.loop_guard:
            return LoopGuardPPOAgent(networks[path], deterministic)
        return PPOAgent(networks[path], deterministic)

    model = network_agent(args.model)
    mode = "actions tirées selon la politique" if args.sample else "actions déterministes"
    if args.loop_guard:
        mode += " avec garde-fou de boucle (tous les réseaux de la table)"
    cap = f", plafond {args.max_steps:,} actions par partie" if args.max_steps else ""
    tables = [name.split("+") if "+" in name else [name] * (n - 1) for name in args.opponents]
    sizes = sorted({len(t) + 1 for t in tables})
    deals = "générateur unique (donnes non appariées)" if args.shared_rng else "donnes appariées"
    print(f"{args.model} : {args.n_games} parties par adversaire, "
          f"{'/'.join(map(str, sizes))} joueurs, seed {args.seed}, {deals}, {mode}{cap}\n")
    if max(sizes) > 4:
        print("HORS DISTRIBUTION : les modèles n'ont été entraînés qu'à 2, 3 ou 4 joueurs.\n")
    if max(sizes) > 2:
        print("ATTENTION : les modèles entraînés avant l'étape 9 (jusqu'au 300k) ne l'ont été "
              "qu'à 2 joueurs ; à plus de 2 joueurs, ils jouent hors de leur distribution "
              "d'entraînement.\n")
    width = max(12, *(len(name) + 2 for name in args.opponents))
    print(f"{'adversaire':<{width}}{'winrate':>10}{'IC 95 %':>18}{'hasard':>8}{'égalités':>10}"
          f"{'écart de score':>16}{'IC 95 %':>18}{'bloquées':>16}{'termine':>9}")
    rows = []
    for name, table in zip(args.opponents, tables):
        size = len(table) + 1
        game_seq, *bot_seqs = np.random.SeedSequence(args.seed).spawn(size)
        bots = []
        for spec, seq in zip(table, bot_seqs):
            if spec.startswith("ppo:"):
                bots.append(network_agent(spec[4:]))
            else:
                bots.append(make_variant(spec, np.random.default_rng(seq)))
        result = simulate([model, *bots], args.n_games, np.random.default_rng(game_seq),
                          max_steps=args.max_steps, deal_seed=None if args.shared_rng else args.seed)
        rows += game_rows(name, size, result)
        lo, hi = wilson_interval(result.wins[0], result.n_games)
        gap, glo, ghi = result.score_gap(0)
        blocked = f"{result.blocked} ({result.blocked / args.n_games:.1%})"
        print(f"{name:<{width}}{result.winrate(0):>10.1%}{f'[{lo:.1%}, {hi:.1%}]':>18}{1 / size:>8.1%}"
              f"{result.ties:>10}{gap:>+16.1f}{f'[{glo:+.1f}, {ghi:+.1f}]':>18}{blocked:>16}"
              f"{result.finish_share(0):>9.1%}")
        if args.loop_guard:
            guards = [a for a in (model, *bots) if isinstance(a, LoopGuardPPOAgent)]
            draws = sum(a.forced_draws for a in guards)
            finishes = sum(a.forced_finishes for a in guards)
            print(f"{'':<{width}}garde-fou : {draws} pioches forcées, {finishes} fins de manche forcées "
                  f"({draws / args.n_games:.2f} et {finishes / args.n_games:.2f} par partie)")
            model.forced_draws = model.forced_finishes = 0
    print("\nÉcart de score = total du modèle - moyenne des totaux des adversaires "
          "(négatif = modèle meilleur) ; hasard = winrate attendu d'un joueur de même niveau (1/N)")
    print("termine = part des manches terminées par le modèle (il retourne sa dernière carte en premier)")
    if args.max_steps:
        print("Winrate et écart sur les parties terminées ; bloquées = part de toutes les parties")
    if args.per_game_csv:
        with open(args.per_game_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["table", "n_players", "game", "blocked", "model_seat", "rounds",
                             "model_rounds_ended", "score_model",
                             *(f"score_opp{i}" for i in range(1, max(sizes)))])
            writer.writerows(rows)


def game_rows(table: str, n: int, result: SimulationResult) -> list[list]:
    """Rows of --per-game-csv for one table, in game order (agent 0 = the model,
    which sits at seat -g mod n in game g)."""
    rows = [[table, n, g, 0, -g % n, result.rounds[i], result.finished_rounds[0][i],
             *(result.scores[a][i] for a in range(n))]
            for i, g in enumerate(result.game_ids)]
    rows += [[table, n, g, 1, -g % n, "", "", *[""] * n] for g in result.blocked_ids]
    return sorted(rows, key=lambda row: row[2])


if __name__ == "__main__":
    main()
