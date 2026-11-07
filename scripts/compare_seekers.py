"""Table of ColumnSeekerAgent settings against a baseline subject (the cautious
bot), from analyze_opportunities.py runs on the same seed and tables.

Each label has, in the directory, <label>_n<N>.csv (per game), .rounds.csv (per
round) and .out (its report). For each label and number of players: wins, score
gap, paired differences of gap and wins against the baseline (95 % intervals),
completed columns per round, share of bets prepared (all bet opportunities,
from the report), bets tried per game by the seeking rule and the share whose
column was completed by the end of the round, and the cost of failed bets: on
the rounds with at least one bet whose column was not completed, the subject's
round score minus the baseline's on the same round of the same game (same
deal), with a 95 % interval from a bootstrap over games.

Usage: python scripts/compare_seekers.py <dir> --baseline cautious --labels s10_b6 s6_b4 ... --n 2 4
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_paired import compare, load  # noqa: E402
from compare_rounds import bootstrap  # noqa: E402

PREPARED = re.compile(r"^  total .*pari : ([\d.]+)/partie, préparée ([\d.]+)%", re.MULTILINE)


def load_rounds(path: str | Path) -> dict[tuple[int, int], dict]:
    """(game, round) -> {"score", "columns", "tried", "completed"}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(int(r["game"]), int(r["round"])): {
            "score": float(r["score_subject"]), "columns": float(r["columns_subject"]),
            "tried": int(r["bets_tried"]), "completed": int(r["bets_completed"])} for r in csv.DictReader(f)}


def failed_bet_cost(rounds: dict, baseline: dict) -> tuple[int, float, float, float]:
    """(rounds, mean, low, high) of subject - baseline round score on the rounds with a failed bet
    that the baseline also played."""
    keys = sorted(k for k, r in rounds.items() if r["tried"] > r["completed"] and k in baseline)
    if not keys:
        return 0, float("nan"), float("nan"), float("nan")
    diffs = np.array([rounds[k]["score"] - baseline[k]["score"] for k in keys])
    low, high = bootstrap(np.array([g for g, _ in keys]), diffs)
    return len(keys), float(diffs.mean()), low, high


def summary(folder: Path, label: str, baseline: str, n: int) -> dict:
    stem, base = folder / f"{label}_n{n}", folder / f"{baseline}_n{n}"
    tables, base_tables = load(f"{stem}.csv"), load(f"{base}.csv")
    table = next(iter(tables))
    r = compare(tables[table], base_tables[table])
    rounds, base_rounds = load_rounds(f"{stem}.rounds.csv"), load_rounds(f"{base}.rounds.csv")
    games = len({g for g, _ in rounds})
    tried = sum(v["tried"] for v in rounds.values())
    completed = sum(v["completed"] for v in rounds.values())
    match = PREPARED.search(Path(f"{stem}.out").read_text(encoding="utf-8"))
    return {"label": label, "n": n, "games": r["both"], "blocked": r["blocked_a"], "blocked_base": r["blocked_b"],
            "win": r["win_a"], "gap": r["gap_a"], "d_gap": r["gap"], "d_win": r["win"],
            "columns": float(np.mean([v["columns"] for v in rounds.values()])),
            "bet_opportunities": float(match.group(1)), "prepared": float(match.group(2)) / 100,
            "tried": tried / games, "success": completed / tried if tried else float("nan"),
            "cost": failed_bet_cost(rounds, base_rounds)}


def interval(d: dict, fmt: str) -> str:
    return f"{d['mean']:{fmt}} [{d['mean'] - d['paired']:{fmt}}, {d['mean'] + d['paired']:{fmt}}]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Table of seeker settings against a baseline subject.")
    parser.add_argument("folder")
    parser.add_argument("--baseline", default="cautious")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--n", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    for n in args.n:
        print(f"\n--- {n} joueurs (différences = réglage - {args.baseline}, mêmes donnes ; IC 95 %)")
        print(f"{'réglage':<14}{'victoires':>10}{'écart':>8}{'Δ écart':>22}{'Δ victoires':>26}{'col./manche':>12}"
              f"{'paris prép.':>12}{'tentés/p.':>10}{'aboutis':>9}   coût des paris ratés (manches)")
        for label in [args.baseline, *args.labels]:
            s = summary(folder, label, args.baseline, n)
            k, mean, low, high = s["cost"]
            cost = f"{mean:+.2f} [{low:+.2f}, {high:+.2f}] ({k})" if k else "-"
            blocked = f"  bloquées {s['blocked']} / {s['blocked_base']}" if s["blocked"] or s["blocked_base"] else ""
            success = "-" if np.isnan(s["success"]) else f"{s['success']:.1%}"
            print(f"{label:<14}{s['win']:>10.1%}{s['gap']:>+8.1f}{interval(s['d_gap'], '+.1f'):>22}"
                  f"{interval(s['d_win'], '+.1%'):>26}{s['columns']:>12.2f}{s['prepared']:>12.1%}"
                  f"{s['tried']:>10.2f}{success:>9}   {cost}{blocked}")
    print("\nécart = score total du sujet - moyenne des adversaires (négatif = meilleur) ; paris prép. = part des "
          "occasions de pari (carte = seule carte visible d'une colonne aux deux autres cases cachées) où la carte "
          "est posée sur une de ces cases, quelle que soit la règle qui l'y pose ; tentés/p. et aboutis = paris de la "
          "règle de recherche, par partie, et part dont la colonne est complétée à la fin de la manche ; coût = "
          "score de manche du réglage - celui de la référence, sur les manches avec au moins un pari raté")


if __name__ == "__main__":
    main()
