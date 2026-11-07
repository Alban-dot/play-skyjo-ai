"""Round-level comparison of two subjects from analyze_opportunities.py --per-round-csv
files (same seed: round r of game g has the same deal for both).

By whether the subject ended the round: mean round score of the subject (as
counted), its completed columns, the mean score of its opponents, and the gap
subject - opponents. Two views of the rounds not ended by the subject:

- each side on its own rounds (the selection depends on the subject's play);
- paired: rounds (g, r) played by both and ended by neither subject, with the
  mean difference A - B and a 95 % interval from a bootstrap over games (the
  rounds of a game are not independent).

Usage: python scripts/compare_rounds.py a.rounds.csv b.rounds.csv [--names forced guard]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

METRICS = {"score": "score du sujet", "columns": "colonnes du sujet", "opps": "score des adversaires",
           "gap": "écart sujet - adv."}


def load(path: str | Path) -> dict[tuple[int, int], dict]:
    """(game, round) -> {"ended", "score", "columns", "opps", "gap"}."""
    rounds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            score, opps = float(r["score_subject"]), float(r["score_opps"])
            rounds[int(r["game"]), int(r["round"])] = {
                "ended": r["subject_ended"] == "1", "score": score, "columns": float(r["columns_subject"]),
                "opps": opps, "gap": score - opps}
    return rounds


def means(rounds: list[dict]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rounds])) for k in METRICS}


def bootstrap(games: np.ndarray, diffs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the mean of ``diffs`` (one per round), resampling whole games."""
    ids, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=diffs)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Round-level comparison of two per-round CSV files.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--names", nargs=2, default=["A", "B"])
    args = parser.parse_args(argv)
    na, nb = args.names
    a, b = load(args.a), load(args.b)

    print(f"{'':<34}" + "".join(f"{label:>24}" for label in METRICS.values()))
    for ended, label in ((False, "non terminées par le sujet"), (True, "terminées par le sujet")):
        for name, rounds in ((na, a), (nb, b)):
            subset = [r for r in rounds.values() if r["ended"] == ended]
            m = means(subset)
            share = len(subset) / len(rounds)
            print(f"{name:<8}{label:<26}" + "".join(f"{m[k]:>24.2f}" for k in METRICS)
                  + f"   ({len(subset)} manches, {share:.1%})")

    common = sorted(k for k in set(a) & set(b) if not a[k]["ended"] and not b[k]["ended"])
    games = np.array([g for g, _ in common])
    print(f"\nApparié : {len(common)} manches (même partie, même manche) non terminées par le sujet "
          f"des deux côtés, sur {len(set(games))} parties")
    for k, label in METRICS.items():
        va = np.array([a[key][k] for key in common])
        vb = np.array([b[key][k] for key in common])
        lo, hi = bootstrap(games, va - vb)
        print(f"  {label:<24} {na} {va.mean():>7.2f}  {nb} {vb.mean():>7.2f}  "
              f"différence {(va - vb).mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
    print("\nScores de manche tels que comptés (doublement compris) ; adversaires = moyenne des "
          "adversaires ; intervalle à 95 % par bootstrap sur les parties")


if __name__ == "__main__":
    main()
