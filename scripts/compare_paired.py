"""Paired comparison of two models from their evaluate.py --per-game-csv files.

Both files must come from the same seed and tables (paired deals): game g of a
table then has exactly the same deals for both models. For each table, on the
games finished by both, the per-game differences A - B give a mean and a
confidence interval; the interval of an independent comparison (same games,
pairing ignored) is shown next to it to measure what the pairing gains.

Usage:
    python scripts/compare_paired.py a.csv b.csv [--names 1200k 300k]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

Z = 1.96


def load(path: str | Path) -> dict[str, dict[int, dict]]:
    """table -> game -> {"blocked", "win", "gap", "ended", "rounds"} (win/gap None if blocked)."""
    tables: dict[str, dict[int, dict]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            game = {"blocked": row["blocked"] == "1", "win": None, "gap": None}
            if not game["blocked"]:
                model = float(row["score_model"])
                opps = [float(v) for k, v in row.items() if k.startswith("score_opp") and v != ""]
                game.update(win=float(model <= min(opps)), gap=model - float(np.mean(opps)),
                            ended=int(row["model_rounds_ended"]), rounds=int(row["rounds"]))
            tables[row["table"]][int(row["game"])] = game
    return tables


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean of a - b with its paired and independent 95 % half-widths."""
    n = len(a)
    diff = a - b
    return {"mean": float(diff.mean()),
            "paired": Z * float(diff.std(ddof=1)) / math.sqrt(n),
            "independent": Z * math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / n)}


def compare(table_a: dict[int, dict], table_b: dict[int, dict]) -> dict:
    common = sorted(set(table_a) & set(table_b))
    both = [g for g in common if not table_a[g]["blocked"] and not table_b[g]["blocked"]]

    def column(table, key):
        return np.array([table[g][key] for g in both], dtype=float)

    def share(table):
        return sum(table[g]["ended"] for g in both) / sum(table[g]["rounds"] for g in both)

    return {"games": len(common), "both": len(both),
            "blocked_a": sum(table_a[g]["blocked"] for g in common),
            "blocked_b": sum(table_b[g]["blocked"] for g in common),
            "win_a": column(table_a, "win").mean(), "win_b": column(table_b, "win").mean(),
            "win": paired(column(table_a, "win"), column(table_b, "win")),
            "gap_a": column(table_a, "gap").mean(), "gap_b": column(table_b, "gap").mean(),
            "gap": paired(column(table_a, "gap"), column(table_b, "gap")),
            "ended_a": share(table_a), "ended_b": share(table_b),
            "ended": paired(column(table_a, "ended") / column(table_a, "rounds"),
                            column(table_b, "ended") / column(table_b, "rounds"))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two per-game CSV files.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--names", nargs=2, default=["A", "B"])
    args = parser.parse_args(argv)
    na, nb = args.names
    tables_a, tables_b = load(args.a), load(args.b)
    for table in [t for t in tables_a if t in tables_b]:
        r = compare(tables_a[table], tables_b[table])
        w, g = r["win"], r["gap"]
        print(f"--- {table} : {r['both']} parties terminées par les deux sur {r['games']} "
              f"(bloquées : {na} {r['blocked_a']}, {nb} {r['blocked_b']})")
        print(f"  victoires  {na} {r['win_a']:.1%}  {nb} {r['win_b']:.1%}  différence {w['mean']:+.1%} "
              f"[{w['mean'] - w['paired']:+.1%}, {w['mean'] + w['paired']:+.1%}] appariée, "
              f"±{w['independent']:.1%} indépendante (largeur x{w['paired'] / w['independent']:.2f})")
        print(f"  écart      {na} {r['gap_a']:+.1f}  {nb} {r['gap_b']:+.1f}  différence {g['mean']:+.1f} "
              f"[{g['mean'] - g['paired']:+.1f}, {g['mean'] + g['paired']:+.1f}] appariée, "
              f"±{g['independent']:.1f} indépendante (largeur x{g['paired'] / g['independent']:.2f})")
        e = r["ended"]
        print(f"  termine    {na} {r['ended_a']:.1%}  {nb} {r['ended_b']:.1%}  différence par partie "
              f"{e['mean']:+.1%} [{e['mean'] - e['paired']:+.1%}, {e['mean'] + e['paired']:+.1%}] appariée")
    print("\nDifférence = A - B par partie (écart négatif = A meilleur) ; indépendante = intervalle "
          "qu'on aurait sans apparier les mêmes parties ; largeur = appariée / indépendante")
    print("termine = part des manches terminées par le modèle sur toutes les parties ; sa différence est "
          "la moyenne, par partie, de l'écart des parts (manches terminées / manches jouées)")


if __name__ == "__main__":
    main()
