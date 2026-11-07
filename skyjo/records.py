"""Record of a human's finished games, whatever holds them (SQLite on the server,
skyjo.storage; the browser's storage in the demo, skyjo.ui.browser). NumPy only."""

from __future__ import annotations

import numpy as np

RECORD_NOTE = ("Victoires : une égalité à la première place compte comme une victoire ; scores et colonnes "
               "par partie (score IA = moyenne des IA) ; écart = ton score - score IA (négatif = tu fais mieux)")


def style_key(styles: list[str]) -> str:
    """Group of a game in the record: its style, or its styles joined if they differ."""
    return styles[0] if len(set(styles)) == 1 else " + ".join(styles)


def record(games: list[dict]) -> list[dict]:
    """Result of the human by (style, number of players), sorted (see RECORD_NOTE):
    wins, mean scores, gap and columns per game."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for g in games:
        groups.setdefault((style_key(g["styles"]), g["n_players"]), []).append(g)
    result = []
    for (style, n), group in sorted(groups.items()):
        mine = np.array([g["scores"][0] for g in group], dtype=float)
        ai = np.array([np.mean(g["scores"][1:]) for g in group])
        result.append({
            "mode": style, "n_players": n, "games": len(group), "wins": sum(g["human_won"] for g in group),
            "chance": 1 / n, "my_score": float(mine.mean()), "ai_score": float(ai.mean()),
            "gap": float((mine - ai).mean()),
            "my_columns": float(np.mean([g["columns"][0] for g in group])),
            "ai_columns": float(np.mean([np.mean(g["columns"][1:]) for g in group]))})
    return result
