"""Record of human games in SQLite (skyjo.storage): kept apart from
skyjo.human_play so that playing does not need SQLite."""

from __future__ import annotations

from skyjo.game.engine import SkyjoGame
from skyjo.human_play import finished_game
from skyjo.records import RECORD_NOTE  # noqa: F401 - re-exported
from skyjo.storage import GameStore


def log_game(store: GameStore, mode: str, seed: int, model: str, game: SkyjoGame, columns: list[int],
             pseudo: str = "console") -> str:
    """Save a finished console game (no actions: it cannot be replayed); return its id."""
    return store.save_game({**finished_game(game, columns), "pseudo": pseudo, "styles": [mode] * (game.n_players - 1),
                            "seed": seed, "model": model, "source": "console"})
