"""Shared by the console (scripts/play_vs_ai.py) and the web UI (skyjo/ui): the
model's opponents by mode, completed columns and the result of a finished game.
NumPy only: the model is loaded elsewhere (PPOAgent.load) and the record of human
games, in SQLite, is in skyjo.human_record."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skyjo.agents.base import Agent
from skyjo.agents.column_rule import ForceSafeAgent
from skyjo.agents.ppo_agent import LoopGuardPPOAgent, PPOAgent
from skyjo.game.constants import N_ROWS
from skyjo.game.engine import SkyjoGame
from skyjo.game.player_state import CardState

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "models/frozen/ppo_multi_1200k_seed4.zip"
MODES = ["guard", "guard-forced", "det", "sample"]
HUMAN = 0
MAX_TABLE = 4


def model_agent(model: MaskablePPO, mode: str, seed: int | list[int]) -> Agent:
    """One opponent playing ``model`` in ``mode`` (sample: drawn with a NumPy generator seeded with ``seed``,
    one seed per seat so that two opponents do not draw the same numbers)."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}.")
    if mode == "sample":
        return PPOAgent(model, deterministic=False, numpy_rng=np.random.default_rng(seed))
    if mode == "guard":
        return LoopGuardPPOAgent(model)
    if mode == "guard-forced":
        return ForceSafeAgent(LoopGuardPPOAgent(model))
    return PPOAgent(model)


def agents_for(model: MaskablePPO, n_players: int, mode: str, seed: int) -> list[Agent]:
    """The n_players - 1 opponents of the human, all playing ``model`` in ``mode``."""
    return [model_agent(model, mode, [seed, seat]) for seat in range(1, n_players)]


def columns_removed(player) -> int:
    """Completed columns removed from a grid during the round (play and final reveal)."""
    return int(np.sum(player.states == CardState.REMOVED)) // N_ROWS


def finished_game(game: SkyjoGame, columns: list[int]) -> dict:
    """Result fields of a finished game for GameStore.save_game."""
    winners = game.winners()
    return {"n_players": game.n_players, "rounds": game.round_number, "scores": list(game.total_scores),
            "columns": list(columns), "winners": winners, "human_won": int(HUMAN in winners)}
