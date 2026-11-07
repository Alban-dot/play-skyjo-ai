"""Styles of the AIs (skyjo/ui/styles.yaml): what the interface shows (name,
level, description) and the agent that plays each style.

The browser build (scripts/build_web_demo.py) ships them as styles.json, read
without PyYAML; styles.json is used when it sits next to this module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skyjo.agents.base import Agent
from skyjo.agents.registry import AGENTS, make_variant
from skyjo.human_play import MODES, model_agent

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

HERE = Path(__file__).resolve().parent
STYLES_FILE = HERE / "styles.json" if (HERE / "styles.json").is_file() else HERE / "styles.yaml"
LEVELS = ["Facile", "Moyen", "Difficile", "Expert"]


@dataclass(frozen=True)
class Style:
    id: str
    name: str
    level: str
    description: str
    agent: str
    agent_2p: str | None = None

    def spec(self, n_players: int) -> str:
        return self.agent_2p if n_players == 2 and self.agent_2p else self.agent

    def public(self) -> dict:
        """What the interface receives (not the agent)."""
        return {k: v for k, v in asdict(self).items() if k in ("id", "name", "level", "description")}


def check_spec(spec: str) -> None:
    kind, _, rest = spec.partition(":")
    if kind == "model":
        if rest not in MODES:
            raise ValueError(f"model:{rest} : mode inconnu ({', '.join(MODES)}).")
    elif kind not in AGENTS:
        raise ValueError(f"Agent inconnu : {spec!r}.")
    else:
        make_variant(spec, np.random.default_rng(0))  # parameters are checked by the constructor


def read_styles_file(path: Path) -> dict:
    """Raw content of a styles file: JSON, or YAML (PyYAML imported only then)."""
    with open(path, encoding="utf-8") as f:
        if Path(path).suffix == ".json":
            return json.load(f)
        import yaml

        return yaml.safe_load(f)


def load_styles(path: Path = STYLES_FILE) -> dict[str, Style]:
    """Styles by id, in the order of the file (.yaml, or .json); ValueError on an invalid file."""
    data = read_styles_file(path)
    styles = {}
    for item in data["styles"]:
        style = Style(**item)
        if style.id in styles:
            raise ValueError(f"Style en double : {style.id}.")
        if style.level not in LEVELS:
            raise ValueError(f"{style.id} : niveau {style.level!r} inconnu ({', '.join(LEVELS)}).")
        for spec in filter(None, (style.agent, style.agent_2p)):
            check_spec(spec)
        styles[style.id] = style
    if not styles:
        raise ValueError(f"Aucun style dans {path}.")
    return styles


def make_style_agent(style: Style, model: MaskablePPO, n_players: int, seed: int, seat: int) -> Agent:
    """The agent of ``style`` at a table of ``n_players``; bots draw from (seed, seat)."""
    spec = style.spec(n_players)
    if spec.startswith("model:"):
        return model_agent(model, spec.removeprefix("model:"), [seed, 2, seat])
    return make_variant(spec, np.random.default_rng([seed, 1, seat]))
