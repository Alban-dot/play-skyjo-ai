"""Bots by name, shared by scripts, training and evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from skyjo.agents.base import Agent
from skyjo.agents.column_seeker import ColumnSeekerAgent
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent, HeuristicAgent
from skyjo.agents.random_agent import RandomAgent

AGENTS: dict[str, Callable[[np.random.Generator], Agent]] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "heuristic-gain": lambda rng: HeuristicAgent(rng, take_discard_for_gain=True),
    "cautious": CautiousHeuristicAgent,
    "seeker": ColumnSeekerAgent,
}


def make_agent(name: str, rng: np.random.Generator) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Known: {', '.join(AGENTS)}.")
    return AGENTS[name](rng)


def make_variant(spec: str, rng: np.random.Generator) -> Agent:
    """Bot from ``name`` or ``name:key=value,...`` (e.g. ``heuristic:min_gain=3``,
    ``cautious:max_refusals=3,take_discard_for_gain=true``, ``seeker:max_bet_value=8``); values
    are ints or booleans."""
    name, _, params = spec.partition(":")
    if not params:
        return make_agent(name, rng)
    classes = {"heuristic": HeuristicAgent, "cautious": CautiousHeuristicAgent, "seeker": ColumnSeekerAgent}
    if name not in classes:
        raise ValueError(f"Parameters are only supported for {', '.join(classes)}, not '{name}'.")
    kwargs = {}
    for item in params.split(","):
        key, _, value = item.partition("=")
        kwargs[key] = value.lower() == "true" if value.lower() in ("true", "false") else int(value)
    return classes[name](rng, **kwargs)
