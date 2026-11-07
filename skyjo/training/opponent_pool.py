"""Opponent factories for the training env.

- WeightedBotFactory: bots drawn by weight.
- PoolFactory (self-play): bots, fixed frozen models and recent snapshots of
  the learning agent, drawn by kind. Snapshots are written by SnapshotCallback
  into ``<pool dir>/recent`` (the last ``keep`` ones) and ``<pool dir>/kept``
  (one every ``keep_every`` steps, never deleted).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from skyjo.agents.base import Agent
from skyjo.agents.column_rule import ColumnRuleAgent
from skyjo.agents.registry import AGENTS, make_agent

BOTS = tuple(AGENTS)
NETWORK_KINDS = ("recent", "frozen")


def _check_weights(weights: dict[str, float]) -> tuple[list[str], list[float]]:
    if not weights or min(weights.values()) < 0 or sum(weights.values()) <= 0:
        raise ValueError("Opponent weights must be non-negative with a positive sum.")
    total = sum(weights.values())
    return list(weights), [w / total for w in weights.values()]


class WeightedBotFactory:
    """Called at each env reset, once per opponent seat: draws a bot by weight."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.names, self.probs = _check_weights(weights)

    def __call__(self, rng: np.random.Generator) -> Agent:
        name = self.names[int(rng.choice(len(self.names), p=self.probs))]
        return make_agent(name, rng)


class PoolFactory:
    """Draws an opponent kind by weight: a bot name, ``frozen`` (one of the
    fixed models, uniformly) or ``recent`` (the latest snapshot with
    probability ``latest_share``, otherwise another snapshot of the pool,
    uniformly). Network opponents sample their actions from their policy.

    The network opponents of the kinds in ``column_rule_kinds`` play with the
    column rule (ColumnRuleAgent), the others without.

    Each returned agent has a ``pool_kind`` attribute. Models are loaded once
    per process (the factory lives in each env subprocess) and dropped once
    their snapshot leaves the pool.
    """

    def __init__(self, weights: dict[str, float], pool_dir: str | Path,
                 frozen: list[str] | None = None, latest_share: float = 0.5,
                 column_rule_kinds: list[str] | None = None) -> None:
        self.kinds, self.probs = _check_weights(weights)
        unknown = set(self.kinds) - set(BOTS) - set(NETWORK_KINDS)
        if unknown:
            raise ValueError(f"Unknown opponent kinds: {sorted(unknown)}.")
        self.frozen = [str(p) for p in frozen or []]
        if "frozen" in self.kinds and not self.frozen:
            raise ValueError("Opponent kind 'frozen' needs at least one frozen model.")
        self.pool_dir = Path(pool_dir)
        self.latest_share = latest_share
        self.column_rule_kinds = set(column_rule_kinds or [])
        if self.column_rule_kinds - set(NETWORK_KINDS):
            raise ValueError(f"column_rule_kinds must be among {NETWORK_KINDS}.")
        self._models: dict[str, Any] = {}

    def __call__(self, rng: np.random.Generator) -> Agent:
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.probs))]
        if kind in BOTS:
            agent = make_agent(kind, rng)
        else:
            agent = self._network(self._pick_path(kind, rng))
            if kind in self.column_rule_kinds:
                agent = ColumnRuleAgent(agent)
        agent.pool_kind = kind
        return agent

    def snapshots(self) -> tuple[Path | None, list[Path]]:
        """(latest snapshot, the other snapshots of the pool)."""
        recent = sorted((self.pool_dir / "recent").glob("snap_*.zip"))
        kept = sorted((self.pool_dir / "kept").glob("snap_*.zip"))
        if not recent:
            return None, kept
        return recent[-1], recent[:-1] + kept

    def _pick_path(self, kind: str, rng: np.random.Generator) -> str:
        if kind == "frozen":
            return self.frozen[int(rng.integers(len(self.frozen)))]
        latest, others = self.snapshots()
        if latest is None:
            raise FileNotFoundError(f"No snapshot in {self.pool_dir / 'recent'}.")
        if others and rng.random() >= self.latest_share:
            return str(others[int(rng.integers(len(others)))])
        return str(latest)

    def _network(self, path: str) -> Agent:
        from skyjo.agents.ppo_agent import PPOAgent  # torch only needed here

        if path not in self._models:
            import torch as th

            th.set_num_threads(1)  # one env per process: no thread oversubscription
            live = set(self.frozen) | {str(p) for p in (self.pool_dir / "recent").glob("*.zip")} \
                | {str(p) for p in (self.pool_dir / "kept").glob("*.zip")}
            self._models = {p: m for p, m in self._models.items() if p in live}
            self._models[path] = PPOAgent.load(path, deterministic=False).model
        return PPOAgent(self._models[path], deterministic=False)


class SnapshotCallback(BaseCallback):
    """Every ``every`` steps, saves the policy into ``<pool_dir>/recent`` and
    keeps the last ``keep`` there; every ``keep_every`` steps, also into
    ``<pool_dir>/kept`` for good. Files are written then renamed, so that env
    processes never read a partial file."""

    def __init__(self, pool_dir: str | Path, every: int, keep: int = 5, keep_every: int | None = None) -> None:
        super().__init__()
        self.pool_dir = Path(pool_dir)
        self.every, self.keep, self.keep_every = every, keep, keep_every
        self._next = every

    def save_snapshot(self, model, steps: int) -> None:
        self._save(model, self.pool_dir / "recent", steps)
        if self.keep_every and steps > 0 and steps % self.keep_every == 0:
            self._save(model, self.pool_dir / "kept", steps)
        for old in sorted((self.pool_dir / "recent").glob("snap_*.zip"))[:-self.keep]:
            old.unlink()

    @staticmethod
    def _save(model, folder: Path, steps: int) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"tmp_{steps:012d}.zip"
        model.save(str(tmp))
        os.replace(tmp, folder / f"snap_{steps:012d}.zip")

    def _on_training_start(self) -> None:
        self._next = (self.num_timesteps // self.every + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self.save_snapshot(self.model, self._next)
            self._next += self.every
        return True
