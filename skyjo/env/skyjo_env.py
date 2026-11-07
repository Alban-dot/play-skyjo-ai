"""Gymnasium environment: one learning agent against configurable opponents.

- One step = one decision of the agent; opponents' moves are played
  automatically until the agent has to decide again or the game ends.
- One episode = one full game.
- At each reset: n_players is drawn from ``n_players`` (uniformly from a
  list, or by weight from a {n: weight} dict) and the agent's seat at
  random; all randomness (game, opponents) derives from the reset seed.
- With ``round_turn_limit`` (turns per player), a round still running after
  limit x n_players turns is ended by force between two turns: every card
  counted, nobody doubled, and the agent gets ``-forced_end_penalty`` on top
  of the round reward (whoever stalled: an equal penalty for all would cancel
  out in a score-gap reward). ``info["forced_round_ends"]`` counts them.
- With ``column_rule``, the agent must seize a safe column opportunity of
  positive gain: when there is one, the action mask holds only those cells
  (see skyjo.agents.column_rule). In any case, ``info`` counts the agent's
  placements, its positive-gain safe opportunities and those it seized.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from skyjo.agents.base import Agent
from skyjo.agents.column_rule import positive_safe_cells, rule_mask
from skyjo.agents.heuristic_agent import HeuristicAgent
from skyjo.env.actions import N_ACTIONS, PLACE, action_mask, apply_action
from skyjo.env.observation import OBS_SIZE, encode
from skyjo.env.rewards import RewardFn, ScoreDiffReward
from skyjo.game.constants import MAX_PLAYERS, MIN_PLAYERS
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.view import make_view

OpponentFactory = Callable[[np.random.Generator], Agent]
OBS_BOUND = 10.0  # every observation field stays well within [-10, 10]


class SkyjoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_players: int | Sequence[int] | dict[int, float] = 2,
        opponent_factory: OpponentFactory = HeuristicAgent,
        reward_fn: RewardFn | None = None,
        max_steps: int = 5_000,
        return_view: bool = False,
        round_turn_limit: int | None = None,
        forced_end_penalty: float = 0.0,
        column_rule: bool = False,
    ) -> None:
        choices = [n_players] if isinstance(n_players, int) else list(n_players)
        if not choices or any(not MIN_PLAYERS <= n <= MAX_PLAYERS for n in choices):
            raise ValueError(f"n_players must be in [{MIN_PLAYERS}, {MAX_PLAYERS}].")
        self.n_players_choices = [int(n) for n in choices]
        self.n_players_probs = None  # uniform
        if isinstance(n_players, dict):
            weights = np.array([n_players[n] for n in choices], dtype=float)
            if weights.min() < 0 or weights.sum() <= 0:
                raise ValueError("n_players weights must be non-negative with a positive sum.")
            self.n_players_probs = weights / weights.sum()
        self.opponent_factory = opponent_factory
        self.reward_fn = reward_fn if reward_fn is not None else ScoreDiffReward()
        self.max_steps = max_steps
        # tests and bot demonstrations only: adds the agent's PlayerView to info
        self.return_view = return_view
        self.round_turn_limit = round_turn_limit
        self.forced_end_penalty = forced_end_penalty
        self.column_rule = column_rule

        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.game: SkyjoGame | None = None
        self.seat = 0
        self.opponents: dict[int, Agent] = {}

    # --- gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        n = int(self.np_random.choice(self.n_players_choices, p=self.n_players_probs))
        self.seat = int(self.np_random.integers(n))
        root = np.random.SeedSequence(int(self.np_random.integers(2**63)))
        game_seq, *opp_seqs = root.spawn(n)

        self.game = SkyjoGame(n, np.random.default_rng(game_seq))
        others = [p for p in range(n) if p != self.seat]
        self.opponents = {}
        for player, seq in zip(others, opp_seqs):
            agent = self.opponent_factory(np.random.default_rng(seq))
            agent.reset()
            self.opponents[player] = agent

        self._round = self.game.round_number
        self._steps = 0
        self._round_scores: list[int] | None = None
        self._forced = 0  # rounds ended by force in this game
        self._forced_pending = False  # the round that just ended was forced
        self._finished = 0  # rounds ended by the agent revealing all its cards
        self._columns = [0, 0, 0]  # agent: placements, positive-gain safe opportunities, seized
        self._play_opponents()
        return self._obs(), self._info()

    def step(self, action: int):
        if self.game is None or self.game.is_over:
            raise RuntimeError("Call reset() before step().")
        self._steps += 1
        self._round_scores = None

        action = int(action)
        if self.game.phase in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
            self._count_columns(action)
        self._apply(action)  # raises ValueError if illegal
        reward = self._collect_rewards()
        reward += self._play_opponents()

        terminated = self.game.is_over
        truncated = not terminated and self._steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Legal actions for the agent (used by MaskablePPO)."""
        if self.game is None:
            return np.zeros(N_ACTIONS, dtype=bool)
        mask = action_mask(self.game)
        if self.column_rule and not self.game.is_over:
            mask = rule_mask(make_view(self.game, self.seat), mask)
        return mask

    # --- internals -------------------------------------------------------------

    def _count_columns(self, action: int) -> None:
        """Count a placement decision of the agent; under the rule, refuse an action
        outside the restricted mask."""
        cells = positive_safe_cells(make_view(self.game, self.seat), action_mask(self.game))
        self._columns[0] += 1
        if cells:
            self._columns[1] += 1
            self._columns[2] += action - PLACE in cells
            if self.column_rule and action - PLACE not in cells:
                raise ValueError(f"Action {action} breaks the column rule (cells {sorted(cells)}).")

    def _play_opponents(self) -> float:
        """Play opponents until the agent must decide or the game ends."""
        reward = 0.0
        game = self.game
        while not game.is_over and game.current_player != self.seat:
            player = game.current_player
            self._apply(self.opponents[player].act(make_view(game, player), action_mask(game)))
            reward += self._collect_rewards()  # a round may end on an opponent's move
        return reward

    def _apply(self, action: int) -> None:
        """Play an action, then end the round by force if it hit the turn limit."""
        game = self.game
        apply_action(game, action)
        if (self.round_turn_limit is not None and game.phase == Phase.CHOOSE_SOURCE
                and game.finisher is None
                and game.turns_played >= self.round_turn_limit * game.n_players):
            game.force_end_round()
            self._forced += 1
            self._forced_pending = True

    def _collect_rewards(self) -> float:
        """Reward for a round / game that ended on the last move, else 0."""
        game = self.game
        if not game.is_over and game.round_number == self._round:
            return 0.0
        self._round = game.round_number
        self._round_scores = list(game.last_round_scores)
        self._finished += game.last_round_finisher == self.seat
        reward = self.reward_fn.on_round_end(game.last_round_scores, self.seat)
        if self._forced_pending:
            reward -= self.forced_end_penalty
            self._forced_pending = False
        if game.is_over:
            reward += self.reward_fn.on_game_end(game.total_scores, self.seat)
        return reward

    def _obs(self) -> np.ndarray:
        return encode(make_view(self.game, self.seat))

    def _info(self) -> dict[str, Any]:
        game = self.game
        info: dict[str, Any] = {
            "seat": self.seat,
            "n_players": game.n_players,
            "round": game.round_number,
            "total_scores": list(game.total_scores),
            "round_scores": list(self._round_scores) if self._round_scores else None,
            "forced_round_ends": self._forced,
            "rounds_finished_by_agent": self._finished,
            "column_placements": self._columns[0],
            "column_opportunities": self._columns[1],
            "column_seized": self._columns[2],
            # pool_kind set by PoolFactory, else the agent's name
            "opponent_kinds": [getattr(a, "pool_kind", a.name) for a in self.opponents.values()],
        }
        if game.is_over:
            info["winners"] = game.winners()
            info["is_winner"] = self.seat in info["winners"]
        if self.return_view:
            info["view"] = make_view(game, self.seat)
        return info
