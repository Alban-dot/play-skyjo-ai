"""Column opportunities and the forced-seizure rule.

At a placement decision (card in hand, drawn or taken from the discard):

- safe opportunity: the card equals two visible cards of a column, whose third
  cell can receive it. Its theoretical gain is what the column stops costing:
  2 x the card value plus the value of the replaced cell (MEAN_CARD_VALUE,
  ~5.07, if hidden);
- bet opportunity: the card equals the only visible card of a column whose two
  other cells are hidden.

The column rule: a safe opportunity of positive gain is always seized.

- rule_mask restricts the legal actions to the positive-gain safe cells when
  there is one; ColumnRuleAgent plays with that mask. This is the rule of
  training (SkyjoEnv(column_rule=True)): the network still chooses among
  several such cells, and its recorded action is the action played.
- ForceSafeAgent (the diagnostic) lets the agent choose freely, then replaces
  its action by the best positive-gain safe cell if it did not take one.
"""

from __future__ import annotations

import numpy as np

from skyjo.agents.base import Agent
from skyjo.env.actions import PLACE
from skyjo.game.constants import MEAN_CARD_VALUE, N_COLS, N_ROWS
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState
from skyjo.game.view import PlayerView

PLACEMENT_PHASES = (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)


def opportunities(view: PlayerView) -> tuple[set[int], set[int]]:
    """Cells that complete a column (safe) and hidden cells that prepare one (bet)."""
    hand, values, states = view.hand, view.my_values, view.my_states
    safe, bet = set(), set()
    for col in range(N_COLS):
        cells = [col + row * N_COLS for row in range(N_ROWS)]
        if states[cells[0]] == CardState.REMOVED:
            continue
        equal = [p for p in cells if states[p] == CardState.VISIBLE and values[p] == hand]
        others = [p for p in cells if p not in equal]
        if len(equal) == 2:
            safe.add(others[0])
        elif len(equal) == 1 and all(states[p] == CardState.HIDDEN for p in others):
            bet.update(others)
    return safe, bet


def safe_gain(view: PlayerView, cell: int) -> float:
    """Points the column stops costing if the hand card completes it on ``cell``."""
    replaced = view.my_values[cell] if view.my_states[cell] == CardState.VISIBLE else MEAN_CARD_VALUE
    return 2 * view.hand + float(replaced)


def positive_safe_cells(view: PlayerView, action_mask: np.ndarray) -> dict[int, float]:
    """Legal safe cells of positive theoretical gain -> gain (empty outside placement phases)."""
    if view.phase not in PLACEMENT_PHASES:
        return {}
    gains = {p: safe_gain(view, p) for p in opportunities(view)[0] if action_mask[PLACE + p]}
    return {p: g for p, g in gains.items() if g > 0}


def rule_mask(view: PlayerView, action_mask: np.ndarray) -> np.ndarray:
    """``action_mask`` restricted to the positive-gain safe cells when there is one."""
    cells = positive_safe_cells(view, action_mask)
    if not cells:
        return action_mask
    only = np.zeros_like(action_mask)
    only[[PLACE + p for p in cells]] = True
    return only


class ColumnRuleAgent(Agent):
    """Plays as ``inner`` with the action mask restricted by rule_mask, as the
    learning agent of SkyjoEnv(column_rule=True)."""

    name = "rule"

    def __init__(self, inner: Agent) -> None:
        self.inner = inner

    def reset(self) -> None:
        self.inner.reset()

    def reseed(self, rng: np.random.Generator) -> None:
        self.inner.reseed(rng)

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        return self.inner.act(view, rule_mask(view, action_mask))


class ForceSafeAgent(Agent):
    """Plays as ``inner``, except that a safe opportunity of positive theoretical
    gain is always seized (on the best such cell). ``inner`` is always asked
    first, so a loop guard keeps its history.

    For reports, when a forced seizure had several candidate cells, the plain
    network (``inner`` as a PPOAgent, without its guard) is asked which one it
    prefers: among the cells of maximal gain (``tied``) and among all
    positive-gain cells (``several``), as [decisions, network preferred another]."""

    name = "forced"

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.forced = 0  # decisions overridden
        self.tied = [0, 0]
        self.several = [0, 0]

    def reset(self) -> None:
        self.inner.reset()

    def reseed(self, rng: np.random.Generator) -> None:
        self.inner.reseed(rng)

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        action = self.inner.act(view, action_mask)
        gains = positive_safe_cells(view, action_mask)
        positive = set(gains)
        if positive and action - PLACE not in positive:
            from skyjo.agents.ppo_agent import PPOAgent  # torch only needed here

            self.forced += 1
            chosen = max(positive, key=gains.get)
            best = {p for p in positive if gains[p] == gains[chosen]}
            for cells, counts in ((best, self.tied), (positive, self.several)):
                if len(cells) > 1 and isinstance(self.inner, PPOAgent):
                    only = np.zeros_like(action_mask)
                    only[[PLACE + p for p in cells]] = True
                    counts[0] += 1
                    counts[1] += PPOAgent.act(self.inner, view, only) != PLACE + chosen
            return PLACE + chosen
        return action
