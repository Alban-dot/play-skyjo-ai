"""Simple rule-based agent (rules 1-4, see the HeuristicAgent docstring)."""

from __future__ import annotations

import numpy as np

from skyjo.agents.base import Agent
from skyjo.env.actions import FLIP, DRAW, TAKE_DISCARD
from skyjo.game.constants import MEAN_CARD_VALUE, N_COLS
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState, PlayerState
from skyjo.game.view import PlayerView

COMPLETE, PAIR, REPLACE, HIDDEN = "complete", "pair", "replace", "hidden"


class HeuristicAgent(Agent):
    """Rule-based player.

    Placement options for a card, tried in order:
      1. complete a column (2 visible cards equal to it)          - rule 4
      2. pair it in a column (1 visible equal card), if the
         replaced cell's expected value is >= the card            - rule 1
      3. replace the highest visible card, if gain >= min_gain    - rule 2
      4. replace a hidden card, if card <= hidden_threshold       - rule 2
    Source: take the discard if it allows 1 or 2, or if it is
    <= discard_threshold and has a placement; otherwise draw      - rule 1
    Drawn card without placement: discard it and flip a card     - rule 3

    With ``take_discard_for_gain``, the discard is also taken when it can
    replace a visible card with a gain >= min_gain (variant of rule 1).

    After each ``act``, ``equivalent`` lists the actions the bot was
    indifferent between (random draw or tie broken by order); it contains
    the chosen action.
    """

    name = "heuristic"

    def __init__(
        self,
        rng: np.random.Generator,
        discard_threshold: int = 3,
        hidden_threshold: int = 4,
        min_gain: int = 2,
        take_discard_for_gain: bool = False,
    ) -> None:
        self.rng = rng
        self.discard_threshold = discard_threshold
        self.hidden_threshold = hidden_threshold
        self.min_gain = min_gain
        self.take_discard_for_gain = take_discard_for_gain
        self.equivalent: list[int] = []

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        action, self.equivalent = self._decide(view, action_mask)
        return action

    def _decide(self, view: PlayerView, action_mask: np.ndarray) -> tuple[int, list[int]]:
        """Return the action and the actions it was equivalent to."""
        if view.phase == Phase.SETUP_FLIP:
            options = np.flatnonzero(action_mask)
            return int(self.rng.choice(options)), options.tolist()

        if view.phase == Phase.CHOOSE_SOURCE:
            source = TAKE_DISCARD if action_mask[TAKE_DISCARD] and self._wants_discard(view) else DRAW
            return source, [source]

        placement = self._placement(view, view.hand)
        if placement is not None:
            return placement[1], placement[2]
        flips = np.flatnonzero(action_mask[FLIP:DRAW])
        if view.phase == Phase.PLACE_DRAWN and flips.size:
            return FLIP + int(self.rng.choice(flips)), (FLIP + flips).tolist()
        return self._least_bad_position(view, view.hand)

    # --- decisions -----------------------------------------------------------

    def _wants_discard(self, view: PlayerView) -> bool:
        card = view.top_discard
        placement = self.choose_placement(view, card)
        return placement is not None and self._worth_taking(placement[0], card)

    def _worth_taking(self, kind: str, card: int) -> bool:
        """Take the discard ``card`` whose best placement is of this kind."""
        if self.take_discard_for_gain and kind == REPLACE:
            return True
        return kind in (COMPLETE, PAIR) or card <= self.discard_threshold

    def choose_placement(self, view: PlayerView, card: int) -> tuple[str, int] | None:
        """Return (kind, position) for ``card`` on my grid, or None."""
        placement = self._placement(view, card)
        return None if placement is None else placement[:2]

    def _placement(self, view: PlayerView, card: int) -> tuple[str, int, list[int]] | None:
        """(kind, position, equivalent positions) for ``card``, or None."""
        return self._column_placement(view, card) or self._grid_placement(view, card)

    def _column_placement(self, view: PlayerView, card: int) -> tuple[str, int, list[int]] | None:
        """Complete a column (rule 4), else pair the card in a column (rule 1), or None."""
        values, states = view.my_values, view.my_states
        expected = self._expected_values(view)

        pair: tuple[int, float] | None = None
        pair_options: list[tuple[int, float]] = []  # (position, gain) of each eligible column
        for col in range(N_COLS):
            cells = PlayerState.column_positions(col)
            if states[cells[0]] == CardState.REMOVED:
                continue
            matches = [p for p in cells if states[p] == CardState.VISIBLE and values[p] == card]
            others = [p for p in cells if p not in matches]
            if not matches or not others:
                continue
            target = max(others, key=lambda p: expected[p])
            ties = [p for p in others if expected[p] == expected[target]]
            if len(matches) == 2:
                return COMPLETE, target, ties
            gain = expected[target] - card
            if gain >= 0:
                pair_options += [(p, gain) for p in ties]
                if pair is None or gain > pair[1]:
                    pair = (target, gain)
        if pair is not None:
            return PAIR, pair[0], [p for p, gain in pair_options if gain == pair[1]]
        return None

    def _grid_placement(self, view: PlayerView, card: int,
                        exclude: frozenset[int] = frozenset()) -> tuple[str, int, list[int]] | None:
        """Replace the highest visible card or a hidden card (rule 2), outside the
        ``exclude`` cells, or None."""
        values, states = view.my_values, view.my_states
        allowed = ~np.isin(np.arange(len(states)), list(exclude))
        visible = np.flatnonzero((states == CardState.VISIBLE) & allowed)
        if visible.size:
            highest = int(visible[np.argmax(values[visible])])
            if values[highest] - card >= self.min_gain:
                return REPLACE, highest, visible[values[visible] == values[highest]].tolist()

        hidden = np.flatnonzero((states == CardState.HIDDEN) & allowed)
        if hidden.size and card <= self.hidden_threshold:
            return HIDDEN, int(self.rng.choice(hidden)), hidden.tolist()
        return None

    def _least_bad_position(self, view: PlayerView, card: int) -> tuple[int, list[int]]:
        """Forced placement: replace the cell with the highest expected value."""
        expected = self._expected_values(view)
        active = np.flatnonzero(view.my_states != CardState.REMOVED)
        best = expected[active].max()
        return int(active[np.argmax(expected[active])]), active[expected[active] == best].tolist()

    @staticmethod
    def _expected_values(view: PlayerView) -> np.ndarray:
        """Known value for visible cards, mean card value for hidden ones."""
        return np.where(
            view.my_states == CardState.VISIBLE,
            view.my_values.astype(float),
            MEAN_CARD_VALUE,
        )


def estimated_score(values: np.ndarray, states: np.ndarray) -> float:
    """Expected round score of a grid: hidden cards count as the mean card,
    fully visible columns of equal cards count as removed."""
    total = 0.0
    for col in range(N_COLS):
        cells = PlayerState.column_positions(col)
        if states[cells[0]] == CardState.REMOVED:
            continue
        visible = all(states[p] == CardState.VISIBLE for p in cells)
        if visible and len({int(values[p]) for p in cells}) == 1:
            continue
        total += sum(
            float(values[p]) if states[p] == CardState.VISIBLE else MEAN_CARD_VALUE
            for p in cells
        )
    return total


class CautiousHeuristicAgent(HeuristicAgent):
    """HeuristicAgent + rule 5: do not reveal my last hidden card (which ends
    the round) unless my estimated score is strictly the lowest or <= 0.

    Instead, the card is placed on my highest visible card. After
    ``max_refusals`` refusals in a round, finishing is allowed again
    (prevents endless rounds between cautious players).
    """

    name = "cautious"

    def __init__(self, rng: np.random.Generator, max_refusals: int = 10, **kwargs) -> None:
        super().__init__(rng, **kwargs)
        self.max_refusals = max_refusals
        self.reset()

    def reset(self) -> None:
        self._round = 0
        self._refusals = 0
        self.refused = False  # rule 5 applied on the last decision

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        self.refused = False
        action = super().act(view, action_mask)
        if view.phase not in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
            return action
        if view.round_number != self._round:
            self._round, self._refusals = view.round_number, 0
        if self._refusals >= self.max_refusals or not self.wants_to_refuse(view, action):
            return action

        visible = np.flatnonzero(view.my_states == CardState.VISIBLE)
        self._refusals += 1
        self.refused = True
        values = view.my_values[visible]
        self.equivalent = visible[values == values.max()].tolist()
        return int(visible[np.argmax(values)])

    def refusals_exhausted(self, view: PlayerView) -> bool:
        """True if no refusal is left in the round of ``view`` (hidden state,
        absent from the observation)."""
        return view.round_number == self._round and self._refusals >= self.max_refusals

    def wants_to_refuse(self, view: PlayerView, action: int) -> bool:
        """True if rule 5 applies to ``action``, ignoring the refusal limit."""
        return (view.finisher is None
                and self.ends_round(view, action)
                and not self._safe_to_finish(view, action)
                and bool(np.any(view.my_states == CardState.VISIBLE)))

    @staticmethod
    def ends_round(view: PlayerView, action: int) -> bool:
        """The action reveals my last hidden card."""
        target = action - FLIP if FLIP <= action < DRAW else action
        hidden = np.flatnonzero(view.my_states == CardState.HIDDEN)
        return hidden.size == 1 and target == hidden[0]

    @staticmethod
    def _safe_to_finish(view: PlayerView, action: int) -> bool:
        values, states = view.my_values.copy(), view.my_states.copy()
        if action < FLIP:  # placement: the final grid is fully known
            values[action] = view.hand
            states[action] = CardState.VISIBLE
        mine = estimated_score(values, states)  # a flipped card stays "mean"
        if mine <= 0:
            return True
        others = [
            estimated_score(view.values[i], view.states[i])
            for i in range(view.n_players) if i != view.player
        ]
        return mine < min(others)
