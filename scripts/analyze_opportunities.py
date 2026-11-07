"""How often a subject seizes column opportunities, against N-1 cautious bots
with paired deals. At each placement decision (card in hand, drawn or taken):

- safe opportunity: the card equals two visible cards of a column, whose third
  cell can receive it -> seized if the card is placed on that third cell. Its
  theoretical gain is what the column stops costing: 2 x the card value plus
  the value of the replaced cell (MEAN_CARD_VALUE, ~5.07, if hidden); with
  several safe cells, the best one counts;
- bet opportunity: the card equals the only visible card of a column whose two
  other cells are hidden -> prepared if the card is placed on one of those two
  hidden cells. Otherwise: completes another column, other hidden cell, other
  visible cell, or discards it (drawn card only).

Subjects: 1200k_guard, 1200k_forced (1200k_guard that always seizes a safe
opportunity of positive gain, a diagnostic), cautious, heuristic, a model
.zip path (played with the loop guard), or rule:<model .zip path> (the same,
under the column rule mask, as the rule run was trained); bots take parameters
(e.g. seeker:seek_until_turn=6,max_bet_value=8, see registry.make_variant).

Opponents (--opponents): N-1 cautious bots (default) or N-1 copies of the 1.2M
model played with the loop guard (1200k_guard).

Usage: python scripts/analyze_opportunities.py <subject> <n_players> <n_games> <seed> [--per-game-csv out.csv]
           [--per-round-csv out.rounds.csv] [--opponents cautious|1200k_guard]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.agents.base import Agent  # noqa: E402
from skyjo.agents.column_seeker import ColumnSeekerAgent  # noqa: E402
from skyjo.agents.column_rule import ColumnRuleAgent, ForceSafeAgent, opportunities, safe_gain  # noqa: E402,F401
from skyjo.agents.ppo_agent import LoopGuardPPOAgent, PPOAgent  # noqa: E402
from skyjo.agents.registry import make_variant  # noqa: E402
from skyjo.env.actions import FLIP, PLACE  # noqa: E402
from skyjo.game.constants import MAX_CARD_VALUE, MEAN_CARD_VALUE, MIN_CARD_VALUE, N_ROWS  # noqa: E402
from skyjo.game.engine import Phase  # noqa: E402
from skyjo.game.player_state import CardState  # noqa: E402
from skyjo.game.view import PlayerView  # noqa: E402
from skyjo.simulation import simulate, wilson_interval  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "models" / "frozen" / "ppo_multi_1200k_seed4.zip"
SOURCES = {Phase.PLACE_DRAWN: "pioche", Phase.PLACE_DISCARD: "défausse"}
BET_OUTCOMES = ("autre case visible", "autre case cachée", "défausse", "complète une colonne sûre")


class Recorder(Agent):
    """Plays as ``inner`` and classifies each placement decision."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.safe: list[tuple[str, int, bool, float]] = []  # (source, hand, seized, best gain)
        self.bet: Counter = Counter()  # (source, hand, outcome) -> decisions

    def reset(self) -> None:
        self.inner.reset()

    def reseed(self, rng: np.random.Generator) -> None:
        self.inner.reseed(rng)

    def act(self, view: PlayerView, action_mask: np.ndarray) -> int:
        action = self.inner.act(view, action_mask)
        if view.phase not in SOURCES:
            return action
        source = SOURCES[view.phase]
        safe, bet = opportunities(view)
        placed = action - PLACE if action < FLIP else None
        if safe:
            self.safe.append((source, view.hand, placed in safe, max(safe_gain(view, p) for p in safe)))
        if bet:
            if placed in bet:
                outcome = "prépare"
            elif placed is None:
                outcome = "défausse"
            elif placed in safe:
                outcome = "complète une colonne sûre"
            elif view.my_states[placed] == CardState.HIDDEN:
                outcome = "autre case cachée"
            else:
                outcome = "autre case visible"
            self.bet[source, view.hand, outcome] += 1
        return action


def share(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson_interval(k, n)
    lo, hi = max(lo, 0.0), min(hi, 1.0)
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def make_subject(name: str) -> Agent:
    if name.partition(":")[0] in ("cautious", "heuristic", "seeker"):
        return make_variant(name, np.random.default_rng(0))  # reseeded per game and seat
    path = name.removeprefix("rule:")
    guard = LoopGuardPPOAgent(PPOAgent.load(MODEL if path.startswith("1200k") else path).model)
    if name == "1200k_forced":
        return ForceSafeAgent(guard)
    return ColumnRuleAgent(guard) if name.startswith("rule:") else guard


def print_by_source(subject: Recorder, n_games: int) -> None:
    for source in (None, "pioche", "défausse"):
        label = "total" if source is None else f"carte de la {source}"
        safe = [r for r in subject.safe if source in (None, r[0])]
        seized = sum(r[2] for r in safe)
        outcomes = Counter()
        for (s, _, outcome), v in subject.bet.items():
            if source in (None, s):
                outcomes[outcome] += v
        n_bet = sum(outcomes.values())
        rest = n_bet - outcomes["prépare"]
        print(f"  {label:<20} sûre : {len(safe) / n_games:.2f}/partie, saisie {share(seized, len(safe))}"
              f"  |  pari : {n_bet / n_games:.2f}/partie, préparée {share(outcomes['prépare'], n_bet)}")
        if rest:
            detail = ", ".join(f"{o} {outcomes[o] / rest:.0%}" for o in BET_OUTCOMES if outcomes[o])
            print(f"  {'':<20} pari non préparé ({rest}) : {detail}")


def print_by_value(subject: Recorder) -> None:
    print("  occasions sûres par valeur de la carte en main (gain théorique = 2 x valeur + case remplacée, "
          f"{MEAN_CARD_VALUE:.2f} si cachée) :")
    print(f"    {'valeur':>6} {'occasions':>10} {'saisie':>24} {'gain moyen':>11} {'gain > 0':>9}")
    rows = [(str(v), [r for r in subject.safe if r[1] == v]) for v in range(MIN_CARD_VALUE, MAX_CARD_VALUE + 1)]
    rows += [("gain>0", [r for r in subject.safe if r[3] > 0]), ("gain<=0", [r for r in subject.safe if r[3] <= 0])]
    for label, safe in rows:
        if safe:
            gains = np.array([r[3] for r in safe])
            print(f"    {label:>6} {len(safe):>10} {share(sum(r[2] for r in safe), len(safe)):>24} "
                  f"{gains.mean():>+11.2f} {np.mean(gains > 0):>9.0%}")
    print("  paris par valeur de la carte en main :")
    print(f"    {'valeur':>6} {'occasions':>10} {'préparée':>24}")
    for v in range(MIN_CARD_VALUE, MAX_CARD_VALUE + 1):
        n_bet = sum(c for (_, h, _), c in subject.bet.items() if h == v)
        prepared = sum(c for (_, h, o), c in subject.bet.items() if h == v and o == "prépare")
        if n_bet:
            print(f"    {v:>6} {n_bet:>10} {share(prepared, n_bet):>24}")


ROUND_FIELDS = ["game", "round", "subject_seat", "finisher", "subject_ended", "score_subject",
                "raw_subject", "columns_subject", "score_opps", "columns_opps", "bets_tried", "bets_completed"]


def round_tracker(n: int, rounds: list[list], seeker: ColumnSeekerAgent | None = None):
    """on_step callback appending one ROUND_FIELDS row per finished round (subject = agent 0,
    at seat -g mod n in game g; scores as counted, doubling included; opps = their mean).
    With ``seeker`` (the subject), its bets of the round and those whose column was
    removed by the end of the round (final reveal included)."""
    state = {"game": None, "index": -1, "ended": 0}

    def on_step(game) -> None:
        if game is not state["game"]:
            state.update(game=game, index=state["index"] + 1, ended=0)
            if seeker is not None:
                seeker.bets.clear()  # bets of a blocked game
        ended = game.round_number - (0 if game.is_over else 1)
        if ended > state["ended"]:
            state["ended"] = ended
            g, seat = state["index"], -state["index"] % n
            cols = [int(np.sum(p.states == CardState.REMOVED)) // N_ROWS for p in game.last_round_players]
            scores = game.last_round_scores
            opps = [s for s in range(n) if s != seat]
            bets = [col for r, col in seeker.bets if r == ended] if seeker is not None else []
            states = game.last_round_players[seat].states
            completed = sum(states[col] == CardState.REMOVED for col in bets)
            if seeker is not None:
                seeker.bets[:] = [b for b in seeker.bets if b[0] != ended]
            rounds.append([g, ended, seat, game.last_round_finisher, int(game.last_round_finisher == seat),
                           scores[seat], game.last_round_players[seat].score(), cols[seat],
                           float(np.mean([scores[s] for s in opps])), float(np.mean([cols[s] for s in opps])),
                           len(bets), completed])

    return on_step


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject")
    parser.add_argument("n_players", type=int)
    parser.add_argument("n_games", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--per-game-csv", help="per-game results in the evaluate.py format")
    parser.add_argument("--per-round-csv", help="one row per round: finisher, scores, columns and bets")
    parser.add_argument("--opponents", choices=["cautious", "1200k_guard"], default="cautious")
    args = parser.parse_args(argv)
    n = args.n_players
    inner = make_subject(args.subject)
    subject = Recorder(inner)
    if args.opponents == "cautious":
        bots = [make_variant("cautious", np.random.default_rng(i + 1)) for i in range(n - 1)]
    else:
        model = PPOAgent.load(MODEL).model
        bots = [LoopGuardPPOAgent(model) for _ in range(n - 1)]  # one loop memory per seat
    seeker = inner if isinstance(inner, ColumnSeekerAgent) else None
    rounds = []
    result = simulate([subject, *bots], args.n_games, np.random.default_rng(args.seed), max_steps=2000,
                      deal_seed=args.seed, on_step=round_tracker(n, rounds, seeker))

    opponents = "prudent(s)" if args.opponents == "cautious" else "1,2M guard"
    print(f"{args.subject} contre {n - 1} {opponents}, {n} joueurs, graine {args.seed}, {args.n_games} parties "
          f"(bloquées {result.blocked}, comptées), victoires {result.winrate(0):.1%}, "
          f"manches terminées par le sujet {result.finish_share(0):.1%}")
    if isinstance(inner, ForceSafeAgent):
        print(f"  saisies forcées : {inner.forced} ({inner.forced / args.n_games:.2f} par partie)")
        for label, (k, m) in (("plusieurs cases au gain maximal", inner.tied),
                              ("plusieurs cases à gain positif", inner.several)):
            print(f"    {label} : {k} ({k / max(inner.forced, 1):.1%} des saisies forcées), "
                  f"le réseau en préférait une autre : {m} ({m / max(k, 1):.1%})")
    print_by_source(subject, args.n_games)
    print_by_value(subject)
    if args.per_game_csv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evaluate import game_rows

        with open(args.per_game_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["table", "n_players", "game", "blocked", "model_seat", "rounds",
                             "model_rounds_ended", "score_model", *(f"score_opp{i}" for i in range(1, n))])
            writer.writerows(game_rows(args.opponents, n, result))
    if args.per_round_csv:
        blocked = set(result.blocked_ids)
        with open(args.per_round_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ROUND_FIELDS)
            writer.writerows(r for r in rounds if r[0] not in blocked)


if __name__ == "__main__":
    main()
