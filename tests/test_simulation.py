from collections import Counter

import numpy as np
import pytest

from skyjo.agents.base import Agent, GameBlocked, play_game
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent, HeuristicAgent
from skyjo.agents.random_agent import RandomAgent
from skyjo.game.constants import MEAN_CARD_VALUE
from skyjo.game.deck import full_deck
from skyjo.simulation import simulate, wilson_interval
from tests.helpers import StallingAgent, all_cards


def make_agents(n_players: int):
    agents = [CautiousHeuristicAgent(np.random.default_rng(0)), HeuristicAgent(np.random.default_rng(1))]
    agents += [RandomAgent(np.random.default_rng(i + 2)) for i in range(n_players - 2)]
    return agents


@pytest.mark.parametrize("n_players", [2, 4])
def test_150_cards_at_every_step_of_simulation(n_players):
    steps = 0

    def check(game):
        nonlocal steps
        steps += 1
        assert all_cards(game) == Counter(full_deck())

    result = simulate(make_agents(n_players), 30, np.random.default_rng(0), on_step=check)
    assert result.n_games == 30
    assert steps > 0


def test_seats_rotate_and_stats_consistent():
    result = simulate(make_agents(2), 10, np.random.default_rng(0))
    assert all(len(s) == 10 for s in result.scores)
    assert sum(result.wins) == 10 + result.ties


def test_same_seed_same_results():
    a = simulate(make_agents(2), 20, np.random.default_rng(5))
    b = simulate(make_agents(2), 20, np.random.default_rng(5))
    assert a.scores == b.scores


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and lo < 1.0


def test_cautious_vs_cautious_games_terminate():
    agents = [CautiousHeuristicAgent(np.random.default_rng(i)) for i in range(2)]
    result = simulate(agents, 20, np.random.default_rng(0))
    assert result.n_games == 20


def test_play_game_max_steps_guard():
    class Stubborn(Agent):
        def act(self, view, action_mask):
            return int(np.flatnonzero(action_mask)[0])

    with pytest.raises(RuntimeError):
        play_game([Stubborn(), Stubborn()], np.random.default_rng(0), max_steps=10)


def test_score_diff_interval():
    result = simulate(make_agents(2), 30, np.random.default_rng(0))
    mean, lo, hi = result.score_diff(0, 1)
    assert lo <= mean <= hi


def test_blocked_games_are_counted_and_excluded():
    agents = [StallingAgent(np.random.default_rng(0)), StallingAgent(np.random.default_rng(1))]
    with pytest.raises(GameBlocked):
        play_game(agents, np.random.default_rng(0), max_steps=500)
    result = simulate(agents, 3, np.random.default_rng(0), max_steps=500)
    assert (result.n_games, result.blocked, result.ties) == (0, 3, 0)
    assert result.wins == [0, 0] and result.scores == [[], []]
    mixed = simulate([agents[0], HeuristicAgent(np.random.default_rng(2))], 6,
                     np.random.default_rng(0), max_steps=5000)
    assert mixed.blocked == 0 and mixed.n_games == 6


def test_score_gap_is_score_minus_mean_of_others():
    result = simulate(make_agents(3), 12, np.random.default_rng(0))
    scores = np.array(result.scores, dtype=float)
    expected = (scores[1] - (scores[0] + scores[2]) / 2).mean()
    mean, lo, hi = result.score_gap(1)
    assert mean == pytest.approx(expected) and lo <= mean <= hi
    two = simulate(make_agents(2), 12, np.random.default_rng(0))
    assert two.score_gap(0) == pytest.approx(two.score_diff(0, 1))


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_diagnose_round_shares_add_up(n_players):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose.py"
    spec = importlib.util.spec_from_file_location("diagnose", path)
    diagnose = importlib.util.module_from_spec(spec)
    sys.modules["diagnose"] = diagnose  # needed by its dataclass
    spec.loader.exec_module(diagnose)
    mine, others = diagnose.diagnose("heuristic", 6, 0, "cautious", n_players=n_players)
    assert others.rounds == (n_players - 1) * mine.rounds  # base figures are per player
    # every round is ended by exactly one player
    assert mine.ended + others.ended == mine.rounds


# --- paired deals --------------------------------------------------------------

def record_deals(agents, n_games, deal_seed, max_steps=None):
    """(game, round) -> deal (every grid by seat, discard, draw pile), seen at the
    first step of each round; and the result."""
    deals, games = {}, []

    def on_step(game):
        if not games or games[-1] is not game:
            games.append(game)
        key = (len(games) - 1, game.round_number)
        if key not in deals and not game.is_over:
            deals[key] = (tuple(tuple(p.values) for p in game.players),
                          tuple(game.deck.discard_pile), tuple(game.deck.draw_pile))

    result = simulate(agents, n_games, np.random.default_rng(123), on_step=on_step,
                      max_steps=max_steps, deal_seed=deal_seed)
    return deals, result


@pytest.mark.parametrize("n_players", [2, 4])
def test_paired_deals_are_the_same_for_different_agents(n_players):
    rng = np.random.default_rng
    a = [RandomAgent(rng(1))] + [HeuristicAgent(rng(i + 2)) for i in range(n_players - 1)]
    b = [CautiousHeuristicAgent(rng(9))] + [HeuristicAgent(rng(i + 20)) for i in range(n_players - 1)]
    deals_a, result_a = record_deals(a, 25, deal_seed=7)
    deals_b, result_b = record_deals(b, 25, deal_seed=7)
    common = deals_a.keys() & deals_b.keys()
    assert {g for g, _ in common} == set(range(25))
    assert all(deals_a[k] == deals_b[k] for k in common)
    # the check is not trivial: the games last a different number of rounds...
    assert result_a.rounds != result_b.rounds
    # ...and another seed changes the deals
    deals_c, _ = record_deals(b, 25, deal_seed=8)
    assert deals_c[(0, 1)] != deals_b[(0, 1)]


def test_paired_bots_are_reseeded_per_game_and_seat():
    def run():
        agents = [HeuristicAgent(np.random.default_rng(1)), RandomAgent(np.random.default_rng(2))]
        return simulate(agents, 10, np.random.default_rng(0), deal_seed=3)

    first, second = run(), run()
    assert first.scores == second.scores
    # the same agents, whatever generator they were built with, play the same games
    other = simulate([HeuristicAgent(np.random.default_rng(50)), RandomAgent(np.random.default_rng(60))],
                     10, np.random.default_rng(70), deal_seed=3)
    assert other.scores == first.scores


@pytest.mark.parametrize("deal_seed", [None, 4])
def test_finished_rounds_add_up_to_rounds(deal_seed):
    result = simulate(make_agents(3), 12, np.random.default_rng(0), deal_seed=deal_seed)
    assert result.game_ids == list(range(12)) and result.blocked_ids == []
    for i in range(12):  # every round is ended by exactly one player
        assert sum(result.finished_rounds[a][i] for a in range(3)) == result.rounds[i]
    assert sum(result.finish_share(a) for a in range(3)) == pytest.approx(1.0)


def test_blocked_game_ids_are_listed_apart():
    agents = [StallingAgent(np.random.default_rng(0)), StallingAgent(np.random.default_rng(1))]
    mixed = simulate([agents[0], HeuristicAgent(np.random.default_rng(2))], 4,
                     np.random.default_rng(0), max_steps=5000, deal_seed=1)
    assert mixed.game_ids == [0, 1, 2, 3] and mixed.blocked_ids == []
    blocked = simulate(agents, 3, np.random.default_rng(0), max_steps=500, deal_seed=1)
    assert blocked.blocked_ids == [0, 1, 2] and blocked.game_ids == []


def load_script(name):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n_players", [2, 3])
def test_per_game_rows_put_the_model_score_first(n_players):
    evaluate = load_script("evaluate")
    agents = make_agents(n_players)
    result = simulate(agents, 7, np.random.default_rng(0), deal_seed=2)
    rows = evaluate.game_rows("t", n_players, result)
    assert [r[2] for r in rows] == list(range(7))
    for i, row in enumerate(rows):
        assert row[7] == result.scores[0][i]  # score_model is agent 0's
        assert row[6] == result.finished_rounds[0][i]


def test_paired_comparison_by_hand(tmp_path):
    compare_paired = load_script("compare_paired")
    header = "table,n_players,game,blocked,model_seat,rounds,model_rounds_ended,score_model,score_opp1\n"
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text(header + "t,2,0,0,0,2,1,10,20\nt,2,1,0,1,3,2,30,20\nt,2,2,0,0,2,1,15,25\nt,2,3,1,1,,,,\n")
    b.write_text(header + "t,2,0,0,0,2,0,20,10\nt,2,1,0,1,3,1,30,20\nt,2,2,1,0,,,,\nt,2,3,0,1,2,1,5,6\n")
    r = compare_paired.compare(compare_paired.load(a)["t"], compare_paired.load(b)["t"])
    assert (r["games"], r["both"], r["blocked_a"], r["blocked_b"]) == (4, 2, 1, 1)
    # games 0 and 1: A wins then loses, B loses then loses; gaps A -10, +10 / B +10, +10
    assert r["win"]["mean"] == pytest.approx(0.5)
    assert r["gap"]["mean"] == pytest.approx(-10.0)
    assert r["ended_a"] == pytest.approx(3 / 5) and r["ended_b"] == pytest.approx(1 / 5)
    # per-game ended shares: A 1/2, 2/3 / B 0/2, 1/3
    assert r["ended"]["mean"] == pytest.approx((1 / 2 + 1 / 3) / 2)


def test_column_opportunities_by_hand():
    import dataclasses

    from skyjo.game.engine import SkyjoGame
    from skyjo.game.view import make_view
    from tests.helpers import rig

    analyze = load_script("analyze_opportunities")
    game = SkyjoGame(2, np.random.default_rng(0))
    # columns are positions (c, c+4, c+8)
    grid = [[5, 5, 7, 1],
            [5, 9, 7, 2],
            [0, 9, 3, 5]]
    # col 0: 5,5 visible, third (pos 8) visible 0 -> safe at 8
    # col 1: 5 visible at pos 1, pos 5 and 9 hidden -> bet at 5 and 9
    # col 2: 7,7 visible (not the hand card), col 3: 5 at pos 11 visible but pos 3 visible -> nothing
    rig(game, [grid, grid], [[0, 4, 8, 1, 2, 6, 3, 11], []])
    view = dataclasses.replace(make_view(game, 0), hand=5)
    safe, bet = analyze.opportunities(view)
    assert safe == {8} and bet == {5, 9}
    view = dataclasses.replace(view, hand=7)
    assert analyze.opportunities(view) == ({10}, set())
    # completing col 2 with a 7 on the hidden pos 10: 2 x 7 + mean hidden card
    assert analyze.safe_gain(view, 10) == pytest.approx(14 + MEAN_CARD_VALUE)
    # col 0 with a -2 on the visible 0: 2 x -2 + 0 -> no gain
    view = dataclasses.replace(make_view(game, 0), hand=-2)
    assert analyze.safe_gain(view, 8) == pytest.approx(-4)


class _Fixed(Agent):
    """Always plays ``action`` (enough for one placement decision)."""

    def __init__(self, action: int) -> None:
        self.action = action

    def act(self, view, action_mask):
        return self.action


def test_force_safe_overrides_only_positive_safe_opportunities():
    import dataclasses

    from skyjo.env.actions import FLIP
    from skyjo.game.engine import Phase, SkyjoGame
    from skyjo.game.view import make_view
    from tests.helpers import rig

    analyze = load_script("analyze_opportunities")
    game = SkyjoGame(2, np.random.default_rng(0))
    grid = [[5, 5, 7, 1],
            [5, 9, 7, 2],
            [0, 9, 3, 5]]
    rig(game, [grid, grid], [[0, 4, 8, 1, 2, 6, 3, 11], []])
    mask = np.ones(26, dtype=bool)
    base = dataclasses.replace(make_view(game, 0), phase=Phase.PLACE_DRAWN)
    # 7 in hand: completing col 2 on hidden pos 10 gains 14 + 5.07 -> forced from a discard (FLIP + 3)
    agent = analyze.ForceSafeAgent(_Fixed(FLIP + 3))
    assert agent.act(dataclasses.replace(base, hand=7), mask) == 10 and agent.forced == 1
    # 5 in hand: col 0 third cell holds a visible 0 -> gain 10 > 0, forced too
    assert agent.act(dataclasses.replace(base, hand=5), mask) == 8 and agent.forced == 2
    # hand equal to nothing useful: inner action kept; same outside placement phases
    assert agent.act(dataclasses.replace(base, hand=12), mask) == FLIP + 3
    assert agent.act(dataclasses.replace(base, hand=7, phase=Phase.CHOOSE_SOURCE), mask) == FLIP + 3
    assert agent.forced == 2
    # two visible -2 over a visible 0: gain 2 x -2 + 0 < 0 -> not forced
    rig(game, [[[-2, 5, 7, 1], [-2, 9, 7, 2], [0, 9, 3, 5]]] * 2, [[0, 4, 8], []])
    view = dataclasses.replace(make_view(game, 0), phase=Phase.PLACE_DISCARD, hand=-2)
    assert analyze.opportunities(view)[0] == {8}
    assert agent.act(view, mask) == FLIP + 3
    assert agent.forced == 2


@pytest.mark.parametrize("n_players", [2, 3])
def test_per_round_csv_matches_per_game_csv(n_players, tmp_path, monkeypatch):
    import csv

    analyze = load_script("analyze_opportunities")
    monkeypatch.setattr(analyze, "make_subject", lambda name: CautiousHeuristicAgent(np.random.default_rng(0)))
    games, rounds = tmp_path / "g.csv", tmp_path / "r.csv"
    analyze.main(["cautious", str(n_players), "6", "3", "--per-game-csv", str(games), "--per-round-csv", str(rounds)])
    with open(games, newline="", encoding="utf-8") as f:
        per_game = list(csv.DictReader(f))
    with open(rounds, newline="", encoding="utf-8") as f:
        per_round = list(csv.DictReader(f))
    for g in per_game:
        mine = [r for r in per_round if r["game"] == g["game"]]
        assert len(mine) == int(g["rounds"])
        assert sum(int(r["subject_ended"]) for r in mine) == int(g["model_rounds_ended"])
        assert all(int(r["subject_seat"]) == int(g["model_seat"]) for r in mine)
        # the subject's counted round scores add up to its game total
        assert sum(int(r["score_subject"]) for r in mine) == int(g["score_model"])


def test_round_comparison_pairs_rounds_not_ended(tmp_path):
    compare_rounds = load_script("compare_rounds")
    header = "game,round,subject_seat,finisher,subject_ended,score_subject,raw_subject,columns_subject,score_opps,columns_opps\n"
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text(header + "0,1,0,1,0,10,10,1,20,0\n0,2,0,0,1,5,5,0,15,0\n1,1,1,0,0,8,8,0,12,1\n")
    b.write_text(header + "0,1,0,1,0,14,14,0,18,0\n0,2,0,1,0,9,9,0,10,0\n1,1,1,1,1,4,4,0,30,0\n")
    ra, rb = compare_rounds.load(a), compare_rounds.load(b)
    assert ra[0, 1]["gap"] == -10 and ra[0, 2]["ended"] and not rb[0, 2]["ended"]
    common = [k for k in set(ra) & set(rb) if not ra[k]["ended"] and not rb[k]["ended"]]
    assert common == [(0, 1)]  # (0, 2) ended by A, (1, 1) ended by B
    lo, hi = compare_rounds.bootstrap(np.array([0, 0, 1]), np.array([1.0, 3.0, 5.0]))
    assert 2.0 <= lo <= hi <= 5.0  # game means 2 and 5, round-weighted means between 2 and 5


def test_seeker_bets_per_round_and_settings_table(tmp_path, capsys):
    import contextlib
    import csv

    analyze, seekers = load_script("analyze_opportunities"), load_script("compare_seekers")
    for subject, label in (("cautious", "cautious"), ("seeker:seek_until_turn=12,max_bet_value=12", "s12")):
        stem = tmp_path / f"{label}_n2"
        with open(f"{stem}.out", "w", encoding="utf-8") as out, contextlib.redirect_stdout(out):
            analyze.main([subject, "2", "8", "4", "--per-game-csv", f"{stem}.csv",
                          "--per-round-csv", f"{stem}.rounds.csv"])
    with open(tmp_path / "s12_n2.rounds.csv", newline="", encoding="utf-8") as f:
        rounds = list(csv.DictReader(f))
    tried = [int(r["bets_tried"]) for r in rounds]
    completed = [int(r["bets_completed"]) for r in rounds]
    assert sum(tried) > 0 and all(c <= t for c, t in zip(completed, tried))
    # a completed bet is one of the subject's removed columns
    assert all(c <= int(r["columns_subject"]) for c, r in zip(completed, rounds))

    seekers.main([str(tmp_path), "--labels", "s12", "--n", "2"])
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("cautious") for line in lines) and any(line.startswith("s12") for line in lines)

    # failed-bet cost: subject - baseline on the rounds with a failed bet, played by both
    mine = {(0, 1): {"score": 10.0, "tried": 2, "completed": 1}, (0, 2): {"score": 4.0, "tried": 1, "completed": 1},
            (1, 1): {"score": 7.0, "tried": 1, "completed": 0}, (1, 3): {"score": 9.0, "tried": 1, "completed": 0}}
    base = {(0, 1): {"score": 6.0}, (0, 2): {"score": 9.0}, (1, 1): {"score": 9.0}}
    k, mean, low, high = seekers.failed_bet_cost(mine, base)
    assert (k, mean) == (2, 1.0) and low <= mean <= high  # (10 - 6 + 7 - 9) / 2; (1, 3) unpaired


def test_rule_mask_keeps_only_positive_gain_safe_cells():
    import dataclasses

    from skyjo.agents.column_rule import ColumnRuleAgent, positive_safe_cells, rule_mask
    from skyjo.game.engine import Phase, SkyjoGame
    from skyjo.game.view import make_view
    from tests.helpers import rig

    game = SkyjoGame(2, np.random.default_rng(0))
    # col 0: two visible -2 over a visible 0 (gain -4); col 2: two visible 7 over a hidden cell
    rig(game, [[[-2, 5, 7, 1], [-2, 9, 7, 2], [0, 9, 3, 5]]] * 2, [[0, 4, 8, 2, 6], []])
    mask = np.ones(26, dtype=bool)
    base = dataclasses.replace(make_view(game, 0), phase=Phase.PLACE_DRAWN)
    seven = dataclasses.replace(base, hand=7)
    assert positive_safe_cells(seven, mask) == {10: pytest.approx(14 + MEAN_CARD_VALUE)}
    assert np.flatnonzero(rule_mask(seven, mask)).tolist() == [10]
    minus_two = dataclasses.replace(base, hand=-2)  # safe at 8 but gain -4: mask unchanged
    assert rule_mask(minus_two, mask) is mask
    choose = dataclasses.replace(seven, phase=Phase.CHOOSE_SOURCE)
    assert positive_safe_cells(choose, mask) == {} and rule_mask(choose, mask) is mask
    illegal = mask.copy()
    illegal[10] = False  # the cell cannot receive the card: no opportunity
    assert rule_mask(seven, illegal) is illegal

    seen = []

    class Spy(Agent):
        def act(self, view, action_mask):
            seen.append(action_mask)
            return int(np.flatnonzero(action_mask)[-1])

    agent = ColumnRuleAgent(Spy())
    assert agent.act(seven, mask) == 10
    assert agent.act(minus_two, mask) == 25  # last legal action, untouched mask
    assert seen[1] is mask
