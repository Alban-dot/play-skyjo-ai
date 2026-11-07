from collections import Counter
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from skyjo.agents.base import Agent
from skyjo.agents.heuristic_agent import HeuristicAgent
from skyjo.agents.random_agent import RandomAgent
from skyjo.env.actions import DRAW, FLIP, N_ACTIONS
from skyjo.env.rewards import ScoreDiffReward
from skyjo.env.skyjo_env import SkyjoEnv
from skyjo.game.engine import Phase
from skyjo.game.view import make_view
from skyjo.simulation import simulate, wilson_interval
from tests.helpers import rig
from tests.test_observation import scramble_hidden


def masked_random_action(env: SkyjoEnv, rng) -> int:
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def play_episode(env: SkyjoEnv, seed: int, policy):
    """Play one episode; return (rewards, final info, number of steps)."""
    obs, info = env.reset(seed=seed)
    rewards = []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(env, obs, info))
        rewards.append(reward)
        if terminated or truncated:
            return rewards, info, len(rewards)


# --- gymnasium compliance ----------------------------------------------------

class UnmaskedSafeEnv(SkyjoEnv):
    """check_env samples actions without a mask: map illegal ones to a legal one.
    Only used for check_env; the real env raises on illegal actions."""

    def step(self, action):
        legal = np.flatnonzero(self.action_masks())
        if not self.action_masks()[action]:
            action = int(legal[int(action) % len(legal)])
        return super().step(action)


@pytest.mark.parametrize("n_players", [2, (2, 3, 4)])
def test_env_checker(n_players):
    check_env(UnmaskedSafeEnv(n_players=n_players), skip_render_check=True)


def test_illegal_action_raises():
    env = SkyjoEnv()
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        SkyjoEnv().step(DRAW)


# --- 1000 games with masked random actions -----------------------------------

@pytest.mark.slow
def test_1000_games_with_masked_random_actions():
    env = SkyjoEnv(n_players=(2, 3, 4), opponent_factory=RandomAgent)
    rng = np.random.default_rng(0)
    seen_n, seen_seats = set(), set()
    for episode in range(1000):
        obs, info = env.reset(seed=episode)
        seen_n.add(info["n_players"])
        seen_seats.add(info["seat"])
        while True:
            mask = env.action_masks()
            assert mask.any() and env.observation_space.contains(obs)
            obs, _, terminated, truncated, info = env.step(masked_random_action(env, rng))
            assert not truncated
            if terminated:
                break
        assert max(info["total_scores"]) >= 100
        assert not env.action_masks().any()
    assert seen_n == {2, 3, 4}
    assert seen_seats == {0, 1, 2, 3}


# --- reproducibility ---------------------------------------------------------

def trajectory(seed: int, n_players=(2, 3, 4)):
    env = SkyjoEnv(n_players=n_players, opponent_factory=RandomAgent)
    rng = np.random.default_rng(123)  # same agent decisions
    obs, info = env.reset(seed=seed)
    observations, rewards = [obs], []
    while True:
        obs, reward, terminated, truncated, info = env.step(masked_random_action(env, rng))
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            return observations, rewards, info


def test_same_seed_same_episode():
    obs_a, rew_a, info_a = trajectory(7)
    obs_b, rew_b, info_b = trajectory(7)
    assert len(obs_a) == len(obs_b)
    assert all(np.array_equal(a, b) for a, b in zip(obs_a, obs_b))
    assert rew_a == rew_b
    assert info_a["total_scores"] == info_b["total_scores"]


def test_different_seed_different_episode():
    obs_a, _, _ = trajectory(7)
    obs_b, _, _ = trajectory(8)
    assert not np.array_equal(obs_a[0], obs_b[0])


# --- rewards -------------------------------------------------------------------

def test_reward_function_by_hand():
    r = ScoreDiffReward()
    assert r.on_round_end([30, 12], me=1) == pytest.approx((30 - 12) / 20)
    assert r.on_round_end([30, 12], me=0) == pytest.approx((12 - 30) / 20)
    assert r.on_round_end([10, 20, 40], me=0) == pytest.approx((30 - 10) / 20)
    assert r.on_game_end([80, 105], me=0) == 1.0  # sole winner
    assert r.on_game_end([80, 105], me=1) == -1.0
    assert r.on_game_end([80, 80, 120], me=0) == 0.0  # shared win
    assert r.on_game_end([80, 80, 120], me=2) == -1.0
    assert r.on_game_end([50, 60, 101], me=1) == -1.0  # 2nd of 3 = not a win
    custom = ScoreDiffReward(scale=10, win_bonus=3)
    assert custom.on_round_end([30, 12], me=1) == pytest.approx(1.8)
    assert custom.on_game_end([80, 105], me=0) == 3.0


def expected_bonus(total_scores, me, win_bonus=1.0):
    best = min(total_scores)
    winners = [i for i, s in enumerate(total_scores) if s == best]
    if me not in winners:
        return -win_bonus
    return win_bonus if len(winners) == 1 else 0.0


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_reward_sum_is_exact_score_gap_plus_bonus(n_players):
    env = SkyjoEnv(n_players=n_players, opponent_factory=RandomAgent)
    rng = np.random.default_rng(1)
    for seed in range(30):
        rewards, info, _ = play_episode(env, seed, lambda e, o, i: masked_random_action(e, rng))
        totals, me = info["total_scores"], info["seat"]
        gap = np.mean([t for i, t in enumerate(totals) if i != me]) - totals[me]
        expected = gap / 20 + expected_bonus(totals, me)
        assert math.fsum(rewards) == pytest.approx(expected, rel=0, abs=1e-9)


# --- round ending during an opponent's turn ---------------------------------

class ScriptedOpponent(Agent):
    """Always draws, then places the card at position 0."""

    def act(self, view, action_mask):
        if view.phase == Phase.SETUP_FLIP:
            return int(np.flatnonzero(action_mask)[0])
        return DRAW if action_mask[DRAW] else 0


ME = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # sum 42
OPP = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]  # sum 6


def rigged_env(my_visible, opp_visible, draw, totals=(0, 0)):
    env = SkyjoEnv(opponent_factory=lambda rng: ScriptedOpponent())
    env.reset(seed=0)
    me, opp = env.seat, 1 - env.seat
    grids, visible = [None, None], [None, None]
    grids[me], grids[opp] = ME, OPP
    visible[me], visible[opp] = my_visible, opp_visible
    rig(env.game, grids, visible, draw=[5] * 10 + draw, discard=[4], current=me)
    env.game.total_scores = [0, 0]
    env.game.total_scores[me], env.game.total_scores[opp] = totals
    return env, me


def test_round_ending_on_opponent_turn_keeps_reward():
    # I reveal my last card; the opponent plays its last turn and the round ends
    # inside my step. Me: 42, not lowest -> 84. Opponent: 0 -> 2 at pos 0 -> 8.
    env, me = rigged_env(list(range(11)), [], draw=[2, 12])
    _, r1, *_ = env.step(DRAW)  # I draw the 12
    _, r2, terminated, _, info = env.step(FLIP + 11)  # discard it, flip my last card
    assert r1 == 0.0
    assert r2 == pytest.approx((8 - 84) / 20)  # -3.8
    assert info["round_scores"][me] == 84 and info["round_scores"][1 - me] == 8
    assert not terminated and info["round"] == 2


def test_game_ending_on_opponent_turn_keeps_reward_and_bonus():
    env, me = rigged_env(list(range(11)), [], draw=[2, 12], totals=(20, 0))
    env.step(DRAW)
    _, reward, terminated, _, info = env.step(FLIP + 11)
    assert terminated and info["total_scores"][me] == 104
    assert reward == pytest.approx((8 - 84) / 20 - 1.0)  # -4.8
    assert info["is_winner"] is False


def test_round_ending_on_my_move_after_opponent_finished():
    # Opponent reveals its last card (pos 0) on its turn; I play my last turn.
    # Me: 42 - 4 - 4 = 34. Opponent finisher with 8, strictly lowest: not doubled.
    env, me = rigged_env(list(range(11)), list(range(1, 12)), draw=[1, 2, 0])
    assert env.step(DRAW)[1] == 0.0  # I draw a 0
    _, r, _, _, info = env.step(3)  # replace my 4; the opponent then finishes
    assert r == 0.0 and info["round"] == 1
    env.step(DRAW)  # I draw a 1
    _, r, _, _, info = env.step(7)  # replace my 5 -> the round ends on my move
    assert r == pytest.approx((8 - 34) / 20)  # -1.3
    assert info["round"] == 2


# --- optional PlayerView in info ---------------------------------------------

def views_equal(a, b) -> bool:
    for field in a.__dataclass_fields__:
        x, y = getattr(a, field), getattr(b, field)
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True


def test_view_absent_by_default():
    _, info = SkyjoEnv().reset(seed=0)
    assert "view" not in info


def test_view_in_info_does_not_leak():
    env = SkyjoEnv(n_players=(2, 4), opponent_factory=RandomAgent, return_view=True)
    rng, scramble_rng = np.random.default_rng(0), np.random.default_rng(1)
    n_checked = 0
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        while True:
            twin = scramble_hidden(env.game, env.seat, scramble_rng)
            assert views_equal(info["view"], make_view(twin, env.seat))
            n_checked += 1
            obs, _, terminated, truncated, info = env.step(masked_random_action(env, rng))
            if terminated or truncated:
                break
    assert n_checked > 100


# --- integration: env vs simulate.py -----------------------------------------

def env_stats(agent: Agent, opponent_factory, n_games: int, seed: int):
    """Agent plays through the env; returns (wins, agent scores, opponent scores)."""
    env = SkyjoEnv(opponent_factory=opponent_factory, return_view=True)
    wins, mine, theirs = 0, [], []
    for g in range(n_games):
        agent.reset()
        _, info, _ = play_episode(
            env, seed + g, lambda e, o, i: agent.act(i["view"], e.action_masks()))
        me = info["seat"]
        wins += info["is_winner"]
        mine.append(info["total_scores"][me])
        theirs.append(info["total_scores"][1 - me])
    return wins, np.array(mine), np.array(theirs)


def assert_same_mean(a, b, n_sigma=4.0):
    """Means of two independent samples agree within n_sigma standard errors."""
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    assert abs(np.mean(a) - np.mean(b)) <= n_sigma * se, (np.mean(a), np.mean(b), se)


@pytest.mark.slow
def test_integration_random_in_env_vs_heuristic():
    n = 200
    wins, mine, theirs = env_stats(RandomAgent(np.random.default_rng(0)), HeuristicAgent, n, 0)
    ref = simulate([RandomAgent(np.random.default_rng(1)), HeuristicAgent(np.random.default_rng(2))],
                   n, np.random.default_rng(3))
    lo, hi = wilson_interval(ref.wins[0], n)
    assert wins / n <= hi + 0.02
    assert_same_mean(mine, np.array(ref.scores[0]))
    assert_same_mean(theirs, np.array(ref.scores[1]))


@pytest.mark.slow
def test_integration_heuristic_in_env_vs_heuristic():
    n = 200
    wins, mine, theirs = env_stats(HeuristicAgent(np.random.default_rng(0)), HeuristicAgent, n, 0)
    lo, hi = wilson_interval(wins, n, z=2.58)  # 99 % interval
    assert lo <= 0.5 <= hi
    ref = simulate([HeuristicAgent(np.random.default_rng(1)), HeuristicAgent(np.random.default_rng(2))],
                   n, np.random.default_rng(3))
    assert_same_mean(mine, np.array(ref.scores[0]))
    assert_same_mean(theirs, np.array(ref.scores[1]))


# --- forced end of rounds that never end ----------------------------------

@pytest.mark.parametrize("n_players", [2, 3])
def test_forced_round_ends_keep_the_reward_sum_exact(n_players):
    from tests.helpers import StallingAgent

    limit, penalty = 10, 0.7
    env = SkyjoEnv(n_players=n_players, opponent_factory=StallingAgent,
                   round_turn_limit=limit, forced_end_penalty=penalty)
    me = StallingAgent(np.random.default_rng(0))
    for seed in range(5):
        turns_seen = []

        def policy(e, obs, info):
            turns_seen.append(e.game.turns_played)
            return me.act(make_view(e.game, e.seat), e.action_masks())

        rewards, info, _ = play_episode(env, seed, policy)
        totals, seat, forced = info["total_scores"], info["seat"], info["forced_round_ends"]
        assert forced == info["round"]  # every round stalled, so every round was forced
        assert max(turns_seen) < limit * n_players  # never past the limit
        gap = np.mean([t for i, t in enumerate(totals) if i != seat]) - totals[seat]
        expected = gap / 20 + expected_bonus(totals, seat) - penalty * forced
        assert math.fsum(rewards) == pytest.approx(expected, rel=0, abs=1e-9)


def test_no_forced_end_without_limit_or_in_normal_play():
    env = SkyjoEnv(opponent_factory=HeuristicAgent, round_turn_limit=50, forced_end_penalty=0.7)
    rng = np.random.default_rng(3)
    for seed in range(5):
        rewards, info, _ = play_episode(env, seed, lambda e, o, i: masked_random_action(e, rng))
        assert info["forced_round_ends"] == 0


def test_rounds_finished_by_agent_and_opponent_kinds_in_info():
    from tests.helpers import StallingAgent

    # stalling opponents never finish: the agent (heuristic) finishes every round
    env = SkyjoEnv(opponent_factory=StallingAgent)
    me = HeuristicAgent(np.random.default_rng(0))
    _, info, _ = play_episode(env, 0, lambda e, o, i: me.act(make_view(e.game, e.seat), e.action_masks()))
    assert info["rounds_finished_by_agent"] == info["round"] and info["forced_round_ends"] == 0
    assert info["opponent_kinds"] == ["agent"]  # no pool_kind: the agent's name
    # stalling on both sides with a turn limit: no round finished by the agent
    env = SkyjoEnv(opponent_factory=StallingAgent, round_turn_limit=10)
    stall = StallingAgent(np.random.default_rng(0))
    _, info, _ = play_episode(env, 0, lambda e, o, i: stall.act(make_view(e.game, e.seat), e.action_masks()))
    assert info["rounds_finished_by_agent"] == 0 and info["forced_round_ends"] == info["round"]


def test_n_players_drawn_by_weight():
    env = SkyjoEnv(n_players={2: 0.2, 3: 0.4, 4: 0.4}, opponent_factory=RandomAgent)
    counts = Counter(env.reset(seed=s)[1]["n_players"] for s in range(3000))
    for n, weight in ((2, 0.2), (3, 0.4), (4, 0.4)):
        assert counts[n] / 3000 == pytest.approx(weight, abs=0.03)
    with pytest.raises(ValueError):
        SkyjoEnv(n_players={2: -1.0, 3: 1.0})


# --- column rule ------------------------------------------------------------------

@pytest.mark.parametrize("column_rule", [False, True])
def test_column_rule_restricts_the_mask_and_counts_opportunities(column_rule):
    from skyjo.agents.column_rule import positive_safe_cells
    from skyjo.env.actions import PLACE, action_mask

    env = SkyjoEnv(n_players=(2, 3, 4), opponent_factory=HeuristicAgent, column_rule=column_rule)
    rng = np.random.default_rng(0)
    totals, restricted = np.zeros(3, dtype=int), 0
    for seed in range(30):
        env.reset(seed=seed)
        while True:
            mask = env.action_masks()
            cells = positive_safe_cells(make_view(env.game, env.seat), action_mask(env.game))
            if column_rule and cells:
                assert set(np.flatnonzero(mask)) == {PLACE + p for p in cells}
            else:
                assert np.array_equal(mask, action_mask(env.game))
            restricted += bool(cells)
            _, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(mask))))
            if terminated or truncated:
                break
        totals += (info["column_placements"], info["column_opportunities"], info["column_seized"])
    placements, opportunities, seized = totals
    assert 0 < opportunities == restricted < placements
    if column_rule:
        assert seized == opportunities
    else:
        assert seized < opportunities  # random choices miss most of them


def test_column_rule_refuses_an_action_outside_the_rule():
    from skyjo.env.actions import action_mask

    env = SkyjoEnv(n_players=2, opponent_factory=HeuristicAgent, column_rule=True)
    rng = np.random.default_rng(1)
    for seed in range(20):
        env.reset(seed=seed)
        while not env.game.is_over:
            mask, legal = env.action_masks(), action_mask(env.game)
            if not np.array_equal(mask, legal):
                other = int(np.flatnonzero(legal & ~mask)[0])
                with pytest.raises(ValueError, match="column rule"):
                    env.step(other)
                return
            env.step(int(rng.choice(np.flatnonzero(mask))))
    pytest.fail("no column rule opportunity met")
