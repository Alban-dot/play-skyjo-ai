from pathlib import Path

import numpy as np
import pytest

from skyjo.game.engine import Phase
from skyjo.training.imitation import Demos, collect_demos
from skyjo.training.train import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ENV_CFG = load_config(CONFIGS / "two_players_b.yaml")["env"]


@pytest.fixture(scope="module")
def demos():
    return collect_demos(ENV_CFG, n_games=6, seed=3)


def test_rows_are_consistent(demos):
    n = len(demos.actions)
    for name in ("obs", "masks", "rewards", "game", "phase", "capped", "cap_decisive"):
        assert len(getattr(demos, name)) == n, name
    assert demos.obs.shape[1] == 628
    assert set(np.unique(demos.game)) == set(range(6))
    assert len(demos.won) == 6


def test_actions_are_legal(demos):
    assert demos.masks[np.arange(len(demos.actions)), demos.actions].all()


def test_rewards_sum_to_game_result(demos):
    # ScoreDiffReward summed over a 2-player game: (opp total - my total) / 20 + bonus
    returns = np.bincount(demos.game, weights=demos.rewards)
    bonus = np.sign(demos.opp_total - demos.my_total)  # +1 sole win, 0 shared, -1 loss
    expected = (demos.opp_total - demos.my_total) / 20 + bonus
    assert np.allclose(returns, expected, atol=1e-4)
    assert np.array_equal(demos.won, demos.my_total <= demos.opp_total)


def test_refusal_flags_only_on_placements(demos):
    place = np.isin(demos.phase, [Phase.PLACE_DRAWN, Phase.PLACE_DISCARD])
    assert not demos.capped[~place].any()
    assert not (demos.cap_decisive & ~demos.capped).any()


def test_collection_is_deterministic(demos):
    again = collect_demos(ENV_CFG, n_games=6, seed=3)
    assert np.array_equal(demos.obs, again.obs)
    assert np.array_equal(demos.actions, again.actions)


def test_save_load_roundtrip(demos, tmp_path):
    demos.save(tmp_path / "d.npz")
    loaded = Demos.load(tmp_path / "d.npz")
    assert np.array_equal(loaded.obs, demos.obs)
    assert np.array_equal(loaded.opponent, demos.opponent)


def test_equivalent_actions_contain_the_action_and_are_legal(demos):
    rows = np.arange(len(demos.actions))
    assert demos.equiv[rows, demos.actions].all()
    assert not (demos.equiv & ~demos.masks).any()
    setup = demos.phase == Phase.SETUP_FLIP  # random flips: every legal action
    assert np.array_equal(demos.equiv[setup], demos.masks[setup])


# --- copy of the policy --------------------------------------------------------

@pytest.fixture(scope="module")
def fresh_model():
    from skyjo.training.train import build_model, make_vec_env
    cfg = load_config(CONFIGS / "two_players_b.yaml")
    env = make_vec_env(cfg["env"], n_envs=1)
    yield build_model(cfg, env, tensorboard_log=None)
    env.close()


def test_masked_loss_matches_maskable_ppo(demos, fresh_model):
    import torch as th
    import torch.nn.functional as F
    from skyjo.training.imitation import demo_tensors, masked_logits

    data = demo_tensors(demos, n_games=1)
    policy = fresh_model.policy
    with th.no_grad():
        ours = F.cross_entropy(masked_logits(policy, data["obs"], data["masks"]),
                               data["actions"], reduction="none")
        _, log_prob, _ = policy.evaluate_actions(data["obs"], data["actions"], data["masks"])
    assert th.allclose(ours, -log_prob, atol=1e-5)


def test_copy_learns_and_leaves_value_network_untouched(demos, fresh_model):
    import copy

    import torch as th
    from skyjo.training.imitation import demo_tensors, evaluate_copy, train_copy

    policy = copy.deepcopy(fresh_model.policy)
    value_before = {k: v.clone() for k, v in policy.state_dict().items()
                    if k.startswith(("mlp_extractor.value_net", "value_net"))}
    data = demo_tensors(demos)
    loss_before, _ = evaluate_copy(policy, data)
    history = train_copy(policy, data, data, demos, epochs=3, log=lambda _: None)
    loss_after, choices = evaluate_copy(policy, data)

    assert loss_after < loss_before
    assert history[-1].val_agreement > history[0].val_agreement - 1e-9
    assert demos.masks[np.arange(len(choices)), choices].all()  # never an illegal choice
    for k, v in value_before.items():
        assert th.equal(policy.state_dict()[k], v), k


def test_end_choice_matches_the_observation(demos):
    from skyjo.training.imitation import FINISHED, REFUSED, ends_round_from_obs

    ends = ends_round_from_obs(demos.obs, demos.actions)
    assert np.array_equal(ends, demos.end_choice == FINISHED)
    refused = demos.end_choice == REFUSED
    assert not ends[refused].any()  # a refusal places the card elsewhere
    assert np.isin(demos.phase[refused], [Phase.PLACE_DRAWN, Phase.PLACE_DISCARD]).all()


# --- targeted demonstrations (end-of-round decisions only) ---------------------

@pytest.fixture(scope="module")
def end_demos():
    return collect_demos(ENV_CFG, n_games=6, seed=3, end_only=True)


def test_end_only_keeps_exactly_the_end_decisions(demos, end_demos):
    from skyjo.training.imitation import NOT_CONCERNED

    kept = demos.end_choice != NOT_CONCERNED  # same seed: same games as the full collection
    assert kept.any()
    assert np.array_equal(end_demos.obs, demos.obs[kept])
    assert np.array_equal(end_demos.actions, demos.actions[kept])
    assert np.array_equal(end_demos.end_choice, demos.end_choice[kept])
    assert end_demos.policy_only.all() and not demos.policy_only.any()


def test_policy_only_rows_are_never_value_targets(demos, end_demos, tmp_path):
    from skyjo.training.imitation import discounted_returns

    end_demos.save(tmp_path / "end.npz")
    merged = Demos.concat([demos, Demos.load(tmp_path / "end.npz")])
    rows, returns = discounted_returns(merged, gamma=1.0)
    assert not merged.policy_only[rows].any()
    assert len(rows) == len(demos.actions)  # every row of the complete games, nothing else
    # gamma = 1: the target of a game's first decision is the sum of its rewards
    first = np.flatnonzero(np.diff(merged.game[rows], prepend=-1) != 0)
    assert np.allclose(returns[first], np.bincount(demos.game, weights=demos.rewards), atol=1e-4)


def test_concat_keeps_games_distinct(demos):
    merged = Demos.concat([demos, demos])
    assert len(np.unique(merged.game)) == 2 * len(demos.won)
    assert len(merged.won) == 2 * len(demos.won)


# --- value network --------------------------------------------------------------

def digest(params) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def test_policy_and_value_share_no_parameters(fresh_model):
    from skyjo.training.imitation import policy_parameters, value_parameters

    policy = fresh_model.policy
    pi = {id(p) for p in policy_parameters(policy)}
    vf = {id(p) for p in value_parameters(policy)}
    assert not pi & vf
    assert pi | vf == {id(p) for p in policy.parameters()}  # nothing else is trainable
    assert not list(policy.features_extractor.parameters())  # the shared Flatten has none


def test_value_training_leaves_the_policy_strictly_identical(demos, fresh_model):
    import copy

    import torch as th
    from skyjo.training.imitation import (
        discounted_returns, policy_parameters, predict_values, train_value, value_parameters,
    )

    policy = copy.deepcopy(fresh_model.policy)
    pi_before, vf_before = digest(policy_parameters(policy)), digest(value_parameters(policy))
    rows, targets = discounted_returns(demos, gamma=0.999)
    obs = th.from_numpy(demos.obs[rows])
    mse_before = np.mean((predict_values(policy, obs) - targets) ** 2)
    history = train_value(policy, [(obs, th.from_numpy(targets))], obs, targets,
                          epochs=3, log=lambda _: None)

    assert digest(policy_parameters(policy)) == pi_before
    assert digest(value_parameters(policy)) != vf_before
    assert min(h.val_mse for h in history) < mse_before


def test_collect_games_of_a_ppo_agent(fresh_model):
    from skyjo.agents.ppo_agent import PPOAgent

    games = collect_demos(ENV_CFG, n_games=2, seed=7, agent=PPOAgent(fresh_model, deterministic=False))
    rows = np.arange(len(games.actions))
    assert games.masks[rows, games.actions].all()
    assert (games.equiv.sum(axis=1) == 1).all() and games.equiv[rows, games.actions].all()
    assert not games.capped.any() and not games.policy_only.any()
    returns = np.bincount(games.game, weights=games.rewards)
    bonus = np.sign(games.opp_total - games.my_total)
    assert np.allclose(returns, (games.opp_total - games.my_total) / 20 + bonus, atol=1e-4)
