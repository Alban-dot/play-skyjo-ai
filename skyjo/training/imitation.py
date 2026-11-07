"""Imitation of the cautious bot (behaviour cloning) before PPO.

Step 1, demonstrations: the bot plays as the learning agent of the training
env, so observations, masks and rewards are exactly those PPO will see.
Step 2, copy of the policy: cross-entropy on the bot's actions, with the
action mask, on the action head of a fresh MaskablePPO.
Step 3, value network: regression on the discounted returns of complete
games, on the value head only (the policy is left strictly unchanged).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from skyjo.agents.base import Agent
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent
from skyjo.env.actions import DRAW, FLIP, N_ACTIONS
from skyjo.env.observation import LAYOUT
from skyjo.game.constants import GRID_SIZE
from skyjo.game.engine import Phase
from skyjo.game.player_state import CardState
from skyjo.training.train import make_env

PLACE_PHASES = (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD)
# Decisions where the bot could end the round (rule 5): it revealed its last
# hidden card (FINISHED) or rule 5 made it place elsewhere (REFUSED).
NOT_CONCERNED, FINISHED, REFUSED = 0, 1, 2


@dataclass
class Demos:
    # one row per decision of the bot
    obs: np.ndarray  # (N, OBS_SIZE) float32
    masks: np.ndarray  # (N, N_ACTIONS) bool
    actions: np.ndarray  # (N,) int64
    equiv: np.ndarray  # (N, N_ACTIONS) bool, actions the bot was indifferent between
    rewards: np.ndarray  # (N,) float32, reward of the env step
    game: np.ndarray  # (N,) int32, game index (train/validation split by game)
    phase: np.ndarray  # (N,) int8
    capped: np.ndarray  # (N,) bool, no refusal left in the round (placement phases)
    cap_decisive: np.ndarray  # (N,) bool, capped and rule 5 would have refused
    end_choice: np.ndarray  # (N,) int8, rule-5 decisions: NOT_CONCERNED, FINISHED or REFUSED
    policy_only: np.ndarray  # (N,) bool, row of an incomplete game: never a value target
    # one row per game
    opponent: np.ndarray  # (G,) str, opponent bot name (2 players)
    n_players: np.ndarray  # (G,) int8
    my_total: np.ndarray  # (G,) int32
    opp_total: np.ndarray  # (G,) float32, mean final total of the opponents
    won: np.ndarray  # (G,) bool, alone or shared

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> Demos:
        with np.load(path) as data:
            arrays = {f.name: data[f.name] for f in fields(cls) if f.name in data.files}
        if "policy_only" not in arrays:  # files collected before the flag: complete games
            arrays["policy_only"] = np.zeros(len(arrays["actions"]), dtype=bool)
        return cls(**arrays)

    @classmethod
    def concat(cls, parts: list[Demos]) -> Demos:
        """Join demonstrations; game indices are shifted to stay distinct."""
        offsets = np.cumsum([0] + [len(d.won) for d in parts[:-1]])
        arrays = {}
        for f in fields(cls):
            values = [getattr(d, f.name) for d in parts]
            if f.name == "game":
                values = [v + off for v, off in zip(values, offsets)]
            arrays[f.name] = np.concatenate(values)
        return cls(**arrays)


def collect_demos(
    env_cfg: dict[str, Any], n_games: int, seed: int, end_only: bool = False,
    agent: Agent | None = None,
) -> Demos:
    """Play ``n_games`` games of the cautious bot in the training env.

    With ``end_only``, only the decisions where the bot could end the round
    (finished or refused) are kept, marked ``policy_only``.
    With ``agent`` (e.g. a PPOAgent), that agent plays instead of the bot:
    ``equiv`` is then its action alone, and refusals are unknown (never set).
    """
    env = make_env(env_cfg, return_view=True)
    is_bot = agent is None
    if is_bot:
        player = CautiousHeuristicAgent(np.random.default_rng(np.random.SeedSequence(seed)))
    else:
        player = agent
        th.manual_seed(seed)  # sampling of a stochastic PPOAgent
    rows: dict[str, list] = {k: [] for k in
                             ("obs", "masks", "actions", "equiv", "rewards", "game", "phase",
                              "capped", "cap_decisive", "end_choice")}
    games: dict[str, list] = {k: [] for k in
                              ("opponent", "n_players", "my_total", "opp_total", "won")}

    for g in range(n_games):
        obs, info = env.reset(seed=seed + 1 if g == 0 else None)
        player.reset()
        done = False
        while not done:
            view, mask = info["view"], env.action_masks()
            capped = is_bot and view.phase in PLACE_PHASES and player.refusals_exhausted(view)
            action = player.act(view, mask)
            equiv = np.zeros(N_ACTIONS, dtype=bool)
            equiv[player.equivalent if is_bot else action] = True
            if is_bot and player.refused:
                end_choice = REFUSED
            elif (view.phase in PLACE_PHASES and view.finisher is None
                    and CautiousHeuristicAgent.ends_round(view, action)):
                end_choice = FINISHED
            else:
                end_choice = NOT_CONCERNED
            next_obs, reward, terminated, truncated, info = env.step(action)
            if not end_only or end_choice != NOT_CONCERNED:
                for key, value in (("obs", obs), ("masks", mask), ("actions", action),
                                   ("equiv", equiv),
                                   ("rewards", reward), ("game", g), ("phase", int(view.phase)),
                                   ("capped", capped),
                                   ("cap_decisive", capped and player.wants_to_refuse(view, action)),
                                   ("end_choice", end_choice)):
                    rows[key].append(value)
            obs, done = next_obs, terminated or truncated
        if not terminated:
            raise RuntimeError(f"Game {g} truncated after {env.max_steps} steps.")

        totals, seat = info["total_scores"], info["seat"]
        others = [t for p, t in enumerate(totals) if p != seat]
        games["opponent"].append(env.opponents[next(iter(env.opponents))].name
                                 if len(env.opponents) == 1 else "mixed")
        games["n_players"].append(len(totals))
        games["my_total"].append(totals[seat])
        games["opp_total"].append(float(np.mean(others)))
        games["won"].append(info["is_winner"])
    env.close()

    return Demos(
        obs=np.asarray(rows["obs"], dtype=np.float32),
        masks=np.asarray(rows["masks"], dtype=bool),
        actions=np.asarray(rows["actions"], dtype=np.int64),
        equiv=np.asarray(rows["equiv"], dtype=bool),
        rewards=np.asarray(rows["rewards"], dtype=np.float32),
        game=np.asarray(rows["game"], dtype=np.int32),
        phase=np.asarray(rows["phase"], dtype=np.int8),
        capped=np.asarray(rows["capped"], dtype=bool),
        cap_decisive=np.asarray(rows["cap_decisive"], dtype=bool),
        end_choice=np.asarray(rows["end_choice"], dtype=np.int8),
        policy_only=np.full(len(rows["actions"]), end_only),
        opponent=np.asarray(games["opponent"]),
        n_players=np.asarray(games["n_players"], dtype=np.int8),
        my_total=np.asarray(games["my_total"], dtype=np.int32),
        opp_total=np.asarray(games["opp_total"], dtype=np.float32),
        won=np.asarray(games["won"], dtype=bool),
    )


def discounted_returns(demos: Demos, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Value targets: (rows, discounted sum of future rewards) for the rows of
    complete games only. Policy-only rows (incomplete games) are never used."""
    rows = np.flatnonzero(~demos.policy_only)
    returns = np.zeros(len(rows), dtype=np.float32)
    running, game = 0.0, None
    for i in range(len(rows) - 1, -1, -1):  # backwards through each game
        row = rows[i]
        if demos.game[row] != game:
            running, game = 0.0, demos.game[row]
        running = demos.rewards[row] + gamma * running
        returns[i] = running
    return rows, returns


def ends_round_from_obs(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per decision, from the observation only: the action reveals my last
    hidden card while nobody has finished yet (so it ends the round)."""
    phase = obs[:, LAYOUT["phase"]].argmax(axis=1)
    grid = obs[:, LAYOUT["my_grid"]].reshape(len(obs), GRID_SIZE, -1)
    hidden = grid[:, :, CardState.HIDDEN] == 1
    target = np.where(actions < FLIP, actions, actions - FLIP)
    target_hidden = hidden[np.arange(len(obs)), np.minimum(target, GRID_SIZE - 1)]
    return (np.isin(phase, PLACE_PHASES) & (obs[:, LAYOUT["last_turn"].start] == 0)
            & (hidden.sum(axis=1) == 1) & (actions < DRAW) & target_hidden)


# --- step 2: behaviour cloning of the policy -----------------------------------

def demo_tensors(demos: Demos, n_games: int | None = None) -> dict[str, th.Tensor]:
    """Decision rows of the first ``n_games`` games as tensors (no copy of obs)."""
    end = len(demos.game) if n_games is None else int(np.searchsorted(demos.game, n_games))
    return {
        "obs": th.from_numpy(demos.obs[:end]),
        "masks": th.from_numpy(demos.masks[:end]),
        "actions": th.from_numpy(demos.actions[:end]),
        "equiv": th.from_numpy(demos.equiv[:end]),
    }


def masked_logits(policy: MaskableActorCriticPolicy, obs: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """Action logits of the policy, illegal actions set to -1e8 as in
    MaskableCategorical (so the softmax is the MaskablePPO distribution)."""
    features = policy.extract_features(obs, policy.pi_features_extractor)
    logits = policy.action_net(policy.mlp_extractor.forward_actor(features))
    return th.where(masks, logits, th.tensor(-1e8, dtype=logits.dtype))


def policy_parameters(policy: MaskableActorCriticPolicy) -> list[th.nn.Parameter]:
    """Parameters of the action head only; the value network is not trained here."""
    return [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()]


@th.no_grad()
def evaluate_copy(
    policy: MaskableActorCriticPolicy, data: dict[str, th.Tensor], batch_size: int = 8192
) -> tuple[float, np.ndarray]:
    """Mean cross-entropy, and the most probable action of each decision
    (the deterministic choice of the copy)."""
    losses, choices = [], []
    for start in range(0, len(data["actions"]), batch_size):
        part = {k: v[start:start + batch_size] for k, v in data.items()}
        logits = masked_logits(policy, part["obs"], part["masks"])
        losses.append(F.cross_entropy(logits, part["actions"], reduction="sum").item())
        choices.append(logits.argmax(dim=1).numpy())
    return sum(losses) / len(data["actions"]), np.concatenate(choices)


def agreement(choices: np.ndarray, equiv: np.ndarray) -> np.ndarray:
    """Per decision: the copy chose an action the bot was indifferent between."""
    return equiv[np.arange(len(choices)), choices]


@dataclass
class EpochStats:
    epoch: int
    train_loss: float  # on a fixed sample of the training rows, after the epoch
    val_loss: float
    val_agreement: float  # setup flips excluded (random actions of the bot)


def train_copy(
    policy: MaskableActorCriticPolicy,
    train: dict[str, th.Tensor],
    val: dict[str, th.Tensor],
    val_demos: Demos,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 3,
    seed: int = 0,
    log=print,
) -> list[EpochStats]:
    """Cross-entropy on the bot's actions with the action mask. Keeps the
    epoch with the lowest validation loss (early stopping after ``patience``)."""
    rng = np.random.default_rng(seed)
    params = policy_parameters(policy)
    optimizer = th.optim.Adam(params, lr=lr)
    n = len(train["actions"])
    sample = rng.choice(n, size=min(n, 100_000), replace=False)
    train_sample = {k: v[th.from_numpy(sample)] for k, v in train.items()}
    scored = val_demos.phase != Phase.SETUP_FLIP

    history: list[EpochStats] = []
    best_loss, best_state = np.inf, None
    for epoch in range(1, epochs + 1):
        policy.train()
        for idx in th.from_numpy(rng.permutation(n)).split(batch_size):
            logits = masked_logits(policy, train["obs"][idx], train["masks"][idx])
            loss = F.cross_entropy(logits, train["actions"][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        train_loss, _ = evaluate_copy(policy, train_sample)
        val_loss, choices = evaluate_copy(policy, val)
        agree = agreement(choices, val_demos.equiv)
        history.append(EpochStats(epoch, train_loss, val_loss, float(agree[scored].mean())))
        log(f"  époque {epoch:>2}  perte entraînement {train_loss:.4f}  validation {val_loss:.4f}"
            f"  accord validation {agree[scored].mean():.2%}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = [p.detach().clone() for p in params]
        elif epoch - min(history, key=lambda h: h.val_loss).epoch >= patience:
            break

    with th.no_grad():
        for p, best in zip(params, best_state):
            p.copy_(best)
    return history


# --- step 3: pre-training of the value network ---------------------------------

def value_parameters(policy: MaskableActorCriticPolicy) -> list[th.nn.Parameter]:
    """Parameters of the value network only (the policy is not trained here)."""
    return [*policy.mlp_extractor.value_net.parameters(), *policy.value_net.parameters()]


def value_of(policy: MaskableActorCriticPolicy, obs: th.Tensor) -> th.Tensor:
    features = policy.extract_features(obs, policy.vf_features_extractor)
    return policy.value_net(policy.mlp_extractor.forward_critic(features))[:, 0]


@th.no_grad()
def predict_values(
    policy: MaskableActorCriticPolicy, obs: th.Tensor, batch_size: int = 8192
) -> np.ndarray:
    return np.concatenate([value_of(policy, obs[i:i + batch_size]).numpy()
                           for i in range(0, len(obs), batch_size)])


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - Var(target - pred) / Var(target), as reported by SB3 during PPO."""
    return float(1 - np.var(target - pred) / np.var(target))


@dataclass
class ValueStats:
    epoch: int
    train_mse: float  # on a fixed sample of the training rows, after the epoch
    val_mse: float
    val_ev: float  # explained variance on the validation rows


def train_value(
    policy: MaskableActorCriticPolicy,
    parts: list[tuple[th.Tensor, th.Tensor]],
    val_obs: th.Tensor,
    val_targets: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 3,
    seed: int = 0,
    log=print,
) -> list[ValueStats]:
    """Mean squared error between the value and the discounted returns.

    ``parts`` are (observations, targets) pairs, sampled together without
    being concatenated (saves memory). Only the value parameters are updated;
    keeps the epoch with the lowest validation error.
    """
    rng = np.random.default_rng(seed)
    params = value_parameters(policy)
    optimizer = th.optim.Adam(params, lr=lr)
    offsets = np.cumsum([0] + [len(t) for _, t in parts])
    n = int(offsets[-1])

    def batch(idx: np.ndarray) -> tuple[th.Tensor, th.Tensor]:
        part = np.searchsorted(offsets, idx, side="right") - 1
        obs, targets = [], []
        for k in np.unique(part):
            local = th.from_numpy(idx[part == k] - offsets[k])
            obs.append(parts[k][0][local])
            targets.append(parts[k][1][local])
        return th.cat(obs), th.cat(targets)

    sample_obs, sample_targets = batch(rng.choice(n, size=min(n, 100_000), replace=False))
    history: list[ValueStats] = []
    best_mse, best_state = np.inf, None
    for epoch in range(1, epochs + 1):
        policy.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            obs, targets = batch(perm[start:start + batch_size])
            loss = F.mse_loss(value_of(policy, obs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        train_mse = float(np.mean((predict_values(policy, sample_obs) - sample_targets.numpy()) ** 2))
        val_pred = predict_values(policy, val_obs)
        val_mse = float(np.mean((val_pred - val_targets) ** 2))
        history.append(ValueStats(epoch, train_mse, val_mse, explained_variance(val_pred, val_targets)))
        log(f"  époque {epoch:>2}  MSE entraînement {train_mse:.4f}  validation {val_mse:.4f}"
            f"  variance expliquée {history[-1].val_ev:.3f}")
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = [p.detach().clone() for p in params]
        elif epoch - min(history, key=lambda h: h.val_mse).epoch >= patience:
            break

    with th.no_grad():
        for p, best in zip(params, best_state):
            p.copy_(best)
    return history


def gae_returns(
    values: np.ndarray, rewards: np.ndarray, game: np.ndarray, gamma: float, lam: float
) -> np.ndarray:
    """Targets of the value as PPO builds them (GAE advantages + values), for
    rows of complete games in order. Used for the explained variance that
    TensorBoard reports during PPO (train/explained_variance)."""
    returns = np.zeros(len(values), dtype=np.float32)
    advantage = 0.0
    for t in range(len(values) - 1, -1, -1):
        last = t == len(values) - 1 or game[t + 1] != game[t]
        next_value = 0.0 if last else values[t + 1]
        if last:
            advantage = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        advantage = delta + gamma * lam * advantage
        returns[t] = advantage + values[t]
    return returns


def parameters_digest(params: list[th.nn.Parameter]) -> str:
    """SHA-256 of the parameter values: equal digests = identical weights."""
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
