"""Pre-train the value network of a copy on the discounted returns of
complete games (the policy is left strictly unchanged).

Usage:
    python scripts/train_value.py --model models/copy/copy_3000_end/model.zip
        --games data/copy_games_train.npz --val data/copy_games_val.npz
        [--extra data/demos_cautious.npz] [--out models/copy/copy_3000_end_vf]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO  # noqa: E402

from skyjo.training.imitation import (  # noqa: E402
    Demos, discounted_returns, explained_variance, gae_returns, parameters_digest,
    policy_parameters, predict_values, train_value,
)
from skyjo.training.train import load_config  # noqa: E402


def value_data(path: str, gamma: float) -> tuple[th.Tensor, np.ndarray]:
    """(observations, targets) of the complete-game rows; policy-only rows are dropped."""
    demos = Demos.load(path)
    rows, targets = discounted_returns(demos, gamma)
    obs = demos.obs if len(rows) == len(demos.obs) else demos.obs[rows]
    return th.from_numpy(obs), targets


def report_ev(label: str, model: MaskablePPO, val: Demos, gamma: float, lam: float) -> None:
    """Both explained variances on the validation games."""
    rows, targets = discounted_returns(val, gamma)
    values = predict_values(model.policy, th.from_numpy(val.obs[rows]))
    gae = gae_returns(values, val.rewards[rows], val.game[rows], gamma, lam)
    print(f"{label} : MSE {np.mean((values - targets) ** 2):.4f}, variance expliquée "
          f"GAE (comme TensorBoard) {explained_variance(values, gae):.3f}, "
          f"retours complets {explained_variance(values, targets):.3f}")


def describe(label: str, targets: np.ndarray) -> None:
    print(f"  {label:<24}{len(targets):>9,} décisions  moyenne {targets.mean():+.3f}  "
          f"variance {targets.var():.3f}  écart type {targets.std():.3f}  "
          f"[{targets.min():+.2f}, {targets.max():+.2f}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pre-train the value network of a copy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/two_players_b.yaml", help="for gamma")
    parser.add_argument("--games", nargs="+", required=True, help="training games")
    parser.add_argument("--val", required=True, help="validation games (judge)")
    parser.add_argument("--extra", nargs="*", default=[], help="additional training games")
    parser.add_argument("--report", nargs="*", default=[], help="other sets, reported only")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ppo_cfg = load_config(args.config)["ppo"]
    gamma, lam = ppo_cfg["gamma"], ppo_cfg["gae_lambda"]
    model = MaskablePPO.load(args.model, device="cpu")
    policy_digest = parameters_digest(policy_parameters(model.policy))
    parts = [value_data(path, gamma) for path in [*args.games, *args.extra]]
    val_obs, val_targets = value_data(args.val, gamma)
    print(f"Cibles : retours actualisés, gamma {gamma}")
    for path, (_, targets) in zip([*args.games, *args.extra], parts):
        describe(Path(path).stem, targets)
    describe(Path(args.val).stem + " (val.)", val_targets)

    val_demos = Demos.load(args.val)
    print()
    report_ev("Avant (validation)", model, val_demos, gamma, lam)
    start = time.perf_counter()
    history = train_value(model.policy, [(obs, th.from_numpy(t)) for obs, t in parts],
                          val_obs, val_targets, epochs=args.epochs, lr=args.lr,
                          batch_size=args.batch_size, patience=args.patience)
    best = min(history, key=lambda h: h.val_mse)
    print(f"\nMeilleure époque {best.epoch} ({time.perf_counter() - start:.0f} s) : "
          f"MSE entraînement {best.train_mse:.4f}, validation {best.val_mse:.4f}, "
          f"variance des cibles {val_targets.var():.4f}, variance expliquée {best.val_ev:.3f}")
    for path in args.report:
        obs, targets = value_data(path, gamma)
        pred = predict_values(model.policy, obs)
        print(f"  sur {Path(path).stem} : MSE {np.mean((pred - targets) ** 2):.4f}, "
              f"variance des cibles {targets.var():.4f}, "
              f"variance expliquée {explained_variance(pred, targets):.3f}")

    report_ev("Après (validation)", model, val_demos, gamma, lam)
    same = parameters_digest(policy_parameters(model.policy)) == policy_digest
    print(f"Empreinte de la politique identique avant/après : {same}")
    if not same:
        raise RuntimeError("The policy changed during value training: model not saved.")
    model.save(str(Path(args.out) / "model"))
    print(f"\nModèle sauvegardé : {args.out}/model.zip")


if __name__ == "__main__":
    main()
