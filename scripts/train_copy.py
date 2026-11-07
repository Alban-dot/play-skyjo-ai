"""Copy the cautious bot into the policy of a fresh MaskablePPO (behaviour
cloning), for several amounts of training games (learning curve).

Usage:
    python scripts/train_copy.py --config configs/two_players_b.yaml
        --demos data/demos_cautious.npz --val data/demos_cautious_val.npz
        [--n-games 1000 2000 3000] [--out models/copy]
        [--extra data/demos_end_train.npz --tag _end]   # add targeted decisions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.env.actions import DRAW, FLIP, TAKE_DISCARD  # noqa: E402
from skyjo.game.engine import Phase  # noqa: E402
from skyjo.training.imitation import (  # noqa: E402
    Demos, agreement, demo_tensors, evaluate_copy, train_copy,
)
from skyjo.training.train import build_model, load_config, make_vec_env  # noqa: E402


def print_phase_agreement(choices: np.ndarray, val: Demos) -> None:
    a = val.actions
    agree = agreement(choices, val.equiv)
    strict = choices == a
    groups = [
        ("CHOOSE_SOURCE", val.phase == Phase.CHOOSE_SOURCE),
        ("  bot pioche", (val.phase == Phase.CHOOSE_SOURCE) & (a == DRAW)),
        ("  bot prend la défausse", (val.phase == Phase.CHOOSE_SOURCE) & (a == TAKE_DISCARD)),
        ("PLACE_DRAWN", val.phase == Phase.PLACE_DRAWN),
        ("  bot place", (val.phase == Phase.PLACE_DRAWN) & (a < FLIP)),
        ("  bot défausse + retourne", (val.phase == Phase.PLACE_DRAWN) & (a >= FLIP)),
        ("PLACE_DISCARD", val.phase == Phase.PLACE_DISCARD),
        ("total (hors SETUP_FLIP)", val.phase != Phase.SETUP_FLIP),
    ]
    print(f"\n{'décisions du bot':<28}{'nombre':>10}{'accord':>10}{'accord strict':>15}")
    for label, sel in groups:
        print(f"{label:<28}{int(sel.sum()):>10,}{agree[sel].mean():>10.2%}{strict[sel].mean():>15.2%}")
    several = val.equiv.sum(axis=1) > 1
    scored = val.phase != Phase.SETUP_FLIP
    print(f"\nDécisions où le bot avait plusieurs actions équivalentes : "
          f"{several[scored].mean():.1%} (hors SETUP_FLIP)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Behaviour cloning of the cautious bot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--demos", default="data/demos_cautious.npz")
    parser.add_argument("--val", default="data/demos_cautious_val.npz")
    parser.add_argument("--n-games", type=int, nargs="+", default=[1000, 2000, 3000])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--out", default="models/copy")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="demonstrations added to every training set (e.g. targeted ones)")
    parser.add_argument("--tag", default="", help="suffix of the saved copy's folder")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    demos, val = Demos.load(args.demos), Demos.load(args.val)
    extra = [demo_tensors(Demos.load(path)) for path in args.extra]
    n_extra = sum(len(e["actions"]) for e in extra)
    val_data = demo_tensors(val)
    env = make_vec_env(cfg["env"], n_envs=1)  # only gives the spaces to the model

    summary = []
    for n_games in args.n_games:
        train = demo_tensors(demos, n_games)
        if extra:
            train = {k: th.cat([train[k]] + [e[k] for e in extra]) for k in train}
        print(f"\n=== {n_games} parties + {n_extra:,} décisions ciblées "
              f"({len(train['actions']):,} décisions), "
              f"validation {len(val.won)} parties ({len(val.actions):,} décisions)")
        model = build_model(cfg, env, tensorboard_log=None)  # fresh network, config seed
        start = time.perf_counter()
        history = train_copy(model.policy, train, val_data, val, epochs=args.epochs,
                             lr=args.lr, batch_size=args.batch_size, patience=args.patience,
                             seed=cfg["seed"])
        best = min(history, key=lambda h: h.val_loss)
        path = Path(args.out) / f"copy_{n_games}{args.tag}" / "model"
        model.save(str(path))
        summary.append((n_games, len(train["actions"]), best, time.perf_counter() - start, model, path))

    print(f"\n{'parties':>8}{'décisions':>11}{'époque':>8}{'perte entr.':>13}{'perte val.':>12}"
          f"{'écart':>9}{'accord val.':>13}{'durée':>9}")
    for n_games, n_rows, best, elapsed, _, _ in summary:
        print(f"{n_games:>8}{n_rows:>11,}{best.epoch:>8}{best.train_loss:>13.4f}{best.val_loss:>12.4f}"
              f"{best.val_loss - best.train_loss:>+9.4f}{best.val_agreement:>13.2%}{elapsed:>8.0f}s")

    n_games, _, best, _, model, path = max(summary, key=lambda s: s[2].val_agreement)
    _, choices = evaluate_copy(model.policy, val_data)
    print(f"\nMeilleure copie : {n_games} parties -> {path}.zip")
    print_phase_agreement(choices, val)
    env.close()


if __name__ == "__main__":
    main()
