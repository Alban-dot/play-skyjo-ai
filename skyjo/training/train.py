"""Train MaskablePPO on SkyjoEnv from a YAML config.

Usage:
    python -m skyjo.training.train --config configs/two_players.yaml
    python -m skyjo.training.train --config configs/two_players.yaml
        --resume models/checkpoints/<run>/model_<steps>_steps.zip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from skyjo.env.rewards import ScoreDiffReward
from skyjo.env.skyjo_env import SkyjoEnv
from skyjo.training.callbacks import EvalCallback, TrainStatsCallback
from skyjo.training.opponent_pool import PoolFactory, SnapshotCallback, WeightedBotFactory


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_opponent_factory(env_cfg: dict[str, Any]) -> WeightedBotFactory | PoolFactory:
    """Bots only, or a self-play pool when ``env_cfg["pool"]`` is set (its
    ``dir`` is filled in by train())."""
    pool = env_cfg.get("pool")
    if pool is None:
        return WeightedBotFactory(env_cfg["opponents"])
    return PoolFactory(env_cfg["opponents"], pool["dir"], frozen=pool.get("frozen"),
                       latest_share=pool.get("latest_share", 0.5),
                       column_rule_kinds=pool.get("column_rule_kinds"))


def column_rule_line(cfg: dict[str, Any]) -> str:
    """Who plays with the column rule (journal line at the start of a run)."""
    env_cfg, eval_cfg = cfg["env"], cfg.get("eval", {})
    if not env_cfg.get("column_rule"):
        return "Règle de colonne : inactive (aucun joueur ne la suit)"
    pool = env_cfg.get("pool", {})
    kinds = set(pool.get("column_rule_kinds", []))
    with_rule = ", ".join(k for k in ("recent", "frozen") if k in kinds) or "aucun"
    without = ", ".join(k for k in ("recent", "frozen") if k not in kinds) or "aucun"
    return ("Règle de colonne : active pour l'agent (masque d'actions) ; adversaires réseau avec la règle : "
            f"{with_rule} ; sans la règle : {without} ; bots sans changement ; évaluation "
            f"{'avec' if eval_cfg.get('column_rule') else 'sans'} la règle")


def make_env(env_cfg: dict[str, Any], return_view: bool = False) -> SkyjoEnv:
    reward_cfg = env_cfg.get("reward", {})
    return SkyjoEnv(
        n_players=env_cfg["n_players"],
        opponent_factory=make_opponent_factory(env_cfg),
        reward_fn=ScoreDiffReward(**reward_cfg),
        max_steps=env_cfg.get("max_steps", 5_000),
        return_view=return_view,
        round_turn_limit=env_cfg.get("round_turn_limit"),
        forced_end_penalty=env_cfg.get("forced_end_penalty", 0.0),
        column_rule=env_cfg.get("column_rule", False),
    )


def make_vec_env(env_cfg: dict[str, Any], n_envs: int) -> VecEnv:
    fns = [lambda: make_env(env_cfg) for _ in range(n_envs)]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(vec)


def build_model(cfg: dict[str, Any], env: VecEnv, tensorboard_log: str | None) -> MaskablePPO:
    ppo = dict(cfg["ppo"])
    net_arch = ppo.pop("net_arch")
    lr = ppo["learning_rate"]
    if isinstance(lr, dict):  # {start, end}: linear decay over the run
        ppo["learning_rate"] = LinearSchedule(lr["start"], lr["end"], 1.0)
    return MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": {"pi": net_arch, "vf": net_arch}},
        tensorboard_log=tensorboard_log,
        seed=cfg["seed"],
        device="cpu",
        verbose=0,
        **ppo,
    )


def load_initial_weights(model: MaskablePPO, path: str | Path) -> None:
    """Copy all the network weights (policy and value) of the model at ``path``
    into ``model``; the optimizer and the step counter of ``model`` are kept."""
    source = MaskablePPO.load(str(path), device="cpu")
    model.policy.load_state_dict(source.policy.state_dict())


def train(
    cfg: dict[str, Any], resume: str | Path | None = None, init_from: str | Path | None = None
) -> Path:
    """Train from scratch, or resume the run of checkpoint ``resume``
    (models/checkpoints/<run>/model_<steps>_steps.zip): same run folders,
    step counter kept, RNGs reseeded deterministically from seed + steps.

    ``init_from`` (or ``cfg["init_from"]``): a new run whose network starts
    from the weights of that model, step counter at 0.
    """
    init_from = init_from or cfg.get("init_from")
    if resume and init_from:
        raise ValueError("Use either resume or init_from, not both.")
    models_dir = Path(cfg.get("models_dir", "models"))
    logs_dir = Path(cfg.get("logs_dir", "logs"))
    run_name = Path(resume).parent.name if resume else f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = models_dir / "runs" / run_name
    n_envs = cfg["n_envs"]
    env_cfg = dict(cfg["env"])
    snapshots = None
    if "pool" in env_cfg:
        pool = env_cfg["pool"] = {**env_cfg["pool"], "dir": str(models_dir / "pool" / run_name)}
        snapshots = SnapshotCallback(pool["dir"], every=pool["snapshot_every"],
                                     keep=pool.get("keep", 5), keep_every=pool.get("keep_every"))

    env = make_vec_env(env_cfg, n_envs)
    if resume:
        model = MaskablePPO.load(str(resume), env=env, device="cpu", tensorboard_log=str(logs_dir))
        model.set_random_seed(cfg["seed"] + model.num_timesteps)  # also seeds the envs
    else:
        model = build_model(cfg, env, str(logs_dir))
        if init_from:
            load_initial_weights(model, init_from)
    start_steps = model.num_timesteps
    remaining = cfg["total_timesteps"] - start_steps
    if remaining <= 0:
        print(f"Run {run_name} déjà à {start_steps:,} pas : rien à faire.")
        env.close()
        return run_dir

    eval_cfg = cfg["eval"]
    n_players = cfg["env"]["n_players"]
    callbacks = CallbackList([
        TrainStatsCallback(print_updates=cfg.get("print_updates", False), alarms=cfg.get("alarms")),
        CheckpointCallback(
            save_freq=max(cfg["checkpoint"]["freq"] // n_envs, 1),
            save_path=str(models_dir / "checkpoints" / run_name),
            name_prefix="model",
        ),
        EvalCallback(
            eval_freq=eval_cfg["freq"],
            n_games=eval_cfg.get("n_games", 0),
            opponents=eval_cfg.get("opponents", []),
            n_players=n_players if isinstance(n_players, int) else min(n_players),
            seed=cfg["seed"] + 1,
            csv_path=run_dir / "eval.csv",
            best_path=models_dir / "best" / run_name / "model",
            stop_if_gap_above=eval_cfg.get("stop_if_gap_above"),
            stop_patience=eval_cfg.get("stop_patience", 2),
            stop_if_winrate_below=eval_cfg.get("stop_if_winrate_below"),
            plateau=eval_cfg.get("plateau"),
            max_steps=eval_cfg.get("max_steps"),
            per_n_players=eval_cfg.get("per_n_players"),
            column_rule=eval_cfg.get("column_rule", False),
        ),
        *([snapshots] if snapshots else []),
    ])
    if snapshots and not resume:
        snapshots.save_snapshot(model, start_steps)  # the pool must not be empty at the first reset

    action = f"reprise à {start_steps:,}" if resume else "nouveau run"
    if init_from:
        action += f", poids initiaux de {init_from}"
    print(f"Run {run_name} ({action}) : objectif {cfg['total_timesteps']:,} pas, "
          f"{n_envs} envs, graine {cfg['seed']}", flush=True)
    print(column_rule_line(cfg), flush=True)
    start = time.perf_counter()
    model.learn(remaining, callback=callbacks, tb_log_name=run_name,
                reset_num_timesteps=not resume)
    elapsed = time.perf_counter() - start
    done = model.num_timesteps - start_steps
    print(f"Terminé en {elapsed / 60:.1f} min au pas {model.num_timesteps:,} "
          f"({done / elapsed:,.0f} pas/s)", flush=True)

    model.save(str(run_dir / "final"))
    env.close()
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Skyjo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-timesteps", type=int, help="override the config")
    parser.add_argument("--seed", type=int, help="override the config")
    parser.add_argument("--n-envs", type=int, help="override the config (must match when resuming)")
    parser.add_argument("--resume", help="checkpoint .zip to resume from (same run)")
    parser.add_argument("--init-from", help="model .zip whose weights start a new run (step 0)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.total_timesteps:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs:
        cfg["n_envs"] = args.n_envs
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
