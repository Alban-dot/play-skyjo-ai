import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from skyjo.agents.base import play_game
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent, HeuristicAgent
from skyjo.agents.ppo_agent import PPOAgent
from skyjo.agents.random_agent import RandomAgent
from skyjo.training.opponent_pool import WeightedBotFactory
from skyjo.training.train import build_model, load_config, make_opponent_factory, make_vec_env, train

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_weighted_bot_factory_follows_weights():
    factory = WeightedBotFactory({"random": 1.0, "heuristic": 3.0, "cautious": 0.0})
    rng = np.random.default_rng(0)
    counts = Counter(type(factory(rng)) for _ in range(4000))
    assert CautiousHeuristicAgent not in counts
    assert counts[HeuristicAgent] / 4000 == pytest.approx(0.75, abs=0.03)
    assert counts[RandomAgent] / 4000 == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize("weights", [{}, {"random": -1.0}, {"random": 0.0}])
def test_weighted_bot_factory_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        WeightedBotFactory(weights)


@pytest.mark.parametrize("name", ["two_players.yaml", "two_players_b.yaml", "two_players_from_copy.yaml",
                                  "two_players_from_copy_cont.yaml", "two_players_selfplay.yaml", "multi_players.yaml",
                                  "sanity_random.yaml"])
def test_configs_load(name, tmp_path):
    cfg = load_config(CONFIGS / name)
    assert isinstance(cfg["total_timesteps"], int)
    assert isinstance(cfg["ppo"]["learning_rate"], float)
    assert cfg["ppo"]["gamma"] == 0.999
    env_cfg = cfg["env"]
    if "pool" in env_cfg:  # train() fills in the pool folder
        env_cfg = {**env_cfg, "pool": {**env_cfg["pool"], "dir": str(tmp_path)}}
    make_opponent_factory(env_cfg)


def test_untrained_ppo_agent_plays_legal_games():
    cfg = load_config(CONFIGS / "sanity_random.yaml")
    env = make_vec_env(cfg["env"], n_envs=1)
    agent = PPOAgent(build_model(cfg, env, tensorboard_log=None))
    for seed in range(3):  # apply_action raises on any illegal action
        game = play_game([agent, RandomAgent(np.random.default_rng(seed))], np.random.default_rng(seed))
        assert game.is_over
    env.close()


@pytest.mark.parametrize("n_players", [5, 6, 8])
def test_ppo_agent_plays_beyond_training_player_counts(n_players):
    # The observation pads opponents up to MAX_PLAYERS (8): a model trained at
    # 2-4 players can sit at any table up to 8 without any change.
    cfg = load_config(CONFIGS / "sanity_random.yaml")
    env = make_vec_env(cfg["env"], n_envs=1)
    agent = PPOAgent(build_model(cfg, env, tensorboard_log=None))
    bots = [CautiousHeuristicAgent(np.random.default_rng(i)) for i in range(n_players - 1)]
    game = play_game([agent, *bots], np.random.default_rng(n_players))
    assert game.is_over and len(game.total_scores) == n_players
    env.close()


@pytest.mark.slow
def test_tiny_training_run(tmp_path):
    cfg = load_config(CONFIGS / "sanity_random.yaml")
    cfg.update(n_envs=2, total_timesteps=256, models_dir=str(tmp_path / "models"),
               logs_dir=str(tmp_path / "logs"))
    cfg["ppo"].update(n_steps=64, batch_size=32, n_epochs=1)
    cfg["eval"].update(freq=128, n_games=4, opponents=["random"])
    cfg["checkpoint"]["freq"] = 128

    run_dir = train(cfg)
    assert (run_dir / "final.zip").exists()
    rows = list(csv.DictReader((run_dir / "eval.csv").open()))
    assert len(rows) >= 2  # baseline + at least one periodic evaluation
    assert 0.0 <= float(rows[-1]["random_winrate"]) <= 1.0
    assert any((tmp_path / "models" / "checkpoints").rglob("*.zip"))
    assert any((tmp_path / "models" / "best").rglob("model.zip"))


# --- resume from a checkpoint ------------------------------------------------

def tiny_cfg(tmp_path, total):
    cfg = load_config(CONFIGS / "sanity_random.yaml")
    cfg.update(n_envs=2, total_timesteps=total, models_dir=str(tmp_path / "models"),
               logs_dir=str(tmp_path / "logs"))
    cfg["ppo"].update(n_steps=64, batch_size=32, n_epochs=1)
    cfg["eval"].update(freq=128, n_games=4, opponents=["random"])
    cfg["checkpoint"]["freq"] = 128
    return cfg


@pytest.mark.slow
def test_resume_continues_the_same_run(tmp_path):
    run_dir = train(tiny_cfg(tmp_path, 256))
    ckpt = next((tmp_path / "models" / "checkpoints").rglob("model_128_steps.zip"))

    resumed_dir = train(tiny_cfg(tmp_path, 512), resume=ckpt)
    assert resumed_dir == run_dir  # same run folders

    from sb3_contrib import MaskablePPO
    assert MaskablePPO.load(str(run_dir / "final"), device="cpu").num_timesteps == 512
    steps = [int(float(r["timesteps"])) for r in csv.DictReader((run_dir / "eval.csv").open())]
    # first run: 0, 128, 256; resume from 128: no duplicate 128, then 256, 384, 512
    assert steps[:3] == [0, 128, 256]
    assert steps[3:] == [256, 384, 512]
    assert (ckpt.parent / "model_512_steps.zip").exists()


@pytest.mark.slow
def test_resume_is_deterministic(tmp_path):
    train(tiny_cfg(tmp_path / "base", 128))
    ckpt = next((tmp_path / "base").rglob("model_128_steps.zip"))

    from sb3_contrib import MaskablePPO
    params = []
    for name in ("a", "b"):
        copy_dir = tmp_path / name / "models" / "checkpoints" / f"run_{name}"
        copy_dir.mkdir(parents=True)
        copy = copy_dir / ckpt.name
        copy.write_bytes(ckpt.read_bytes())
        run_dir = train(tiny_cfg(tmp_path / name, 384), resume=copy)
        params.append(MaskablePPO.load(str(run_dir / "final"), device="cpu").policy.state_dict())
    for key in params[0]:
        assert np.array_equal(params[0][key].numpy(), params[1][key].numpy()), key


# --- start from given weights, automatic stop ------------------------------------

def test_load_initial_weights_copies_the_whole_network(tmp_path):
    import torch as th
    from skyjo.training.train import load_initial_weights

    cfg = load_config(CONFIGS / "sanity_random.yaml")
    env = make_vec_env(cfg["env"], n_envs=1)
    source = build_model(cfg, env, tensorboard_log=None)
    source.save(str(tmp_path / "source"))
    cfg["seed"] += 1  # different initial weights
    model = build_model(cfg, env, tensorboard_log=None)
    load_initial_weights(model, tmp_path / "source.zip")
    for key, value in source.policy.state_dict().items():
        assert th.equal(model.policy.state_dict()[key], value), key
    assert model.num_timesteps == 0
    env.close()


def test_stop_after_consecutive_evaluations_above_threshold():
    from skyjo.training.callbacks import EvalCallback

    cb = EvalCallback(eval_freq=1, n_games=1, opponents=["cautious"],
                      stop_if_gap_above={"cautious": 12.0}, stop_patience=2)
    gaps = [13.0, 5.0, 12.0, 13.0, 14.0]  # 12.0 is not above; only the last two in a row count
    stopped = [cb.check_stop({"timesteps": i, "cautious_score_gap": g}) for i, g in enumerate(gaps)]
    assert stopped == [False, False, False, False, True]


def test_plateau_stop_compares_last_window_with_the_one_before():
    from skyjo.training.callbacks import EvalCallback

    def run(gaps):
        cb = EvalCallback(eval_freq=1, n_games=1, opponents=["cautious"],
                          plateau={"bot": "cautious", "window": 3, "min_gain": 2.0})
        return [cb.check_stop({"timesteps": i, "cautious_score_gap": g}) for i, g in enumerate(gaps)]

    assert run([10, 9, 8, 7, 6, 5]) == [False] * 6        # 9 -> 6: gain 3
    assert run([10, 9, 8, 8, 7, 6.5]) == [False] * 5 + [True]  # 9 -> 7.17: gain < 2
    assert run([0, 0, 0, 0, 0]) == [False] * 5             # fewer than 2 windows: no decision


@pytest.mark.slow
def test_init_from_starts_a_new_run_at_step_zero(tmp_path):
    first = train(tiny_cfg(tmp_path / "first", 128))
    ckpt = next((tmp_path / "first").rglob("model_128_steps.zip"))
    run_dir = train(tiny_cfg(tmp_path / "second", 256), init_from=ckpt)
    assert run_dir.name != first.name
    from sb3_contrib import MaskablePPO
    assert MaskablePPO.load(str(run_dir / "final"), device="cpu").num_timesteps == 256
    steps = [int(float(r["timesteps"])) for r in csv.DictReader((run_dir / "eval.csv").open())]
    assert steps == [0, 128, 256]


@pytest.mark.slow
def test_training_stops_automatically(tmp_path):
    cfg = tiny_cfg(tmp_path, 1024)
    cfg["eval"].update(stop_if_gap_above={"random": -1000.0}, stop_patience=2)  # always above
    run_dir = train(cfg)
    from sb3_contrib import MaskablePPO
    # evaluations at 0 and 128 are both above: stop right after the second one
    assert MaskablePPO.load(str(run_dir / "final"), device="cpu").num_timesteps < 1024
    steps = [int(float(r["timesteps"])) for r in csv.DictReader((run_dir / "eval.csv").open())]
    assert steps == [0, 128]


# --- self-play pool ---------------------------------------------------------------

def tiny_model_file(tmp_path, name="tiny"):
    cfg = tiny_cfg(tmp_path, 64)
    cfg["ppo"]["net_arch"] = [8]
    env = make_vec_env(cfg["env"], 1)
    path = tmp_path / f"{name}.zip"
    build_model(cfg, env, None).save(str(path))
    env.close()
    return path


def test_snapshots_keep_the_last_ones_plus_one_every_keep_every(tmp_path):
    from sb3_contrib import MaskablePPO
    from skyjo.training.opponent_pool import SnapshotCallback

    model = MaskablePPO.load(str(tiny_model_file(tmp_path)), device="cpu")
    snaps = SnapshotCallback(tmp_path / "pool", every=100, keep=2, keep_every=400)
    for steps in range(0, 900, 100):
        snaps.save_snapshot(model, steps)
    names = lambda d: [p.name for p in sorted((tmp_path / "pool" / d).glob("*.zip"))]  # noqa: E731
    assert names("recent") == ["snap_000000000700.zip", "snap_000000000800.zip"]
    assert names("kept") == ["snap_000000000400.zip", "snap_000000000800.zip"]


def test_pool_factory_draws_kinds_by_weight_and_tags_them(tmp_path):
    from skyjo.training.opponent_pool import PoolFactory, SnapshotCallback
    from sb3_contrib import MaskablePPO

    frozen = tiny_model_file(tmp_path, "frozen")
    model = MaskablePPO.load(str(frozen), device="cpu")
    snaps = SnapshotCallback(tmp_path / "pool", every=100, keep=5)
    for steps in (0, 100, 200):
        snaps.save_snapshot(model, steps)
    factory = PoolFactory({"recent": 0.4, "frozen": 0.2, "cautious": 0.3, "random": 0.1},
                          tmp_path / "pool", frozen=[str(frozen)], latest_share=0.5)
    rng = np.random.default_rng(0)
    agents = [factory(rng) for _ in range(2000)]
    kinds = Counter(a.pool_kind for a in agents)
    for kind, weight in (("recent", 0.4), ("frozen", 0.2), ("cautious", 0.3), ("random", 0.1)):
        assert kinds[kind] / 2000 == pytest.approx(weight, abs=0.03)
    assert isinstance(next(a for a in agents if a.pool_kind == "cautious"), CautiousHeuristicAgent)
    network = [a for a in agents if a.pool_kind in ("recent", "frozen")]
    assert all(isinstance(a, PPOAgent) and not a.deterministic for a in network)
    # recent: the latest snapshot half of the time, the 2 others otherwise
    latest = factory._models[str(tmp_path / "pool" / "recent" / "snap_000000000200.zip")]
    recent = [a for a in agents if a.pool_kind == "recent"]
    assert sum(a.model is latest for a in recent) / len(recent) == pytest.approx(0.5, abs=0.06)
    with pytest.raises(ValueError):
        PoolFactory({"frozen": 1.0}, tmp_path / "pool")  # no frozen model given


@pytest.mark.slow
def test_self_play_training_fills_the_pool_and_reports_kinds(tmp_path, capsys):
    frozen = tiny_model_file(tmp_path, "frozen")
    cfg = tiny_cfg(tmp_path, 512)
    cfg["print_updates"] = True
    cfg["env"]["opponents"] = {"recent": 0.5, "frozen": 0.3, "random": 0.2}
    cfg["env"]["pool"] = {"frozen": [str(frozen)], "snapshot_every": 128, "keep": 2, "keep_every": 256}
    cfg["env"].update(round_turn_limit=50, forced_end_penalty=0.7)
    cfg["eval"].update(opponents=["random", f"ppo:{frozen}"], max_steps=500)
    run_dir = train(cfg)
    pool = tmp_path / "models" / "pool" / run_dir.name
    assert len(list((pool / "recent").glob("snap_*.zip"))) == 2
    assert sorted(p.name for p in (pool / "kept").glob("snap_*.zip")) == ["snap_000000000256.zip", "snap_000000000512.zip"]
    header = next(csv.reader((run_dir / "eval.csv").open()))
    assert "frozen_winrate" in header and "frozen_blocked" in header
    out = capsys.readouterr().out
    assert "[pool total]" in out and "recent" in out and "frozen" in out


def test_winrate_alarm_after_consecutive_evaluations_below_threshold():
    from skyjo.training.callbacks import EvalCallback

    cb = EvalCallback(eval_freq=1, n_games=1, opponents=["heuristic"],
                      stop_if_winrate_below={"heuristic": 0.65}, stop_patience=2)
    rates = [0.60, 0.70, 0.64, 0.65, 0.62, 0.61]  # 0.65 is not below; then two in a row
    rows = [{"timesteps": i, "heuristic_winrate": r, "heuristic_score_gap": -20.0} for i, r in enumerate(rates)]
    assert [cb.check_stop(r) for r in rows] == [False] * 5 + [True]


def test_training_alarms_on_explained_variance_and_forced_rounds():
    from skyjo.training.callbacks import TrainStatsCallback

    cb = TrainStatsCallback(alarms={"ev_below": 0.3, "ev_patience": 3})
    assert [cb.check_alarms(ev) for ev in (0.2, 0.1, 0.5, 0.2, 0.29, 0.25)] == [False] * 5 + [True]

    cb = TrainStatsCallback(alarms={"forced_above": 0.02, "forced_min_games": 11})
    cb.games.extend([(("all", "2j", "recent"), 10, 0, 4)] * 9 + [(("all", "2j", "recent"), 10, 3, 0)])  # 3 forced / 100 rounds
    assert not cb.check_alarms(0.5)  # 10 games: fewer than forced_min_games
    cb.games.append((("all", "2j", "cautious"), 10, 0, 5))  # 3 / 110 rounds = 2.7 %
    assert cb.check_alarms(0.5)
    cb = TrainStatsCallback(alarms={"forced_above": 0.02, "forced_min_games": 1})
    cb.games.extend([(("all", "2j", "recent"), 10, 0, 4)] * 99 + [(("all", "2j", "recent"), 10, 2, 0)])  # 2 / 1000 = 0.2 %
    assert not cb.check_alarms(0.5)


def test_evaluation_per_number_of_players(tmp_path):
    from skyjo.training.callbacks import EvalCallback
    from stable_baselines3.common.logger import configure

    cfg = tiny_cfg(tmp_path, 64)
    env = make_vec_env(cfg["env"], 1)
    model = build_model(cfg, env, None)
    model.set_logger(configure(None, []))
    cb = EvalCallback(eval_freq=1, n_games=0, opponents=[], csv_path=tmp_path / "eval.csv",
                      per_n_players={2: {"n_games": 4, "opponents": ["random"], "max_steps": 1200},
                                     3: {"n_games": 3, "opponents": ["random", "cautious"]}},
                      max_steps=2000,
                      stop_if_gap_above={"n3_random": 1000.0})
    cb.init_callback(model)
    row = cb.evaluate()
    assert cb.labels == ["n2_random", "n3_random", "n3_cautious"]
    assert [t[4] for t in cb.tables] == [1200, 2000, 2000]  # per-table cap, else the global one
    for label in cb.labels:
        assert f"{label}_winrate" in row and f"{label}_score_gap" in row
    assert not cb.check_stop(row)
    header = next(csv.reader((tmp_path / "eval.csv").open()))
    assert "n3_cautious_winrate" in header
    env.close()


def test_round_stats_per_number_of_players_and_kind():
    from skyjo.training.callbacks import TrainStatsCallback

    games = [(("all", "2j", "cautious"), 5, 0, 2), (("all", "4j", "cautious", "recent"), 4, 1, 1)]
    shares = TrainStatsCallback.round_shares(games)
    assert shares["all"] == (2, 1 / 9, 3 / 9)
    assert shares["4j"] == (1, 1 / 4, 1 / 4) and shares["recent"] == (1, 1 / 4, 1 / 4)
    assert shares["cautious"] == (2, 1 / 9, 3 / 9)  # a mixed table counts for each kind


def test_analyze_opportunities_rule_subject_seizes_every_positive_safe_opportunity(tmp_path):
    import importlib.util

    from skyjo.agents.column_rule import ColumnRuleAgent
    from skyjo.agents.ppo_agent import LoopGuardPPOAgent
    from skyjo.agents.registry import make_variant
    from skyjo.simulation import simulate

    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_opportunities.py"
    spec = importlib.util.spec_from_file_location("analyze_opportunities", path)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)

    model = str(tiny_model_file(tmp_path))
    assert type(analyze.make_subject(model)) is LoopGuardPPOAgent
    subject = analyze.make_subject(f"rule:{model}")
    assert isinstance(subject, ColumnRuleAgent) and isinstance(subject.inner, LoopGuardPPOAgent)

    recorder = analyze.Recorder(subject)
    bots = [make_variant("cautious", np.random.default_rng(i + 1)) for i in range(2)]
    simulate([recorder, *bots], 6, np.random.default_rng(0), max_steps=2000, deal_seed=0)
    positive = [seized for _, _, seized, gain in recorder.safe if gain > 0]
    assert positive and all(positive)


def test_analyze_opportunities_against_guarded_model_copies(tmp_path, monkeypatch, capsys):
    import csv
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_opportunities.py"
    spec = importlib.util.spec_from_file_location("analyze_opportunities", path)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    monkeypatch.setattr(analyze, "MODEL", tiny_model_file(tmp_path))

    games = tmp_path / "g.csv"
    analyze.main(["seeker", "3", "3", "1", "--opponents", "1200k_guard", "--per-game-csv", str(games)])
    assert "contre 2 1,2M guard, 3 joueurs" in capsys.readouterr().out
    with open(games, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and {r["table"] for r in rows} == {"1200k_guard"}


@pytest.mark.parametrize("mode", ["guard", "guard-forced", "det", "sample"])
def test_play_vs_ai_runs_full_games(mode, tmp_path, monkeypatch, capsys):
    # The human seat is played by a random agent; the narration covers every AI action.
    import importlib.util
    import sys

    from skyjo.env.actions import action_mask, apply_action
    from skyjo.game.view import make_view

    path = Path(__file__).resolve().parents[1] / "scripts" / "play_vs_ai.py"
    spec = importlib.util.spec_from_file_location("play_vs_ai", path)
    play_vs_ai = importlib.util.module_from_spec(spec)
    sys.modules["play_vs_ai"] = play_vs_ai
    spec.loader.exec_module(play_vs_ai)

    cfg = load_config(CONFIGS / "sanity_random.yaml")
    env = make_vec_env(cfg["env"], n_envs=1)
    build_model(cfg, env, tensorboard_log=None).save(tmp_path / "m.zip")
    env.close()
    human = RandomAgent(np.random.default_rng(0))

    def random_human(game, names):
        apply_action(game, human.act(make_view(game, 0), action_mask(game)))
        while game.current_player == 0 and game.phase.name.startswith("PLACE"):
            apply_action(game, human.act(make_view(game, 0), action_mask(game)))

    monkeypatch.setattr(play_vs_ai, "human_turn", random_human)
    monkeypatch.setattr(play_vs_ai, "human_setup_flip", random_human)
    monkeypatch.setattr(play_vs_ai, "read", lambda prompt: "")
    from skyjo.storage import GameStore

    store = GameStore(tmp_path / "skyjo.db", legacy_csv=None)
    finals = []
    for n in (2, 3, 4):
        agents = play_vs_ai.make_agents(str(tmp_path / "m.zip"), n, mode, seed=n)
        game, columns = play_vs_ai.play(n, n, agents)
        assert game.is_over and len(columns) == n and all(0 <= c <= 4 * game.round_number for c in columns)
        play_vs_ai.log_game(store, mode, n, "m.zip", game, columns)
        finals.append((game.total_scores, columns, game.winners()))
    out = capsys.readouterr().out
    assert "Fin de la manche" in out and ("a gagné" in out or "Égalité" in out)

    games = list(reversed(store.games()))
    assert [g["n_players"] for g in games] == [2, 3, 4]
    assert [(g["scores"], g["columns"], g["winners"]) for g in games] == finals
    assert all(g["pseudo"] == "console" and g["styles"] == [mode] * (g["n_players"] - 1) and g["source"] == "console"
               and g["human_won"] == int(0 in g["winners"]) and not g["replayable"] for g in games)
    play_vs_ai.print_record(store)
    record = capsys.readouterr().out
    assert f"{mode:<8}" in record and "Bilan de 3 parties" in record
    play_vs_ai.main(["--bilan", "--db", str(store.path), "--pseudo", "personne"])
    assert "Aucune partie" in capsys.readouterr().out


def test_columns_removed_counts_removed_columns():
    import importlib.util
    import sys

    from skyjo.game.player_state import PlayerState

    path = Path(__file__).resolve().parents[1] / "scripts" / "play_vs_ai.py"
    spec = importlib.util.spec_from_file_location("play_vs_ai", path)
    play_vs_ai = importlib.util.module_from_spec(spec)
    sys.modules["play_vs_ai"] = play_vs_ai
    spec.loader.exec_module(play_vs_ai)

    player = PlayerState(list(range(12)))
    assert play_vs_ai.columns_removed(player) == 0
    player.remove_column(0)
    player.remove_column(3)
    assert play_vs_ai.columns_removed(player) == 2


# --- step 11: resume with the column rule ---------------------------------------

def test_resume_configs_differ_only_by_the_column_rule(tmp_path):
    rule, control = load_config(CONFIGS / "resume_rule.yaml"), load_config(CONFIGS / "resume_control.yaml")
    assert rule["env"]["column_rule"] and rule["eval"]["column_rule"]
    assert rule["env"]["pool"]["column_rule_kinds"] == ["recent"]
    assert not control["env"]["column_rule"] and not control["eval"]["column_rule"]
    assert control["env"]["pool"]["column_rule_kinds"] == []
    for cfg in (rule, control):
        cfg["name"] = None
        cfg["env"]["column_rule"] = cfg["eval"]["column_rule"] = None
        cfg["env"]["pool"]["column_rule_kinds"] = None
    assert rule == control
    assert rule["init_from"] == "models/frozen/ppo_multi_1200k_seed4.zip"
    assert rule["total_timesteps"] == 1_000_000 and rule["checkpoint"]["freq"] == 250_000
    assert rule["eval"]["freq"] == 500_000
    assert rule["env"]["n_players"] == {2: 0.1, 3: 0.3, 4: 0.6}
    assert rule["ppo"]["learning_rate"] == {"start": 1.0e-4, "end": 1.0e-5}


def test_linear_learning_rate_from_config():
    cfg = load_config(CONFIGS / "sanity_random.yaml")
    cfg["ppo"]["learning_rate"] = {"start": 1.0e-4, "end": 1.0e-5}
    env = make_vec_env(cfg["env"], 1)
    model = build_model(cfg, env, tensorboard_log=None)
    env.close()
    # progress_remaining goes from 1 (start of the run) to 0 (end)
    assert model.lr_schedule(1.0) == pytest.approx(1.0e-4)
    assert model.lr_schedule(0.5) == pytest.approx(5.5e-5)
    assert model.lr_schedule(0.0) == pytest.approx(1.0e-5)


def test_column_rule_journal_line():
    from skyjo.training.train import column_rule_line

    rule, control = load_config(CONFIGS / "resume_rule.yaml"), load_config(CONFIGS / "resume_control.yaml")
    line = column_rule_line(rule)
    assert "active pour l'agent" in line and "avec la règle : recent" in line
    assert "sans la règle : frozen" in line and "évaluation avec la règle" in line
    assert "inactive" in column_rule_line(control)


def test_pool_factory_applies_the_column_rule_to_chosen_kinds(tmp_path):
    from sb3_contrib import MaskablePPO

    from skyjo.agents.column_rule import ColumnRuleAgent
    from skyjo.training.opponent_pool import PoolFactory, SnapshotCallback

    frozen = tiny_model_file(tmp_path, "frozen")
    SnapshotCallback(tmp_path / "pool", every=100).save_snapshot(MaskablePPO.load(str(frozen), device="cpu"), 0)
    factory = PoolFactory({"recent": 1.0, "frozen": 1.0, "cautious": 1.0}, tmp_path / "pool",
                          frozen=[str(frozen)], column_rule_kinds=["recent"])
    rng = np.random.default_rng(0)
    agents = [factory(rng) for _ in range(60)]
    by_kind = {k: [a for a in agents if a.pool_kind == k] for k in ("recent", "frozen", "cautious")}
    assert all(by_kind.values())
    assert all(isinstance(a, ColumnRuleAgent) and isinstance(a.inner, PPOAgent) for a in by_kind["recent"])
    assert all(type(a) is PPOAgent for a in by_kind["frozen"])
    assert all(isinstance(a, CautiousHeuristicAgent) for a in by_kind["cautious"])
    with pytest.raises(ValueError):
        PoolFactory({"recent": 1.0}, tmp_path / "pool", column_rule_kinds=["cautious"])


@pytest.mark.slow
def test_tiny_training_run_with_the_column_rule(tmp_path, capsys):
    cfg = tiny_cfg(tmp_path, 512)
    cfg["env"]["column_rule"] = True
    cfg["eval"]["column_rule"] = True
    cfg["print_updates"] = True
    cfg["ppo"]["learning_rate"] = {"start": 1.0e-4, "end": 1.0e-5}
    run_dir = train(cfg)
    out = capsys.readouterr().out
    assert "Règle de colonne : active pour l'agent" in out
    assert "[règle" in out and "saisies 100.0%" in out
    assert (run_dir / "final.zip").exists()
