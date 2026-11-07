"""Training callbacks: periodic evaluation against bots, training stats."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from skyjo.agents.column_rule import ColumnRuleAgent
from skyjo.agents.ppo_agent import PPOAgent
from skyjo.agents.registry import make_agent
from skyjo.simulation import simulate, wilson_interval


class EvalCallback(BaseCallback):
    """Every ``eval_freq`` env steps, play ``n_games`` deterministic games
    against each bot (seats alternate), log winrate + Wilson interval and the
    score gap to TensorBoard and to ``csv_path``. The model with the best mean
    winrate over the bots is saved to ``best_path``. An opponent can also be a
    frozen model, ``ppo:<path>``, labelled by its file name. With ``max_steps``,
    games longer than that are counted as blocked and excluded.

    ``per_n_players`` ({n: {"n_games", "opponents", optional "max_steps"}}) replaces n_games /
    opponents / n_players: one table per number of players, the model against
    n-1 copies of each opponent, labelled ``n<n>_<opponent>`` (e.g.
    ``n3_cautious``, also the key prefix in the CSV and in the stop rules).
    The best model is then the one with the best mean winrate above chance (1/n).

    ``stop_if_gap_above`` ({bot: gap}) stops the training once the score gap
    against that bot exceeds the threshold at ``stop_patience`` consecutive
    evaluations; ``stop_if_winrate_below`` ({bot: winrate}) likewise once the
    winrate falls below the threshold (an alarm).

    ``plateau`` ({"bot", "window", "min_gain"}) stops it once the mean score
    gap against that bot over the last ``window`` evaluations is not lower
    (better) by at least ``min_gain`` than over the ``window`` before.
    """

    def __init__(
        self,
        eval_freq: int,
        n_games: int,
        opponents: list[str],
        n_players: int = 2,
        seed: int = 0,
        csv_path: str | Path | None = None,
        best_path: str | Path | None = None,
        stop_if_gap_above: dict[str, float] | None = None,
        stop_patience: int = 2,
        stop_if_winrate_below: dict[str, float] | None = None,
        plateau: dict | None = None,
        max_steps: int | None = None,
        per_n_players: dict[int, dict] | None = None,
        column_rule: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.column_rule = column_rule  # the model plays with the column rule (ColumnRuleAgent)
        # one entry per table: (n_players, opponent spec, label, n_games, max_steps)
        if per_n_players:
            self.tables = [(int(n), spec, f"n{n}_{self._label(spec)}", t["n_games"],
                            t.get("max_steps", max_steps))
                           for n, t in sorted(per_n_players.items()) for spec in t["opponents"]]
        else:
            self.tables = [(n_players, spec, self._label(spec), n_games, max_steps) for spec in opponents]
        self.labels = [t[2] for t in self.tables]
        self._frozen_agents: dict[str, PPOAgent] = {}
        self.max_steps = max_steps
        self.seed = seed
        self.csv_path = Path(csv_path) if csv_path else None
        self.best_path = Path(best_path) if best_path else None
        self.best_score = -np.inf
        self.stop_if_gap_above = stop_if_gap_above or {}
        self.stop_patience = stop_patience
        self._over_threshold = 0  # consecutive evaluations above a stop threshold
        self.stop_if_winrate_below = stop_if_winrate_below or {}
        self._under_threshold = 0  # consecutive evaluations below a winrate threshold
        self.plateau = plateau
        self._gaps: list[float] = []  # score gaps against plateau["bot"], one per evaluation
        self.stopped = False
        self._next_eval = eval_freq

    def _on_training_start(self) -> None:
        # On resume: continue the schedule and the best score of the run.
        self._next_eval = (self.num_timesteps // self.eval_freq + 1) * self.eval_freq
        previous = self._read_csv()
        if previous:
            self.best_score = max(
                float(np.mean([float(r[f"{label}_winrate"]) - 1 / n for n, _, label, _, _ in self.tables]))
                for r in previous
            )
            if self.plateau:
                self._gaps = [float(r[f"{self.plateau['bot']}_score_gap"]) for r in previous]
        if not any(int(float(r["timesteps"])) == self.num_timesteps for r in previous):
            self.check_stop(self.evaluate())  # baseline (or state at the resumed checkpoint)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self.check_stop(self.evaluate())
        return not self.stopped

    def check_stop(self, row: dict[str, float]) -> bool:
        """Count consecutive evaluations above a threshold; True once stopped."""
        over = [f"{name} {row[f'{name}_score_gap']:+.1f} > {limit:+.1f}"
                for name, limit in self.stop_if_gap_above.items()
                if row[f"{name}_score_gap"] > limit]
        self._over_threshold = self._over_threshold + 1 if over else 0
        if over and self._over_threshold >= self.stop_patience:
            self.stopped = True
            print(f"*** ARRÊT AUTOMATIQUE au pas {int(row['timesteps']):,} : écart de score "
                  f"au-dessus du seuil à {self._over_threshold} évaluations consécutives "
                  f"({', '.join(over)}) ***", flush=True)
        below = [f"{name} {row[f'{name}_winrate']:.1%} < {limit:.0%}"
                 for name, limit in self.stop_if_winrate_below.items()
                 if row[f"{name}_winrate"] < limit]
        self._under_threshold = self._under_threshold + 1 if below else 0
        if below and self._under_threshold >= self.stop_patience and not self.stopped:
            self.stopped = True
            print(f"*** ALARME, ARRÊT au pas {int(row['timesteps']):,} : winrate sous le seuil "
                  f"à {self._under_threshold} évaluations consécutives ({', '.join(below)}) ***",
                  flush=True)
        if self.plateau and not self.stopped:
            self._check_plateau(row)
        return self.stopped

    def _check_plateau(self, row: dict[str, float]) -> None:
        bot, window, min_gain = self.plateau["bot"], self.plateau["window"], self.plateau["min_gain"]
        self._gaps.append(row[f"{bot}_score_gap"])
        if len(self._gaps) < 2 * window:
            return
        last = float(np.mean(self._gaps[-window:]))
        before = float(np.mean(self._gaps[-2 * window:-window]))
        if last > before - min_gain:
            self.stopped = True
            print(f"*** ARRÊT : PLATEAU au pas {int(row['timesteps']):,} : écart moyen contre "
                  f"{bot} {last:+.1f} sur les {window} dernières évaluations, {before:+.1f} sur "
                  f"les {window} précédentes (amélioration < {min_gain}). Signal, pas une preuve : "
                  f"cette différence est dans le bruit de l'évaluation ***", flush=True)

    def evaluate(self) -> dict[str, float]:
        agent = PPOAgent(self.model, deterministic=True)
        if self.column_rule:
            agent = ColumnRuleAgent(agent)
        rng = np.random.default_rng(self.seed)  # same games at every evaluation
        row: dict[str, float] = {"timesteps": self.num_timesteps}
        excess = []  # winrate above chance (1/n), per table
        for n, spec, name, n_games, max_steps in self.tables:
            if spec.startswith("ppo:"):
                bots = [self._frozen(spec[4:])] * (n - 1)
            else:
                bots = [make_agent(spec, rng) for _ in range(n - 1)]
            result = simulate([agent, *bots], n_games, rng, max_steps=max_steps)
            if result.n_games:
                wr = result.winrate(0)
                lo, hi = wilson_interval(result.wins[0], result.n_games)
                gap = result.score_gap(0)[0]
            else:  # every game blocked
                wr = lo = hi = gap = float("nan")
            values = [("winrate", wr), ("winrate_low", lo), ("winrate_high", hi), ("score_gap", gap)]
            if max_steps:
                values.append(("blocked", result.blocked / n_games))
            for key, value in values:
                self.logger.record(f"eval_{name}/{key}", value)
                row[f"{name}_{key}"] = value
            excess.append(wr - 1 / n)
            if self.verbose:
                blocked = f"  bloquées {result.blocked}" if max_steps else ""
                chance = f" (hasard {1 / n:.0%})" if n > 2 else ""
                print(f"[eval {self.num_timesteps:>9,}] vs {name:<10} winrate {wr:6.1%}{chance} "
                      f"[{lo:.1%}, {hi:.1%}]  écart de score {gap:+6.1f}{blocked}", flush=True)
        self.logger.dump(self.num_timesteps)
        self._write_csv(row)
        if self.verbose and len({t[0] for t in self.tables}) > 1:
            self._print_summary(row)

        mean_wr = float(np.nanmean(excess)) if not np.all(np.isnan(excess)) else -np.inf
        if self.best_path and mean_wr > self.best_score:
            self.best_score = mean_wr
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_path))
        return row

    def _print_summary(self, row: dict[str, float]) -> None:
        """One line per number of players: winrate / score gap side by side for each table."""
        for n in sorted({t[0] for t in self.tables}):
            cells = []
            for tn, _, name, _, _ in self.tables:
                if tn == n:
                    blocked = row.get(f"{name}_blocked", 0.0)
                    games = next(t[3] for t in self.tables if t[2] == name)
                    extra = f", {round(blocked * games)} bloquées" if blocked else ""
                    cells.append(f"{name[len(f'n{n}_'):]} {row[f'{name}_winrate']:.1%} / "
                                 f"{row[f'{name}_score_gap']:+.1f}{extra}")
            print(f"[bilan {self.num_timesteps:>9,}] {n} joueurs (hasard {1 / n:.0%}) : "
                  f"victoires / écart  " + "  |  ".join(cells), flush=True)

    @staticmethod
    def _label(spec: str) -> str:
        """"ppo:<path>" = a frozen model playing deterministically, labelled by its file name."""
        return Path(spec[4:]).stem if spec.startswith("ppo:") else spec

    def _frozen(self, path: str) -> PPOAgent:
        if path not in self._frozen_agents:
            self._frozen_agents[path] = PPOAgent.load(path, deterministic=True)
        return self._frozen_agents[path]

    def _read_csv(self) -> list[dict[str, str]]:
        if self.csv_path is None or not self.csv_path.exists():
            return []
        with self.csv_path.open(newline="") as f:
            return list(csv.DictReader(f))

    def _write_csv(self, row: dict[str, float]) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


class TrainStatsCallback(BaseCallback):
    """Logs the winrate and final score gap of finished training episodes.
    With ``print_updates``, prints the PPO stats of each update (one line)
    and the share of rounds ended by force / finished by the agent, overall,
    per number of players and per opponent kind (a game counts for each kind
    at its table), over the last ``round_window`` games (one line).
    A summary over the whole run is printed at the end of training.

    ``alarms`` stops the training with a message in the log when
    explained_variance stays below ``ev_below`` for ``ev_patience``
    consecutive updates, or when the share of rounds ended by force over the
    last ``round_window`` games exceeds ``forced_above`` (once at least
    ``forced_min_games`` games are counted)."""

    PRINTED = ("explained_variance", "approx_kl", "clip_fraction", "entropy_loss")

    def __init__(self, window: int = 200, print_updates: bool = False, round_window: int = 1000,
                 alarms: dict | None = None) -> None:
        super().__init__()
        self.alarms = alarms or {}
        self._low_ev = 0  # consecutive updates with explained_variance below ev_below
        self.stopped = False
        self.window = window
        self.wins: dict[int, deque[float]] = {}  # per number of players
        self.gaps: dict[int, deque[float]] = {}
        # per game: (groups, rounds, rounds ended by force, rounds finished by the agent);
        # groups = "all", the number of players ("3j") and each opponent kind at the table
        self.games: deque[tuple[tuple[str, ...], int, int, int]] = deque(maxlen=round_window)
        self.totals: dict[str, np.ndarray] = {}  # group -> [games, rounds, forced, finished], whole run
        self.print_updates = print_updates
        self._n_updates = 0
        # per game: agent's placement decisions, positive-gain safe opportunities, seized
        self.columns: deque[tuple[int, int, int]] = deque(maxlen=round_window)

    @staticmethod
    def round_shares(games) -> dict[str, tuple[int, float, float]]:
        """group -> (games, share of rounds ended by force, share finished by the agent)."""
        sums: dict[str, np.ndarray] = {}
        for groups, rounds, forced, finished in games:
            for key in groups:
                sums.setdefault(key, np.zeros(4))
                sums[key] += (1, rounds, forced, finished)
        return {k: (int(s[0]), s[2] / s[1], s[3] / s[1]) for k, s in sums.items()}

    @staticmethod
    def _format(shares: dict[str, tuple[int, float, float]]) -> str:
        n, forced, finished = shares.pop("all")
        by_n = {k: v for k, v in shares.items() if k[0].isdigit()}
        kinds = {k: v for k, v in shares.items() if not k[0].isdigit()}
        forced_n = "  ".join(f"{k} {f:.1%} ({g})" for k, (g, f, _) in sorted(by_n.items()))
        forced_k = "  ".join(f"{k} {f:.1%} ({g})" for k, (g, f, _) in sorted(kinds.items()))
        finished_n = "  ".join(f"{k} {e:.1%}" for k, (_, _, e) in sorted(by_n.items()))
        return (f"{n} parties : clôtures forcées {forced:.1%} [{forced_n}] [{forced_k}]  "
                f"manches finies par l'agent {finished:.1%} [{finished_n}]")

    @staticmethod
    def column_line(columns) -> str:
        """Column opportunities of positive gain of the agent over the games of ``columns``.
        Under the column rule, every one is seized because the mask imposes it."""
        placements, opportunities, seized = np.sum(list(columns), axis=0)
        share = opportunities / placements if placements else 0.0
        rate = seized / opportunities if opportunities else 0.0
        return (f"{len(columns)} parties : occasions sûres à gain positif {opportunities / len(columns):.2f}"
                f"/partie ({share:.1%} des placements), saisies {rate:.1%}")

    def _round_line(self, games) -> str:
        return self._format(self.round_shares(games))

    def _on_training_start(self) -> None:
        self._n_updates = self.model._n_updates  # non-zero when resuming

    def _on_rollout_start(self) -> None:
        # The stats of the previous update are still in the logger (dumped later).
        stats = self.logger.name_to_value
        if self.print_updates and "train/approx_kl" in stats:
            # epochs actually run: fewer than n_epochs when target_kl stopped the update
            epochs = int(stats["train/n_updates"]) - self._n_updates
            self._n_updates = int(stats["train/n_updates"])
            values = "  ".join(f"{key} {stats[f'train/{key}']:+.4f}" for key in self.PRINTED)
            print(f"[ppo  {self.num_timesteps:>9,}] {values}  époques {epochs}", flush=True)
            if self.games:
                print(f"[pool {self.num_timesteps:>9,}] {self._round_line(self.games)}", flush=True)
            if self.columns:
                print(f"[règle {self.num_timesteps:>8,}] {self.column_line(self.columns)}", flush=True)
        if "train/approx_kl" in stats:
            self.check_alarms(stats["train/explained_variance"])

    def check_alarms(self, explained_variance: float) -> bool:
        """Called once per update; True once an alarm stopped the training."""
        if "ev_below" in self.alarms:
            low = explained_variance < self.alarms["ev_below"]
            self._low_ev = self._low_ev + 1 if low else 0
            if self._low_ev >= self.alarms.get("ev_patience", 5) and not self.stopped:
                self.stopped = True
                print(f"*** ALARME, ARRÊT au pas {self.num_timesteps:,} : explained_variance sous "
                      f"{self.alarms['ev_below']} à {self._low_ev} mises à jour consécutives "
                      f"(dernière {explained_variance:+.3f}) ***", flush=True)
        if "forced_above" in self.alarms and len(self.games) >= self.alarms.get("forced_min_games", 200):
            forced = self.round_shares(self.games)["all"][1]
            if forced > self.alarms["forced_above"] and not self.stopped:
                self.stopped = True
                print(f"*** ALARME, ARRÊT au pas {self.num_timesteps:,} : {forced:.1%} de manches "
                      f"clôturées d'office sur les {len(self.games)} dernières parties "
                      f"(seuil {self.alarms['forced_above']:.0%}) ***", flush=True)
        return self.stopped

    def _on_training_end(self) -> None:
        if self.print_updates and self.totals:
            shares = {k: (int(t[0]), t[2] / t[1], t[3] / t[1]) for k, t in self.totals.items()}
            print(f"[pool total] {self._format(shares)}", flush=True)

    def _on_step(self) -> bool:
        ended = set()
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "is_winner" in info:
                totals, me, n = info["total_scores"], info["seat"], info["n_players"]
                others = [t for i, t in enumerate(totals) if i != me]
                for d, value in ((self.wins, float(info["is_winner"])),
                                 (self.gaps, totals[me] - float(np.mean(others)))):
                    d.setdefault(n, deque(maxlen=self.window)).append(value)
                ended.add(n)
                groups = ("all", f"{n}j", *sorted(set(info.get("opponent_kinds", ["?"]))))
                game = (groups, info["round"], info.get("forced_round_ends", 0),
                        info.get("rounds_finished_by_agent", 0))
                self.games.append(game)
                if "column_placements" in info:
                    self.columns.append((info["column_placements"], info["column_opportunities"],
                                         info["column_seized"]))
                for key in groups:
                    self.totals.setdefault(key, np.zeros(4))
                    self.totals[key] += (1, *game[1:])
        if ended:
            single = set(self.wins) == {2}  # keep the 2-player tags of earlier runs
            for n in self.wins:
                suffix = "" if single else f"_{n}j"
                self.logger.record(f"train/winrate{suffix}", float(np.mean(self.wins[n])))
                self.logger.record(f"train/score_gap{suffix}", float(np.mean(self.gaps[n])))
            for key, (_, forced, finished) in self.round_shares(self.games).items():
                self.logger.record(f"pool/forced_share_{key}", forced)
                self.logger.record(f"pool/agent_finish_share_{key}", finished)
            if self.columns:
                placements, opportunities, seized = np.sum(list(self.columns), axis=0)
                self.logger.record("pool/column_opportunity_share", opportunities / max(placements, 1))
                self.logger.record("pool/column_seized_share", seized / max(opportunities, 1))
        return not self.stopped
