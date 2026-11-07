"""Finished games in SQLite: the console and the web UI write here, --bilan,
the record page and the admin page read here.

The database path is SKYJO_DB (default logs/skyjo.db). Tables:
    games         one row per finished game (scores, columns and winners by
                  seat as JSON lists, seat 0 = the human; styles = one per AI seat)
    game_actions  every engine action of a web game, in order (seat, action of
                  skyjo.env.actions, texts of its events): with the seed, the
                  game can be replayed exactly
    meta          one-time markers (import of logs/human_games.csv)
Only parameterised queries.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

from skyjo.pseudo import PSEUDO_MAX, PSEUDO_MIN, clean_pseudo  # noqa: F401 - re-exported
from skyjo.records import record, style_key  # noqa: F401 - style_key re-exported

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "logs" / "skyjo.db"
LEGACY_CSV = ROOT / "logs" / "human_games.csv"
CSV_KEYS = ["toi", "ia1", "ia2", "ia3"]  # columns of the old journal, by seat

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id         TEXT PRIMARY KEY,
    pseudo     TEXT NOT NULL,
    date       TEXT NOT NULL,
    n_players  INTEGER NOT NULL,
    styles     TEXT NOT NULL,
    seed       INTEGER,
    model      TEXT,
    rounds     INTEGER NOT NULL,
    scores     TEXT NOT NULL,
    columns    TEXT NOT NULL,
    winners    TEXT NOT NULL,
    human_won  INTEGER NOT NULL,
    source     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS games_pseudo ON games (pseudo);
CREATE TABLE IF NOT EXISTS game_actions (
    game_id  TEXT NOT NULL REFERENCES games (id) ON DELETE CASCADE,
    idx      INTEGER NOT NULL,
    seat     INTEGER NOT NULL,
    action   INTEGER NOT NULL,
    texts    TEXT NOT NULL,
    PRIMARY KEY (game_id, idx)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
GAME_FIELDS = ["id", "pseudo", "date", "n_players", "styles", "seed", "model", "rounds", "scores", "columns",
               "winners", "human_won", "source"]
JSON_FIELDS = ("styles", "scores", "columns", "winners")


def db_path() -> Path:
    return Path(os.environ.get("SKYJO_DB") or DEFAULT_DB)


class GameStore:
    """Access to the database; one short connection per call (safe across threads)."""

    def __init__(self, path: str | Path | None = None, legacy_csv: Path | None = LEGACY_CSV) -> None:
        self.path = Path(path) if path is not None else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(SCHEMA)
        if legacy_csv is not None and not self._meta("csv_imported"):
            n = self.import_csv(legacy_csv) if Path(legacy_csv).exists() else 0
            self._set_meta("csv_imported", f"{n} parties de {legacy_csv}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _meta(self, key: str) -> str | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with closing(self._connect()) as con, con:
            con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # --- writing ---------------------------------------------------------------------

    def save_game(self, game: dict, actions: list[tuple[int, int, list[str]]] = ()) -> str:
        """Insert a finished game (keys of GAME_FIELDS; id, date and source have
        defaults) and its actions (seat, action, texts). Return its id."""
        row = {"id": uuid.uuid4().hex, "date": datetime.now().isoformat(timespec="seconds"), "source": "ui",
               "seed": None, "model": None, **game}
        row["pseudo"] = clean_pseudo(row["pseudo"])
        values = [json.dumps(row[f]) if f in JSON_FIELDS else row[f] for f in GAME_FIELDS]
        with closing(self._connect()) as con, con:
            con.execute(f"INSERT INTO games ({', '.join(GAME_FIELDS)}) VALUES ({', '.join('?' * len(GAME_FIELDS))})",
                        values)
            con.executemany("INSERT INTO game_actions (game_id, idx, seat, action, texts) VALUES (?, ?, ?, ?, ?)",
                            [(row["id"], i, seat, action, json.dumps(texts, ensure_ascii=False))
                             for i, (seat, action, texts) in enumerate(actions)])
        return row["id"]

    def import_csv(self, path: Path, pseudo: str = "console") -> int:
        """Import the old journal (scripts/play_vs_ai.py format); return the number of games."""
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            n = int(r["n_players"])
            keys = CSV_KEYS[:n]
            self.save_game({
                "pseudo": pseudo, "date": r["date"], "n_players": n, "styles": [r["mode"]] * (n - 1),
                "seed": int(r["seed"]) if r["seed"] else None, "model": r["model"], "rounds": int(r["rounds"]),
                "scores": [int(r[f"score_{k}"]) for k in keys], "columns": [int(r[f"columns_{k}"]) for k in keys],
                "winners": [keys.index(w) for w in r["winners"].split(";") if w],
                "human_won": int(r["human_won"]), "source": "csv"})
        return len(rows)

    # --- reading ---------------------------------------------------------------------

    @staticmethod
    def _game(row: sqlite3.Row) -> dict:
        game = dict(row)
        for f in JSON_FIELDS:
            game[f] = json.loads(game[f])
        return game

    def game(self, game_id: str) -> dict | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
            if row is None:
                return None
            game = self._game(row)
            game["replayable"] = con.execute("SELECT COUNT(*) FROM game_actions WHERE game_id = ?",
                                             (game_id,)).fetchone()[0] > 0
        return game

    def actions(self, game_id: str) -> list[tuple[int, int]]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT seat, action FROM game_actions WHERE game_id = ? ORDER BY idx",
                               (game_id,)).fetchall()
        return [(r["seat"], r["action"]) for r in rows]

    def games(self, pseudo: str | None = None, limit: int | None = None) -> list[dict]:
        """Games, most recent first, with ``replayable``."""
        query = ("SELECT g.*, EXISTS (SELECT 1 FROM game_actions a WHERE a.game_id = g.id) AS replayable "
                 "FROM games g")
        params: list = []
        if pseudo is not None:
            query += " WHERE g.pseudo = ?"
            params.append(pseudo)
        query += " ORDER BY g.date DESC, g.rowid DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with closing(self._connect()) as con:
            rows = con.execute(query, params).fetchall()
        games = []
        for row in rows:
            game = self._game(row)
            game["replayable"] = bool(game["replayable"])
            games.append(game)
        return games

    def pseudos(self) -> list[dict]:
        """Every pseudo with its number of games, wins and last game date."""
        with closing(self._connect()) as con:
            rows = con.execute("SELECT pseudo, COUNT(*) AS games, SUM(human_won) AS wins, MAX(date) AS last "
                               "FROM games GROUP BY pseudo ORDER BY last DESC, MAX(rowid) DESC").fetchall()
        return [dict(r) for r in rows]

    def record(self, pseudo: str | None = None) -> list[dict]:
        """Result of the human by (style, number of players): see skyjo.records.record."""
        return record(self.games(pseudo))
