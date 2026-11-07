import csv
import sqlite3

import pytest

from skyjo.storage import GameStore, clean_pseudo, db_path, style_key

GAME = {"pseudo": "Camille", "n_players": 3, "styles": ["prudent", "expert"], "seed": 7, "model": "m.zip",
        "rounds": 5, "scores": [80, 104, 95], "columns": [2, 1, 0], "winners": [0], "human_won": 1}


def test_clean_pseudo():
    assert clean_pseudo("  Zoé   la-Rapide ") == "Zoé la-Rapide"
    assert clean_pseudo("R2-D2") == "R2-D2"
    assert clean_pseudo("<i>Bob</i>") == "iBobi"
    for raw in ("", "x", "<>", "a" * 21, None, 3, "   "):
        with pytest.raises(ValueError):
            clean_pseudo(raw)


def test_save_and_read_back(tmp_path):
    store = GameStore(tmp_path / "s.db", legacy_csv=None)
    game_id = store.save_game(GAME, [(0, 12, ["Camille retourne un 3"]), (0, 13, [])])
    assert store.save_game({**GAME, "pseudo": "Zoé", "human_won": 0, "winners": [1], "styles": ["expert"] * 2})
    saved = store.game(game_id)
    assert {k: saved[k] for k in GAME} == GAME and saved["source"] == "ui" and saved["replayable"]
    assert store.actions(game_id) == [(0, 12), (0, 13)]
    assert store.game("missing") is None
    assert [g["pseudo"] for g in store.games()] == ["Zoé", "Camille"]
    assert [p["pseudo"] for p in store.pseudos()] == ["Zoé", "Camille"]
    record = store.record("Camille")
    assert len(record) == 1 and record[0] == {
        "mode": "prudent + expert", "n_players": 3, "games": 1, "wins": 1, "chance": 1 / 3, "my_score": 80.0,
        "ai_score": 99.5, "gap": -19.5, "my_columns": 2.0, "ai_columns": 0.5}
    assert [g["mode"] for g in store.record()] == ["expert", "prudent + expert"]
    # parameterised queries: a hostile pseudo is just a value that matches nothing
    assert store.games("Camille' OR '1'='1") == [] and store.record("x'; DROP TABLE games; --") == []
    assert len(store.games()) == 2
    with sqlite3.connect(store.path) as con:  # the stored pseudo is the cleaned one
        assert con.execute("SELECT COUNT(*) FROM games WHERE pseudo = ?", ("Camille",)).fetchone()[0] == 1


def test_style_key():
    assert style_key(["expert", "expert"]) == "expert"
    assert style_key(["prudent", "expert"]) == "prudent + expert"


def test_path_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYJO_DB", str(tmp_path / "elsewhere" / "x.db"))
    assert db_path() == tmp_path / "elsewhere" / "x.db"
    store = GameStore(legacy_csv=None)
    assert store.path == tmp_path / "elsewhere" / "x.db" and store.path.exists()


def test_old_journal_is_imported_once(tmp_path):
    journal = tmp_path / "human_games.csv"
    fields = ["date", "mode", "n_players", "seed", "model", "rounds", "winners", "human_won", "score_toi",
              "score_ia1", "score_ia2", "score_ia3", "columns_toi", "columns_ia1", "columns_ia2", "columns_ia3"]
    with open(journal, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerow(["2026-11-01T10:00:00", "guard", 2, 1, "m", 5, "toi", 1, 80, 104, "", "", 2, 1, "", ""])
        writer.writerow(["2026-11-01T11:00:00", "ui-det", 4, 3, "m", 4, "toi;ia2", 1, 60, 90, 60, 101, 1, 0, 2, 1])
    store = GameStore(tmp_path / "s.db", legacy_csv=journal)
    games = sorted(store.games(), key=lambda g: g["date"])
    assert [(g["pseudo"], g["styles"], g["scores"], g["winners"], g["source"], g["replayable"]) for g in games] == [
        ("console", ["guard"], [80, 104], [0], "csv", False),
        ("console", ["ui-det"] * 3, [60, 90, 60, 101], [0, 2], "csv", False)]
    assert GameStore(tmp_path / "s.db", legacy_csv=journal).games() and len(store.games()) == 2  # not twice
