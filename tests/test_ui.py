import base64
import json
import secrets
from urllib.parse import urlencode

import numpy as np
import pytest

from skyjo.agents.column_rule import ForceSafeAgent
from skyjo.agents.column_seeker import ColumnSeekerAgent
from skyjo.agents.heuristic_agent import CautiousHeuristicAgent, HeuristicAgent
from skyjo.agents.ppo_agent import LoopGuardPPOAgent, PPOAgent
from skyjo.agents.random_agent import RandomAgent
from skyjo.game.engine import Phase
from skyjo.storage import GameStore
from skyjo.ui.browser import BrowserBackend
from skyjo.ui.server import SessionPool, create_app
from skyjo.ui.session import GameSession, IllegalAction
from skyjo.ui.styles import load_styles
from tests.test_training import tiny_model_file

STYLES = load_styles()
ADMIN = secrets.token_urlsafe(16)  # admin token of the tests, never written in a file


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return PPOAgent.load(tiny_model_file(tmp_path_factory.mktemp("ui"))).model


@pytest.fixture
def store(tmp_path):
    return GameStore(tmp_path / "skyjo.db", legacy_csv=None)


def client_for(model, store, **kwargs):
    app = create_app("tiny.zip", store, model=model, **kwargs)
    return app.config["SESSIONS"], app.test_client()


class BrowserClient:
    """The browser demo's game (skyjo.ui.browser) behind the interface of Flask's test client."""

    def __init__(self, backend: BrowserBackend) -> None:
        self.backend = backend

    def get(self, url, query_string=None):
        return self.backend.request("GET", url + (f"?{urlencode(query_string)}" if query_string else ""))

    def post(self, url, json=None):
        return self.backend.request("POST", url, json)


def clients(kind, model, store):
    """(sessions, client, store) of the Flask server or of the browser demo."""
    if kind == "flask":
        return (*client_for(model, store), store)
    backend = BrowserBackend(model, STYLES, model_name="tiny.zip")
    return backend.pool, BrowserClient(backend), backend.store


BACKENDS = ["flask", "browser"]


def random_action(state, rng, slow=False):
    """A random legal action of the human, from the legal moves of the state. A slow
    human never flips and replaces visible cards when it can: the AIs end the rounds."""
    legal = state["legal"]
    if state["phase"] == "setup":
        return {"type": "setup", "positions": [int(p) for p in rng.choice(legal["setup"], 2, replace=False)]}
    if state["phase"] == "choose_source":
        return {"type": "take_discard" if legal["take_discard"] and rng.random() < 0.5 else "draw"}
    options = [("place", p) for p in legal["place"]] + [("flip", p) for p in legal["flip"]]
    if slow:
        grid = state["players"][0]["grid"]
        options = [("place", p) for p in legal["place"] if grid[p]["state"] == "visible"] or \
            [("place", p) for p in legal["place"]]
    kind, pos = options[rng.integers(len(options))]
    return {"type": kind, "pos": int(pos)}


def start(client, styles, seed, pseudo="Testeur"):
    r = client.post("/api/games", json={"pseudo": pseudo, "styles": styles, "seed": seed})
    assert r.status_code == 200, r.json
    return r.json["state"]


def play(client, styles, seed, on_response=None, slow=False, pseudo="Testeur", max_requests=5000):
    """A full game through the API with a random human; return the final state and every event."""
    rng = np.random.default_rng(seed)
    state, events = start(client, styles, seed, pseudo), []
    for _ in range(max_requests):
        if state["game_over"]:
            return state, events
        assert state["your_turn"] and state["current_player"] == 0
        r = client.post(f"/api/games/{state['id']}/action", json=random_action(state, rng, slow))
        assert r.status_code == 200, r.json
        state = r.json["state"]
        events += r.json["events"]
        if on_response is not None:
            on_response(r.json)
    raise AssertionError("game not over")


# --- styles ------------------------------------------------------------------------

def test_styles_build_the_right_agents(model):
    from skyjo.ui.styles import make_style_agent

    assert [s.level for s in STYLES.values()] == ["Facile", "Facile", "Moyen", "Moyen", "Difficile", "Expert"]
    for n in (2, 3, 4):
        agents = {sid: make_style_agent(style, model, n, seed=1, seat=1) for sid, style in STYLES.items()}
        assert type(agents["distrait"]) is RandomAgent
        assert type(agents["presse"]) is HeuristicAgent
        assert type(agents["prudent"]) is CautiousHeuristicAgent
        assert type(agents["chasseur"]) is ColumnSeekerAgent
        assert (agents["chasseur"].seek_until_turn, agents["chasseur"].max_bet_value) == (14, 8)
        assert type(agents["imprevisible"]) is PPOAgent and not agents["imprevisible"].deterministic
        expert = agents["expert"]
        if n == 2:
            assert type(expert) is LoopGuardPPOAgent
        else:
            assert type(expert) is ForceSafeAgent and type(expert.inner) is LoopGuardPPOAgent
        for agent in (agents["imprevisible"], expert if n == 2 else expert.inner):
            assert agent.model is model


def test_styles_file_is_checked(tmp_path):
    bad = tmp_path / "styles.yaml"
    for text in ("styles:\n  - {id: a, name: A, level: Champion, description: x, agent: random}\n",
                 "styles:\n  - {id: a, name: A, level: Facile, description: x, agent: model:best}\n",
                 "styles:\n  - {id: a, name: A, level: Facile, description: x, agent: genius}\n"):
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_styles(bad)


# --- full games, storage and replay ---------------------------------------------------------

@pytest.mark.parametrize("styles", [["expert"], ["imprevisible"], ["distrait"], ["chasseur"], ["expert", "expert"],
                                    ["prudent", "presse"], ["expert", "prudent", "chasseur"],
                                    ["distrait", "imprevisible", "presse"]])
@pytest.mark.parametrize("kind", BACKENDS)
def test_full_games_are_saved_and_replayable(model, store, styles, kind):
    _, client, store = clients(kind, model, store)
    n = len(styles) + 1
    state, events = play(client, styles, seed=10 + n)
    kinds = [e["kind"] for e in events]
    assert {"setup", "first_player", "turn", "round_end"} <= set(kinds) and kinds[-1] == "game_end"
    assert kinds.count("round_end") == state["round"]
    assert state["seed"] == 10 + n and state["saved_id"] == state["id"]
    assert [p["style"]["id"] for p in state["players"][1:]] == styles and state["players"][0]["style"] is None
    columns = [sum(sum(c["state"] == "removed" for c in e["grids"][seat]) // 3
                   for e in events if e["kind"] == "round_end") for seat in range(n)]

    saved = store.game(state["id"])
    assert (saved["pseudo"], saved["styles"], saved["n_players"], saved["seed"], saved["rounds"]) == \
        ("Testeur", styles, n, state["seed"], state["round"])
    assert saved["scores"] == state["total_scores"] and saved["winners"] == state["winners"]
    assert saved["columns"] == columns and saved["human_won"] == int(0 in state["winners"])
    assert saved["source"] == "ui" and saved["replayable"]

    history = client.get(f"/api/games/{state['id']}/history").json
    steps = history["steps"]
    assert len(steps) == len(store.actions(state["id"])) + 1
    assert steps[-1]["state"]["total_scores"] == state["total_scores"]
    assert sum("round_end" in s for s in steps) == state["round"]
    assert history["game"]["names"][1:] == [p["name"] for p in state["players"][1:]]
    # the replay shows every card: hidden ones keep their value
    first = steps[0]["state"]["players"]
    assert all(c["state"] == "hidden" and isinstance(c["value"], int) for p in first for c in p["grid"])
    assert all(s["texts"] for s in steps)

    record = client.get("/api/record", query_string={"pseudo": "Testeur"}).json
    assert [(g["n_players"], g["games"]) for g in record["groups"]] == [(n, 1)]
    assert record["games"][0]["id"] == state["id"] and record["games"][0]["replayable"]


def test_history_is_refused_until_the_game_is_over(model, store):
    from skyjo.human_record import log_game

    _, client = client_for(model, store)
    state = start(client, ["prudent"], seed=4)
    r = client.get(f"/api/games/{state['id']}/history")
    assert r.status_code == 403 and "pas terminée" in r.json["error"]
    assert client.get("/api/games/nope/history").status_code == 404
    final, _ = play(client, ["prudent"], seed=4)
    assert client.get(f"/api/games/{final['id']}/history").status_code == 200
    # a console game has no actions: no replay
    from skyjo.agents.base import play_game

    rng = np.random.default_rng(2)
    game = play_game([RandomAgent(rng), RandomAgent(rng)], rng)
    console_id = log_game(store, "guard", 2, "m.zip", game, [0, 0])
    r = client.get(f"/api/games/{console_id}/history")
    assert r.status_code == 404 and "relecture" in r.json["error"]


def test_events_are_told_from_each_seat(model):
    session = GameSession.new(model, "Camille", [STYLES["prudent"], STYLES["prudent"]], seed=3)
    assert session.names == ["Camille", "Le Prudent", "Le Prudent 2"]
    events = session.act(0, {"type": "setup", "positions": [0, 1]})
    mine = events[0]
    assert mine["kind"] == "setup" and mine["text"].startswith("Tu retournes")
    assert session.text(session.events[0], None).startswith("Camille retourne")
    assert any(e["text"].startswith("Le Prudent 2 retourne") for e in events)
    with pytest.raises(IllegalAction, match="pas ton tour"):
        session.act(1, {"type": "draw"})


# --- no leak of hidden information -----------------------------------------------------

def scramble(game, rng):
    """Shuffle every hidden card of the current grids together with the draw pile:
    nothing public changes, so the state sent must not change either."""
    cells = [(p, pos) for p in game.players for pos in p.hidden_positions()]
    pool = [int(p.values[pos]) for p, pos in cells] + game.deck.draw_pile
    rng.shuffle(pool)
    for (p, pos), value in zip(cells, pool):
        p.values[pos] = value
    game.deck.draw_pile = pool[len(cells):]


@pytest.mark.parametrize("styles", [["expert"], ["expert", "chasseur"], ["imprevisible", "prudent", "expert"]])
@pytest.mark.parametrize("kind", BACKENDS)
def test_state_never_leaks_hidden_cards(model, store, styles, kind):
    sessions, client, _ = clients(kind, model, store)
    n_players = len(styles) + 1
    rng = np.random.default_rng(n_players)
    seen = {"checked": 0, "last_turn": 0, "after_reshuffle": 0, "ai_turns": 0}
    previous = {}

    received = []

    def check(response):
        state = response["state"]
        session = sessions[state["id"]]
        if state["id"] != previous.get("id"):
            received.clear()
            previous["id"] = state["id"]
        game = session.game
        # each event carries the state right after it (built by the same state()), the last one = the reply's
        snapshots = [e["state"] for e in response["events"]]
        assert not snapshots or snapshots[-1] == state
        for snapshot in [state, *snapshots]:
            for player in snapshot["players"]:
                assert all((c["value"] is None) == (c["state"] != "visible") for c in player["grid"])
            assert ("seed" in snapshot) == snapshot["game_over"]
        assert all("state" not in e and "states" not in e for e in session.events)  # no snapshot kept
        # the log sent to a reloaded page: the same events as sent during the game, without snapshots
        received.extend({k: v for k, v in e.items() if k != "state"} for e in response["events"])
        if len(received) % 10 < len(response["events"]) or state["game_over"]:  # about every 10 events
            assert client.get(f"/api/games/{state['id']}?log=1").json["log"] == received
        if state["game_over"]:
            return
        assert "seed" not in state and all("seed" not in e for e in response["events"])
        # the same public situation with other hidden cards and another draw order gives the same state
        before = json.dumps(state)
        scramble(game, rng)
        assert json.dumps(client.get(f"/api/games/{state['id']}").json["state"]) == before
        seen["checked"] += 1
        seen["last_turn"] += state["finisher"] is not None
        if previous.get("round") == state["round"] and state["draw_pile_size"] > previous["draw_pile_size"]:
            seen["after_reshuffle"] += 1
        previous.update(round=state["round"], draw_pile_size=state["draw_pile_size"])
        # once per round, nearly empty the draw pile (its cards go under the discard) to force a reshuffle
        if state["phase"] == "choose_source" and len(game.deck.draw_pile) > 2 and previous.get("emptied") != \
                state["round"]:
            game.deck.discard_pile[:0] = game.deck.draw_pile[:-1]
            del game.deck.draw_pile[:-1]
            previous["emptied"] = state["round"]
            previous["draw_pile_size"] = 1
        # 2 players: the card values of the AI's turn are public once it is over
        events = response["events"]
        if n_players == 2 and events and events[-1]["kind"] == "turn" and events[-1]["player"] == 1:
            e, grid = events[-1], state["players"][1]["grid"]
            if e["action"] == "place":
                assert (grid[e["pos"]]["value"], state["top_discard"]) == (e["card"], e["replaced"])
            else:
                assert (grid[e["pos"]]["value"], state["top_discard"]) == (e["flipped"], e["card"])
            seen["ai_turns"] += 1

    for seed in range(3):
        play(client, styles, seed, on_response=check, slow=seed == 2)
    assert seen["checked"] > 100 and seen["last_turn"] > 0 and seen["after_reshuffle"] > 0
    assert seen["ai_turns"] > 20 or n_players > 2


# --- pseudos and illegal requests -------------------------------------------------------------

@pytest.mark.parametrize("raw, clean", [("Camille", "Camille"), ("  Jean   Pierre ", "Jean Pierre"), ("Zoé-42", "Zoé-42"),
                                        ("<b>Al</b>", "bAlb"), ("Al'; DROP TABLE x;--", "Al DROP TABLE x--"),
                                        ("<script>alert(1)</script>", "scriptalert1script")])
def test_pseudos_are_cleaned(model, store, raw, clean):
    _, client = client_for(model, store)
    state = start(client, ["prudent"], seed=1, pseudo=raw)
    assert state["names"][0] == clean


@pytest.mark.parametrize("raw", ["", "a", "<>", "   x   ", "x" * 21, 42, None, "!!!!"])
def test_invalid_pseudos_are_refused(model, store, raw):
    _, client = client_for(model, store)
    r = client.post("/api/games", json={"pseudo": raw, "styles": ["prudent"]})
    assert r.status_code == 400 and "seudo" in r.json["error"]


def test_illegal_actions_are_refused_cleanly(model, store):
    sessions, client = client_for(model, store)
    state = start(client, ["expert"], seed=1)
    url = f"/api/games/{state['id']}"

    def refused(body, code=400, raw=False):
        before = client.get(url).json["state"]
        r = client.post(f"{url}/action", data=body, content_type="application/json") if raw else \
            client.post(f"{url}/action", json=body)
        assert r.status_code == code and isinstance(r.json["error"], str) and r.json["error"], body
        assert client.get(url).json["state"] == before  # nothing played

    def ok(body):
        r = client.post(f"{url}/action", json=body)
        assert r.status_code == 200, r.json
        return r.json["state"]

    for body in ({"type": "draw"}, {"type": "take_discard"}, {"type": "place", "pos": 0}, {"type": "flip", "pos": 0},
                 {"type": "setup", "positions": [0, 0]}, {"type": "setup", "positions": [0]},
                 {"type": "setup", "positions": [0, 12]}, {"type": "setup", "positions": [0, "1"]},
                 {"type": "setup", "positions": [True, 1]}, {"type": "setup"}, {"type": "foo"}, {}, [0, 1]):
        refused(body)
    refused("{not json", raw=True)
    state = ok({"type": "setup", "positions": [0, 1]})
    assert state["phase"] == "choose_source" and state["your_turn"]
    for body in ({"type": "place", "pos": 2}, {"type": "flip", "pos": 2}, {"type": "setup", "positions": [2, 3]}):
        refused(body)

    game = sessions[state["id"]].game
    state = ok({"type": "draw"})
    assert state["phase"] == "place_drawn" and state["hand"] is not None
    for pos in (12, -1, "3", True, None, 1.0):
        refused({"type": "place", "pos": pos})
    refused({"type": "flip", "pos": 0})  # visible card
    refused({"type": "take_discard"})
    game.players[0].remove_column(3)
    refused({"type": "place", "pos": 3})
    refused({"type": "flip", "pos": 7})

    state = ok({"type": "flip", "pos": 2})
    while not (state["your_turn"] and state["phase"] == "choose_source"):
        state = ok(random_action(state, np.random.default_rng(0)))
    state = ok({"type": "take_discard"})
    refused({"type": "flip", "pos": state["legal"]["place"][0]})
    state = ok({"type": "place", "pos": state["legal"]["place"][0]})
    if state["phase"] == "choose_source":
        game.deck.discard_pile.clear()
        refused({"type": "take_discard"})

    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/action", json={"type": "draw"}).status_code == 404
    for params in ({"styles": []}, {"styles": ["expert"] * 4}, {"styles": ["champion"]}, {"styles": "expert"},
                   {"styles": [1]}, {"styles": ["expert"], "seed": -1}, {"styles": ["expert"], "seed": "abc"},
                   {"styles": ["expert"], "seed": 1.5}):
        r = client.post("/api/games", json={"pseudo": "Testeur", **params})
        assert r.status_code == 400 and r.json["error"], params
    assert client.post("/api/games", data="x", content_type="application/json").status_code == 400

    final, _ = play(client, ["expert"], seed=3)
    r = client.post(f"/api/games/{final['id']}/action", json={"type": "draw"})
    assert r.status_code == 400 and "terminée" in r.json["error"]


# --- admin, pages, memory --------------------------------------------------------------------------

def basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def test_admin_requires_its_token(model, store):
    _, client = client_for(model, store, admin_token="")
    for url in ("/admin", "/api/admin/players"):
        assert client.get(url).status_code == 404  # no token configured: disabled

    _, client = client_for(model, store, admin_token=ADMIN)
    play(client, ["prudent"], seed=2, pseudo="Camille")
    for url in ("/admin", "/api/admin/players", "/api/admin/players/Camille"):
        r = client.get(url)
        assert r.status_code == 401 and r.headers["WWW-Authenticate"].startswith("Basic")
        assert client.get(url, headers=basic("admin", "wrong")).status_code == 401
        assert client.get(url, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get(url, headers=basic("admin", ADMIN)).status_code == 200
        assert client.get(url, headers={"Authorization": f"Bearer {ADMIN}"}).status_code == 200
    players = client.get("/api/admin/players", headers=basic("x", ADMIN)).json["players"]
    assert [(p["pseudo"], p["games"]) for p in players] == [("Camille", 1)]
    detail = client.get("/api/admin/players/Camille", headers=basic("x", ADMIN)).json
    assert detail["groups"][0]["mode"] == "Le Prudent" and detail["games"][0]["replayable"]


def test_admin_token_from_the_environment(model, store, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN)
    _, client = client_for(model, store)
    assert client.get("/admin").status_code == 401
    assert client.get("/admin", headers=basic("a", ADMIN)).status_code == 200
    monkeypatch.delenv("ADMIN_TOKEN")
    _, client = client_for(model, store)
    assert client.get("/admin").status_code == 404


def test_pool_limits_games_and_drops_abandoned_ones(model, store):
    now = [0.0]
    pool = SessionPool(max_games=2, idle_seconds=100, finished_seconds=10, clock=lambda: now[0])
    _, client = client_for(model, store, pool=pool)

    def new():
        r = client.post("/api/games", json={"pseudo": "Testeur", "styles": ["prudent"]})
        if r.status_code == 200:
            pool[r.json["state"]["id"]].last_active = now[0]
        return r

    a, b = new(), new()
    assert a.status_code == b.status_code == 200 and len(pool) == 2
    r = new()
    assert r.status_code == 503 and "Trop de parties" in r.json["error"]
    now[0] = 50
    pool[a.json["state"]["id"]].last_active = 50  # a is still played, b is not
    now[0] = 120
    assert new().status_code == 200  # b, idle for 120 s, was dropped
    assert b.json["state"]["id"] not in pool and a.json["state"]["id"] in pool
    assert client.get(f"/api/games/{b.json['state']['id']}").status_code == 404
    # a finished game is kept a little, then dropped (it is in the store)
    game = pool[a.json["state"]["id"]]
    game.game.phase = Phase.GAME_OVER
    game.last_active = now[0]
    now[0] += 11
    pool.purge()
    assert a.json["state"]["id"] not in pool


def test_pages_and_static_files(model, store):
    _, client = client_for(model, store)
    for url, marker in (("/", b'<script src="/static/app.js">'), ("/aide", b"Questions courtes"),
                        ("/replay/abc", b'<script src="/static/replay.js">')):
        r = client.get(url)
        assert r.status_code == 200 and marker in r.data
        assert r.headers["Content-Security-Policy"].startswith("default-src 'self'")
    for name in ("common.js", "app.js", "replay.js", "admin.js", "style.css", "admin.html", "help.html"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200
        text = r.get_data(as_text=True)
        assert "http://" not in text and "https://" not in text and "innerHTML" not in text  # offline, no HTML injection
    styles = client.get("/api/styles").json
    assert [s["name"] for s in styles["styles"]] == [s.name for s in STYLES.values()]
    assert all(set(s) == {"id", "name", "level", "description"} for s in styles["styles"])  # agents stay server side


def test_run_ui_host_option_and_no_debug(model, store, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    from flask import Flask

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ui.py"
    spec = importlib.util.spec_from_file_location("run_ui", path)
    run_ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_ui)
    calls = []
    monkeypatch.setattr(run_ui, "create_app", lambda *a, **k: create_app("tiny.zip", store, model=model))
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("SKYJO_DB", str(store.path))
    run_ui.main(["--port", "8123"])
    run_ui.main(["--port", "8124", "--host", "0.0.0.0"])
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8123" in out and str(store.path) in out
    assert calls == [{"host": "127.0.0.1", "port": 8123, "threaded": True, "debug": False},
                     {"host": "0.0.0.0", "port": 8124, "threaded": True, "debug": False}]
