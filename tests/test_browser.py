"""The game of the browser demo (skyjo.ui.browser): same answers as the Flask
server, games kept by the page, games rebuilt after a reload."""

import json

import numpy as np
import pytest

from skyjo.agents.ppo_agent import PPOAgent
from skyjo.storage import GameStore
from skyjo.ui.browser import DEFAULT_PSEUDO, BrowserBackend, BrowserStore
from skyjo.ui.styles import load_styles
from tests.test_training import tiny_model_file
from tests.test_ui import BrowserClient, client_for, play, random_action, start

STYLES = load_styles()


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return PPOAgent.load(tiny_model_file(tmp_path_factory.mktemp("browser"))).model


def backend_for(model, games=()):
    return BrowserBackend(model, STYLES, games, model_name="tiny.zip")


def without_ids(data):
    """Answers compared without the ids (games get random ones)."""
    if isinstance(data, dict):
        return {k: None if k in ("id", "saved_id") else without_ids(v) for k, v in data.items()}
    if isinstance(data, list):
        return [without_ids(v) for v in data]
    return data


@pytest.mark.parametrize("styles", [["expert", "prudent"], ["chasseur", "imprevisible", "distrait"]])
def test_same_answers_as_the_flask_server(model, tmp_path, styles):
    _, flask = client_for(model, GameStore(tmp_path / "s.db", legacy_csv=None))
    browser = BrowserClient(backend_for(model))
    rng = np.random.default_rng(5)
    a, b = start(flask, styles, 21), start(browser, styles, 21)
    while True:
        assert without_ids({"state": a}) == without_ids({"state": b})
        if a["game_over"]:
            break
        action = random_action(a, rng)
        ra, rb = flask.post(f"/api/games/{a['id']}/action", json=action), \
            browser.post(f"/api/games/{b['id']}/action", json=action)
        assert ra.status_code == rb.status_code == 200
        assert without_ids(ra.json) == without_ids(rb.json)
        a, b = ra.json["state"], rb.json["state"]
    ha = flask.get(f"/api/games/{a['id']}/history").json
    hb = browser.get(f"/api/games/{b['id']}/history").json
    assert ha["steps"] == hb["steps"]


def test_pseudo_is_optional_and_the_record_covers_the_browser(model):
    browser = BrowserClient(backend_for(model))
    r = browser.post("/api/games", json={"styles": ["prudent"], "seed": 1})
    assert r.status_code == 200 and r.json["state"]["names"][0] == DEFAULT_PSEUDO
    play(browser, ["prudent"], seed=2, pseudo="Camille")
    play(browser, ["expert"], seed=3, pseudo="")
    record = browser.get("/api/record").json
    assert record["pseudo"] is None and sum(g["games"] for g in record["groups"]) == 2
    assert [g["games"] for g in browser.get("/api/record?pseudo=Camille").json["groups"]] == [1]
    assert browser.post("/api/games", json={"styles": ["prudent"], "pseudo": "<>"}).status_code == 400


def test_finished_games_go_to_the_page_and_come_back(model):
    backend = backend_for(model)
    answer = json.loads(backend.handle(json.dumps({"method": "POST", "url": "/api/games",
                                                   "body": {"styles": ["prudent"], "seed": 7}})))
    assert answer["status"] == 200 and "games" not in answer
    state, _ = play(BrowserClient(backend), ["prudent"], seed=8)
    games = backend.store.export()
    assert [g["id"] for g in games] == [state["id"]] and all(len(a) == 2 for a in games[0]["actions"])
    # a new page: the games kept in localStorage are given back to the worker
    again = BrowserClient(backend_for(model, json.loads(json.dumps(games))))
    history = again.get(f"/api/games/{state['id']}/history")
    assert history.status_code == 200 and history.json["steps"][-1]["state"]["total_scores"] == state["total_scores"]
    assert again.get("/api/record").json["games"][0]["id"] == state["id"]


def test_the_page_keeps_the_last_50_games():
    store = BrowserStore()
    for i in range(55):
        store.save_game({"id": f"g{i}", "date": f"2026-11-06T10:{i:02d}:00", "pseudo": "Camille", "n_players": 2,
                         "styles": ["prudent"], "rounds": 3, "scores": [40, 101], "columns": [0, 1],
                         "winners": [0], "human_won": 1}, [(0, 1)])
    kept = store.export()
    assert len(kept) == 50 and kept[0]["id"] == "g5" and kept[-1]["id"] == "g54"
    assert [g["id"] for g in store.games(limit=2)] == ["g54", "g53"]


def test_a_game_in_progress_is_rebuilt_after_a_reload(model):
    backend = backend_for(model)
    client = BrowserClient(backend)
    settings = {"pseudo": "Camille", "styles": ["expert", "prudent"], "seed": 12}
    state = client.post("/api/games", json=settings).json["state"]
    rng, requests = np.random.default_rng(0), []
    for _ in range(25):
        action = random_action(state, rng)
        requests.append(action)
        state = client.post(f"/api/games/{state['id']}/action", json=action).json["state"]
    before = client.get(f"/api/games/{state['id']}?log=1").json
    reloaded = backend_for(model)  # a new worker
    r = reloaded.resume(state["id"], settings, requests)
    assert r.status_code == 200 and r.json == {"resumed": True}
    assert BrowserClient(reloaded).get(f"/api/games/{state['id']}?log=1").json == before
    assert reloaded.resume(state["id"], settings, requests).json == {"resumed": False}
    bad = backend_for(model).resume("x", settings, [{"type": "place", "pos": 99}])
    assert bad.status_code == 400 and "reprendre" in bad.json["error"]
