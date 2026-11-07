"""The game in the browser demo (web_demo/), run by Pyodide in a Web Worker: the
same requests and answers as the Flask server (skyjo.ui.api.GameApi), carried by
messages instead of HTTP. NumPy only.

Finished games live in the player's browser (web_demo/demo.js keeps them in
localStorage): BrowserStore holds the last MAX_GAMES of them in memory, each with
its actions as (seat, action) pairs (the texts are rebuilt by the replay).
No database, no administration, no rooms.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from urllib.parse import parse_qs, unquote, urlsplit

from skyjo.pseudo import clean_pseudo
from skyjo.records import record
from skyjo.ui.api import GameApi, NotFound, RequestError, SessionPool
from skyjo.ui.session import IllegalAction

MAX_GAMES = 50
DEFAULT_PSEUDO = "Joueur"
GAME_FIELDS = ["id", "pseudo", "date", "n_players", "styles", "seed", "model", "rounds", "scores", "columns",
               "winners", "human_won", "source"]


class BrowserStore:
    """Finished games in memory, the most recent MAX_GAMES (same reading methods as GameStore)."""

    def __init__(self, games: list[dict] = (), limit: int = MAX_GAMES) -> None:
        self.limit = limit
        self._games: list[dict] = []  # oldest first
        for game in games:
            self._keep({**{f: game.get(f) for f in GAME_FIELDS}, "actions": [list(a) for a in game["actions"]]})

    def _keep(self, game: dict) -> None:
        self._games = [g for g in self._games if g["id"] != game["id"]] + [game]
        del self._games[:-self.limit]

    def save_game(self, game: dict, actions=()) -> str:
        row = {"id": secrets.token_hex(16), "date": datetime.now().isoformat(timespec="seconds"), "source": "ui",
               "seed": None, "model": None, **game}
        row["pseudo"] = clean_pseudo(row["pseudo"])
        self._keep({**{f: row[f] for f in GAME_FIELDS}, "actions": [[int(a[0]), int(a[1])] for a in actions]})
        return row["id"]

    def export(self) -> list[dict]:
        """Every game kept, oldest first, as the page stores them."""
        return [dict(g) for g in self._games]

    def game(self, game_id: str) -> dict | None:
        for g in self._games:
            if g["id"] == game_id:
                return {**{f: g[f] for f in GAME_FIELDS}, "replayable": bool(g["actions"])}
        return None

    def actions(self, game_id: str) -> list[tuple[int, int]]:
        for g in self._games:
            if g["id"] == game_id:
                return [(seat, action) for seat, action in g["actions"]]
        return []

    def games(self, pseudo: str | None = None, limit: int | None = None) -> list[dict]:
        """Games, most recent first, with ``replayable``."""
        chosen = [g for g in reversed(self._games) if pseudo is None or g["pseudo"] == pseudo]
        chosen.sort(key=lambda g: g["date"], reverse=True)  # stable: same date, latest saved first
        return [{**{f: g[f] for f in GAME_FIELDS}, "replayable": bool(g["actions"])} for g in chosen[:limit]]

    def record(self, pseudo: str | None = None) -> list[dict]:
        return record(self.games(pseudo))


class Response:
    def __init__(self, status: int, data: dict) -> None:
        self.status_code, self.json = status, data


class BrowserBackend:
    """Answers the page's requests (method, URL of the Flask API, JSON body)."""

    def __init__(self, model, styles, games: list[dict] = (), model_name: str | None = None) -> None:
        self.store = BrowserStore(games)
        self.pool = SessionPool(max_games=20, idle_seconds=float("inf"), finished_seconds=float("inf"))
        kwargs = {"model_name": model_name} if model_name else {}
        self.api = GameApi(model, self.store, styles, self.pool, default_pseudo=DEFAULT_PSEUDO, **kwargs)

    def _route(self, method: str, url: str, body):
        parts = urlsplit(url)
        path = [unquote(p) for p in parts.path.strip("/").split("/")]
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if path[:1] != ["api"]:
            raise NotFound(f"Adresse inconnue : {url}")
        route = (method, *path[1:])
        if route == ("GET", "styles"):
            return self.api.list_styles()
        if route == ("POST", "games"):
            return self.api.start(body)
        if route == ("GET", "record"):
            return self.api.record(query.get("pseudo"))
        if len(path) >= 3 and path[1] == "games":
            game_id, rest = path[2], tuple(path[3:])
            if method == "GET" and rest == ():
                return self.api.read(game_id, log=query.get("log") == "1")
            if method == "POST" and rest == ("action",):
                return self.api.act(game_id, body)
            if method == "GET" and rest == ("history",):
                return self.api.history(game_id)
        raise NotFound(f"Adresse inconnue : {method} {url}")

    def request(self, method: str, url: str, body=None) -> Response:
        """The answer the Flask server would give (status and JSON data)."""
        try:
            return Response(200, self._route(method, url, body))
        except IllegalAction as err:
            return Response(400, {"error": str(err)})
        except RequestError as err:
            return Response(err.status, {"error": str(err)})

    def resume(self, game_id: str, settings: dict, requests: list[dict]) -> Response:
        """Rebuild a game in progress after a reload of the page: same settings (seed
        included), then the same requests of the player, in order (everything else is
        deterministic). The page then reads it as before (``?log=1``)."""
        if game_id in self.pool:
            return Response(200, {"resumed": False})
        try:
            self.api.start(settings, game_id=game_id)
            for data in requests:
                self.api.act(game_id, data)
        except (IllegalAction, RequestError) as err:
            self.pool.games.pop(game_id, None)
            return Response(400, {"error": f"Partie impossible à reprendre : {err}"})
        return Response(200, {"resumed": True})

    # --- for the worker (JSON in, JSON out) -----------------------------------------------

    def handle(self, message: str) -> str:
        """``message``: {"method", "url", "body"} or {"resume": {"id", "settings", "requests"}};
        answer: {"status", "data", "games" (every finished game kept, when one was just saved)}."""
        request = json.loads(message)
        saved_before = [g["id"] for g in self.store.export()]
        if "resume" in request:
            r = request["resume"]
            response = self.resume(r["id"], r["settings"], r["requests"])
        else:
            response = self.request(request["method"], request["url"], request.get("body"))
        answer = {"status": response.status_code, "data": response.json}
        games = self.store.export()
        if [g["id"] for g in games] != saved_before:
            answer["games"] = games
        return json.dumps(answer, ensure_ascii=False)
