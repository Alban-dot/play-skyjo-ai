"""Local web server to play against the AIs (Flask).

Pages: / (game), /aide (help), /replay/<id> (replay of a finished game, every
card revealed), /admin (players, records and games; HTTP Basic authentication
with the ADMIN_TOKEN environment variable as password, disabled without it).

API (JSON):
    GET  /api/styles                -> {"styles": [...], "levels": [...]}
    POST /api/games                 {"pseudo": "Camille", "styles": ["prudent", "expert"], "seed": null}
                                    -> {"state": ..., "events": [...]}   (one style per AI seat)
    GET  /api/games/<id>[?log=1]    -> {"state": ...[, "log": [...]]} (log: the events so far, as sent)
    POST /api/games/<id>/action     {"type": "setup", "positions": [i, j]}
                                    {"type": "draw"} | {"type": "take_discard"}
                                    {"type": "place", "pos": i} | {"type": "flip", "pos": i}
                                    -> {"state": ..., "events": [...]} (the AIs' moves included,
                                       each event with the public state right after it)
    GET  /api/games/<id>/history    -> {"game": ..., "steps": [...]} (403 while the game is not over)
    GET  /api/record?pseudo=Camille  -> {"groups": [...], "games": [...], "note": ...}
    GET  /api/admin/players         -> {"players": [...]}              (admin)
    GET  /api/admin/players/<pseudo> -> {"groups": [...], "games": [...]} (admin)
Cells are numbered 0-11 (row * 4 + column); event texts number them 1-12. A
refused request answers 400 (404 unknown game, 403 history of a game in
progress, 503 too many games) with {"error": message}; the game is unchanged.

Games in memory: at most ``max_games``; a game left alone ``idle_seconds`` is
dropped (abandoned), a finished one after ``finished_seconds`` (it is saved).
The answers come from skyjo.ui.api.GameApi, shared with the browser demo.
"""

from __future__ import annotations

import hmac
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, send_from_directory

from skyjo.agents.ppo_agent import PPOAgent
from skyjo.human_play import DEFAULT_MODEL, ROOT
from skyjo.storage import GameStore
from skyjo.ui.api import (  # noqa: F401 - SessionPool and the errors are re-exported
    RECENT_GAMES,
    SEAT,
    Forbidden,
    Full,
    GameApi,
    NotFound,
    RequestError,
    SessionPool,
)
from skyjo.ui.session import IllegalAction
from skyjo.ui.styles import Style, load_styles

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO


def create_app(model_path: str = DEFAULT_MODEL, store: GameStore | None = None, model: MaskablePPO | None = None,
               styles: dict[str, Style] | None = None, admin_token: str | None = None,
               pool: SessionPool | None = None) -> Flask:
    """The app, with the model loaded once (``model`` skips the loading, for tests).
    Finished games go to ``store`` (default: SKYJO_DB); ``admin_token`` defaults to
    the ADMIN_TOKEN environment variable (none: /admin disabled)."""
    if model is None:
        path = Path(model_path)
        model = PPOAgent.load(path if path.is_absolute() else ROOT / path).model
    store = store if store is not None else GameStore()
    styles = styles if styles is not None else load_styles()
    admin_token = admin_token if admin_token is not None else os.environ.get("ADMIN_TOKEN") or None
    pool = pool if pool is not None else SessionPool()
    api = GameApi(model, store, styles, pool, model_name=model_path)

    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # a modified page is reloaded at once
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    app.config.update(SESSIONS=pool, STORE=store)
    pages = Path(app.static_folder)

    def body():
        return request.get_json(silent=True)

    def admin_only(view):
        @wraps(view)
        def guarded(*args, **kwargs):
            if not admin_token:
                return jsonify(error="Administration désactivée (ADMIN_TOKEN non défini)."), 404
            auth = request.authorization
            given = None
            if auth is not None:
                given = auth.password if auth.type == "basic" else auth.token
            if not given or not hmac.compare_digest(given.encode(), admin_token.encode()):
                return Response("Jeton d'administration requis.", 401,
                                {"WWW-Authenticate": 'Basic realm="Skyjo admin", charset="UTF-8"'})
            return view(*args, **kwargs)
        return guarded

    @app.errorhandler(IllegalAction)
    def refused(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(RequestError)
    def request_error(error):
        return jsonify(error=str(error)), error.status

    @app.after_request
    def headers(response):
        # pages only load their own files: no injected script can run
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    # --- pages -------------------------------------------------------------------------

    @app.get("/")
    def index():
        return send_from_directory(pages, "index.html")

    @app.get("/aide")
    def help_page():
        return send_from_directory(pages, "help.html")

    @app.get("/replay/<game_id>")
    def replay_page(game_id: str):
        return send_from_directory(pages, "replay.html")

    @app.get("/admin")
    @admin_only
    def admin_page():
        return send_from_directory(pages, "admin.html")

    # --- games -------------------------------------------------------------------------

    @app.get("/api/styles")
    def list_styles():
        return jsonify(api.list_styles())

    @app.post("/api/games")
    def start():
        return jsonify(api.start(body()))

    @app.get("/api/games/<game_id>")
    def read(game_id: str):
        return jsonify(api.read(game_id, log=request.args.get("log") == "1"))

    @app.post("/api/games/<game_id>/action")
    def act(game_id: str):
        return jsonify(api.act(game_id, body()))

    @app.get("/api/games/<game_id>/history")
    def history(game_id: str):
        return jsonify(api.history(game_id))

    @app.get("/api/record")
    def bilan():
        return jsonify(api.record(request.args.get("pseudo", "")))

    # --- administration -------------------------------------------------------------------

    @app.get("/api/admin/players")
    @admin_only
    def admin_players():
        return jsonify(players=store.pseudos())

    @app.get("/api/admin/players/<pseudo>")
    @admin_only
    def admin_player(pseudo: str):
        return jsonify(api.record_of(pseudo))

    return app
