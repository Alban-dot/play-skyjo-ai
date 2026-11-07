"""The requests of the game page, whatever carries them: HTTP with the Flask server
(skyjo/ui/server.py), messages to a Web Worker in the browser demo
(skyjo/ui/browser.py, web_demo/). Both send exactly the same answers. NumPy only.

The store is anything with ``game``, ``actions``, ``games`` and ``record`` (and
``save_game``, called by the sessions): skyjo.storage.GameStore on the server,
skyjo.ui.browser.BrowserStore in the browser.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from skyjo.human_play import DEFAULT_MODEL
from skyjo.pseudo import clean_pseudo
from skyjo.records import RECORD_NOTE
from skyjo.ui.session import GameSession, IllegalAction, replay, table_names
from skyjo.ui.styles import LEVELS, Style

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

SEAT = 0  # the seat of the (single) human for now; rooms will map each player to a seat
RECENT_GAMES = 20


class RequestError(Exception):
    status = 400


class NotFound(RequestError):
    status = 404


class Forbidden(RequestError):
    status = 403


class Full(RequestError):
    status = 503


class SessionPool:
    """Games in memory, bounded, with abandoned and finished games dropped."""

    def __init__(self, max_games: int = 100, idle_seconds: float = 7200, finished_seconds: float = 600,
                 clock=time.monotonic) -> None:
        self.max_games, self.idle_seconds, self.finished_seconds, self.clock = \
            max_games, idle_seconds, finished_seconds, clock
        self.games: dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def purge(self) -> None:
        now = self.clock()
        with self._lock:
            for game_id, game in list(self.games.items()):
                idle = now - game.last_active
                if idle > self.idle_seconds or (game.game.is_over and idle > self.finished_seconds):
                    del self.games[game_id]

    def add(self, game: GameSession) -> None:
        self.purge()
        with self._lock:
            if len(self.games) >= self.max_games:
                raise Full("Trop de parties en cours sur ce serveur : réessaie dans quelques minutes.")
            self.games[game.id] = game

    def get(self, game_id: str) -> GameSession:
        self.purge()
        with self._lock:
            if game_id not in self.games:
                raise NotFound(f"Partie inconnue ou expirée : {game_id}")
            return self.games[game_id]

    def __contains__(self, game_id: str) -> bool:
        return game_id in self.games

    def __getitem__(self, game_id: str) -> GameSession:
        return self.games[game_id]

    def __len__(self) -> int:
        return len(self.games)


class GameApi:
    """The answers to the page's requests. ``default_pseudo``: name of a player who
    gives none, and a record without pseudo covers every game of the store (the
    browser, where the store is the player's own); None: a pseudo is required."""

    def __init__(self, model: MaskablePPO, store, styles: dict[str, Style], pool: SessionPool | None = None,
                 model_name: str = DEFAULT_MODEL, default_pseudo: str | None = None) -> None:
        self.model, self.store, self.styles, self.model_name = model, store, styles, model_name
        self.pool = pool if pool is not None else SessionPool()
        self.default_pseudo = default_pseudo

    def style_name(self, key: str) -> str:
        return " + ".join(self.styles[s].name if s in self.styles else s for s in key.split(" + "))

    def game_summary(self, g: dict) -> dict:
        return {"id": g["id"], "date": g["date"], "pseudo": g["pseudo"], "n_players": g["n_players"],
                "styles": [self.style_name(s) for s in g["styles"]], "scores": g["scores"], "winners": g["winners"],
                "human_won": bool(g["human_won"]), "rounds": g["rounds"], "replayable": g["replayable"]}

    def record_of(self, pseudo: str | None) -> dict:
        """Record and recent games of ``pseudo`` (None: every game of the store)."""
        groups = [{**g, "mode": self.style_name(g["mode"])} for g in self.store.record(pseudo)]
        return {"pseudo": pseudo, "groups": groups, "note": RECORD_NOTE,
                "games": [self.game_summary(g) for g in self.store.games(pseudo, limit=RECENT_GAMES)]}

    # --- requests -------------------------------------------------------------------

    def list_styles(self) -> dict:
        return {"styles": [s.public() for s in self.styles.values()], "levels": LEVELS}

    def start(self, data, game_id: str | None = None) -> dict:
        """New game; ``game_id`` gives it a known id (a game rebuilt after a reload of the browser page)."""
        if not isinstance(data, dict):
            raise IllegalAction("Corps de requête JSON (objet) attendu.")
        chosen = data.get("styles")
        if not isinstance(chosen, list) or not all(isinstance(s, str) and s in self.styles for s in chosen):
            raise IllegalAction(f"Styles inconnus : {chosen!r} (connus : {', '.join(self.styles)}).")
        pseudo = data.get("pseudo")
        if self.default_pseudo is not None and pseudo in (None, ""):
            pseudo = self.default_pseudo
        game = GameSession.new(self.model, pseudo, [self.styles[s] for s in chosen], data.get("seed"),
                               model_name=self.model_name, store=self.store)
        if game_id is not None:
            game.id = game_id
        self.pool.add(game)
        return {"state": game.state(SEAT), "events": []}

    def read(self, game_id: str, log: bool = False) -> dict:
        game = self.pool.get(game_id)
        with game.lock:
            if log:  # to rebuild the log of a reloaded page
                return {"state": game.state(SEAT), "log": game.log_for(SEAT)}
            return {"state": game.state(SEAT)}

    def act(self, game_id: str, data) -> dict:
        if not isinstance(data, dict):
            raise IllegalAction("Corps de requête JSON (objet) attendu.")
        game = self.pool.get(game_id)
        events = game.act(SEAT, data)
        with game.lock:
            return {"state": game.state(SEAT), "events": events}

    def history(self, game_id: str) -> dict:
        if game_id in self.pool and not self.pool[game_id].game.is_over:
            raise Forbidden("La partie n'est pas terminée : son historique n'est pas disponible.")
        record = self.store.game(game_id)
        if record is None:
            raise NotFound(f"Partie inconnue ou non terminée : {game_id}")
        if not record["replayable"]:
            raise NotFound("Cette partie (console ou import) n'a pas de relecture.")
        names = table_names(record["pseudo"], [self.styles[s].name if s in self.styles else s
                                               for s in record["styles"]])
        steps = replay(record, self.store.actions(game_id), names)
        return {"game": {**self.game_summary(record), "names": names, "seed": record["seed"]}, "steps": steps}

    def record(self, raw_pseudo: str | None) -> dict:
        if self.default_pseudo is not None and raw_pseudo in (None, ""):
            return self.record_of(None)
        try:
            pseudo = clean_pseudo(raw_pseudo if raw_pseudo is not None else "")
        except ValueError as err:
            raise IllegalAction(str(err)) from None
        return self.record_of(pseudo)
