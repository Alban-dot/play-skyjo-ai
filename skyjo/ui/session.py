"""One game of the web UI, played server side: a list of seats, each human or AI.

For now one human sits at seat 0 against AIs of chosen styles; rooms will only
have to give other seats to other humans: every human seat has its own view
(``state(seat)``), acts with ``act(seat, request)``, and reads the events in
its own words (``events_for(events, seat)``: "Tu pioches" for its own moves).

A client only ever receives public information: ``state(seat)`` is built from
that seat's PlayerView (hidden cards have no value, the draw pile is a count,
the hand only when this seat holds it), and the full grids only appear in the
summary of a finished round, after the final reveal. The seed is sent once the
game is over: with it, every deal of the game could be recomputed.

Every engine action is recorded; a finished game is saved with them in the
store, so that ``replay`` can rebuild it step by step, every card revealed.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from skyjo.agents.base import Agent
from skyjo.env.actions import DRAW, FLIP, PLACE, TAKE_DISCARD, action_mask, apply_action
from skyjo.game.constants import GRID_SIZE, N_COLS
from skyjo.game.engine import Phase, SkyjoGame
from skyjo.game.player_state import CardState
from skyjo.game.view import make_view
from skyjo.human_play import DEFAULT_MODEL, MAX_TABLE, columns_removed, finished_game
from skyjo.pseudo import clean_pseudo
from skyjo.ui.styles import Style, make_style_agent

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

    from skyjo.storage import GameStore

PHASES = {Phase.SETUP_FLIP: "setup", Phase.CHOOSE_SOURCE: "choose_source", Phase.PLACE_DRAWN: "place_drawn",
          Phase.PLACE_DISCARD: "place_discard", Phase.GAME_OVER: "game_over"}
STATES = {CardState.HIDDEN: "hidden", CardState.VISIBLE: "visible", CardState.REMOVED: "removed"}
# action type -> phases where a human may send it
ACTIONS = {"setup": (Phase.SETUP_FLIP,), "draw": (Phase.CHOOSE_SOURCE,), "take_discard": (Phase.CHOOSE_SOURCE,),
           "place": (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD), "flip": (Phase.PLACE_DRAWN,)}
MAX_SEED = 2**63
HUMAN, AI, REPLAY = "human", "ai", "replay"  # kinds of seat (REPLAY: moves read from the store)


class IllegalAction(ValueError):
    """A request the game refuses; the message is shown to the player."""


@dataclass
class Seat:
    kind: str
    name: str
    style: Style | None = None
    agent: Agent | None = None


def card(value: int, hidden: bool = False) -> str:
    return f"un {value}{' caché' if hidden else ''}"


def table_names(human: str, ai_names: list[str]) -> list[str]:
    """Names by seat; a name used twice gets a number ("Le Prudent", "Le Prudent 2")."""
    names, seen = [], {}
    for name in [human, *ai_names]:
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name} {seen[name]}")
    return names


def check_seed(seed) -> int:
    if seed is None:
        return secrets.randbelow(1_000_000)
    if type(seed) is not int or not 0 <= seed < MAX_SEED:
        raise IllegalAction(f"Graine invalide : {seed!r} (entier positif ou nul).")
    return seed


class GameSession:
    """One game: ``seats`` in seat order, deals drawn from ``seed``."""

    def __init__(self, seats: list[Seat], seed: int, model_name: str = DEFAULT_MODEL,
                 store: GameStore | None = None) -> None:
        if not 2 <= len(seats) <= MAX_TABLE:
            raise IllegalAction(f"Nombre de joueurs invalide : {len(seats)} (2 à {MAX_TABLE}).")
        self.id = secrets.token_urlsafe(12)
        self.seats, self.seed, self.model_name, self.store = seats, check_seed(seed), model_name, store
        self.n_players = len(seats)
        self.names = [s.name for s in seats]
        self.game = SkyjoGame(self.n_players, np.random.default_rng(self.seed),
                              round_rng=lambda r: np.random.default_rng([self.seed, r]))
        for seat in seats:
            if seat.agent is not None:
                seat.agent.reset()
        self.columns = [0] * self.n_players
        self.lock = threading.Lock()
        self.last_active = time.monotonic()
        self.saved_id: str | None = None  # id in the store once finished
        self.actions: list[tuple[int, int, list[str]]] = []  # every engine action: seat, action, texts
        self.events: list[dict] = []  # neutral events, without the state snapshots
        self._pending: dict | None = None  # source of the turn in progress
        self._setup: list[tuple[int, int]] = []  # setup flips of the current player
        self.events += [{k: v for k, v in e.items() if k != "states"} for e in self._play_ais()]

    @classmethod
    def new(cls, model: MaskablePPO, pseudo, styles: list[Style], seed=None, model_name: str = DEFAULT_MODEL,
            store: GameStore | None = None) -> GameSession:
        """The human ``pseudo`` at seat 0, then one AI per style."""
        try:
            pseudo = clean_pseudo(pseudo)
        except ValueError as err:
            raise IllegalAction(str(err)) from None
        if not 1 <= len(styles) <= MAX_TABLE - 1:
            raise IllegalAction(f"Il faut de 1 à {MAX_TABLE - 1} styles d'IA (un par siège).")
        seed = check_seed(seed)
        n = len(styles) + 1
        names = table_names(pseudo, [s.name for s in styles])
        seats = [Seat(HUMAN, names[0])] + [Seat(AI, names[i], style, make_style_agent(style, model, n, seed, i))
                                           for i, style in enumerate(styles, start=1)]
        return cls(seats, seed, model_name, store)

    @property
    def human_seats(self) -> list[int]:
        return [i for i, s in enumerate(self.seats) if s.kind == HUMAN]

    # --- actions -------------------------------------------------------------

    def act(self, seat: int, request: dict) -> list[dict]:
        """Play the action of the human at ``seat``, then the AIs until a human must
        decide. Return the events in ``seat``'s words; raise IllegalAction (game
        unchanged) if refused."""
        with self.lock:
            self.last_active = time.monotonic()
            actions = self._parse(seat, request)
            events = []
            for action in actions:
                events += self._apply(seat, action)
            events += self._play_ais()
            self.events += [{k: v for k, v in e.items() if k != "states"} for e in events]
            return self.events_for(events, seat)

    def _parse(self, seat: int, request: dict) -> list[int]:
        game = self.game
        if not isinstance(request, dict):
            raise IllegalAction("L'action doit être un objet JSON.")
        if game.is_over:
            raise IllegalAction("La partie est terminée.")
        if self.seats[seat].kind != HUMAN or game.current_player != seat:
            raise IllegalAction("Ce n'est pas ton tour.")
        kind = request.get("type")
        if kind not in ACTIONS:
            raise IllegalAction(f"Type d'action inconnu : {kind!r} ({', '.join(ACTIONS)}).")
        if game.phase not in ACTIONS[kind]:
            raise IllegalAction(f"Action « {kind} » impossible maintenant (phase {PHASES[game.phase]}).")
        if kind == "setup":
            positions = request.get("positions")
            if (not isinstance(positions, list) or len(positions) != 2
                    or not all(type(p) is int for p in positions) or positions[0] == positions[1]):
                raise IllegalAction("Il faut deux cases différentes à retourner : \"positions\": [i, j].")
            legal = game.legal_setup_positions()
            if any(p not in legal for p in positions):
                raise IllegalAction(f"Cases non retournables : {positions} (cases 0 à {GRID_SIZE - 1}, cachées).")
            return [FLIP + p for p in positions]
        if kind in ("draw", "take_discard"):
            action = DRAW if kind == "draw" else TAKE_DISCARD
            if not action_mask(game)[action]:
                raise IllegalAction("La défausse est vide.")
            return [action]
        pos = request.get("pos")
        if type(pos) is not int or not 0 <= pos < GRID_SIZE:
            raise IllegalAction(f"Case invalide : {pos!r} (entier de 0 à {GRID_SIZE - 1}).")
        action = (PLACE if kind == "place" else FLIP) + pos
        if not action_mask(game)[action]:
            what = "retirée" if kind == "place" else "déjà visible ou retirée"
            raise IllegalAction(f"Case {pos} {what} : action refusée.")
        return [action]

    def _play_ais(self) -> list[dict]:
        events = []
        game = self.game
        while not game.is_over and self.seats[game.current_player].kind == AI:
            seat = game.current_player
            action = self.seats[seat].agent.act(make_view(game, seat), action_mask(game))
            events += self._apply(seat, action)
        return events

    # --- one engine action and its events --------------------------------------

    def _apply(self, seat: int, action: int) -> list[dict]:
        """Play one action and record it; each of its events carries, for every human
        seat, the public state right after it, so that the client can replay the
        AIs' moves one by one."""
        events = self._events_of(seat, action)
        self.actions.append((seat, int(action), [self.text(e, None) for e in events]))
        if self.game.is_over and self.store is not None and self.saved_id is None:
            self._save()
        if events:
            states = {s: self.state(s) for s in self.human_seats}
            for event in events:
                event["states"] = states
        return events

    def _events_of(self, seat: int, action: int) -> list[dict]:
        game = self.game
        player = game.players[seat]
        round_before, finisher_before, phase = game.round_number, game.finisher, game.phase
        removed_before = player.states == CardState.REMOVED
        target = action - FLIP if action >= FLIP else action
        # value at the target cell, public once the action is played (flipped or discarded)
        before = (int(player.values[target]), bool(player.states[target] == CardState.HIDDEN)) \
            if action < DRAW else None
        apply_action(game, action)

        events = []
        if phase == Phase.SETUP_FLIP:
            self._setup.append((target, before[0]))
            if len(self._setup) == 2:
                (p1, v1), (p2, v2) = self._setup
                events.append({"kind": "setup", "player": seat, "positions": [p1, p2], "values": [v1, v2]})
                self._setup = []
            if game.phase == Phase.CHOOSE_SOURCE:
                events.append({"kind": "first_player", "player": game.current_player,
                               "sums": [p.visible_score() for p in game.players]})
            return events
        if phase == Phase.CHOOSE_SOURCE:
            self._pending = {"source": "draw" if action == DRAW else "discard", "card": game.hand}
            return events

        event = {"kind": "turn", "player": seat, **self._pending}
        if action >= FLIP:
            event.update(action="flip", pos=target, flipped=before[0])
        else:
            event.update(action="place", pos=target, replaced=before[0], replaced_hidden=before[1])
        events.append(event)
        self._pending = None
        if game.is_over or game.round_number != round_before:
            self.columns = [c + columns_removed(p) for c, p in zip(self.columns, game.last_round_players)]
            ended = game.round_number if game.is_over else game.round_number - 1
            events.append({"kind": "round_end", "round": ended, **self.last_round()})
            if game.is_over:
                events.append({"kind": "game_end", "winners": game.winners(),
                               "total_scores": list(game.total_scores), "seed": self.seed})
            return events
        new = np.flatnonzero((player.states == CardState.REMOVED) & ~removed_before)
        for col in sorted({int(p) % N_COLS for p in new}):
            events.append({"kind": "column", "player": seat, "column": col, "value": game.deck.top_discard})
        if finisher_before is None and game.finisher is not None:
            events.append({"kind": "last_turn", "player": seat})
        return events

    def _save(self) -> None:
        styles = [s.style.id if s.style else s.kind for s in self.seats[1:]]
        self.saved_id = self.store.save_game(
            {**finished_game(self.game, self.columns), "id": self.id, "pseudo": self.names[0], "styles": styles,
             "seed": self.seed, "model": self.model_name, "source": "ui"}, self.actions)

    # --- words -------------------------------------------------------------------------

    def text(self, event: dict, viewer: int | None) -> str:
        """The event in the words of ``viewer`` ("Tu ..." for its own moves; None: third person)."""
        names = [("Toi" if i == viewer else n) for i, n in enumerate(self.names)]
        seat = event.get("player")
        you = seat is not None and seat == viewer
        name = self.names[seat] if seat is not None else ""
        kind = event["kind"]
        if kind == "setup":
            (p1, p2), (v1, v2) = event["positions"], event["values"]
            verb = "Tu retournes" if you else f"{name} retourne"
            return f"{verb} {card(v1)} en case {p1 + 1} et {card(v2)} en case {p2 + 1}."
        if kind == "first_player":
            detail = ", ".join(f"{n} {s}" for n, s in zip(names, event["sums"]))
            return f"Sommes des cartes retournées : {detail}. {'Tu commences' if you else f'{name} commence'}."
        if kind == "turn":
            value = event["card"]
            if event["source"] == "draw":
                start = f"Tu pioches {card(value)}" if you else f"{name} pioche {card(value)}"
            else:
                start = f"Tu prends le {value} de la défausse" if you else f"{name} prend le {value} de la défausse"
            pos = event["pos"] + 1
            if event["action"] == "flip":
                verb = "le jettes et retournes" if you else "le jette et retourne"
                return f"{start}, {verb} {card(event['flipped'])} en case {pos}."
            verb = "le places" if you else "le place"
            return f"{start} et {verb} en case {pos} (à la place d'{card(event['replaced'], event['replaced_hidden'])})."
        if kind == "column":
            who = "toi" if you else name
            return f"Colonne de {event['value']} complétée par {who} : elle est retirée."
        if kind == "last_turn":
            who = "Tu as retourné toutes tes cartes" if you else f"{name} a retourné toutes ses cartes"
            return f"{who} : dernier tour !"
        if kind == "round_end":
            finisher = event["finisher"]
            scores = " | ".join(f"{n} {s}" for n, s in zip(names, event["scores"]))
            by = "manche arrêtée" if finisher is None else f"terminée par {'toi' if finisher == viewer else self.names[finisher]}"
            return f"Fin de la manche {event['round']}, {by} : {scores}."
        if kind == "game_end":
            winners = event["winners"]
            if viewer is not None and winners == [viewer]:
                result = "Tu as gagné !"
            elif len(winners) > 1:
                result = f"Égalité entre : {', '.join(names[i] for i in winners)}."
            else:
                result = f"{self.names[winners[0]]} a gagné."
            return f"Partie terminée. {result}"
        raise ValueError(f"Unknown event {kind!r}.")

    def events_for(self, events: list[dict], viewer: int) -> list[dict]:
        """Events as sent to ``viewer``: its words and its own state snapshot."""
        return [{**{k: v for k, v in e.items() if k != "states"}, "text": self.text(e, viewer),
                 "state": e["states"][viewer]} for e in events]

    def log_for(self, viewer: int) -> list[dict]:
        """Every event so far as it was sent to ``viewer``, without the state snapshots."""
        return [{**e, "text": self.text(e, viewer)} for e in self.events]

    # --- what a client sees ----------------------------------------------------

    def last_round(self) -> dict | None:
        """The finished round after its final reveal: every card is public."""
        game = self.game
        if game.last_round_players is None:
            return None
        grids = [[{"state": STATES[CardState(s)], "value": None if s == CardState.REMOVED else int(v)}
                  for v, s in zip(p.values, p.states)] for p in game.last_round_players]
        return {"finisher": game.last_round_finisher, "grids": grids,
                "raw_scores": [p.score() for p in game.last_round_players],
                "scores": list(game.last_round_scores), "total_scores": list(game.total_scores)}

    def _players(self, values, states) -> list[dict]:
        players = []
        for seat in range(self.n_players):
            style = self.seats[seat].style
            grid = [{"state": STATES[CardState(s)], "value": None if s == CardState.REMOVED or v is None else int(v)}
                    for v, s in zip(values[seat], states[seat])]
            players.append({"name": self.names[seat], "kind": HUMAN if self.seats[seat].kind == HUMAN else AI,
                            "style": style.public() if style else None, "grid": grid,
                            "visible_score": int(sum(c["value"] for c in grid if c["state"] == "visible"))})
        return players

    def state(self, seat: int) -> dict:
        """Public state from ``seat`` (see the module docstring)."""
        self.last_active = time.monotonic()
        game = self.game
        view = make_view(game, seat)
        mask = action_mask(game) if game.current_player == seat and not game.is_over \
            else np.zeros(TAKE_DISCARD + 1, dtype=bool)
        hidden = [[None if s != CardState.VISIBLE else v for v, s in zip(vals, sts)]
                  for vals, sts in zip(view.values, view.states)]
        setup = view.phase == Phase.SETUP_FLIP
        state = {
            "id": self.id, "you": seat, "n_players": self.n_players, "model": self.model_name,
            "names": self.names, "round": view.round_number, "phase": PHASES[view.phase],
            "current_player": view.current_player, "your_turn": bool(mask.any()),
            "hand": view.hand, "top_discard": view.top_discard, "discard_pile": list(view.discard_pile),
            "draw_pile_size": view.draw_pile_size, "total_scores": list(view.total_scores),
            "finisher": view.finisher, "players": self._players(hidden, view.states),
            "legal": {"setup": [int(p) for p in np.flatnonzero(mask[FLIP:DRAW])] if setup else [],
                      "draw": bool(mask[DRAW]), "take_discard": bool(mask[TAKE_DISCARD]),
                      "place": [int(p) for p in np.flatnonzero(mask[PLACE:FLIP])],
                      "flip": [] if setup else [int(p) for p in np.flatnonzero(mask[FLIP:DRAW])]},
            "last_round": self.last_round(), "game_over": game.is_over,
        }
        if game.is_over:
            state.update(winners=game.winners(), seed=self.seed, saved_id=self.saved_id)
        return state

    def full_state(self) -> dict:
        """Every card with its value, hidden ones included: only for replays of finished games."""
        game = self.game
        values = [p.values for p in game.players]
        states = [p.states for p in game.players]
        return {"round": game.round_number, "phase": PHASES[game.phase], "current_player": game.current_player,
                "hand": game.hand, "top_discard": game.deck.top_discard, "draw_pile_size": len(game.deck.draw_pile),
                "total_scores": list(game.total_scores), "finisher": game.finisher,
                "players": self._players(values, states)}


def replay(record: dict, actions: list[tuple[int, int]], names: list[str]) -> list[dict]:
    """Steps of a finished game rebuilt from its seed and actions: the initial deal,
    then one step per action (seat, texts, full state; ``round_end`` when it ended
    a round). ValueError if the rebuilt game does not end with the recorded scores."""
    session = GameSession([Seat(REPLAY, n) for n in names], record["seed"], record.get("model") or DEFAULT_MODEL)
    steps = [{"seat": None, "pos": None, "texts": ["Donne de la manche 1."], "state": session.full_state()}]
    for seat, action in actions:
        flipped = int(session.game.players[seat].values[action - FLIP]) if FLIP <= action < DRAW else None
        events = session._apply(seat, action)
        texts = [session.text(e, None) for e in events]
        if not texts:  # first half of a turn or of the setup: no event yet
            name, hand = names[seat], session.game.hand
            texts = [f"{name} pioche {card(hand)}." if action == DRAW
                     else f"{name} prend le {hand} de la défausse." if action == TAKE_DISCARD
                     else f"{name} retourne {card(flipped)} en case {action - FLIP + 1}."]
        step = {"seat": seat, "pos": action % GRID_SIZE if action < DRAW else None, "texts": texts,
                "state": session.full_state()}
        ends = [e for e in events if e["kind"] == "round_end"]
        if ends:
            step["round_end"] = {k: v for k, v in ends[0].items() if k != "states"}
        steps.append(step)
    if not session.game.is_over or list(session.game.total_scores) != list(record["scores"]):
        raise ValueError("La partie rejouée ne retrouve pas les scores enregistrés.")
    return steps
