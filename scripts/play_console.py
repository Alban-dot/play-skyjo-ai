"""Play a 2-player game of Skyjo in the console against a random bot.

Usage: python scripts/play_console.py [--seed N]
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.game.constants import N_COLS, N_ROWS  # noqa: E402
from skyjo.game.engine import Phase, SkyjoGame  # noqa: E402
from skyjo.game.player_state import CardState, PlayerState  # noqa: E402

HUMAN, BOT = 0, 1
NAMES = ["Toi", "Bot"]
COL_WIDTH = 20


class Quit(Exception):
    pass


# --- display -----------------------------------------------------------------

def card_str(player: PlayerState, pos: int) -> str:
    state = player.states[pos]
    if state == CardState.REMOVED:
        return "--"
    if state == CardState.HIDDEN:
        return "?"
    return str(int(player.values[pos]))


def grid_lines(player: PlayerState) -> list[str]:
    return [
        " ".join(f"{card_str(player, r * N_COLS + c):>3}" for c in range(N_COLS))
        for r in range(N_ROWS)
    ]


def position_lines() -> list[str]:
    return [
        " ".join(f"{r * N_COLS + c + 1:>3}" for c in range(N_COLS))
        for r in range(N_ROWS)
    ]


def print_blocks(blocks: list[tuple[str, list[str]]]) -> None:
    """Print several grids side by side, each with a title."""
    print("".join(f"{title:<{COL_WIDTH}}" for title, _ in blocks))
    for row in zip(*(lines for _, lines in blocks)):
        print("".join(f"{line:<{COL_WIDTH}}" for line in row))


def scores_str(scores: list[int]) -> str:
    return " | ".join(f"{NAMES[i]} {s}" for i, s in enumerate(scores))


def show_table(game: SkyjoGame) -> None:
    print()
    print(f"======== Manche {game.round_number} ========")
    print(f"Scores cumulés : {scores_str(game.total_scores)}")
    if game.last_round_scores is not None:
        print(f"Manche précédente : {scores_str(game.last_round_scores)}")
    top = game.deck.top_discard
    print(f"Défausse : {top if top is not None else '(vide)'}    "
          f"Pioche : {len(game.deck.draw_pile)} cartes")
    if game.finisher is not None and game.current_player != game.finisher:
        print(f"!! {NAMES[game.finisher]} a tout retourné : DERNIER TOUR !!")
    print()
    bot, me = game.players[BOT], game.players[HUMAN]
    print_blocks([
        (f"Bot (visible: {bot.visible_score()})", grid_lines(bot)),
        (f"Toi (visible: {me.visible_score()})", grid_lines(me)),
        ("Positions", position_lines()),
    ])
    print()


# --- input -------------------------------------------------------------------

def read(prompt: str) -> str:
    raw = input(prompt).strip().lower()
    if raw == "q":
        raise Quit
    return raw


def ask(prompt: str, options: dict[str, str]) -> str:
    while True:
        raw = read(prompt)
        if raw in options:
            return options[raw]
        print(f"Réponds par : {', '.join(options)} (ou q pour quitter).")


def ask_position(prompt: str, legal: list[int]) -> int:
    while True:
        raw = read(prompt)
        if raw.isdigit() and int(raw) - 1 in legal:
            return int(raw) - 1
        print(f"Position invalide. Possibles : {', '.join(str(p + 1) for p in legal)}")


# --- narration helpers -------------------------------------------------------

def describe_old(player: PlayerState, pos: int) -> str:
    """Describe the card about to be replaced (before the move)."""
    hidden = player.states[pos] == CardState.HIDDEN
    return f"un {int(player.values[pos])}{' caché' if hidden else ''} en position {pos + 1}"


# --- setup -------------------------------------------------------------------

def setup_flip(game: SkyjoGame, bot_rng: np.random.Generator) -> None:
    legal = game.legal_setup_positions()
    if game.current_player == HUMAN:
        if len(legal) == 12:
            show_table(game)
            print("Début de manche : retourne 2 cartes.")
        pos = ask_position("Carte à retourner (1-12) : ", legal)
        print(f"Tu retournes un {game.setup_flip(pos)}.")
    else:
        pos = int(bot_rng.choice(legal))
        print(f"Le bot retourne un {game.setup_flip(pos)} en position {pos + 1}.")


def announce_first_player(game: SkyjoGame) -> None:
    sums = [p.visible_score() for p in game.players]
    print()
    print(f"Sommes des 2 cartes retournées : {scores_str(sums)}")
    if sums[HUMAN] == sums[BOT]:
        if game.total_scores[HUMAN] != game.total_scores[BOT]:
            print("Égalité : départage au plus grand score cumulé.")
        else:
            print("Égalité parfaite : tirage au sort.")
    print(f"=> {NAMES[game.current_player]} commence.")


# --- turns -------------------------------------------------------------------

def human_turn(game: SkyjoGame) -> None:
    show_table(game)
    me = game.players[HUMAN]
    options = {"p": "draw"}
    prompt = "[p] piocher"
    if game.can_take_discard():
        options["d"] = "discard"
        prompt += f"  /  [d] prendre le {game.deck.top_discard} de la défausse"
    source = ask(prompt + " : ", options)

    if source == "draw":
        card = game.draw_from_deck()
        print(f"Tu pioches un {card}.")
        if game.legal_flip_positions():
            keep = ask("[g] garder et placer  /  [j] jeter et retourner une carte : ",
                       {"g": "keep", "j": "flip"})
        else:
            keep = "keep"
        if keep == "flip":
            pos = ask_position("Carte cachée à retourner : ", game.legal_flip_positions())
            value = int(me.values[pos])
            game.discard_and_flip(pos)
            print(f"Tu jettes le {card} et retournes un {value} en position {pos + 1}.")
            return
    else:
        card = game.take_discard()
        print(f"Tu prends le {card} de la défausse (tu dois le placer).")

    pos = ask_position(f"Où placer le {card} ? ", game.legal_place_positions())
    old = describe_old(me, pos)
    game.place(pos)
    print(f"Tu places le {card} à la place d'{old}.")


def bot_turn(game: SkyjoGame, rng: np.random.Generator) -> None:
    bot = game.players[BOT]
    if game.can_take_discard() and rng.random() < 0.5:
        card = game.take_discard()
        source = f"prend le {card} de la défausse"
    else:
        card = game.draw_from_deck()
        source = f"pioche un {card}"

    flips = game.legal_flip_positions()
    if flips and rng.random() < 0.5:
        pos = int(rng.choice(flips))
        value = int(bot.values[pos])
        game.discard_and_flip(pos)
        print(f"Le bot {source}, le jette et retourne un {value} en position {pos + 1}.")
    else:
        pos = int(rng.choice(game.legal_place_positions()))
        old = describe_old(bot, pos)
        game.place(pos)
        print(f"Le bot {source} et remplace {old}.")


# --- end of round ------------------------------------------------------------

def show_round_end(game: SkyjoGame, finisher: int) -> None:
    ended = game.round_number if game.is_over else game.round_number - 1
    players = game.last_round_players
    print()
    print(f"======== Fin de la manche {ended} ========")
    print(f"{NAMES[finisher]} a terminé la manche. Cartes retournées, colonnes complètes retirées :")
    print()
    print_blocks([(NAMES[i], grid_lines(p)) for i, p in enumerate(players)])
    print()
    for i, p in enumerate(players):
        raw, score = p.score(), game.last_round_scores[i]
        note = f"  (doublé : {raw} x 2, pas strictement le plus petit score)" if score != raw else ""
        print(f"  {NAMES[i]} : {score}{note}")
    print(f"Scores cumulés : {scores_str(game.total_scores)}")

    if game.is_over:
        win = game.winners()
        print()
        if len(win) > 1:
            print("Égalité ! Vous gagnez tous les deux.")
        else:
            print("Tu as gagné !" if win == [HUMAN] else "Le bot a gagné.")
    else:
        read("\nEntrée pour la manche suivante... ")


# --- main --------------------------------------------------------------------

def play(seed: int) -> SkyjoGame:
    game_seq, bot_seq = np.random.SeedSequence(seed).spawn(2)
    game = SkyjoGame(2, np.random.default_rng(game_seq))
    bot_rng = np.random.default_rng(bot_seq)

    while not game.is_over:
        round_before = game.round_number
        finisher_before = game.finisher

        if game.phase == Phase.SETUP_FLIP:
            setup_flip(game, bot_rng)
            if game.phase == Phase.CHOOSE_SOURCE:
                announce_first_player(game)
            continue

        player = game.players[game.current_player]
        n_active = len(player.active_positions())
        if game.current_player == HUMAN:
            human_turn(game)
        else:
            bot_turn(game, bot_rng)

        if game.is_over or game.round_number != round_before:
            show_round_end(game, finisher_before)
            continue
        if len(player.active_positions()) < n_active:
            print(f"Colonne complète ! Elle est retirée (défausse : {game.deck.top_discard}).")
        if game.finisher is not None and finisher_before is None:
            print(f"{NAMES[game.finisher]} a retourné toutes ses cartes !")
    return game


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play Skyjo against a random bot.")
    parser.add_argument("--seed", type=int, default=None, help="seed to replay a game")
    args = parser.parse_args(argv)
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)

    print(f"Graine : {seed}  (python scripts/play_console.py --seed {seed} pour rejouer)")
    print("Positions numérotées de 1 à 12. Tape q pour quitter.")
    try:
        play(seed)
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nPartie abandonnée.")


if __name__ == "__main__":
    main()
