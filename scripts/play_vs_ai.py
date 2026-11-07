"""Play Skyjo in the console against the trained model, at 2, 3 or 4 players.

You are "Toi" (seat 1); every other seat is the model. The deal of each round
comes from (seed, round), so --seed replays the same deals.

Modes of the model:
    guard   deterministic, with the loop guard (default; see LoopGuardPPOAgent)
    guard-forced  guard, and always completes a column when the theoretical gain is
            positive (see skyjo.agents.column_rule.ForceSafeAgent)
    det     deterministic: always its preferred action
    sample  actions drawn from its policy (seeded by --seed)

Each finished game is saved in the SQLite database (SKYJO_DB, default
logs/skyjo.db; see skyjo.storage) under --pseudo (default "console"), with the
mode as the style of the AIs; --bilan shows the cumulative result by style and
number of players (of every pseudo, or of --pseudo if given), web games included.

Usage: python scripts/play_vs_ai.py [--players 2|3|4] [--mode guard|guard-forced|det|sample]
                                    [--seed N] [--model models/frozen/ppo_multi_1200k_seed4.zip]
                                    [--pseudo NAME] [--db logs/skyjo.db]
       python scripts/play_vs_ai.py --bilan [--pseudo NAME]
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from play_console import (  # noqa: E402
    Quit,
    ask,
    ask_position,
    describe_old,
    grid_lines,
    position_lines,
    print_blocks,
    read,
)

from skyjo.agents.base import Agent  # noqa: E402
from skyjo.agents.ppo_agent import PPOAgent  # noqa: E402
from skyjo.env.actions import DRAW, FLIP, TAKE_DISCARD, action_mask  # noqa: E402
from skyjo.game.engine import Phase, SkyjoGame  # noqa: E402
from skyjo.game.view import make_view  # noqa: E402
from skyjo.human_play import (  # noqa: E402,F401
    DEFAULT_MODEL,
    HUMAN,
    MODES,
    agents_for,
    columns_removed,
)
from skyjo.human_record import RECORD_NOTE, log_game  # noqa: E402,F401
from skyjo.storage import GameStore, clean_pseudo  # noqa: E402


def scores_str(names: list[str], scores) -> str:
    return " | ".join(f"{name} {s}" for name, s in zip(names, scores))


def show_table(game: SkyjoGame, names: list[str]) -> None:
    print()
    print(f"======== Manche {game.round_number} ========")
    print(f"Scores cumulés : {scores_str(names, game.total_scores)}")
    if game.last_round_scores is not None:
        print(f"Manche précédente : {scores_str(names, game.last_round_scores)}")
    top = game.deck.top_discard
    print(f"Défausse : {top if top is not None else '(vide)'}    "
          f"Pioche : {len(game.deck.draw_pile)} cartes")
    if game.finisher is not None and game.current_player != game.finisher:
        print(f"!! {names[game.finisher]} a tout retourné : DERNIER TOUR !!")
    print()
    blocks = [(f"{names[i]} (visible: {p.visible_score()})", grid_lines(p))
              for i, p in enumerate(game.players) if i != HUMAN]
    me = game.players[HUMAN]
    blocks += [(f"Toi (visible: {me.visible_score()})", grid_lines(me)), ("Positions", position_lines())]
    print_blocks(blocks)
    print()


def human_setup_flip(game: SkyjoGame, names: list[str]) -> None:
    legal = game.legal_setup_positions()
    if len(legal) == 12:
        show_table(game, names)
        print("Début de manche : retourne 2 cartes.")
    pos = ask_position("Carte à retourner (1-12) : ", legal)
    print(f"Tu retournes un {game.setup_flip(pos)}.")


def human_turn(game: SkyjoGame, names: list[str]) -> None:
    show_table(game, names)
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
        keep = "keep"
        if game.legal_flip_positions():
            keep = ask("[g] garder et placer  /  [j] jeter et retourner une carte : ",
                       {"g": "keep", "j": "flip"})
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


def ai_action(game: SkyjoGame, agent: Agent, name: str) -> None:
    """Let the model play one action and describe it."""
    player = game.players[game.current_player]
    action = agent.act(make_view(game, game.current_player), action_mask(game))
    if game.phase == Phase.SETUP_FLIP:
        pos = action - FLIP
        print(f"{name} retourne un {game.setup_flip(pos)} en position {pos + 1}.")
    elif action == DRAW:
        print(f"{name} pioche un {game.draw_from_deck()}.")
    elif action == TAKE_DISCARD:
        print(f"{name} prend le {game.take_discard()} de la défausse.")
    elif action >= FLIP:
        pos, card = action - FLIP, game.hand
        value = int(player.values[pos])
        game.discard_and_flip(pos)
        print(f"{name} jette le {card} et retourne un {value} en position {pos + 1}.")
    else:
        card, old = game.hand, describe_old(player, action)
        game.place(action)
        print(f"{name} place le {card} à la place d'{old}.")


def announce_first_player(game: SkyjoGame, names: list[str]) -> None:
    sums = [p.visible_score() for p in game.players]
    print()
    print(f"Sommes des 2 cartes retournées : {scores_str(names, sums)}")
    print(f"=> {names[game.current_player]} commence (plus grande somme ; égalité : plus grand "
          "score cumulé, puis tirage au sort).")


def show_round_end(game: SkyjoGame, names: list[str], finisher: int) -> None:
    ended = game.round_number if game.is_over else game.round_number - 1
    print()
    print(f"======== Fin de la manche {ended} ========")
    print(f"{names[finisher]} a terminé la manche. Cartes retournées, colonnes complètes retirées :")
    print()
    print_blocks([(names[i], grid_lines(p)) for i, p in enumerate(game.last_round_players)])
    print()
    for i, p in enumerate(game.last_round_players):
        raw, score = p.score(), game.last_round_scores[i]
        note = f"  (doublé : {raw} x 2, pas strictement le plus petit score)" if score != raw else ""
        print(f"  {names[i]} : {score}{note}")
    print(f"Scores cumulés : {scores_str(names, game.total_scores)}")
    if game.is_over:
        win = game.winners()
        print()
        if win == [HUMAN]:
            print("Tu as gagné !")
        elif len(win) > 1:
            print(f"Égalité entre : {', '.join(names[i] for i in win)}.")
        else:
            print(f"{names[win[0]]} a gagné.")
    else:
        read("\nEntrée pour la manche suivante... ")


def make_agents(model_path: str, n_players: int, mode: str, seed: int) -> list[Agent]:
    return agents_for(PPOAgent.load(model_path).model, n_players, mode, seed)


def play(seed: int, n_players: int, agents: list[Agent]) -> tuple[SkyjoGame, list[int]]:
    """Play one game; return it with the columns completed by each player over all rounds."""
    names = ["Toi"] + [f"IA {i}" for i in range(1, n_players)] if n_players > 2 else ["Toi", "IA"]
    game = SkyjoGame(n_players, np.random.default_rng(seed),
                     round_rng=lambda r: np.random.default_rng([seed, r]))
    columns = [0] * n_players
    for agent in agents:
        agent.reset()

    while not game.is_over:
        round_before, finisher_before = game.round_number, game.finisher
        current = game.current_player
        if game.phase == Phase.SETUP_FLIP:
            if current == HUMAN:
                human_setup_flip(game, names)
            else:
                ai_action(game, agents[current - 1], names[current])
            if game.phase == Phase.CHOOSE_SOURCE:
                announce_first_player(game, names)
            continue

        player = game.players[current]
        n_active = len(player.active_positions())
        if current == HUMAN:
            human_turn(game, names)
        else:
            ai_action(game, agents[current - 1], names[current])
            if game.phase in (Phase.PLACE_DRAWN, Phase.PLACE_DISCARD):
                continue  # the same turn goes on

        if game.is_over or game.round_number != round_before:
            columns = [c + columns_removed(p) for c, p in zip(columns, game.last_round_players)]
            show_round_end(game, names, game.last_round_finisher)
            continue
        if len(player.active_positions()) < n_active:
            print(f"Colonne complète ! Elle est retirée (défausse : {game.deck.top_discard}).")
        if game.finisher is not None and finisher_before is None:
            print(f"{names[game.finisher]} a retourné toutes ses cartes !")
    return game, columns


def print_record(store: GameStore, pseudo: str | None = None) -> None:
    """Cumulative result of the human by style and number of players (every pseudo if None)."""
    groups = store.record(pseudo)
    who = f", pseudo {pseudo}" if pseudo is not None else ""
    if not groups:
        print(f"Aucune partie enregistrée ({store.path}{who}).")
        return
    print(f"Bilan de {sum(g['games'] for g in groups)} parties ({store.path}{who})")
    print(f"{'mode':<14}{'joueurs':>8}{'parties':>9}{'victoires':>11}{'hasard':>8}"
          f"{'ton score':>11}{'score IA':>10}{'écart':>8}{'tes col.':>10}{'col. IA':>9}")
    for g in groups:
        wins = f"{g['wins']} ({g['wins'] / g['games']:.0%})"
        print(f"{g['mode']:<14}{g['n_players']:>8}{g['games']:>9}{wins:>11}{g['chance']:>8.0%}"
              f"{g['my_score']:>11.1f}{g['ai_score']:>10.1f}{g['gap']:>+8.1f}{g['my_columns']:>10.1f}"
              f"{g['ai_columns']:>9.1f}")
    print(f"\n{RECORD_NOTE}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play Skyjo against the trained model.")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--mode", default="guard", choices=MODES,
                        help="guard: deterministic with loop guard; guard-forced: guard that always "
                             "completes a column of positive gain; det: deterministic; "
                             "sample: actions drawn from the policy")
    parser.add_argument("--seed", type=int, default=None, help="seed to replay the same deals")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default=None, help="SQLite database (default: SKYJO_DB, else logs/skyjo.db)")
    parser.add_argument("--pseudo", default=None,
                        help="your pseudo (games saved as 'console' without it; --bilan: every pseudo)")
    parser.add_argument("--bilan", action="store_true",
                        help="show the cumulative result by style and number of players, then exit")
    args = parser.parse_args(argv)
    store = GameStore(args.db)
    pseudo = clean_pseudo(args.pseudo) if args.pseudo is not None else None
    if args.bilan:
        print_record(store, pseudo)
        return
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000)

    print(f"Graine : {seed}  (python scripts/play_vs_ai.py --players {args.players} "
          f"--mode {args.mode} --seed {seed} pour rejouer les mêmes donnes)")
    print(f"Modèle : {args.model}, mode {args.mode}")
    print("Positions numérotées de 1 à 12. Tape q pour quitter.")
    agents = make_agents(args.model, args.players, args.mode, seed)
    try:
        game, columns = play(seed, args.players, agents)
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nPartie abandonnée (non enregistrée).")
        return
    log_game(store, args.mode, seed, args.model, game, columns, pseudo or "console")
    print(f"\nPartie enregistrée dans {store.path} ; ton bilan : python scripts/play_vs_ai.py --bilan")


if __name__ == "__main__":
    main()
