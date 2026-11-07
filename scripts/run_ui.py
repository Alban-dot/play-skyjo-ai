"""Start the web server to play Skyjo against the AIs.

By default only this computer can reach it (--host 127.0.0.1); never in debug
mode. Finished games go to the SQLite database (SKYJO_DB, default
logs/skyjo.db); /admin is enabled when ADMIN_TOKEN is set (it is the password
asked by the browser).

Usage: python scripts/run_ui.py [--host 127.0.0.1] [--port 8000] [--db logs/skyjo.db]
                                [--model models/frozen/ppo_multi_1200k_seed4.zip]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skyjo.human_play import DEFAULT_MODEL  # noqa: E402
from skyjo.storage import GameStore  # noqa: E402
from skyjo.ui.server import create_app  # noqa: E402

HOST = "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Web server to play Skyjo against the AIs.")
    parser.add_argument("--host", default=HOST, help="address to listen on (0.0.0.0: every network interface)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default=None, help="SQLite database (default: SKYJO_DB, else logs/skyjo.db)")
    args = parser.parse_args(argv)
    store = GameStore(args.db)
    app = create_app(args.model, store)
    print(f"Modèle : {args.model}")
    print(f"Base : {store.path}")
    print(f"Administration : {'/admin' if os.environ.get('ADMIN_TOKEN') else 'désactivée (ADMIN_TOKEN non défini)'}")
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Ouvre http://{shown}:{args.port} dans ton navigateur (Ctrl+C pour arrêter).")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
