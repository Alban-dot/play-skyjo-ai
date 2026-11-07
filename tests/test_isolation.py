"""The game, the bots, the agents and the session logic run with NumPy only (as in
the browser, with Pyodide): importing them in a fresh interpreter loads none of
torch, Stable-Baselines3, gymnasium, Flask, SQLite or PyYAML."""

import json
import subprocess
import sys
from pathlib import Path

from skyjo.ui.styles import read_styles_file

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["torch", "stable_baselines3", "sb3_contrib", "gymnasium", "flask", "sqlite3", "_sqlite3", "yaml", "_yaml"]
MODULES = ["skyjo.ui.session", "skyjo.ui.styles", "skyjo.human_play", "skyjo.pseudo", "skyjo.simulation",
           "skyjo.agents.registry", "skyjo.agents.ppo_agent", "skyjo.agents.column_rule",
           "skyjo.agents.numpy_policy", "skyjo.ui.api", "skyjo.ui.browser", "skyjo.records"]

CHILD = """
import importlib, json, sys
from pathlib import Path
for name in {modules!r}:
    importlib.import_module(name)
from skyjo.ui.styles import load_styles
styles = load_styles(Path({styles!r}))
loaded = sorted({{m.split(".")[0] for m in sys.modules}} & set({forbidden!r}))
print(json.dumps({{"loaded": loaded, "styles": len(styles)}}))
"""


def test_session_logic_and_bots_need_numpy_only(tmp_path):
    styles = tmp_path / "styles.json"
    styles.write_text(json.dumps(read_styles_file(ROOT / "skyjo" / "ui" / "styles.yaml")), encoding="utf-8")
    code = CHILD.format(modules=MODULES, styles=str(styles), forbidden=FORBIDDEN)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["loaded"] == []
    assert result["styles"] > 0
