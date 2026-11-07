"""Download the files of Pyodide the browser demo needs (web_demo/), at a fixed
version, and check their SHA-256 before keeping them: the site serves them itself,
no CDN at play time. Used by the publication workflow
(.github/workflows/pages.yml) and for local development. The files are not in git.

Only the Python standard library: it runs on a bare GitHub runner.

Usage: python scripts/fetch_pyodide.py [--out web_demo/pyodide]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

VERSION = "0.29.5"  # same as PYODIDE_VERSION in scripts/build_web_demo.py
SOURCE = f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/"
FILES = {  # name -> SHA-256 (numpy: the same as in pyodide-lock.json)
    "pyodide.js": "7f832a350240263d9946a9c3c877f7bcdab6c37d6dc65f72cd3be5905dca62dd",
    "pyodide.asm.js": "356c42f69e1695397e9d8670bd3c2e678248cde76e18bdae1731e848537b47d7",
    "pyodide.asm.wasm": "54309a5a2cfd757b1f0fdbc9093c92503f7b341500110329d932487183912718",
    "python_stdlib.zip": "831fd1e535084b972f87c6a35275de3d236efd5166f5e77ce89497710635e73c",
    "pyodide-lock.json": "14d2c2dba101277999e17135e653d8f15389ad1437f53eae213bf0c3cdff723d",
    "numpy-2.2.5-cp313-cp313-pyemscripten_2025_0_wasm32.whl":
        "800c98edc0c864dfa49f07005680c699b4b42b84eae1f8cb19d35b3634e7f05c",
}
ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(out: Path, source: str = SOURCE) -> list[str]:
    """Download the missing or altered files into ``out``; problems (nothing kept for a bad file)."""
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, expected in FILES.items():
        target = out / name
        if target.is_file() and sha256(target.read_bytes()) == expected:
            continue
        try:
            with urllib.request.urlopen(source + name, timeout=60) as response:
                data = response.read()
        except OSError as err:
            problems.append(f"{name} : téléchargement impossible ({err})")
            continue
        if sha256(data) != expected:
            problems.append(f"{name} : empreinte SHA-256 inattendue, fichier refusé")
            continue
        target.write_bytes(data)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and check the Pyodide files of the browser demo.")
    parser.add_argument("--out", default=str(ROOT / "web_demo" / "pyodide"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    problems = fetch(out)
    if problems:
        print("Pyodide incomplet :", *problems, sep="\n  - ")
        return 1
    total = sum((out / name).stat().st_size for name in FILES)
    print(f"Pyodide {VERSION} : {len(FILES)} fichiers vérifiés dans {out} ({total / 2**20:.1f} Mo).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
