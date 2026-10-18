"""Produce the frontend/dist bundle served by the FastAPI dashboard.

Invoke before packaging or launching a FastAPI-hosted deployment. A source-only
Git checkout is not enough because fxbot.api serves frontend/dist as static files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use existing node_modules.")
    args = parser.parse_args()

    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        parser.error("Node.js/npm is required to build frontend/dist")

    if not args.skip_install:
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "test:theme"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)

    dist = FRONTEND / "dist"
    index = dist / "index.html"
    assets = dist / "assets"
    expected = [index, dist / "theme-init.js"]
    if not all(path.is_file() for path in expected):
        raise RuntimeError("The frontend build is incomplete: index.html or theme-init.js missing")
    if not assets.is_dir() or not list(assets.glob("*.js")) or not list(assets.glob("*.css")):
        raise RuntimeError("The frontend build is incomplete: bundled JS/CSS assets missing")
    if "theme-init.js" not in index.read_text(encoding="utf8"):
        raise RuntimeError("Compiled index.html does not reference theme-init.js")

    print(f"Verified FastAPI static frontend: {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
