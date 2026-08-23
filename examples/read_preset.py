"""Minimal preset consumer for a second application."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presets import parse_preset


def main(path):
    runtime = parse_preset(Path(path).read_bytes())
    print(json.dumps({
        "parameters": runtime["parameters"],
        "settings": runtime["settings"],
        "extensions": runtime["extensions"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 examples/read_preset.py PRESET.json")
    main(sys.argv[1])

