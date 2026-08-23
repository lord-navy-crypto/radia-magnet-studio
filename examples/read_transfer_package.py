"""Minimal downstream reader for a Magnet Studio transfer ZIP."""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export.exporters import validate_research_package_bytes


def read_package(path):
    raw = Path(path).read_bytes()
    status = validate_research_package_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        config = json.loads(archive.read("device_config.json"))
        text = archive.read("on_axis_field.csv").decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
    field = {
        "z_mm": [float(row["z_mm"]) for row in rows],
        "Bx_T": [float(row["Bx_T"]) for row in rows],
        "By_T": [float(row["By_T"]) for row in rows],
        "Bz_T": [float(row["Bz_T"]) for row in rows],
    }
    return status, config, field


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 examples/read_transfer_package.py PACKAGE.zip")
    package_status, device_config, on_axis_field = read_package(sys.argv[1])
    print(json.dumps(package_status, indent=2))
    print("Device:", device_config["parameters"].get("device"))
    print("On-axis samples:", len(on_axis_field["z_mm"]))
