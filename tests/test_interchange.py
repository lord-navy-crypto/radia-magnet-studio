import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import analyze
from export.exporters import (
    json_bytes, research_package_bytes, validate_research_package_bytes,
)


def run():
    z = np.linspace(-100.0, 100.0, 401)
    phase = 2 * np.pi * z / 50.0
    field = np.column_stack((0.1 * np.cos(phase), 0.15 * np.sin(phase), np.zeros_like(z)))
    metrics = analyze(z, field, 50.0, 3.0)
    metrics["diagnostic_nan"] = float("nan")
    params = {"device": "Helical", "period_mm": 50.0, "segmentation": (1, 1, 1)}

    strict = json_bytes(params, metrics)
    assert b"NaN" not in strict
    assert json.loads(strict)["metrics"]["diagnostic_nan"] is None

    package = research_package_bytes(
        params, metrics, z, field, blocks=[],
        comparison={"K_peak": {"delta": 0.0}},
        run_metadata={"python_version": "test"},
    )
    status = validate_research_package_bytes(package)
    assert status["schema_version"] == "1.0.0"
    assert status["has_realized_geometry"] is True
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        expected = {"manifest.json", "device_config.json", "analysis_summary.json",
                    "on_axis_field.csv", "device_geometry.json",
                    "ideal_error_comparison.json", "run_metadata.json"}
        assert expected.issubset(set(archive.namelist()))
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema"]["version"] == "1.0.0"
        for item in manifest["files"]:
            data = archive.read(item["path"])
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
        config = json.loads(archive.read("device_config.json"))
        assert config["units"]["magnetic_field"] == "T"
        assert config["coordinate_system"]["z"] == "longitudinal beam direction"

    corrupted = bytearray(package)
    corrupted[-20] ^= 1
    try:
        validate_research_package_bytes(bytes(corrupted))
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted package was not rejected")

    print("INTERCHANGE PACKAGE TESTS PASSED")


if __name__ == "__main__":
    run()
