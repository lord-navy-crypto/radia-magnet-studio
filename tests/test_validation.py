import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from analysis.metrics import analyze
from export.exporters import fieldmap3d_csv_bytes
from fake_radia import FakeRadia
from devices.factory import build_device
from solver.pipeline import sample_points


def _must_fail(call, text):
    try:
        call()
    except (ValueError, RuntimeError) as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"Expected failure containing {text!r}")


def run():
    field = np.zeros((3, 3))
    _must_fail(lambda: analyze([0, 2, 1], field, 50.0, 3.0), "strictly increasing")
    _must_fail(lambda: analyze([0, 1, 2], np.zeros((2, 3)), 50.0, 3.0), "shape")
    _must_fail(lambda: sample_points(FakeRadia(), 1, [[0, 1]], chunk_size=2), "shape")
    _must_fail(
        lambda: fieldmap3d_csv_bytes([0, 1], [0], [0], np.zeros((1, 1, 1, 3))),
        "shape",
    )
    base = {
        "period_mm": 50.0, "periods": 2, "gap_mm": 12.0,
        "block_width_mm": 10.0, "block_height_mm": 10.0, "br_t": 1.2,
        "material_mode": "Fixed remanence", "blocks_per_period": 4,
    }
    invalid = dict(base, longitudinal_fill=1.2)
    _must_fail(lambda: build_device(FakeRadia(), "Planar", invalid), "longitudinal_fill")
    invalid = dict(base, field_error_pct=-1.0)
    _must_fail(lambda: build_device(FakeRadia(), "Planar", invalid), "field_error_pct")
    print("INPUT VALIDATION TESTS PASSED")


if __name__ == "__main__":
    run()
