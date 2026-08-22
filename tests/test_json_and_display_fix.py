import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export.exporters import json_bytes
from analysis.metrics import analyze

def run():
    # Reproduce the shape that caused the real Mac error:
    # metrics contains nested NumPy arrays in trajectory/electron_phase.
    z = np.linspace(-100.0, 100.0, 1001)
    p = 2*np.pi*z/50.0
    B = np.column_stack([
        0.1*np.cos(p),
        0.1552*np.sin(p),
        np.zeros_like(z),
    ])
    metrics = analyze(z, B, 50.0, 3.0)
    params = {
        "period_mm": np.float64(50.0),
        "periods": np.int64(20),
        "segmentation": (1, 1, 1),
    }

    blob = json_bytes(params, metrics)
    parsed = json.loads(blob.decode("utf-8"))

    assert parsed["parameters"]["period_mm"] == 50.0
    assert isinstance(parsed["metrics"]["trajectory"]["x_mm"], list)
    assert isinstance(parsed["metrics"]["electron_phase"]["positions_mm"], list)
    assert parsed["metrics"]["K_peak"] == float(metrics["K_peak"])
    assert parsed["metrics"]["Bperp_peak_T"] == float(metrics["Bperp_peak_T"])

    print("JSON NUMPY-ARRAY SERIALIZATION TEST PASSED")

if __name__ == "__main__":
    run()
