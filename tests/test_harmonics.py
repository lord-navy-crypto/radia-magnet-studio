import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import harmonic_ratios


def run():
    period = 50.0
    z = np.linspace(-287.3, 301.9, 5003)
    u = 2 * np.pi * z / period
    signal = 0.7 + 2e-4 * z + 1.2 * np.sin(u + 0.3)
    signal += 0.24 * np.sin(3 * u - 0.7) + 0.06 * np.cos(5 * u + 0.2)
    result = harmonic_ratios(z, signal, period)
    assert abs(result["H3/H1"] - 0.20) < 0.005, result
    assert abs(result["H5/H1"] - 0.05) < 0.005, result
    print("HARMONIC LEAKAGE-RESISTANCE TEST PASSED")


if __name__ == "__main__":
    run()
