import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import trapezoid_integral, analyze

def run():
    x = np.linspace(0.0, 1.0, 1001)
    y = x**2
    integral = trapezoid_integral(y, x)
    assert abs(integral - 1.0/3.0) < 1e-6, integral

    z_mm = np.linspace(-100.0, 100.0, 2001)
    phase = 2*np.pi*z_mm/50.0
    B = np.column_stack([
        0.10*np.cos(phase),
        0.15*np.sin(phase),
        np.zeros_like(z_mm),
    ])
    result = analyze(z_mm, B, 50.0, 3.0)
    for key in ("I1x_Tm", "I1y_Tm", "I2x_Tm2", "I2y_Tm2"):
        assert np.isfinite(result[key]), (key, result[key])

    print("NUMPY INTEGRATION COMPATIBILITY TEST PASSED")

if __name__ == "__main__":
    run()
