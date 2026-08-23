import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import analyze
from examples.read_transfer_package import read_package
from export.exporters import research_package_bytes


def run():
    z = np.linspace(-75.0, 75.0, 301)
    phase = 2 * np.pi * z / 50.0
    field = np.column_stack((
        0.1 * np.cos(phase),
        0.15 * np.sin(phase),
        np.zeros_like(z),
    ))
    params = {"device": "Helical", "period_mm": 50.0, "electron_energy_GeV": 3.0}
    metrics = analyze(z, field, 50.0, 3.0)
    package = research_package_bytes(params, metrics, z, field)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "transfer.zip"
        path.write_bytes(package)
        status, config, loaded = read_package(path)

    assert status["payload_files_checked"] == 3
    assert config["parameters"]["device"] == "Helical"
    assert np.allclose(loaded["z_mm"], z)
    assert np.allclose(loaded["Bx_T"], field[:, 0])
    assert np.allclose(loaded["By_T"], field[:, 1])
    assert np.allclose(loaded["Bz_T"], field[:, 2])
    print("DOWNSTREAM READER ROUND-TRIP TEST PASSED")


if __name__ == "__main__":
    run()
