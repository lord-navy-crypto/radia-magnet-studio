"""Dependency-aware self-check that does not require pytest."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = [
    "test_numpy_compat.py",
    "test_json_and_display_fix.py",
    "test_interchange.py",
    "test_validation.py",
    "test_harmonics.py",
    "test_downstream_reader.py",
    "test_strict_correctness.py",
    "test_studies.py",
    "test_presets.py",
    "test_integrated_ui.py",
    "test_smoke.py",
]


def main():
    failures = []
    for name in TESTS:
        print(f"RUN  {name}")
        result = subprocess.run([sys.executable, str(ROOT / "tests" / name)], cwd=ROOT)
        if result.returncode:
            failures.append(name)
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print("ALL AVAILABLE SELF-CHECKS PASSED")


if __name__ == "__main__":
    main()
