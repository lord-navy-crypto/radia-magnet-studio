"""Run the repository's deterministic mock and compatibility checks."""

from test_numpy_compat import run as run_numpy_compat
from test_json_and_display_fix import run as run_json_display
from test_strict_correctness import run as run_strict_correctness
from test_smoke import run as run_smoke


def main():
    run_numpy_compat()
    run_json_display()
    run_strict_correctness()
    run_smoke()
    print("ALL TEST SUITES PASSED")


if __name__ == "__main__":
    main()
