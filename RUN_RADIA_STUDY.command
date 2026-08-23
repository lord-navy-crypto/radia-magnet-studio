#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"

export RADIA_PYTHONPATH="${RADIA_PYTHONPATH:-$HOME/Desktop/Radia-master/cpp/gcc}"
export PYTHONPATH="$PWD:$RADIA_PYTHONPATH:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-$PWD/studies/example_parameter_scan.json}"
OUTPUT="${2:-$PWD/.radia_studies/terminal}"
WORKERS="${RADIA_STUDY_WORKERS:-2}"

echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo "Workers: $WORKERS"
echo "Press Control-C to stop safely; run this command again to resume."
exec "$PYTHON_BIN" -m studies.cli \
  --config "$CONFIG" --output-dir "$OUTPUT" --workers "$WORKERS"

