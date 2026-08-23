#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"

export RADIA_PYTHONPATH="${RADIA_PYTHONPATH:-$HOME/Desktop/Radia-master/cpp/gcc}"
export PYTHONPATH="$PWD:$RADIA_PYTHONPATH:$PYTHONPATH"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import importlib.util, subprocess, sys
mods = {
    "numpy":"numpy",
    "pandas":"pandas",
    "streamlit":"streamlit",
    "plotly":"plotly",
    "h5py":"h5py",
    "matplotlib":"matplotlib",
}
missing=[pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing Python packages:", ", ".join(missing))
    subprocess.check_call([sys.executable,"-m","pip","install","--user",*missing])
PY

"$PYTHON_BIN" - <<'PY'
from radia_support import load_radia
rad = load_radia()
print("RADIA import OK:", getattr(rad, "__file__", "built-in module"))
PY

echo "Starting RADIA Magnet Studio..."
echo "RADIA path: $RADIA_PYTHONPATH"
exec "$PYTHON_BIN" -m streamlit run app/studio.py
