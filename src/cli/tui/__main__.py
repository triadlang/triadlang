
import os
import sys

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from cli.tui import run_tui

if __name__ == "__main__":
    run_tui(sys.argv[1:])
