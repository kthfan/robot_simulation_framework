"""``python -m rsf`` for interpreters without this repository on their path (Isaac Sim's
bundled ``python.sh``, which sets its own PYTHONPATH):

    ~/isaacsim/python.sh tools/rsf_cli.py teach cells/tm_tray/layouts/strict.yaml --workflow ... --out ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsf.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
