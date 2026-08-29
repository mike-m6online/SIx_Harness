"""Test bootstrap: make the kit root importable regardless of pytest cwd.

Running ``python -m pytest harness_tests`` from the kit root already puts
the root on ``sys.path`` (via ``sys.path[0] = cwd``); this shim keeps the
suite working when pytest is invoked from elsewhere or via an IDE runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KIT_ROOT = str(Path(__file__).resolve().parent.parent)
if _KIT_ROOT not in sys.path:
    sys.path.insert(0, _KIT_ROOT)
