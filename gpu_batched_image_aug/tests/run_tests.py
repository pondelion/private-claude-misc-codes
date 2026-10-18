"""Minimal test runner (no pytest required): python tests/run_tests.py [name_filter ...]
With pytest installed, ``pytest tests`` runs the same tests."""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    filters = sys.argv[1:]
    failed, passed = [], 0
    for path in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or not callable(getattr(mod, name)):
                continue
            full = f"{path.stem}::{name}"
            if filters and not any(f in full for f in filters):
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {full} ({time.time() - t0:.2f}s)")
            except Exception:  # noqa: BLE001
                failed.append(full)
                print(f"FAIL {full} ({time.time() - t0:.2f}s)")
                traceback.print_exc(limit=3)
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print("  FAILED", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
