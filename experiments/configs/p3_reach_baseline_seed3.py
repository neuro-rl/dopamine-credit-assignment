"""p3_reach_baseline_seed3.py — p3_reach_baseline.py pinned to seed 3, no
env var needed on the command line. For manually reproducing/comparing
against the locked 20261022-122938_p3_reach_baseline_7x7_seed3_seed4 run.
"""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("P12_SEED", "3")

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_base", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    return _base.Config()
