"""p3_reach_baseline_smalltarget50mm_batched.py -- degradation-curve
point 3 of 3 (+/-5mm -> +/-25mm -> +/-50mm), run only if +/-25mm learns.

Same pattern as p3_reach_baseline_smalltarget25mm_batched.py, built
from p3_reach_baseline.py (single import): actor_batched_d1=True,
target_box_halfwidth_m=0.050, same box center (0.29, 0.125).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_smalltarget50mm_batched")
    cfg.actor_batched_d1 = True
    cfg.target_box_halfwidth_m = 0.050
    cfg.target_box_center = (0.29, 0.125)
    return cfg
