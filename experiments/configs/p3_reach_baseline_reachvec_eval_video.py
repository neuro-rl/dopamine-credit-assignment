"""p3_reach_baseline_reachvec_eval_video.py -- SCRATCH: same greedy
(exploration=0), weights-frozen eval pass as p3_reach_baseline_reachvec_eval.py,
except video_interval is turned back on (every episode) instead of off, so
every eval episode gets a multi-panel spike-raster video -- used to pick a
genuine frozen-policy success episode for a README demo, not for any
reported number (the locked eval run this is based on already reports
numbers with video_interval=0, for speed).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_reachvec_eval",
    Path(__file__).resolve().parent / "p3_reach_baseline_reachvec_eval.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.video_interval = 1
    return cfg
