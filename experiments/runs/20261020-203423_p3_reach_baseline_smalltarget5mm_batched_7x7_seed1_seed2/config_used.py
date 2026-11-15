"""p3_reach_baseline_smalltarget5mm_batched.py -- does the batched D1
learner rescue the cortex path that native STDP broke on the small
target box (flat ~311mm training curve)? One factor changed vs. the
native-STDP version: actor_batched_d1=True instead of False. Everything
else (real recurrent cortex, no gain-field, Python critic, +/-5mm
target box, random start, continuous heads) is identical.

Built from p3_reach_baseline_smalltarget5mm.py (single import).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_smalltarget5mm", Path(__file__).resolve().parent / "p3_reach_baseline_smalltarget5mm.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline_smalltarget5mm", "p3_reach_baseline_smalltarget5mm_batched")
    cfg.actor_batched_d1 = True
    return cfg
