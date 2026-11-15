"""p3_reach_baseline_smalltarget5mm_batched.py -- small-target grid
completion, Cell 1 (2026-10-20, external-collaborator spec): does the
BATCHED D1 learner rescue the cortex path that native STDP broke
(SMALLTARGET_ABLATION_FINDING.md's Arm B: ~311mm, flat training
curve)? One factor changed vs Arm B: actor_batched_d1=True instead of
False. Everything else (real recurrent cortex, NOT bypassed, no
gain-field, Python critic, ±5mm target box, random start, continuous
heads) identical to Arm B.

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
