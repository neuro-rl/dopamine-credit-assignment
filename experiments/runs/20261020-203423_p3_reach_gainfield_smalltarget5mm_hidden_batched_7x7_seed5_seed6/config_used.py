"""p3_reach_gainfield_smalltarget5mm_hidden_batched.py -- small-target
grid completion, Cell 3 (2026-10-20, external-collaborator spec): does
the BATCHED D1 learner rescue the added-depth case that native STDP
broke (SMALLTARGET_ABLATION_FINDING.md's Arm C: ~293mm, flat training
curve, indistinguishable from untrained)? One factor changed vs Arm C:
actor_batched_d1=True instead of False. Everything else (gain-field
representation, cortex bypassed, frozen fixed-hidden layer stacked
after the gain-field relay, Python critic, ±5mm target box, random
start, continuous heads) identical to Arm C.

Built from p3_reach_gainfield_smalltarget5mm_hidden.py (single
import).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_smalltarget5mm_hidden", Path(__file__).resolve().parent / "p3_reach_gainfield_smalltarget5mm_hidden.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_gainfield_smalltarget5mm_hidden", "p3_reach_gainfield_smalltarget5mm_hidden_batched")
    cfg.actor_batched_d1 = True
    return cfg
