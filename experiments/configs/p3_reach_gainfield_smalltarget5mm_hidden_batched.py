"""p3_reach_gainfield_smalltarget5mm_hidden_batched.py -- does the
batched D1 learner rescue the added-depth case that native STDP broke
(flat ~293mm training curve, indistinguishable from untrained)? One
factor changed vs. the native-STDP version: actor_batched_d1=True
instead of False. Everything else (gain-field representation, cortex
bypassed, frozen fixed-hidden layer stacked after the gain-field relay,
Python critic, +/-5mm target box, random start, continuous heads) is
identical.

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
