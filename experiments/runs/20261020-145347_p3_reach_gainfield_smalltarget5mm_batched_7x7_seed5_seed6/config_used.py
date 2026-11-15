"""p3_reach_gainfield_smalltarget5mm_batched.py -- small-target
ablation, Arm A (2026-10-20, external-collaborator spec): does the
BATCHED D1 learner (BatchedD1Learning, a Python reimplementation of
dopamine-modulated STDP) also learn the localized ±5mm reach, same as
the native stdp_dopamine_synapse reference
(p3_reach_gainfield_smalltarget5mm.py, SMALLTARGET_FINDING.md)?
Expected ~equivalent per the earlier batched/native equivalence
findings -- this re-confirms it on the working small-target task
specifically, rather than assuming it carries over.

Built from p3_reach_gainfield_smalltarget5mm.py (single import) --
ONE change: actor_batched_d1=True (reverts the nativestdp chain's own
override back to reach_state.py's dataclass default). Everything else
(gain-field representation, cortex bypassed, Python critic, continuous
heads, ±5mm target box, random start) is identical to the reference.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_smalltarget5mm", Path(__file__).resolve().parent / "p3_reach_gainfield_smalltarget5mm.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_gainfield_smalltarget5mm", "p3_reach_gainfield_smalltarget5mm_batched")
    cfg.actor_batched_d1 = True
    return cfg
