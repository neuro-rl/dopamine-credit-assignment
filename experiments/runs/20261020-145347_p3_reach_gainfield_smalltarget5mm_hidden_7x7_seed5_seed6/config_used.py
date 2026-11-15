"""p3_reach_gainfield_smalltarget5mm_hidden.py -- small-target
ablation, Arm C (2026-10-20, external-collaborator spec): inserts a
FROZEN random fixed-hidden layer (actor_bypass_fixed_hidden, N_H=750
default) between the gain-field relay and D1/D2, on the working ±5mm
task. Question: does an extra fixed nonlinear layer help, hurt, or do
nothing, compared to the gain-field-direct-to-D1/D2 reference
(p3_reach_gainfield_smalltarget5mm.py, SMALLTARGET_FINDING.md)?

Required a small network.py change (2026-10-20): the existing
actor_gainfield_hidden / actor_bypass_fixed_hidden branches in
Network.__init__ were mutually exclusive (if/elif) -- neither let the
fixed-hidden layer sit AFTER the gain-field relay. Added a new combined
branch (checked first) that builds the gain-field relay exactly as the
gainfield-only path does, then feeds it (not the raw bypass encoder)
into a fixed_indegree frozen random projection to a second
iaf_psc_alpha population, which D1/D2 then reads. No other existing
branch's behavior changed (the combined branch only fires when BOTH
flags are set, which no other existing config does).

Built from p3_reach_gainfield_smalltarget5mm.py (single import) --
ONE change: actor_bypass_fixed_hidden=True (triggers the new combined
network.py branch; actor_gainfield_hidden=True is already inherited).
Everything else (native STDP, cortex bypassed, Python critic,
continuous heads, ±5mm target box, random start) is identical to the
reference. actor_fixed_hidden_size/indegree/weight use their existing
defaults (750 / 30 / 300.0) -- not recalibrated for this stack, since
the spec's own framing treats this as "does adding this help/hurt",
not a condition requiring its own calibration pass.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_smalltarget5mm", Path(__file__).resolve().parent / "p3_reach_gainfield_smalltarget5mm.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_gainfield_smalltarget5mm", "p3_reach_gainfield_smalltarget5mm_hidden")
    cfg.actor_bypass_fixed_hidden = True
    return cfg
