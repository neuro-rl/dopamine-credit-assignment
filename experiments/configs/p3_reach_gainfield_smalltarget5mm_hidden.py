"""p3_reach_gainfield_smalltarget5mm_hidden.py -- small-target
ablation, Arm C: inserts a frozen random fixed-hidden layer
(actor_bypass_fixed_hidden, N_H=750 default) between the gain-field
relay and D1/D2, on the working +/-5mm task. Question: does an extra
fixed nonlinear layer help, hurt, or do nothing, compared to the
gain-field-direct-to-D1/D2 reference
(p3_reach_gainfield_smalltarget5mm.py)?

Required a small network.py change: the existing
actor_gainfield_hidden / actor_bypass_fixed_hidden branches in
Network.__init__ were mutually exclusive, so neither let the
fixed-hidden layer sit after the gain-field relay. Added a combined
branch (checked first) that builds the gain-field relay exactly as the
gainfield-only path does, then feeds it (not the raw bypass encoder)
into a fixed_indegree frozen random projection to a second
iaf_psc_alpha population, which D1/D2 then reads; the combined branch
only fires when both flags are set, so no other existing branch's
behavior changes.

Built from p3_reach_gainfield_smalltarget5mm.py (single import) -- one
change: actor_bypass_fixed_hidden=True (actor_gainfield_hidden=True is
already inherited). Everything else (native STDP, cortex bypassed,
Python critic, continuous heads, +/-5mm target box, random start) is
identical to the reference. actor_fixed_hidden_size/indegree/weight
use their existing defaults (750 / 30 / 300.0), not recalibrated for
this stack, since the question here is whether adding this helps or
hurts, not a condition requiring its own calibration pass.
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
