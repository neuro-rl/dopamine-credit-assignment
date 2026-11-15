"""p3_reach_gainfield_nativestdp.py -- the load-bearing control the external
collaborator flagged (2026-10-19): every arm result so far, including the
just-completed gain-field negative, trained D1 with BatchedD1Learning (a
Python reimplementation of dopamine-modulated STDP, built as a diagnostic,
with two deliberate simplifications vs the real synapse -- see
batched_actor_learning.py's own docstring) -- never validated against the
native rule, and inconsistent with the paper's own positive spine
(CartPole), which the collaborator confirmed runs on the native
stdp_dopamine_synapse directly. A reviewer would correctly object that the
arm's "credit assignment can't exploit even an ideal representation"
conclusion is a claim about the batched reimplementation, not about
dopamine-STDP, unless dopamine-STDP itself is what's tested.

This config is p3_reach_gainfield.py (the best representation this
investigation produced -- offline decode R^2=0.983) with ONE change:
actor_batched_d1=False. main.py only ever re-zeros dopsyn's real A_plus/
A_minus when actor_batched_d1=True (main.py:2138, every step, since
control_volume_transmitter would otherwise re-enable them past warmup) --
setting it False means the native, continuously-online, three-factor
dopamine-modulated STDP governs cortex->D1 the whole run, with no other
code path changes needed (actor.py always builds the dopsyn connection
itself regardless of this flag; BatchedD1Learning's Python-side
bookkeeping simply doesn't get constructed). D2 is already native
(actor_batched_d2=False is the standing default) -- this makes BOTH
channels native, matching CartPole's spine exactly.

Two decisive outcomes (collaborator's framing, not re-derived here):
native fails too (~similar to the ~260mm batched/untrained-control
result) -> the negative becomes implementation-independent and airtight,
and the arm now runs the same rule as CartPole. Native does meaningfully
better -> the batched detour handicapped the actor and the "rule can't
learn" story needs revising before publication.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield", Path(__file__).resolve().parent / "p3_reach_gainfield.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_gainfield", "p3_reach_gainfield_nativestdp")
    cfg.actor_batched_d1 = False
    return cfg
