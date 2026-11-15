"""p3_reach_gainfield_nativestdp.py -- the load-bearing control: every
arm result so far, including the gain-field negative, trained D1 with
BatchedD1Learning (a Python reimplementation of dopamine-modulated
STDP, with two deliberate simplifications vs. the real synapse -- see
batched_actor_learning.py's own docstring), never validated against the
native rule, and inconsistent with CartPole's own positive spine, which
runs on the native stdp_dopamine_synapse directly. The arm's "credit
assignment can't exploit even an ideal representation" conclusion is a
claim about the batched reimplementation, not about dopamine-STDP,
unless dopamine-STDP itself is what's tested.

This config is p3_reach_gainfield.py (the best representation this
investigation produced -- offline decode R^2=0.983) with one change:
actor_batched_d1=False. main.py only re-zeros dopsyn's real A_plus/
A_minus when actor_batched_d1=True, so setting it False means the
native, continuously-online, three-factor dopamine-modulated STDP
governs cortex->D1 the whole run (actor.py always builds the dopsyn
connection regardless of this flag; BatchedD1Learning's Python-side
bookkeeping simply doesn't get constructed). D2 is already native
(actor_batched_d2=False is the standing default), so this makes both
channels native, matching CartPole's spine exactly.

Two decisive outcomes: native fails too, similar to the batched/
untrained-control result, which makes the negative implementation-
independent and puts the arm on the same rule as CartPole. Or native
does meaningfully better, in which case the batched detour handicapped
the actor and the "rule can't learn" story needs revising.
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
