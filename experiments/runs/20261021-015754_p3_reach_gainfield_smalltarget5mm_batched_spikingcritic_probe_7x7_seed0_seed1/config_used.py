"""p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py --
spiking-critic-hardening spec (2026-10-21, external-collaborator):
FALLBACK Phase 1. The primary test (cortex + batched actor + spiking
critic, p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py)
FAILED the spec's gate: aggregate greedy-eval 273.2mm/1.7%@15mm,
indistinguishable from the untrained control (288.9mm/1.7%@15mm) --
only 1/3 seeds showed any real improvement, and even that seed (164.5mm)
fell far short of Cell 1's own Python-critic positive (64.7mm/45%@15mm).
Per the spec's own decision rule, this triggers the fallback: pair the
spiking critic with the representation that was cleanest for the actor
instead -- gain-field + batched (Arm A's actor, 122.1mm w/ Python
critic). The gain-field gives both actor AND critic clean task-space
geometry, so if the spiking critic can work anywhere on this task, it
should work here.

Built from p3_reach_gainfield_smalltarget5mm_batched.py (Arm A's own
small-target config, single import) -- confirmed via its full import
chain (-> p3_reach_gainfield_smalltarget5mm -> p3_reach_gainfield_
nativestdp -> p3_reach_gainfield -> p3_reach_bypass_cortex ->
p3_reach_combined_encoder -> p3_reach_baseline) that critic_reach_vec_
input=True and critic_single_layer_centered=True are ALREADY inherited
unchanged -- same critic infrastructure as the primary test's own base.
Only addition: critic_centered_wmin=-15.0 (P1.4 signed-weight
architecture) + critic_vstar_probe_at_end=True, identical pattern to
p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py.

Same two-phase structure as the primary test: this phase leaves the
actor on delta_python (unchanged Arm A training) while the spiking
critic rides along passively, so its sign/fidelity/scale can be
measured before the actor is committed to it in Phase 2.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_smalltarget5mm_batched",
    Path(__file__).resolve().parent / "p3_reach_gainfield_smalltarget5mm_batched.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace(
        "p3_reach_gainfield_smalltarget5mm_batched", "p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe")
    cfg.critic_centered_wmin = -15.0
    cfg.critic_vstar_probe_at_end = True
    return cfg
