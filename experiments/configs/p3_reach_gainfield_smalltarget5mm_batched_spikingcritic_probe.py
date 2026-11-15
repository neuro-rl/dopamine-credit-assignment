"""p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py --
fallback phase 1 for the spiking-critic test. The primary test (cortex +
batched actor + spiking critic) failed to beat the untrained control
(273.2mm/1.7%@15mm vs. 288.9mm/1.7%@15mm), so this pairs the spiking
critic with the cleanest actor representation instead: gain-field +
batched (Arm A's actor, 122.1mm w/ Python critic). The gain-field gives
both actor and critic clean task-space geometry, so if the spiking
critic works anywhere on this task, it should work here.

Built from p3_reach_gainfield_smalltarget5mm_batched.py (Arm A's own
small-target config, single import), which already inherits
critic_reach_vec_input=True through its own chain -- same critic
infrastructure as the primary test's base. Only addition:
critic_centered_wmin=-15.0 (signed-weight critic architecture) +
critic_vstar_probe_at_end=True.

Same two-phase structure as the primary test: this phase leaves the
actor on delta_python (unchanged Arm A training) while the spiking
critic rides along passively, so its sign/fidelity/scale can be
measured before the actor is committed to it in phase 2.
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
