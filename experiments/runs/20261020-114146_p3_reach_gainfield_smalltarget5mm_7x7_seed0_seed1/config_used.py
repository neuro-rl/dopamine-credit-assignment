"""p3_reach_gainfield_smalltarget5mm.py -- does removing target spread
let the local rule learn, on the continuous-action arm? An earlier
discrete-primitive test and a sparsity sweep both failed to rescue the
local rule -- this isolates a different variable, target
variance/state-dependence, instead of representation or action-space
structure.

Variant: gain-field representation (already cortex-bypassed, offline
decode R^2=0.983) + native stdp_dopamine_synapse actor (matching
CartPole's spine) + Python critic (teacher="python", so a learned
critic during training isolates the actor's own learnability) +
continuous per-joint heads (not the discrete-primitive variant). Built
from p3_reach_gainfield_nativestdp.py (single import) -- inherits
actor_bypass_cortex_joints=True, actor_gainfield_hidden=True,
actor_batched_d1=False, reward_shaping="jacobian" unchanged, so any
improvement here is attributable to target spread alone, not a
reward-SNR confound vs. the main continuous runs.

New: cfg.target_box_halfwidth_m=0.005 (+/-5mm box, the strongest,
decisive go/no-go condition), cfg.target_box_center=(0.29, 0.125) (the
full workspace box's own center, verified reachable at all 9 sampled
corners/center of the +/-5mm box). Start configuration stays fully
random every episode, so this tests a genuine state-dependent reach to
a near-fixed goal, not trajectory memorization.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_nativestdp", Path(__file__).resolve().parent / "p3_reach_gainfield_nativestdp.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_gainfield_nativestdp", "p3_reach_gainfield_smalltarget5mm")
    cfg.target_box_halfwidth_m = 0.005
    cfg.target_box_center = (0.29, 0.125)
    return cfg
