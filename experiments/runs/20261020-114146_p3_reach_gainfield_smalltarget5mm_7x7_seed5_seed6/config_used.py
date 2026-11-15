"""p3_reach_gainfield_smalltarget5mm.py -- external-collaborator small-
target go/no-go spec (2026-10-20): does removing TARGET SPREAD let the
local rule learn, on the continuous-action arm? Both the discrete-primitive
test (DISCRETE_PRIMITIVE_FINDING.md) and the sparsity sweep failed to
rescue the local rule -- this spec isolates a different variable, target
variance/state-dependence, instead of representation or action-space
structure.

Variant pinned explicitly by the collaborator (not "same as main runs" --
that was underspecified): gain-field representation (already
cortex-bypassed, offline decode R^2=0.983) + native stdp_dopamine_synapse
actor (the biological rule under test, matching CartPole's spine) +
Python critic (teacher="python", the standing default through this whole
config chain -- a privileged/learned critic during training isolates the
ACTOR's learnability, same logic as the torch reference pipeline) +
CONTINUOUS per-joint heads (unchanged from every other arm config in this
chain -- this is NOT the discrete-primitive variant). Built from
p3_reach_gainfield_nativestdp.py (single import) -- inherits
actor_bypass_cortex_joints=True, actor_gainfield_hidden=True,
actor_batched_d1=False, reward_shaping="jacobian" (unchanged, so any
improvement here is attributable to target spread alone, not a reward-SNR
confound vs the main continuous runs).

New: cfg.target_box_halfwidth_m=0.005 (+-5mm box, the spec's own strongest,
decisive go/no-go condition), cfg.target_box_center=(0.29, 0.125) (the
FULL workspace box's own center -- (TARGET_LO+TARGET_HI)/2 in arm.py --
verified reachable at all 9 sampled corners/center of the +-5mm box before
use, not assumed). Start configuration stays fully random every episode
(env_reach.py's ReachAdapter.reset -> arm.reset is untouched by
target_box_halfwidth_m by construction) -- per the spec's own explicit
requirement, so this tests a genuine state-dependent reach to a
near-fixed goal, not trajectory memorization.
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
