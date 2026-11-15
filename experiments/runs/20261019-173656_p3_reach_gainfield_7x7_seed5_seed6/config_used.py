"""p3_reach_gainfield.py -- external-collaborator gain-field spec
(2026-10-19): replaces the frozen random/CANN hidden layer with a FIXED
conjunctive gain-field basis (place(q) + place(q)*target, see encoders.py's
GainFieldEncoder) that spans the resolved-rate IK function class by
construction, instead of hoping a random or recurrent basis spans it by
luck. Offline decode gate passed decisively (spike-count R^2=0.983 vs the
0.62 random-hidden reference, vs the >=0.9 "ideal" bar) before this
training config was built, per the spec's own order of work.

Built from p3_reach_bypass_cortex.py (single import) -- inherits
actor_bypass_cortex_joints=True, actor_disable_efference=False (efference
ON, safe: targets the real iaf_psc_alpha relay, not bare parrots -- see
network.py). New: actor_gainfield_hidden=True, and reward_shaping=
"jacobian" (the spec's "fold in the reward-SNR fix": train on the clean
per-joint/jacobian reward instead of the noisy finite-difference delta --
this flag already existed, see env_reach.py, just never combined with an
arm hidden-layer variant before).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_bypass_cortex", Path(__file__).resolve().parent / "p3_reach_bypass_cortex.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_bypass_cortex", "p3_reach_gainfield")
    cfg.actor_gainfield_hidden = True
    cfg.actor_disable_efference = False
    cfg.reward_shaping = "jacobian"
    # ff_weight calibrated via a standalone isolated smoke check (one_to_one
    # feedforward, same "needs far more per-synapse weight than the random
    # hidden layer's indegree=30" pattern as CANN): swept 300->20000 at a
    # representative mid-workspace state; 1500 lands the driven population
    # rate at ~9.3Hz / ~20% active -- healthy, non-saturated (ff>=3000
    # saturates further without raising active_frac).
    cfg.actor_gainfield_ff_weight = 1500.0
    return cfg
