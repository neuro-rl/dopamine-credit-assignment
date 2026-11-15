"""p3_reach_gainfield.py -- replaces the frozen random/CANN hidden layer
with a fixed conjunctive gain-field basis (place(q) + place(q)*target,
see encoders.py's GainFieldEncoder) that spans the resolved-rate IK
function class by construction, instead of hoping a random or recurrent
basis spans it by luck. Offline decode passed decisively (spike-count
R^2=0.983 vs. the 0.62 random-hidden reference, vs. the >=0.9 "ideal"
bar) before this training config was built.

Built from p3_reach_bypass_cortex.py (single import) -- inherits
actor_bypass_cortex_joints=True, actor_disable_efference=False
(efference on is safe here: targets the real iaf_psc_alpha relay, not
bare parrots). New: actor_gainfield_hidden=True, and
reward_shaping="jacobian" to train on the clean per-joint/jacobian
reward instead of the noisy finite-difference delta.
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
    # ff_weight calibrated by sweeping 300-20000 at a representative
    # mid-workspace state; 1500 lands the driven population rate at
    # ~9.3Hz / ~20% active, healthy and non-saturated.
    cfg.actor_gainfield_ff_weight = 1500.0
    return cfg
