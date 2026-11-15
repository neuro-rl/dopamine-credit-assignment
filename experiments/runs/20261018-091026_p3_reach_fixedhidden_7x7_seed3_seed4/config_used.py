"""p3_reach_fixedhidden.py -- trains place-field encoder -> frozen
random hidden layer (N_H=750, indegree=30, iaf_psc_alpha) -> D1/D2,
cortex bypassed, and measures distance directly rather than gating on
another offline decode probe. Decode R^2 is non-predictive of control
on this system: the flat baseline has the worst action-decode R^2
(0.35) of any condition tested and is the best controller (154mm), so
the fixed-hidden probe's 0.62-0.68 decode isn't a reason to withhold
training. This is the direct test.

Built from p3_reach_bypass_cortex.py (single import) -- only two
changes: (1) actor_bypass_fixed_hidden=True (inserts the frozen hidden
layer, see network.py), (2) actor_disable_efference=False (efference
back on -- the bypass's runaway bug was bare encoder parrots as the
efference target; here the target is the hidden layer, real
iaf_psc_alpha neurons with inhibition/refractory, which should break
the runaway loop without disabling efference entirely).

Last arm experiment in this family: 9 seeds, greedy eval, compared
directly to the 154mm flat baseline.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_bypass_cortex", Path(__file__).resolve().parent / "p3_reach_bypass_cortex.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_bypass_cortex", "p3_reach_fixedhidden")
    cfg.actor_bypass_fixed_hidden = True
    cfg.actor_disable_efference = False
    return cfg
