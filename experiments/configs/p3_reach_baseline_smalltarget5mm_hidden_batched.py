"""p3_reach_baseline_smalltarget5mm_hidden_batched.py -- does a frozen
fixed-hidden layer after the real recurrent cortex, trained with the
batched D1 learner, learn the localized reach? There's no direct
native-vs-batched contrast for this cell (an earlier hidden-layer
negative was on the gain-field path, not the cortex path), so this
stands on its own as a data point on whether depth-after-cortex ever
learns, under either rule.

actor_cortex_fixed_hidden (separate from actor_bypass_fixed_hidden,
which only fires under cortex-bypass) inserts a frozen random
fixed_indegree hidden layer between self.cortex.output() and D1/D2,
cortex otherwise unchanged. Reuses the existing
actor_fixed_hidden_size/indegree/weight fields.

Built from p3_reach_baseline_smalltarget5mm.py (single import) -- two
changes: actor_batched_d1=True and actor_cortex_fixed_hidden=True.
Everything else (real recurrent cortex, Python critic, +/-5mm target
box, random start, continuous heads) identical to the base config.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_smalltarget5mm", Path(__file__).resolve().parent / "p3_reach_baseline_smalltarget5mm.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline_smalltarget5mm", "p3_reach_baseline_smalltarget5mm_hidden_batched")
    cfg.actor_batched_d1 = True
    cfg.actor_cortex_fixed_hidden = True
    return cfg
