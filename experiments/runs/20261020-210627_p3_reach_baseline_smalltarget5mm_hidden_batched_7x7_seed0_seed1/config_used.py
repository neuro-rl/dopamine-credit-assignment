"""p3_reach_baseline_smalltarget5mm_hidden_batched.py -- small-target
grid completion, Cell 2 (2026-10-20, external-collaborator spec): does
a frozen fixed-hidden layer AFTER the real recurrent cortex, trained
with the BATCHED D1 learner, learn the localized reach? Completes the
{cortex/bypass} x {hidden/no-hidden} x {native/batched} grid's
cortex+hidden cell (no native-STDP counterpart was run for this
specific cell -- Arm C's hidden-layer negative was on the gain-field
path, not the cortex path -- so there is no direct native-vs-batched
contrast for Cell 2 the way Cells 1/3 have one; it stands on its own
as a data point on whether depth-after-cortex ever learns, under
either rule).

Required a small model/network.py addition (2026-10-20):
actor_cortex_fixed_hidden (new flag, separate from
actor_bypass_fixed_hidden, which only fires under cortex-bypass)
inserts a frozen random fixed_indegree hidden layer between
self.cortex.output() and D1/D2, cortex otherwise unchanged/not
bypassed. Reuses the existing actor_fixed_hidden_size/indegree/weight
fields.

Built from p3_reach_baseline_smalltarget5mm.py (single import) -- two
changes: actor_batched_d1=True (same flip as Cell 1), and
actor_cortex_fixed_hidden=True (new flag above). Everything else
(real recurrent cortex, Python critic, ±5mm target box, random start,
continuous heads) identical to Arm B / Cell 1.
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
