"""p3_reach_baseline_multigoal2_batched_eval_swap.py -- swapped-target
confirmatory control (2026-10-21 spec): identical to the normal eval
except cfg.target_swap_eval=True -- every perception channel is fed
the OTHER goal in the 2-goal set instead of the true commanded one,
while reward/distance scoring still uses the real commanded target.
A target-conditioned policy follows the swapped input (so distance-to-
true-target gets worse); a centroid-homing policy is roughly unaffected.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_multigoal2_batched_eval",
    Path(__file__).resolve().parent / "p3_reach_baseline_multigoal2_batched_eval.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.target_swap_eval = True
    return cfg
