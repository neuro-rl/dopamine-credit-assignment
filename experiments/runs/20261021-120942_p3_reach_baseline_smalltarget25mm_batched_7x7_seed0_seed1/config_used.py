"""p3_reach_baseline_smalltarget25mm_batched.py -- target-conditioning
spec (2026-10-21, external-collaborator): Part 1, degradation curve
point 2 of 3 (±5mm -> ±25mm -> ±50mm). Does Cell 1's actor (cortex +
batched D1, the best actor) learn at a target box too big to solve by
servoing to a fixed point? Center-servoing costs real distance as the
box grows, so a positive here (beating untrained) is evidence the
policy reads the target, not just a fixed-goal reach.

Built from p3_reach_baseline.py (single import), same pattern as
p3_reach_baseline_smalltarget5mm_batched.py but skipping the native-
STDP intermediate (not needed -- only the batched/Cell-1 actor is
under test here). Two changes: actor_batched_d1=True (Cell 1's rule)
and target_box_halfwidth_m=0.025, same box center (0.29, 0.125) as
every other arm in this family. Random start, unchanged.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_smalltarget25mm_batched")
    cfg.actor_batched_d1 = True
    cfg.target_box_halfwidth_m = 0.025
    cfg.target_box_center = (0.29, 0.125)
    return cfg
