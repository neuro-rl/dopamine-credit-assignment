"""p3_reach_baseline_multigoal3_batched.py -- multi-goal target-
conditioning spec (2026-10-21), 3-goal follow-up. The 2-goal test
(p3_reach_baseline_multigoal2_batched.py) showed genuine discrete
target-conditioning in 2/3 seeds (selectivity 100%->5% under the
swap control, decisive) with one seed failing to converge to a
working two-goal discrimination at all. Per the spec's own order
("only if 2 learns with real selectivity: 3 goals, for the stronger
claim"), this extends to 3 goals (chance=33%, a stronger claim than
2 goals' 50%).

Built the same way as the 2-goal config, from p3_reach_baseline.py
directly (NOT the smalltarget5mm chain -- same target_lo/target_hi
normalization concern as the 2-goal config's own docstring explains).
Goal set: A=(0.26,0.05), B=(0.36,0.20), C=(0.30,0.30) -- all three
independently verified reachable, pairwise distances 180/253/117mm
(all >=100mm, the spec's own separation bar).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_multigoal3_batched")
    cfg.actor_batched_d1 = True
    cfg.target_goal_set = [(0.26, 0.05), (0.36, 0.20), (0.30, 0.30)]
    return cfg
