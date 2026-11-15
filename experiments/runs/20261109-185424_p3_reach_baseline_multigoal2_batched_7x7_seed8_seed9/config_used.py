"""p3_reach_baseline_multigoal2_batched.py -- multi-goal target-
conditioning: does the batched-D1 actor (cortex + batched D1, Python
critic) use the target when a single fixed spot can no longer solve
the task? A prior ablation showed the +/-5/+/-25/+/-50mm positives are
centroid/fixed-goal homing, not target-conditioned. This forces the
issue: 2 discrete goals, far enough apart (180mm) that homing to their
midpoint is clearly worse than reaching the commanded one, and their
Gaussian/place-field codes don't overlap.

Built from p3_reach_baseline.py (single import), not the smalltarget5mm
chain: that chain's target_box_halfwidth_m shrinks arm.target_lo/
target_hi, which model/python_critic.py uses to normalize the target
into [-1,1] for the critic's own features. Shrinking that box to +/-5mm
while these goals sit 100-250mm away would badly miscalibrate the
critic, so target_lo/target_hi stay at the full-workspace default.

Goal set (model/env_reach.py's new target_goal_set override --
reset() picks one uniformly at random instead of rejection-sampling a
continuous box): two points, each independently reachable (verified
directly via arm.is_reachable), 180mm apart:
  A = (0.26, 0.05)
  B = (0.36, 0.20)
Random start (unchanged), 3 seeds, 400 ep (set at launch).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_multigoal2_batched")
    cfg.actor_batched_d1 = True
    cfg.target_goal_set = [(0.26, 0.05), (0.36, 0.20)]
    return cfg
