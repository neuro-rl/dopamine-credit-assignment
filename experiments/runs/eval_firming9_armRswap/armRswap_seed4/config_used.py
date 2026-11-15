"""p3_reach_baseline_reachvec_eval_swap.py -- Control 3 (reach-vector
spec): swap-control confirming vector-following. cfg.target_continuous_
swap_eval=True feeds a fixed-per-episode random DECOY target to
perception (the reach_vec computation) instead of the true commanded
one, while reward/distance still score against the real target. A
target-conditioned policy follows the decoy -> distance-to-true-target
gets markedly worse; a policy that ignores the target (e.g. homing to a
fixed/centroid point) is roughly unaffected.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_reachvec_eval",
    Path(__file__).resolve().parent / "p3_reach_baseline_reachvec_eval.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.target_continuous_swap_eval = True
    return cfg
