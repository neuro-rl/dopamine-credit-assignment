"""p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --
swapped-target confirmatory control (the spec's own guardrail): same
pattern as the baseline's own swap eval. cfg.target_swap_eval=True feeds
the NEXT goal in rotation (index+1 mod 3) instead of the true commanded
one.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_multigoal3_batched_contextgated_eval",
    Path(__file__).resolve().parent / "p3_reach_baseline_multigoal3_batched_contextgated_eval.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.target_swap_eval = True
    return cfg
