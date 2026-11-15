"""p3_reach_baseline_nocenter.py -- centering ablation (Section 5d), built
from p3_reach_baseline.py directly (not a deep chain, same pattern as the
multigoal configs). Flips only actor_batched_d1_presyn_baseline_correction
to False; everything else (grid, critic, action-perturbation scale, axis-
aligned preference, velocity drive, exploration schedule) stays identical
to the locked baseline so the centered-vs-uncentered comparison isolates
exactly one mechanism.

Needed because the only prior centered-vs-uncentered data (the pre-P2
reach_state_fix6_biggrid(_nocenter) batch, Oct 10-12) predates fix1_action-
pert/fix2_presyn/fix3_axispref/fix5_velocity and isn't comparable to
today's locked regime (see experiments/stats/README.md's fig3_weights
blocker note, 2026-10-24).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_nocenter")
    cfg.actor_batched_d1_presyn_baseline_correction = False
    return cfg
