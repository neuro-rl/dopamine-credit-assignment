"""p3_reach_baseline_nocenter.py -- centering ablation, built from
p3_reach_baseline.py directly. Flips only
actor_batched_d1_presyn_baseline_correction to False; everything else
(grid, critic, action-perturbation scale, axis-aligned preference,
velocity drive, exploration schedule) stays identical to the locked
baseline so the centered-vs-uncentered comparison isolates exactly one
mechanism. See experiments/stats/README.md for why an earlier
centered-vs-uncentered batch predating this config isn't comparable.
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
