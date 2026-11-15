"""p3_reach_baseline_reachvec_batchedd2_test.py -- SCRATCH, one-off probe:
identical to p3_reach_baseline_reachvec.py except actor_batched_d2=True,
switching D2 from the native, continuously-online adopsyn rule (every
arm run to date, including the best single-goal result this is built
from, used the native rule for D2) to BatchedD1Learning's own
batched-STDP machinery -- the same treatment D1 already gets. Tests
whether batching D2 too helps or hurts the best-performing single-goal
seed found so far (seed7, 58.7mm mean distance / 35% success,
eval_firming9_armR/armR_seed7, p3_reach_baseline_reachvec.py).

NOT a locked config -- not referenced by run.py or the 9-seed sweep,
exists only for this probe.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_reachvec",
    Path(__file__).resolve().parent / "p3_reach_baseline_reachvec.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.name = cfg.name + "_batchedd2"
    cfg.actor_batched_d2 = True
    return cfg
