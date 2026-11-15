"""p3_reach_baseline_tauc_test.py -- SCRATCH, one-off probe: identical to
the locked p3_reach_baseline.py except da_tau_c is raised from the locked
1.0ms default, to test whether the batched D1 eligibility trace's
near-zero memory across control steps (rho = exp(-50ms/1ms) ~= 1.9e-22,
see eq:arm-elig in the paper) is actually costing performance, or whether
the locked value was already fine. NOT a locked config -- not referenced
by run.py or the 9-seed sweep, exists only for this probe.

Env vars: same as p3_reach_baseline.py (P12_SEED, GRID), plus
TAU_C_MS (float, default 20.0 -- a 20x increase over the locked 1.0ms,
giving rho = exp(-50/20) ~= 0.082).
"""
import importlib.util
import os
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.da_tau_c = float(os.environ.get("TAU_C_MS", "20.0"))
    cfg.name = cfg.name + f"_tauc{cfg.da_tau_c:g}"
    return cfg
