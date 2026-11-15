"""p3_reach_burst.py -- Path B (2026-10-17, external-collaborator spec):
burst-dependent plasticity hidden layer (cortex -> H -> D1/D2, replacing
the flat cortex -> D1/D2 projection) in the arm actor. Built from
p3_reach_baseline.py (one level, not a deep chain) -- only
actor_burst_hidden=True differs; teacher stays "python"/Python critic
(kept as baseline, per spec, so the critic isn't a confound -- this
tests the actor's hidden layer only). Grid stays the locked 7x7 default
via the same GRID env var p3_reach_baseline.py reads.

See model/burst_hidden.py and the plan this was implemented from
(Path B -- burst-plasticity hidden layer in the arm actor) for the full
design/integration-point writeup.

RECALIBRATION (measured, not guessed -- calibrate_burst_std.py/
measure_delta_python.py, scratchpad, 2026-10-17): CartPole's own
BASE_STD=35.0 produced near-total H saturation here (burst_frac~0.95,
H rate~190Hz) regardless of cortex->H weight scale (swept 1.0x down to
0.05x the default range, zero effect) -- NOT a cortex-drive problem.
Isolated to NEST's noise_generator: this project's kernel resolution is
1.0ms (CartPole's was 0.1ms, with dt set to match); at dt=1.0ms the SAME
std drives roughly a 10x-longer-held current fluctuation, pushing the
same izhikevich IB bifurcation CartPole's own docstring warns about
(burst_frac jumps sharply, not linearly, with std) much harder. Measured
std=0 -> rate=12Hz/burst_frac=0.08 (cortex-only, confirms cortex drive
itself is mild) through std=20 -> rate=100Hz/burst_frac=0.70 (already
near-saturated); std=13.0 lands closest to CartPole's own target
burst_frac~0.40 (measured 0.40 mean, 0.356-0.451 range across 5
different episode resets -- reasonably state-independent, same property
CartPole's own calibration required). GAIN_TD=40 (CartPole's value) was
ALSO re-derived: delta_python's real magnitude here (measured over 500
steps, untrained) is abs_mean=0.0067/p90=0.0168 -- tiny compared to
CartPole's reward-scale TD error, the same gap actor_batched_d1_delta_
python_scale=236.5 already compensates for on the OTHER (H->D1/D2)
learning pathway. GAIN_TD=900 targets a ~6-unit typical std swing
(roughly half the new STD_MIN/STD_MAX headroom around BASE_STD=13),
not pinned at the clip bounds for a typical step but able to reach them
for a large one. STD_MIN/MAX rescaled proportionally to BASE_STD's own
35->13 change (5->2, 80->30). P_BASELINE=0.40 (CartPole's value) was
NOT changed -- the measured mean (0.402) already matches it almost
exactly, so that part of the reference calibration still holds here.

D1-drive recalibration (calibrate_burst_kctx.py, scratchpad): H fires
much harder than real cortex (measured ~57-63Hz vs cortex's own
calibrated ~5.6-5.8Hz baseline), so the 7x7 baseline's own actor_k_ctx_
d1/d2_scale=2.070 (tuned for CORTEX-driven D1) drove D1 to ~33Hz, well
above the 10-15Hz target band. Swept 0.5/0.7/0.8/0.9/1.1 -> 0.8 lands at
11.33Hz, closest to band center.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_baseline_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_mod)


def Config():
    cfg = _baseline_mod.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_burst")
    cfg.actor_burst_hidden = True
    cfg.burst_base_std = 13.0
    cfg.burst_std_min = 2.0
    cfg.burst_std_max = 30.0
    cfg.burst_gain_td = 900.0
    cfg.actor_k_ctx_d1_scale = 0.8
    cfg.actor_k_ctx_d2_scale = 0.8
    return cfg
