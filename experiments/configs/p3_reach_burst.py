"""p3_reach_burst.py -- burst-dependent plasticity hidden layer
(cortex -> H -> D1/D2, replacing the flat cortex -> D1/D2 projection)
in the arm actor. Built from p3_reach_baseline.py (one level, not a
deep chain) -- only actor_burst_hidden=True differs; teacher stays
"python"/Python critic so the critic isn't a confound, testing the
actor's hidden layer only. Grid stays the locked 7x7 default.

See model/burst_hidden.py for the mechanism itself.

Noise/burst calibration: CartPole's own BASE_STD=35.0 near-saturates H
here (burst_frac~0.95) regardless of cortex->H weight scale, because
this project's coarser kernel resolution (1.0ms vs CartPole's 0.1ms)
drives a much longer-held current fluctuation for the same std, pushing
the izhikevich IB bifurcation harder. BASE_STD=13.0 lands closest to the
target burst_frac~0.40 (measured 0.40 mean, 0.356-0.451 range across
resets). GAIN_TD was re-derived from this project's much smaller
delta_python magnitude (abs_mean=0.0067 vs CartPole's reward-scale TD
error) to 900, and STD_MIN/MAX rescaled proportionally with BASE_STD
(5->2, 80->30). P_BASELINE=0.40 (CartPole's value) already matched the
measured mean here, so it's unchanged.

D1-drive recalibration: H fires much harder than real cortex
(~57-63Hz vs cortex's ~5.6-5.8Hz baseline), so the 7x7 baseline's
actor_k_ctx_d1/d2_scale=2.070 (tuned for cortex-driven D1) drove D1 to
~33Hz, well above the 10-15Hz target band; 0.8 lands at 11.33Hz,
closest to band center.
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
