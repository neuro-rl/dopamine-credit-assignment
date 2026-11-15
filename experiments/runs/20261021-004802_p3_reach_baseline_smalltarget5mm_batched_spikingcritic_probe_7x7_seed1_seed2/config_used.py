"""p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py --
phase 1 of the fully-spiking small-target test. Before committing the
actor to the spiking critic's own delta (a real behavioral risk if the
critic's sign/scale are wrong), this phase trains the spiking-critic
architecture passively: actor_batched_d1_signal_source stays at its
default "delta_python" (actor training is byte-identical to the
completed p3_reach_baseline_smalltarget5mm_batched.py run) while the
critic rides along, same as every p3_reach_* config (only
critic_centered_wmin differs here).

Two things this phase measures rather than assumes:
  1. critic_sign per seed, via critic_vstar_probe_at_end=True's
     rollout-based V* comparison, not the online signflip probe (which
     is confounded by this task's negative-skewed reward + time-in-
     episode artifacts).
  2. actor_batched_d1_delta_spiking_scale: read delta_python_std/
     delta_spiking_std directly from stats.csv and scale so
     delta_spiking*scale lands at the same magnitude as
     delta_python*236.5, this project's standing convention for
     comparing signal sources.

Only change vs. the base config: critic_centered_wmin=-15.0, a single-
pool signed-weighted readout (mixed-sign cortex/reach-vec->L1 weights,
subtracting at the input-current level so no two-rate-cancellation
mechanism applies) that hit the ridge fidelity ceiling on the
full-workspace task (mean corr(v_spiking, V*)=+0.58, 9/9 correct sign).
critic_vstar_probe_at_end=True added to run that same measurement here.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_smalltarget5mm_batched",
    Path(__file__).resolve().parent / "p3_reach_baseline_smalltarget5mm_batched.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace(
        "p3_reach_baseline_smalltarget5mm_batched", "p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe")
    cfg.critic_centered_wmin = -15.0
    cfg.critic_vstar_probe_at_end = True
    return cfg
