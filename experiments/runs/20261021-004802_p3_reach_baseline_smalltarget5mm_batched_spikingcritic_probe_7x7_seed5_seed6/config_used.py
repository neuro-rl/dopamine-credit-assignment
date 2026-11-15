"""p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py --
spiking-critic-hardening spec (2026-10-21, external-collaborator): PHASE 1
of the fully-spiking small-target test. Before committing the actor to
the spiking critic's own delta (a real behavioral risk if the critic's
sign/scale are wrong), this phase trains the P1.4-winning spiking-critic
architecture PASSIVELY -- actor_batched_d1_signal_source stays at its
default "delta_python" (Cell 1's own actor training is byte-identical to
the completed p3_reach_baseline_smalltarget5mm_batched.py run) -- while
the critic rides along, exactly as every p3_reach_* config already does
(p3_reach_baseline.py locks critic_reach_vec_input=True + critic_single_
layer_centered=True for every config in this chain; only critic_centered_
wmin differs here).

Two things this phase measures, per the project's own "measure, don't
assume" discipline (P1.2b/P1.4 precedent -- sign assumed from an earlier
measurement, never re-derived from data the actor's own training already
shaped):
  1. critic_sign per seed: via critic_vstar_probe_at_end=True's rollout-
     based V* comparison (p1_1_vstar_probe.json), NOT the online signflip
     probe (confounded by this task's negative-skewed reward + time-in-
     episode artifacts -- P1.3's own documented lesson).
  2. actor_batched_d1_delta_spiking_scale: read delta_python_std/delta_
     spiking_std directly from stats.csv (now logged every episode,
     unconditionally -- see main.py's ep_delta_python/ep_delta_spiking
     additions) and scale so delta_spiking*scale lands at the same
     magnitude as delta_python*236.5 (this project's standing convention
     for comparing signal sources, used identically in P1.3/P1.4).

Only change vs Cell 1 (p3_reach_baseline_smalltarget5mm_batched.py):
critic_centered_wmin=-15.0 -- the P1.4 "signed weighted readout"
architecture (single pool, mixed-sign cortex/reach-vec->L1 weights,
subtracts at the input-CURRENT level so no two-rate-cancellation
mechanism applies, unlike the abandoned opponent-pools variant) that hit
the ridge fidelity ceiling on the full-workspace task (mean corr(v_
spiking,V*)=+0.58, 9/9 correct sign, p1_4_probe_seeds.csv). critic_
vstar_probe_at_end=True added to run that same measurement here.
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
