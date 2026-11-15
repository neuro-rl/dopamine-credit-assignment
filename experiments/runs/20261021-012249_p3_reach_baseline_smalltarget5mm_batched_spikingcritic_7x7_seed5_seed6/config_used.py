"""p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py -- spiking-
critic-hardening spec (2026-10-21, external-collaborator): PHASE 2, the
actual fully-spiking closed-loop test. Cell 1's actor (cortex + batched
D1, the best actor, 64.7mm with the Python critic) now trains from the
P1.4-style spiking critic's OWN delta instead of the Python critic's --
the only change from the Phase 1 probe config
(p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py), which
left the actor on delta_python so the critic's sign/fidelity/scale could
be measured without risking the actor's own training.

Measured from Phase 1 (3 seeds x 400 episodes, NOT assumed):
  - critic_sign: all 3 seeds came back POSITIVE (corr(v_spiking,V*) =
    +0.81/+0.63/+0.85 for seed 0/1/5) -- no flip needed anywhere, unlike
    the full-workspace P1.4 result where 2/9 seeds needed a flip never
    materializing. Fidelity itself is also materially HIGHER here (mean
    +0.76) than full-workspace P1.4's own ceiling (+0.58) -- consistent
    with the spec's own prediction that a near-fixed target makes the
    critic's job easier too, not just the actor's.
  - actor_batched_d1_delta_spiking_scale: delta_python_std/delta_spiking_
    std ratio (last 200 episodes, both now logged every episode via
    main.py's ep_delta_python/ep_delta_spiking) gave per-seed scales
    0.6159/0.6744/0.5291 (same ~1.3x seed-to-seed spread as every other
    calibration measurement in this project) -- averaged to 0.606, scaled
    against delta_python's own actor_batched_d1_delta_python_scale=236.5
    convention, same as every P1.3/P1.4 calibration.

critic_signflip_probe_episodes is NOT set here -- critic_sign_override
being set already freezes critic_sign from episode 0 and skips the
(confounded) online probe entirely, per main.py's own logic.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe",
    Path(__file__).resolve().parent / "p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# Measured 2026-10-21 from the Phase 1 probe's own p1_1_vstar_probe.json
# (real rollout-based V*, same methodology as P1.1/P1.2b/P1.4) -- all 3
# positive, so this is a no-op for every seed here, but it is NOT
# skipping the check, it's recording the result of actually running it.
_SIGN_BY_SEED = {0: 1.0, 1: 1.0, 5: 1.0}


def Config():
    cfg = _base.Config()
    import os
    seed = int(os.environ.get("P12_SEED", "-1"))
    cfg.name = cfg.name.replace(
        "p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe",
        "p3_reach_baseline_smalltarget5mm_batched_spikingcritic")
    cfg.actor_batched_d1_signal_source = "delta_spiking"
    cfg.critic_sign_override = _SIGN_BY_SEED.get(seed, 1.0)
    cfg.actor_batched_d1_delta_spiking_scale = 0.606
    return cfg
