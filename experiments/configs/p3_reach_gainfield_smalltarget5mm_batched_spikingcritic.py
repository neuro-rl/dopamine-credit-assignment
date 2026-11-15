"""p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py -- the
closed-loop spiking-critic test. The actor (gain-field + batched D1,
122.1mm with the Python critic) now trains from the spiking critic's own
delta instead of the Python critic's -- the only change from the probe
config (p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py),
which left the actor on delta_python so the critic's sign/fidelity/scale
could be measured without risking the actor's own training.

Measured from the probe run (3 seeds x 400 episodes):
  - critic sign: all 3 seeds came back positive (corr(v_spiking, V*) =
    +0.36/+0.85/+0.91 for seed 0/1/5, mean +0.71) -- no flip needed.
  - delta_spiking_scale: delta_python_std/delta_spiking_std ratio (last
    200 episodes) gave per-seed scales 0.756/0.746/0.463, averaged to
    0.655.

critic_signflip_probe_episodes is not set here -- critic_sign_override
being set already freezes critic_sign from episode 0 and skips the
online probe entirely.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe",
    Path(__file__).resolve().parent / "p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# Measured from the probe run's rollout-based V* check -- all 3 seeds
# came back positive, so this records the result rather than skipping it.
_SIGN_BY_SEED = {0: 1.0, 1: 1.0, 5: 1.0}


def Config():
    cfg = _base.Config()
    import os
    seed = int(os.environ.get("P12_SEED", "-1"))
    cfg.name = cfg.name.replace(
        "p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe",
        "p3_reach_gainfield_smalltarget5mm_batched_spikingcritic")
    cfg.actor_batched_d1_signal_source = "delta_spiking"
    cfg.critic_sign_override = _SIGN_BY_SEED.get(seed, 1.0)
    cfg.actor_batched_d1_delta_spiking_scale = 0.655
    return cfg
