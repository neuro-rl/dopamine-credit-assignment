"""p3_reach_bypass_cortex.py -- external-collaborator fallback spec
(2026-10-17): the combined encoder's reach_vec signal (R^2=0.99 at the
encoder's own output) collapses to 0.015 through the real recurrent
cortex, and stays at 0.03 even with spatial-masked connectivity (see
findings/reachvec_cortex_probe.csv, findings/reachvec_cortex_spatial_
probe.csv) -- the recurrent cortex cannot carry this place-coded signal,
in or out of spatial register. Fallback, per spec: don't fix the
cortex, don't go through it at all for the actor's pathway -- D1/D2
reads the combined encoder (place-field joints + raw linear target)
DIRECTLY (actor_bypass_cortex_joints=True, see network.py/main.py).
This is guaranteed to reach the readout at the already-measured
encoder-direct R^2=0.99 (no new offline probe needed -- the bypass IS
that condition, by construction).

Built from p3_reach_combined_encoder.py (single import) -- same
encoder_type/joints_encoder_kw/object_raw_encoder_kw as that config,
only new field is the bypass flag. cortex itself is still built and
still fed by joints/object as before (network.py doesn't skip that) --
only the actor's own input is redirected away from it.

Biology framing (per spec): this isn't "delete the cortex," it's
"place-coded parietal input -> striatum," skipping the recurrent
reservoir hop for this pathway -- a cleaner story than a random
reservoir in between, not a dodge.

Env vars: same P12_SEED/GRID convention as p3_reach_baseline.py.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_combined_encoder", Path(__file__).resolve().parent / "p3_reach_combined_encoder.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_combined_encoder", "p3_reach_bypass_cortex")
    cfg.actor_bypass_cortex_joints = True

    # actor_disable_efference (REQUIRED for bypass, not optional): D1/D2's
    # efference-copy loop (actor.py's _build_efference_copy) defaults its
    # target to whatever population was passed in as the Actor's "cortex"
    # constructor arg -- under bypass that's actor_bypass_pop, the bare
    # encoder parrot neurons. Feeding D1/D2's own output back into THOSE
    # (no inhibition, no refractory ceiling, just 1:1 relay) creates a
    # runaway positive feedback loop: D1/D2 fires -> parrots relay it
    # straight back -> D1/D2 fires more. MEASURED: D1 at 333Hz (fully
    # saturated) and the parrot population at ~3300Hz average (impossible
    # for a rate-coded Poisson source with max_rate=400 -- confirmed via
    # isolated reproduction that the encoder class itself is NOT the
    # source; this efference loop is). actor.py's own comment records
    # that disabling efference entirely already measured ZERO effect on
    # behavior in the normal (cortex-target) architecture, so this isn't
    # a new unvalidated assumption -- it's the safe, precedented choice.
    cfg.actor_disable_efference = True
    return cfg
