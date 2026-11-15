"""p3_reach_bypass_cortex.py -- the combined encoder's reach_vec signal
(R^2=0.99 at the encoder's own output) collapses to 0.015 through the
real recurrent cortex, and stays at 0.03 even with spatial-masked
connectivity -- the recurrent cortex cannot carry this place-coded
signal, in or out of spatial register. So instead of fixing the cortex,
this config routes around it: D1/D2 reads the combined encoder
(place-field joints + raw linear target) directly
(actor_bypass_cortex_joints=True, see network.py/main.py), which
reaches the readout at the already-measured encoder-direct R^2=0.99 by
construction.

Built from p3_reach_combined_encoder.py (single import) -- same
encoder_type/joints_encoder_kw/object_raw_encoder_kw as that config,
only new field is the bypass flag. Cortex itself is still built and fed
by joints/object as before; only the actor's own input is redirected
away from it.

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

    # Required under bypass, not optional: D1/D2's efference-copy loop
    # defaults its target to whatever population was passed in as the
    # actor's "cortex" arg -- under bypass that's the bare encoder parrot
    # neurons, with no inhibition or refractory ceiling, so feeding D1/D2's
    # output straight back into them creates a runaway feedback loop (D1
    # saturates at 333Hz, parrots at ~3300Hz). Disabling efference entirely
    # has no measured effect on behavior in the normal architecture either,
    # so this is a safe default, not a new assumption.
    cfg.actor_disable_efference = True
    return cfg
