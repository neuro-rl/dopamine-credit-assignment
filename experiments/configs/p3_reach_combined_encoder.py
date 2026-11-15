"""p3_reach_combined_encoder.py -- the one change the offline
reach-vector decode probes recommended: swap the actor's cortex-side
joints encoder from TopographicPlaceFieldEncoder (separable per-axis
code) to Grid2DPlaceFieldEncoder (a true joint 2-D Gaussian receptive
field over (q1,q2)), and swap the object (target) encoder from
TopographicPlaceFieldEncoder to LinearRateEncoder (raw 2-feature linear
rate code, no place field) -- the pairing that measured reach_vec
spike-count R^2=0.99 offline, at the smallest width tested (bins=12,
144 joint-PF units), vs. ~0.76 for the shipped topo_both pairing.

Built from p3_reach_baseline.py (single import) -- every other locked
decision (critic, actor rule, grid, exploration schedule, calibration)
is inherited unchanged. This is deliberately a narrow, isolated A/B:
only the joints/object encoder swap differs from the 154.0mm/10.6%
flat baseline, so any change in control is attributable to the
representation fix, not a confound.

Width: bins=12 for the joints Grid2DPlaceFieldEncoder (144 units), the
smallest width the offline probe tested, already clears the ~0.9 gate
(0.993 combined/0.995 raw_target spike-count R^2); no cortex widening
needed.

connect_kw: Grid2DPlaceFieldEncoder.connect_to and
LinearRateEncoder.connect_to are both plain random fixed_indegree
(neither supports the spatial/topographic radius+spatial_indegree
masking TopographicPlaceFieldEncoder.connect_to does), so
joints_connect_kw/object_connect_kw must drop those two keys for this
encoder_type, or nest.Connect's syn_spec receives invalid keys and
raises.

joints_vel (velocity_input) stream is untouched -- stays
TopographicPlaceFieldEncoder regardless of encoder_type.

Env vars: same P12_SEED/GRID convention as p3_reach_baseline.py.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_combined_encoder")

    cfg.encoder_type = "grid2d_joints_lineartarget"

    # joints: Grid2DPlaceFieldEncoder, bins=12 (144 units, the width that
    # cleared the offline decode gate) -- same lo/hi/max_rate/sigma_bins
    # as the shipped joints_encoder_kw; only the tuning form changes to
    # a joint 2-D Gaussian instead of separable per-axis.
    cfg.joints_encoder_kw = dict(
        bins=12, lo=(-1.745, -1.690), hi=(1.745, 1.690), max_rate=400.0, sigma_bins=1.0)

    # object/target: LinearRateEncoder, raw (tx,ty), same lo/hi/max_rate
    # range as the shipped object_encoder_kw.
    cfg.object_raw_encoder_kw = dict(
        lo=(0.20, 0.00), hi=(0.40, 0.25), min_rate=0.0, max_rate=400.0)

    # connect_kw: same weight the baseline's drive scaling already
    # produced, delay/indegree unchanged; radius/spatial_indegree dropped
    # since neither new encoder class's connect_to supports them.
    cfg.joints_connect_kw = dict(weight=cfg.joints_connect_kw["weight"], delay=0.5, indegree=10)
    cfg.object_connect_kw = dict(weight=cfg.object_connect_kw["weight"], delay=0.5, indegree=10)

    return cfg
