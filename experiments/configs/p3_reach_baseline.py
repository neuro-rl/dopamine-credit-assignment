"""p3_reach_baseline.py — the locked baseline config (Python/analytic TD
critic). Flat, self-contained: every value here is inlined directly
rather than pulled through a parent-config chain.

What's locked in:
  - critic: Python/analytic TD (teacher="python", reach_state.py's own
    default -- this config never touches the critic signal source).
  - actor rule: centered REINFORCE + vec2dproj readout (also
    reach_state.py's defaults, unchanged here).
  - grid: 7x7 by default (GRID env var overrides; 12x12 is the fallback
    if 7x7 doesn't hold up at 9 seeds).

Env vars: P12_SEED (int seed), GRID ("7x7" default, "6x6"/"8x8"/"10x10"/
"12x12"/"16x16" also valid).
"""
import importlib.util
import os
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "reach_state_base", Path(__file__).resolve().parent / "reach_state.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_GRID_PARAMS = {
    "6x6":   dict(shape=(6, 6),   k_ctx_scale=1.152, readout_scale=1.0),
    "7x7":   dict(shape=(7, 7),   k_ctx_scale=2.070, readout_scale=0.6952),
    "8x8":   dict(shape=(8, 8),   k_ctx_scale=3.440, readout_scale=0.5074),
    "10x10": dict(shape=(10, 10), k_ctx_scale=8.039, readout_scale=0.2998),
    "12x12": dict(shape=(12, 12), k_ctx_scale=16.08, readout_scale=0.1949),
    "16x16": dict(shape=(16, 16), k_ctx_scale=48.0,  readout_scale=0.099),
}

_DRIVE_SCALE = 0.65  # cortex-input recalibration for the 3-stream (joints+object+joints_vel) drive


def Config():
    cfg = _base.Config()
    seed = int(os.environ.get("P12_SEED", "-1"))
    grid_key = os.environ.get("GRID", "7x7")
    g = _GRID_PARAMS[grid_key]

    cfg.name = f"p3_reach_baseline_{grid_key}_seed{seed}"
    cfg.seed = seed + 1 if seed >= 0 else seed

    # online action-perturbation rule for the batched D1 update
    cfg.actor_batched_d1_use_action_perturbation = True
    cfg.actor_batched_d1_action_perturbation_scale = 60.0
    cfg.actor_batched_d1_presyn_baseline_correction = True

    # axis-aligned D1 preference directions
    cfg.actor_axis_aligned_pref = True

    # 3rd cortex stream (joint velocity); all three input drive weights
    # scaled down together to hold pre-learning D1 rate constant
    cfg.velocity_input = True
    cfg.joints_connect_kw = dict(cfg.joints_connect_kw,
                                  weight=cfg.joints_connect_kw["weight"] * _DRIVE_SCALE)
    cfg.object_connect_kw = dict(cfg.object_connect_kw,
                                  weight=cfg.object_connect_kw["weight"] * _DRIVE_SCALE)
    cfg.joints_vel_connect_kw = dict(cfg.joints_vel_connect_kw,
                                      weight=cfg.joints_vel_connect_kw["weight"] * _DRIVE_SCALE)

    # grid size (locked default 7x7); k_ctx_scale/readout_scale per grid
    # are D1-rate-calibrated, not free parameters
    cfg.actor_grid_shape = g["shape"]
    cfg.actor_k_ctx_d1_scale = g["k_ctx_scale"]
    cfg.actor_k_ctx_d2_scale = g["k_ctx_scale"]
    cfg.actor_vector_readout_scale_factor = g["readout_scale"]

    # fast eligibility trace
    cfg.da_tau_c = 1.0

    # critic: reach-vector place-field input, Python-teacher path
    # otherwise untouched
    cfg.critic_reach_vec_input = True
    cfg.critic_probe_at_end = True

    # exploration schedule: no decay, fixed floor
    cfg.exploration_start = 0.3
    cfg.exploration_min = 0.3
    cfg.exploration_decay = 1.0

    return cfg
