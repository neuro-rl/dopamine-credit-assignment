"""p3_reach_baseline_eval.py -- greedy (exploration_rate=0), STDP-frozen
eval pass for a checkpoint saved by p3_reach_baseline.py. Same pattern as
p3_reach_fixedhidden_eval.py.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def Config():
    cfg = _mod.Config()
    cfg.exploration_start = 0.0
    cfg.exploration_min = 0.0
    cfg.freeze_weights_episode = 1
    cfg.video_interval = 0
    cfg.checkpoint_every = 0
    cfg.critic_probe_at_end = False
    return cfg
