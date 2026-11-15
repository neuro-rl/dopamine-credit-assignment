"""p3_reach_burst_eval_greedy.py -- greedy (exploration=0), weights-frozen
POOLED eval pass for a checkpoint saved by p3_reach_burst.py, matching the
project's locked greedy-eval convention (p3_reach_baseline_eval.py's own
pattern): restore checkpoint, 1 discarded warmup episode + N clean greedy
episodes (run with --episodes 21 for the usual 20 clean), no video (unlike
the existing p3_reach_burst_eval.py, which is a 2-episode video-generation
pass, not a pooled-stats one).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_burst", Path(__file__).resolve().parent / "p3_reach_burst.py")
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
