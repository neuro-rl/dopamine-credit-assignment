"""p3_reach_burst_eval.py -- greedy (exploration_rate=0), STDP/burst-
frozen eval/video-generation pass for a checkpoint saved by
p3_reach_burst.py. Same pattern as the legacy reach_state_fix6_biggrid_
criticfix_reachvec_eval.py (experiments/configs/legacy/).

Run with --restore-checkpoint pointing at that run's final_checkpoint.pkl
and --episodes 2, with the SAME P12_SEED/GRID env vars the training run
used (cfg.name must match exactly, or NestCheckpointer's restore crashes
-- see that legacy eval config's own docstring for why):
  - episode 0: exploration already 0, so the policy is greedy from the
    start; freeze_weights_episode=1 then zeroes all learning (including
    the burst cortex->H update and BatchedD1Learning) from here on.
  - episode 1+: fully frozen AND greedy -- the clean eval episode(s).
    Use THIS episode's video.
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
    cfg.video_interval = 1
    cfg.checkpoint_every = 0
    cfg.critic_probe_at_end = False
    return cfg
