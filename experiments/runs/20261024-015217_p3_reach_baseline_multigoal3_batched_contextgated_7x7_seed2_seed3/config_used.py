"""p3_reach_baseline_multigoal3_batched_contextgated.py -- context-gated
plasticity spec (2026-10-21, external-collaborator): the 3-goal baseline
(MULTIGOAL_TARGET_CONDITIONING_FINDING.md, 80% selectivity/142mm) pools
its reward-centering baseline and eligibility accumulator across whatever
goals land in a 10-episode batch -- the spec's hypothesis is that this
cross-goal pooling is itself a source of interference, diluting each
goal's own credit. BatchedD1Learning now supports a SEPARATE baseline +
accumulator per goal (model/batched_actor_learning.py's n_goals param);
this config turns that on, nothing else.

Built from p3_reach_baseline_multigoal3_batched.py (single import) --
deliberately does NOT inherit max_steps=200 or target_settle_amplitude
(MULTIGOAL_TIGHTEN_LANDING_FINDING.md: both regressed everything) --
building from the 3-goal baseline directly, not that chain, keeps this
config at the spec's explicitly fixed 100 steps/episode automatically.

checkpoint_every=10 (overriding reach_state.py's default of 3), aligned
exactly with actor_batched_d1_batch_episodes=10 -- every periodic
rebuild now lands ON a commit boundary (every accumulator, including the
new per-goal ones, already zero), per the spec's own explicit request
("keep checkpoint_every aligned with the batch boundary so no rebuild
lands mid-batch for any goal") and matching this project's own prior
rebuild-invariance convention (p3_reach_baseline_ckpt10.py).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline_multigoal3_batched",
    Path(__file__).resolve().parent / "p3_reach_baseline_multigoal3_batched.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline_multigoal3_batched",
                                 "p3_reach_baseline_multigoal3_batched_contextgated")
    cfg.actor_batched_d1_context_gated = True
    cfg.checkpoint_every = 10
    return cfg
