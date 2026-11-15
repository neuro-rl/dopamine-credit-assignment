"""p3_reach_baseline_multigoal3_batched_contextgated.py -- context-gated
plasticity. The 3-goal baseline (80% selectivity/142mm) pools its
reward-centering baseline and eligibility accumulator across whatever
goals land in a 10-episode batch, which is itself a plausible source of
interference, diluting each goal's own credit. BatchedD1Learning now
supports a separate baseline + accumulator per goal
(model/batched_actor_learning.py's n_goals param); this config turns
that on, nothing else.

Built from p3_reach_baseline_multigoal3_batched.py (single import) --
deliberately does not inherit max_steps=200 or target_settle_amplitude,
which regressed everything in an earlier test; building from the 3-goal
baseline directly keeps this config at a fixed 100 steps/episode.

checkpoint_every=10 (overriding reach_state.py's default of 3), aligned
exactly with actor_batched_d1_batch_episodes=10, so every periodic
rebuild lands on a commit boundary (every accumulator, including the new
per-goal ones, already zero) -- same rebuild-invariance convention as
p3_reach_baseline_ckpt10.py.
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
