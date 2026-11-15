"""p3_reach_baseline_reachvec.py -- reach-vector-input spec (2026-10-22,
external-collaborator): the full-workspace continuous reach (this
config's own parent, p3_reach_baseline.py, never learned well: 164.5mm/
8.3% success, 9-seed confirmation) never gave the actor the error
vector directly -- cortex sees raw joints + raw target, and has to
discover the forward-kinematics subtraction itself. This swaps cortex's
"object" (raw target) stream for reach_vec = fk(joints)-target (the
same population already validated for the critic, critic_reach_vec_
input), removing the FK-error step while keeping joints (q) -- the
map from error to joint motion is still q-dependent (see Control 1,
findings/REACHVEC_INPUT_FINDING.md), so q is required, not optional.

Unchanged: continuous targets, wide random start (p3_reach_baseline.py's
own existing defaults -- arm.reset()/sample_target() already sample the
FULL range every episode, no box-shrinking flag is set here), standard
delta reward (reward_shaping stays at its "delta" default -- NOT the
exponential settling term, which already regressed a different task
and would confound this input change), cortex + batched D1 + Python
critic, 3 seeds (0,1,5)/400ep/100 steps.

Built from p3_reach_baseline.py (single import).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p3_reach_baseline", Path(__file__).resolve().parent / "p3_reach_baseline.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def Config():
    cfg = _base.Config()
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_reachvec")
    cfg.actor_reach_vec_input = True
    return cfg
