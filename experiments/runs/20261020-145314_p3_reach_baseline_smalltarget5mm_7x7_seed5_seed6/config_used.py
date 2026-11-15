"""p3_reach_baseline_smalltarget5mm.py -- small-target ablation, Arm B
(2026-10-20, external-collaborator spec): the ±5mm small-target reach
learns under the gain-field/cortex-bypassed/native-STDP variant
(SMALLTARGET_FINDING.md). This arm asks the most-informative ablation
question: does the STANDARD architecture -- place-field encoder ->
recurrent cortex -> D1/D2, cortex NOT bypassed, NO gain-field -- also
learn the localized reach on its own? If it does, the gain-field +
bypass engineering was never load-bearing for localized targets. If it
fails where the gain-field/bypass succeeds, the representation
engineering is shown to be load-bearing.

Built from p3_reach_baseline.py (single import) -- that IS this
project's own "standard architecture" config (place-field encoder,
real recurrent cortex, Python critic; see its own docstring for the
full locked P2 decision set: fix1_actionpert, fix3_axispref,
fix5_velocity, grid=7x7, da_tau_c, critic_reach_vec_input, fixed
0.3 exploration floor -- all held as-is, since "standard architecture"
means the architecture's own standard config, not a stripped-down
variant). Two changes only:
  (1) actor_batched_d1=False -- native stdp_dopamine_synapse (the rule
      under test), matching every other arm in this ablation and
      CartPole's own spine. p3_reach_baseline.py's default (True,
      BatchedD1Learning) is NOT what this ablation holds fixed --
      the spec requires native STDP in every arm.
  (2) target_box_halfwidth_m=0.005 / target_box_center=(0.29, 0.125) --
      same ±5mm box, same full-workspace-box-center, as every other
      arm in this ablation (see env_reach.py's ReachAdapter.__init__).
      Start config stays fully random (untouched by this override).

Held fixed per the spec: Python critic (teacher="python", unchanged --
p3_reach_baseline.py never overrides it), continuous per-joint heads
(unchanged), 3 seeds, 400 episodes (set at launch, not here).

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
    cfg.name = cfg.name.replace("p3_reach_baseline", "p3_reach_baseline_smalltarget5mm")
    cfg.actor_batched_d1 = False
    cfg.target_box_halfwidth_m = 0.005
    cfg.target_box_center = (0.29, 0.125)
    return cfg
