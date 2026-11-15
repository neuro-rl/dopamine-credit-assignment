"""Config for cart_pole/run.py: '+action-identity+centering' (the locked
"centered" condition) PLUS real, static, bidirectional D1<->D2 lateral
inhibition -- the mechanism CartPole never had (see model.py's
build_network docstring note). Quick-gate test: does an actual
neural opponent circuit beat the "opponent" realised purely as a Python
sign test?

mutual_inhibition_weight=-30.0: calibrated by sweeping D1/D2 firing rate
on a fresh (untrained) network -- a clear, non-bistable ~9% rate
reduction (276->250Hz each).
IMPORTANT, found during calibration: weights at/above ~-200 (all-to-all,
flat population, no topology) are bistable -- initial weight noise alone
snowballs into one population fully silencing the other, independent of
cortex's actual input. That would be catastrophic here (the actor would
stop reading state at all), unlike the arm's own cross-inhibition which
is topographic/Gaussian (spreads weakly over many pairs, not all-to-all)
and never measured to do this. -30 sits well clear of that cliff.
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="mutual_inhibition",
        action_identity=True,
        centered=True,
        population_readout=False,
        mutual_inhibition=True,
        mutual_inhibition_weight=-30.0,
        n_episodes=1000,
        n_eval_episodes=30,
        warmup_episodes=50,
        anneal=False,
        eps_start=0.5,
        eps_end=0.15,
        wmax_override=None,
    )
