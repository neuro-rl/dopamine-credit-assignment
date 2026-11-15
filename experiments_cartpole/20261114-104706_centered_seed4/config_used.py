"""Config for cart_pole/run.py: the '+action-identity+centering' E1-lock
condition.

warmup_episodes=50, matching the other three conditions: the original
version used warmup_episodes=0, which left this the only cell with no
critic-warmup phase, confounding the centering comparison. Set to 50 for
an apples-to-apples ablation.
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="centered",
        action_identity=True,
        centered=True,
        population_readout=False,
        n_episodes=1000,
        n_eval_episodes=30,
        warmup_episodes=50,
        anneal=False,
        eps_start=0.5,
        eps_end=0.15,
        wmax_override=None,
    )
