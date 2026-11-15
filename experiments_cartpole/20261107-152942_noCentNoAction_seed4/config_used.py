"""Config for cart_pole/run.py: the 'neither' E1-lock condition (no
centering, no action-identity-aware credit) -- confirmed worst of the
ablation chain at 9 seeds (mean 9.7, std 0.9).
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="noCentNoAction",
        action_identity=False,
        centered=False,
        population_readout=False,
        n_episodes=1000,
        n_eval_episodes=30,
        warmup_episodes=50,
        anneal=False,
        eps_start=0.5,
        eps_end=0.15,
        wmax_override=None,
    )
