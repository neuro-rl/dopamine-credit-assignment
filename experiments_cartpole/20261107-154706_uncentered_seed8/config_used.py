"""Config for cart_pole/run.py: the '+action-identity' E1-lock condition
(action-identity-aware credit, no centering).
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="uncentered",
        action_identity=True,
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
