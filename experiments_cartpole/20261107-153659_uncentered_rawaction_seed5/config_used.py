"""Config for cart_pole/run.py: "action-dependent, uncentered": credit =
td0 * a (a in {-1,+1}), no tanh(g/g_scale) subtraction. Same protocol as
configs/uncentered.py ("action-dependent, surrogate-centered": credit =
td0 * (a - tanh(g/g_scale))) in every other respect -- same presynaptic
centered=False, same warmup/episode/eval counts, same fixed eps=0.3
(anneal=False) -- so the uncentered-vs-this comparison isolates exactly
the tanh(g/g_scale) subtraction's own effect, holding everything else
fixed.
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="uncentered_rawaction",
        action_identity=True,
        action_centered=False,
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
