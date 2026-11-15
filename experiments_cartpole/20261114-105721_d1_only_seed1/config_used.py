"""Config for cart_pole/run.py: the locked "centered" condition, with D2
removed from the decision and from learning entirely (model.py's
run_episode(d1_only=True)). D2 is still built and simulated (nothing in
the shared run.py/run_io.py harness special-cases its absence), but its
spikes are discarded, its weights are frozen at their initial random
values forever, and it has zero influence on the action taken or on D1's
own credit. g becomes D1's own count minus a running mean of that same
count (g_bar), the self-referential analogue of D1-D2.

Tests the question this project had never actually answered: does the
opponent D1/D2 architecture itself matter, or would a single feedforward
population do just as well? Every other flag matches "centered" (the
current best CartPole result, median 118.2) exactly, so this is a single-
factor change from that condition.
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="d1_only",
        action_identity=True,
        centered=True,
        population_readout=False,
        d1_only=True,
        n_episodes=1000,
        n_eval_episodes=30,
        warmup_episodes=50,
        anneal=False,
        eps_start=0.5,
        eps_end=0.15,
        wmax_override=None,
    )
