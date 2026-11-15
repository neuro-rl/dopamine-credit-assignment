"""Config for cart_pole/run.py: the 'all three' E1-lock condition
(action-identity + centering + population/vector readout). 9-seed result:
mean 160.6, std 112.5; the population-readout's marginal effect on top of
+action-identity+centering is not significant (p=0.724).

wmax_override: the original train_and_eval_vec2d_proj.py harness took an
optional 5th positional CLI arg to override model.WMAX (tested whether
growing weight saturation was capping further improvement). Set a float
here (or pass --wmax on run.py's CLI) to reproduce that; None leaves the
module's own default untouched.
"""
from types import SimpleNamespace


def Config():
    return SimpleNamespace(
        label="vec2d_proj",
        action_identity=True,
        centered=True,
        population_readout=True,
        n_episodes=1000,
        n_eval_episodes=30,
        warmup_episodes=50,
        anneal=False,
        eps_start=0.5,
        eps_end=0.15,
        wmax_override=None,
    )
