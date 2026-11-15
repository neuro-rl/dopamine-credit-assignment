"""CartPole state -> spike-rate encoding: a 6-channel Gaussian population
code (x, xdot, theta, thetadot, theta/4, thetadot/4) driving relay
neurons into model/cortex.py's Cortex population, plus a constant-rate
bias channel. Byte-identical across all four E1-lock conditions
(noCentNoAction/uncentered/centered/vec2d_proj) in the original
nest_cartpole_*_test.py modules -- this is that shared code, pulled into
its own file so it exists in exactly one place.

Not named cortex.py / encoder.py: model/cortex.py and model/encoders.py
already exist in the arm's model/ directory, which every cart_pole
module adds to sys.path at position 0 -- a same- or close-named file
here would risk a silent import shadowing bug.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import nest
import numpy as np
from cortex import Cortex

N_POP = 6
BASELINE_RATE_HZ = 20.0
MAX_RATE_HZ = 120.0
SIGMA = 0.4
PREF = np.linspace(-1, 1, N_POP)
SIGMA_Q = 0.12
PREF_Q = np.linspace(-0.3, 0.3, N_POP)
CHANNEL_PREF = [PREF, PREF, PREF, PREF, PREF_Q, PREF_Q]
CHANNEL_SIGMA = [SIGMA, SIGMA, SIGMA, SIGMA, SIGMA_Q, SIGMA_Q]
N_STATE_CHANNELS = 6

STATE_SCALE = np.array([2.4, 3.0, 0.2095, 3.0])

CORTEX_EXC_SHAPE = (6, 6)
CORTEX_INH_SHAPE = (3, 3)
INPUT_TO_CORTEX_P = 0.3
INPUT_CORTEX_INIT = 150.0
N_EXC = 36   # 6x6 cortex, fixed for this architecture


def normalize_state(s):
    return np.clip(np.asarray(s) / STATE_SCALE, -1.0, 1.0)


def extended_obs(state_norm):
    x, xdot, theta, thetadot = state_norm
    return np.array([x, xdot, theta, thetadot, theta / 4.0, thetadot / 4.0])


# Example: state=(0.5, 0.3, 0.05, 0.2) -> obs6=(0.2083, 0.1000, 0.2387,
# 0.0667, 0.0597, 0.0167) -> per-channel rates (Hz, 6 neurons each):
#   x          21.0  33.0  79.4 120.0  81.9  34.1
#   xdot       22.3  41.6  95.5 116.9  65.8  28.0
#   theta      20.8  31.1  74.8 119.5  86.5  36.3
#   thetadot   22.9  44.9 100.1 114.6  61.1  26.6
#   theta/4    21.1  33.6  80.8 120.0  80.5  33.5
#   thetadot/4 23.1  46.1 101.5 113.7  59.6  26.2
def set_state_rates(pg_vars, obs6):
    for i in range(N_STATE_CHANNELS):
        pref, sigma = CHANNEL_PREF[i], CHANNEL_SIGMA[i]
        gauss = np.exp(-((obs6[i] - pref) ** 2) / (2 * sigma ** 2))
        rate = BASELINE_RATE_HZ + (MAX_RATE_HZ - BASELINE_RATE_HZ) * gauss
        pg_vars[i].rate = rate


def build_input_and_cortex():
    pg_vars = [nest.Create("poisson_generator", N_POP) for _ in range(N_STATE_CHANNELS)]
    relay_vars = [nest.Create("iaf_psc_alpha", N_POP) for _ in range(N_STATE_CHANNELS)]
    for pg, relay in zip(pg_vars, relay_vars):
        nest.Connect(pg, relay, "one_to_one", {"weight": 1000.0})
    all_relay = relay_vars[0]
    for rv in relay_vars[1:]:
        all_relay = all_relay + rv

    # Bias channel for cortex, analogous to the critic's bias
    # feature -- fires at a CONSTANT rate regardless of state, giving
    # cortex->D1/D2 a state-independent tonic input the plasticity rule
    # can use to build an overall offset, the same role the critic's bias
    # weight plays for V(s).
    bias_pg = nest.Create("poisson_generator", N_POP, {"rate": MAX_RATE_HZ})
    bias_relay = nest.Create("iaf_psc_alpha", N_POP)
    nest.Connect(bias_pg, bias_relay, "one_to_one", {"weight": 1000.0})
    all_relay = all_relay + bias_relay

    cortex = Cortex(exc_shape=CORTEX_EXC_SHAPE, inh_shape=CORTEX_INH_SHAPE,
                     r_ei=0.40, w_ei=80.0, name_prefix="cartpole_toy_v2_")
    nest.Connect(all_relay, cortex.exc,
                 {"rule": "pairwise_bernoulli", "p": INPUT_TO_CORTEX_P},
                 {"weight": INPUT_CORTEX_INIT, "delay": 1.0})
    return pg_vars, cortex
