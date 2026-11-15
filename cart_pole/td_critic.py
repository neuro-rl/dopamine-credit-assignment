"""CartPole critic: 14-feature linear value function, analytic TD(0).
Byte-identical across all four E1-lock conditions in the original
nest_cartpole_*_test.py modules -- pulled into its own file so it exists
in exactly one place.

Not named critic.py: model/critic.py already exists in the arm's model/
directory, which every cart_pole module adds to sys.path at position 0 --
a same-named file here would risk a silent import shadowing bug.
"""
import numpy as np

GAMMA = 0.97
CRITIC_LR = 0.05
N_CRITIC_FEATURES = 14
# past attempts to trim these features (fewer rescaled theta/thetadot
# copies, or dropping the bias term) measurably hurt success rate.
# Keep all 14 features; don't trim without re-testing.


def critic_features(obs6):
    x, xdot, theta, thetadot, theta_q, thetadot_q = obs6
    # half-wave rectified pos/neg copies: lets the (linear) critic learn a
    # different slope for positive vs negative tilt, instead of one
    # symmetric line through theta=0
    theta_pos, theta_neg = max(theta, 0.0), max(-theta, 0.0)
    thetadot_pos, thetadot_neg = max(thetadot, 0.0), max(-thetadot, 0.0)
    return np.array([1.0, x, xdot, theta, thetadot, theta ** 2,
                      theta_pos, theta_neg, thetadot_pos, thetadot_neg,
                      theta_q, thetadot_q, theta / 8.0, thetadot / 8.0])


def critic_update(w_critic, feat_s, feat_next, reward, terminated):
    """Linear TD(0): V(s)=w.feat(s), same update family as the pole-balancing
    actor-critic in Barto, Sutton & Anderson (1983). In-place update of
    w_critic (mutated via numpy's += in
    place -- the caller's array object is updated, nothing is returned
    for it). v_next is zeroed on true termination (pole fell) but NOT on
    truncation (hit max_steps) -- bootstrapping continues through a
    timeout, matching the original `done and terminated` condition
    (algebraically identical to `terminated`, since done = terminated or
    truncated). Returns the TD error `td0`.
    """
    v_s = w_critic @ feat_s
    v_next = 0.0 if terminated else w_critic @ feat_next
    td0 = reward + GAMMA * v_next - v_s
    w_critic += CRITIC_LR * td0 * feat_s
    return td0
