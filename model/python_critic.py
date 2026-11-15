"""python_critic.py — a small linear TD(0) critic computed entirely in
Python/numpy, meant to run alongside the real spiking critic (critic.py)
at all times rather than replace it in the codebase. `cfg.teacher`
selects which signal actually drives dopamine, so the spiking critic's
own learning can be tracked against this consistent reference across
any config.

phi(s) = [q1_norm, q2_norm, tx_norm, ty_norm, bias] -- joints normalized
via the arm's own joint limits, target normalized via arm.py's
TARGET_LO/TARGET_HI, both to [-1, 1]. V(s) = w . phi(s), trained online
every step via standard semi-gradient TD(0): delta = r + gamma*V(s') -
V(s); w += alpha*delta*phi(s). Task-specific (couples to ReachAdapter's
observation shape) -- not meant as a general-purpose critic.

feature_set="rich": the reward's real state-dependence comes from the
arm's Jacobian, which is trigonometric in the joint angles -- a purely
linear function of raw q/target coordinates structurally cannot
represent it. Adding sin/cos(q1), sin/cos(q2), and joint-target cross
terms substantially improves the fit to the true Monte-Carlo return.
Default "linear": every existing config's behavior unchanged.
"""
import numpy as np

class PythonCritic:
    def __init__(self, arm, gamma: float = 0.99, alpha: float = 0.02,
                 feature_set: str = "linear"):
        self.arm = arm
        self.gamma = gamma
        self.alpha = alpha
        self.feature_set = feature_set
        # feature_set="rich" hardcodes q[0]/q[1] trig pairs and 2x2
        # joint-target cross terms below -- not generalized to n_joints
        # != 2. No existing config sets
        # feature_set="rich" by default; guard rather than silently
        # compute something wrong for a 3-DOF obs.
        if feature_set == "rich" and arm.n_joints != 2:
            raise NotImplementedError(
                "PythonCritic feature_set='rich' is hardcoded to 2 joints "
                "(trig/cross-term features) -- not yet generalized for "
                f"arm.n_joints={arm.n_joints}")
        n = arm.n_joints
        self.w = np.zeros(13 if feature_set == "rich" else 2 * n + 1)

    def phi(self, obs):
        # self.arm.target_lo/target_hi (not the module-level TARGET_LO/HI
        # import this used to use) -- an instance property sized 2 or 3 by
        # arm.include_shoulder_pan, so this normalizes correctly against a
        # 3-vector obs["object"] too (the
        # module constants stayed a fixed 2-vector and would otherwise
        # hard-crash here on a shape mismatch for a 3-DOF obs).
        q_norm = (obs["joints"] - self.arm._lo) / (self.arm._hi - self.arm._lo) * 2 - 1
        t_norm = (obs["object"] - self.arm.target_lo) / (self.arm.target_hi - self.arm.target_lo) * 2 - 1
        if self.feature_set == "rich":
            q = obs["joints"]
            trig = np.array([np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])])
            cross = np.array([q_norm[0] * t_norm[0], q_norm[0] * t_norm[1],
                               q_norm[1] * t_norm[0], q_norm[1] * t_norm[1]])
            return np.concatenate([q_norm, t_norm, trig, cross, [1.0]])
        return np.concatenate([q_norm, t_norm, [1.0]])

    def value(self, obs) -> float:
        return float(self.w @ self.phi(obs))

    def update(self, obs, reward: float, next_obs) -> float:
        """Online semi-gradient TD(0) step. Returns delta (the TD error) --
        callers that want V(s) specifically should call value(obs) BEFORE
        update() (update() mutates self.w, so a value() call after it no
        longer reflects the weights that produced this delta)."""
        phi_s = self.phi(obs)
        v_s = float(self.w @ phi_s)
        v_next = self.value(next_obs)
        delta = reward + self.gamma * v_next - v_s
        self.w += self.alpha * delta * phi_s
        return delta
