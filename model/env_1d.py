"""env_1d.py — Simple1DReachAdapter: a deliberately MINIMAL sanity-check
task, reusing this project's exact D1/D2 SNN architecture, cortex, critic,
and dopamine machinery completely unchanged (only cfg.n_actions=1 and this
trivial 1-DOF dynamics differ from env_reach.py's real 2-joint task).

Motivation: after an extensive investigation into why reward-only
learning fails on the real 2-DOF reaching task, this asks the sharpest
possible version of that question -- removing every source of difficulty
this session identified (2D nonlinear kinematics / Jacobian-driven
state-dependent reward, needing two joints coordinated) while keeping
everything else (the REAL main.py training loop, REAL STDP/dopamine
machinery, REAL episode structure) unchanged. If THIS also fails via real
training, that's evidence about the architecture itself, not about this
specific task's difficulty.

obs      = {"joints": [q, 0],  "object": [target, 0]}  (reuses the existing
           2D encoders UNCHANGED -- only the FIRST component of each is
           ever driven, matching every earlier oracle test's convention of
           holding the unused slot fixed)
action   = heads (1,) in [-1,+1] -> dq = heads[0] * cfg.q_step
reward   = |target-q_prev| - |target-q_now|      dense, signed both ways,
           computed directly in q-space -- no kinematics, no Jacobian, no
           state-dependent nonlinearity of any kind
success  = |target-q| < cfg.success_radius
"""
import numpy as np

from base_util import Base
from env_adapter import EnvAdapter


class _FakeArm:
    """Duck-typed shim so main.py's existing `adapter.arm.fk(adapter.arm.q)`
    stats-logging code (shared with env_reach.py's ReachAdapter) works
    unchanged for this task too, with zero further main.py edits. `fk(q)`
    embeds the scalar q as a 2D point [q, 0] so `np.linalg.norm(target -
    fk(q))` reduces to the plain 1D distance |target - q| when `target` is
    ALSO stored the same way (see Simple1DReachAdapter.target)."""

    def __init__(self):
        self.q = 0.0

    def fk(self, q):
        return np.array([float(np.asarray(q).reshape(-1)[0]), 0.0])


class Simple1DReachAdapter(EnvAdapter):

    def __init__(self, cfg):
        self.cfg = cfg
        # Reuses joints_encoder_kw's/object_encoder_kw's own first-axis
        # bounds so the SAME two encoders (unchanged) see values inside
        # their own calibrated ranges. reach1d_sanity.py
        # sets object_encoder_kw's lo/hi equal to joints_encoder_kw's --
        # a real, measured bug in the first version of this task used
        # object_encoder_kw's own (much narrower, off-center) default range
        # directly as the target-sampling range while q lived in joints'
        # much wider range, so `reward=|target-q|` was comparing two
        # different numeric spaces, and the target was always confined to
        # a narrow band well inside q's full range -- inflating measured
        # "success" via exploration luck (starting near the band) rather
        # than learned control. Both self.lo/self.hi (q's range) and
        # self.obj_lo/self.obj_hi (target's range) are read from the
        # config so they track whatever the config sets, instead of being
        # silently assumed equal in code.
        self.lo = float(cfg.joints_encoder_kw["lo"][0])
        self.hi = float(cfg.joints_encoder_kw["hi"][0])
        self.obj_lo = float(cfg.object_encoder_kw["lo"][0])
        self.obj_hi = float(cfg.object_encoder_kw["hi"][0])
        self.arm = _FakeArm()   # see _FakeArm's docstring
        self.target = None
        self.d_prev = None
        self.step_count = 0
        # DEDICATED, cfg.seed-derived RNG for exploration noise -- see
        # env_reach.py's identical fix for why np.random.uniform (global,
        # unseeded-by-cfg state) broke run-to-run reproducibility.
        seed = getattr(cfg, "seed", -1)
        self._explore_rng = np.random.default_rng(seed if seed >= 0 else None)

    def reset(self, rng) -> dict:
        self.arm.q = float(rng.uniform(self.lo, self.hi))
        target_val = float(rng.uniform(self.obj_lo, self.obj_hi))
        self.target = np.array([target_val, 0.0])
        self.d_prev = float(abs(target_val - self.arm.q))
        self.step_count = 0
        return self._obs()

    def _obs(self) -> dict:
        return {"joints": np.array([self.arm.q, 0.0]), "object": self.target.copy()}

    def step(self, action):
        dq = float(np.clip(action[0], -1.0, 1.0)) * self.cfg.q_step
        self.arm.q = float(np.clip(self.arm.q + dq, self.lo, self.hi))
        d_now = float(abs(self.target[0] - self.arm.q))
        reward = self.d_prev - d_now      # dense, signed both ways, same shape as env_reach.py
        # actor_per_channel_reward: with n_actions=1 there's nothing to
        # decompose -- the single channel's reward IS the pooled reward.
        self.last_per_joint_reward = np.array([reward])
        self.d_prev = d_now
        self.step_count += 1
        success = d_now < self.cfg.success_radius
        terminated = success or self.step_count >= self.cfg.max_steps
        return self._obs(), reward, terminated, success

    def select_action(self, actor, cfg, t0: float, t1: float, exploration_rate: float):
        heads = self.compute_heads(actor, cfg, t0, t1, scale_factor=1.0)
        noise = exploration_rate * self._explore_rng.uniform(-1.0, 1.0, size=heads.shape)
        action = np.clip(heads + noise, -1.0, 1.0)
        return action, heads.copy()

    def record_action_stats(self, episode_actions: list, stats: dict) -> str:
        if not episode_actions:
            return ""
        heads = np.asarray(episode_actions)
        mean_abs = np.mean(np.abs(heads), axis=0)
        for ch in range(heads.shape[1]):
            Base.append_stat(stats, f"head_ch{ch}_mean_abs", float(mean_abs[ch]))
        return "  ".join(f"h{ch}={mean_abs[ch]:+.2f}" for ch in range(heads.shape[1]))
