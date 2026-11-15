"""env_reach.py — ReachAdapter: planar reaching task.

obs      = {"joints": q (2,),  "object": target_xy (2,)}
         (3, and 3, when cfg.arm_include_shoulder_pan=True -- shoulder_pan
         added as a 3rd DOF, target becomes a full (x,y,z) world position)
action   = heads (2,) in [-1,+1] from actor.compute_heads() -> dq = heads * cfg.q_step
         (3, likewise, when cfg.n_actions=3 to match)
reward   = d_prev - d_now                    dense, signed, both directions
         (cfg.reward_shaping="jacobian": see step()'s docstring)
success  = d < cfg.success_radius
terminate= success or step >= max_steps

No terminal bonus -- the point of this task is that the dense signed reward
IS the experiment; a sparse bonus reintroduces the Acrobot problem.

There is no external gym env here (main.py drives this directly) --
ReachAdapter owns the SO101PlanarArm and the target itself.
"""
import numpy as np

from base_util import Base
from env_adapter import EnvAdapter
from arm import SO101PlanarArm


class ReachAdapter(EnvAdapter):

    def __init__(self, cfg):
        self.cfg = cfg
        self.arm = SO101PlanarArm(
            include_shoulder_pan=getattr(cfg, "arm_include_shoulder_pan", False))
        # target_box_halfwidth_m: shrinks sample_target()'s rejection box to a small square around
        # cfg.target_box_center (default: the full box's own center,
        # measured reachable) instead of the full workspace -- start
        # config (arm.reset) is untouched, stays full-range random, so
        # this isolates target spread alone. None (default, every other
        # config) leaves the arm's own full-workspace box untouched.
        halfwidth = getattr(cfg, "target_box_halfwidth_m", None)
        if halfwidth is not None:
            center = getattr(cfg, "target_box_center",
                              (self.arm.target_lo + self.arm.target_hi) / 2.0)
            center = np.asarray(center, dtype=float)
            self.arm.target_lo = center - halfwidth
            self.arm.target_hi = center + halfwidth
        # target_goal_set: a small set of fixed, discrete goal points -- when
        # set, reset() picks one uniformly at random instead of rejection-
        # sampling a continuous box. Mutually exclusive with
        # target_box_halfwidth_m in practice (not enforced -- just don't
        # set both). None (default, every other config) leaves the
        # continuous-box behavior above untouched.
        goal_set = getattr(cfg, "target_goal_set", None)
        self._goal_set = np.asarray(goal_set, dtype=float) if goal_set is not None else None
        self.target = None
        self.d_prev = None
        self.step_count = 0
        # q_vel (velocity_input): actual_dq from the
        # just-completed step, in rad/step (same units as q_step, not
        # normalized by wall-clock dt) -- 0 at reset (no motion yet).
        self.q_vel = np.zeros(self.arm.n_joints)
        # DEDICATED, cfg.seed-derived RNG for exploration noise -- was
        # np.random.uniform (global, unseeded-by-cfg state) in
        # select_action, silently breaking run-to-run reproducibility at a
        # fixed cfg.seed: exploration noise is part of the actual env
        # dynamics AND, under use_action_perturbation, the credit signal
        # itself.
        seed = getattr(cfg, "seed", -1)
        self._explore_rng = np.random.default_rng(seed if seed >= 0 else None)

    def reset(self, rng) -> dict:
        self.arm.reset(rng)
        if self._goal_set is not None:
            self.target = self._goal_set[rng.integers(len(self._goal_set))].copy()
        else:
            self.target = self.arm.sample_target(rng)
        self.d_prev = float(np.linalg.norm(self.target - self.arm.fk(self.arm.q)))
        self.step_count = 0
        self.q_vel = np.zeros(self.arm.n_joints)
        return self._obs()

    def _obs(self) -> dict:
        return {"joints": self.arm.q, "object": self.target, "joints_vel": self.q_vel}

    def step(self, action):
        """cfg.reward_shaping (default "delta", every existing config's
        behavior unchanged):

        The default reward (d_prev-d_now, a finite difference of the
        actual post-step distance) has only a small fraction of its
        variance explained by either channel's own current action under a
        random policy -- most of it comes from the current arm
        configuration/target position, because the Jacobian relating joint
        velocity to end-effector velocity varies sharply across the
        workspace: the same dq produces very different real distance
        changes depending on where the arm currently is.

        reward_shaping="jacobian": replaces the finite difference with its
        first-order (linearized) approximation -- the INSTANTANEOUS
        directional derivative of distance along the actual joint motion,
        computed from arm.jacobian() at the PRE-step configuration:
        reward = (unit vector toward target) . (J @ actual_dq). This is
        the SAME physical quantity (rate of distance
        change, same units/similar magnitude to the old reward for this
        task's small q_step=0.08rad step size) but computed via
        linearization instead of a finite difference through the arm's
        true nonlinear kinematics -- by construction an EXACT linear
        function of the actual joint displacement at a given
        configuration, removing the higher-order/nonlinear noise a real
        step introduces. actual_dq (not the raw clipped-to-[-1,1] action)
        is used so joint-limit clipping in arm.step() is correctly
        accounted for. success/termination always reflect the TRUE
        post-step distance regardless of this flag -- only the REWARD
        computation differs, matching potential-based-shaping practice of
        not touching the actual task-completion criterion.
        """
        dq = np.clip(action, -1.0, 1.0) * self.cfg.q_step
        reward_shaping = getattr(self.cfg, "reward_shaping", "delta")
        q_before = self.arm.q
        p_before = self.arm.fk(q_before)
        J = self.arm.jacobian(q_before)
        self.arm.step(dq)
        d_now = float(np.linalg.norm(self.target - self.arm.fk(self.arm.q)))
        actual_dq = self.arm.q - q_before
        self.q_vel = actual_dq.copy()
        direction = self.target - p_before
        dist_before = float(np.linalg.norm(direction))
        unit_dir = direction / dist_before if dist_before > 1e-9 else np.zeros_like(direction)
        # self.last_per_joint_reward (actor_per_channel_reward): ALWAYS
        # computed (cheap -- pure linear algebra, no extra arm.fk() calls),
        # regardless of reward_shaping -- an EXACT additive per-joint
        # decomposition of the jacobian-style reward: unit_dir @ (J @
        # actual_dq) == sum_i unit_dir @ (J[:,i]*actual_dq[i]) == sum(self.
        # last_per_joint_reward), since both the matrix-vector product and
        # the dot product distribute over the per-joint sum. r_i is "how
        # much did joint i's OWN motion move the end-effector toward the
        # target," decoupled from what the other joint(s) did.
        self.last_per_joint_reward = np.array(
            [float(unit_dir @ (J[:, i] * actual_dq[i])) for i in range(len(actual_dq))])
        if reward_shaping == "jacobian":
            reward = float(unit_dir @ (J @ actual_dq))
        else:
            reward = self.d_prev - d_now      # dense, signed both ways
        self.d_prev = d_now
        self.step_count += 1
        success = d_now < self.cfg.success_radius
        terminated = success or self.step_count >= self.cfg.max_steps
        return self._obs(), reward, terminated, success

    def select_action(self, actor, cfg, t0: float, t1: float, exploration_rate: float):
        """Both channels act every step -- no argmax/winner-take-all like the
        discrete-action adapters (Acrobot/CartPole/Pendulum). Each channel's
        tanh(d1-d2) IS the continuous dq for its own joint."""
        heads = self.compute_heads(actor, cfg, t0, t1, scale_factor=1.0)
        noise = exploration_rate * self._explore_rng.uniform(-1.0, 1.0, size=heads.shape)
        action = np.clip(heads + noise, -1.0, 1.0)
        return action, heads.copy()

    def record_action_stats(self, episode_actions: list, stats: dict) -> str:
        if not episode_actions:
            return ""
        heads = np.asarray(episode_actions)   # (T, n_actions)
        mean_abs = np.mean(np.abs(heads), axis=0)
        for ch in range(heads.shape[1]):
            Base.append_stat(stats, f"head_ch{ch}_mean_abs", float(mean_abs[ch]))
        return "  ".join(f"h{ch}={mean_abs[ch]:+.2f}" for ch in range(heads.shape[1]))
