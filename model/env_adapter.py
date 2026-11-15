from abc import ABC, abstractmethod

import numpy as np

from base_util import Base


class EnvAdapter(ABC):
    """
    Encapsulates everything that differs between tasks: environment lifecycle
    (reset/step), action selection (SNN activity -> action), reward. main.py
    drives this without knowing what task it is.

    obs is a dict of named arrays, e.g. {"joints": q, "object": target_xy} --
    matches SensoryInput's encoder-name keying (encoders.py), not a single
    flat feature vector. There is no obs_min/obs_max/preprocess here (the old
    ndarray-obs + MinMaxScaler design doesn't apply): each encoder normalises
    its own stream via its own lo/hi at construction time.

    There is also no external gym env to wrap (main.py drives this directly,
    no gymnasium) -- reset()/step() own the task's own simulated world
    (e.g. ReachAdapter owns an SO101PlanarArm + target).
    """

    @abstractmethod
    def reset(self, rng) -> dict:
        """Start a new episode. Returns the initial obs dict."""
        ...

    @abstractmethod
    def step(self, action):
        """Advance one control step.

        Returns (obs: dict, reward: float, terminated: bool, success: bool).
        """
        ...

    @abstractmethod
    def select_action(self, actor, cfg, t0: float, t1: float,
                      exploration_rate: float):
        """
        Return (action, stat_action).
        action       -- passed directly to step()
        stat_action  -- used for action-distribution stats
        """
        ...

    @abstractmethod
    def record_action_stats(self, episode_actions: list, stats: dict) -> str:
        """
        Write action stats into stats dict and return a one-line summary
        string for the episode banner.
        """
        ...

    # ── Shared helper ─────────────────────────────────────────────────────────

    def compute_heads(self, actor, cfg, t0: float, t1: float,
                      scale_factor: float = 10.0) -> np.ndarray:
        """
        Return N values ∈ [-1, +1], one per D1/D2 channel:
            head[i] = tanh((d1_rate - d2_rate) / scale)

        scale = N * window_s * scale_factor
        Discrete-choice envs use scale_factor=10 (only sign matters).
        Continuous envs use scale_factor=1 so small rate differences
        produce meaningful action magnitude, not near-zero output.

        actor.vector_readout=True (population-vector decode, see actor.py's
        own comment): replaces the plain population rate (Base.rate_in_
        window -- identical weight per neuron, so a uniform activity shift
        moves it directly) with a per-neuron-WEIGHTED sum (actor.
        weighted_rate_in_window) using each neuron's fixed random preferred
        value -- decoupled from overall activity level (MEASURED offline,
        random_vector_decode_check.py). actor_vector_readout_scale_factor
        (getattr, default equal to scale_factor) lets this mode use its own
        calibration -- weighted sums are typically smaller in magnitude
        than a plain rate difference (cancellation across +/- preferred
        neurons), so the SAME scale_factor a plain-rate config was tuned
        with likely under-drives tanh here; unvalidated starting point,
        expect to need real-data recalibration.
        """
        vector_readout = getattr(actor, "vector_readout", False)
        # d2_stabilizer (actor.py): D2 stops being a second vote subtracted
        # from the readout -- its influence on the action is now entirely
        # indirect, via suppressing specific D1 neurons (a wide-radius,
        # asymmetric D2->D1 connection built in actor.py) BEFORE D1's own
        # weighted sum is computed here. Dropping D2 from this formula
        # entirely reflects that: D2's activity, once it's not itself
        # cortex-driven-and-subtracted in a matched/symmetric way, doesn't
        # carry directional content worth subtracting -- it already did its
        # job upstream, in shaping which D1 neurons got to fire at all.
        d2_stabilizer = getattr(actor, "d2_stabilizer", False)
        # unified_readout (actor.py): exactly ONE D1/D2 pair is built
        # (channel index 0 always) and D1's preferred value is a
        # dim_actions-D VECTOR per neuron, so weighted_rate_in_window(0,
        # "d1", ...) already returns the full n_actions-length action
        # vector directly from this one population -- no per-channel loop
        # needed. D2 is dropped from the formula exactly as in the plain
        # d2_stabilizer case above (same rationale) unless d2_stabilizer
        # is off, in which case actor.py built a matching D2 vector
        # preference as a fallback.
        unified_readout = getattr(actor, "unified_readout", False)
        if vector_readout:
            scale_factor = getattr(cfg, "actor_vector_readout_scale_factor", scale_factor)
        scale = actor.N * max((t1 - t0) / 1000.0, 1e-6) * scale_factor
        if unified_readout:
            d1 = actor.weighted_rate_in_window(0, "d1", t0, t1)
            d2 = 0.0 if d2_stabilizer else actor.weighted_rate_in_window(0, "d2", t0, t1)
            # per_dim_normalize (actor.py): corrects the per-DIMENSION
            # gain mismatch a small-population random pref draw produces
            # (q1/q2 not equally "spread out" by chance) -- multiplies
            # the shared scalar `scale` by each dimension's own
            # normalized std instead of applying the same divisor to
            # both. No-op (dim_scale=None or all-1s) for scalar pref or
            # when disabled.
            if getattr(actor, "per_dim_normalize", False):
                dim_scale = actor.d1_pref_dim_scale(0)
                if dim_scale is not None:
                    scale = scale * dim_scale
            return np.tanh((d1 - d2) / (scale + 1e-6))
        heads = np.empty(cfg.n_actions)
        for ch in range(cfg.n_actions):
            if vector_readout:
                d1 = actor.weighted_rate_in_window(ch, "d1", t0, t1)
            else:
                d1 = Base.rate_in_window(actor.get_recorder(ch, "d1"), actor.N, t0, t1)
            if d2_stabilizer:
                d2 = 0.0
            elif vector_readout:
                d2 = actor.weighted_rate_in_window(ch, "d2", t0, t1)
            else:
                d2 = Base.rate_in_window(actor.get_recorder(ch, "d2"), actor.N, t0, t1)
            heads[ch] = np.tanh((d1 - d2) / (scale + 1e-6))
        return heads
