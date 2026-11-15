"""The linear-baseline control (paper Section 5.1, "Control 1").

Question: could a single fixed linear map already solve the reach task,
making the reach-vector input trivial and any later positive result a
disguised linear fit? Run this BEFORE any training to find out.

What it does, step by step:
1. Sample 3000 random (joint-angle, target) pairs. For each, compute the
   reach vector e = target - hand(q) and the TRUE per-step joint-velocity
   a resolved-rate controller would take: dq_oracle = pinv(J(q)) @ e,
   using the actual state-dependent Jacobian J(q).
2. Fit ONE global linear map dq = W @ e to all those pairs at once by
   ordinary least squares, ignoring q entirely -- that's the candidate
   trivial solution. Three variants, because a naive fit is numerically
   broken here: raw OLS (misleading -- a few near-singular-J outliers
   dominate it), direction-only (unit-normalized targets, magnitude
   discarded), and winsorized (magnitude capped at its own 90th
   percentile, direction kept -- the one reported in the paper, since
   it isn't outlier-dominated but still reflects "bigger gap, bigger
   step").
3. Roll out each fitted W as a fixed policy in the real reach
   environment (60 episodes each) and record final distance/success.
4. For reference, also roll out an untrained (random-action) policy and
   the true oracle policy (the real q-dependent resolved-rate
   controller, recomputed every step) -- the floor and the ceiling.
5. Print mean distance and success rate (at several tolerances) for
   every policy.

Result: the fixed linear map reaches 3.3-5.0% success, indistinguishable
from untrained, while the true oracle reaches 86.7% -- so a disguised
linear fit cannot explain any later positive result on this task.

Pure numpy, no NEST needed (model/arm.py + model/env_reach.py's
ReachAdapter are both NEST-free on this path).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
import numpy as np
from arm import SO101PlanarArm
from env_reach import ReachAdapter


class _Cfg:
    q_step = 0.08
    success_radius = 0.015
    max_steps = 100
    reward_shaping = "delta"
    arm_include_shoulder_pan = False
    seed = -1


def _sample_oracle_pairs(n_samples=3000, seed=20261022):
    arm = SO101PlanarArm()
    rng = np.random.default_rng(seed)
    E, DQ = [], []
    for _ in range(n_samples):
        q = rng.uniform(arm._lo, arm._hi)
        target = arm.sample_target(rng)
        hand = arm.fk(q)
        e = target - hand                        # spec's own sign convention
        J = arm.jacobian(q)
        dq_oracle = np.linalg.pinv(J) @ e          # resolved-rate oracle, full (unclipped) e
        E.append(e)
        DQ.append(dq_oracle)
    return np.array(E), np.array(DQ)


def _fit(E, Y):
    W, *_ = np.linalg.lstsq(E, Y, rcond=None)
    pred = E @ W
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return W, r2


def fit_raw(E, DQ):
    """Naive OLS on raw dq_oracle -- MISLEADING, see module docstring."""
    return _fit(E, DQ)


def fit_direction(E, DQ):
    """Unit-normalize dq_oracle before fitting -- direction only."""
    norms = np.linalg.norm(DQ, axis=1)
    mask = norms > 1e-9
    return _fit(E[mask], DQ[mask] / norms[mask, None])


def fit_winsorized(E, DQ, pct=90):
    """Cap |dq_oracle| at its own p{pct} (direction unchanged) -- the
    reported Control 1 fit: robust to the near-singular-J outliers
    without discarding magnitude/distance information entirely."""
    norms = np.linalg.norm(DQ, axis=1)
    cap = np.percentile(norms, pct)
    scale = np.minimum(1.0, cap / np.maximum(norms, 1e-9))
    return _fit(E, DQ * scale[:, None])


def rollout(policy_fn, n_episodes=60, seed=12345):
    adapter = ReachAdapter(_Cfg())
    rng = np.random.default_rng(seed)
    finals, successes = [], []
    for ep in range(n_episodes):
        obs = adapter.reset(rng)
        for step in range(_Cfg.max_steps):
            action = policy_fn(adapter)
            obs, reward, terminated, success = adapter.step(action)
            if terminated:
                break
        d_mm = float(np.linalg.norm(adapter.target - adapter.arm.fk(adapter.arm.q))) * 1000.0
        finals.append(d_mm)
        successes.append(bool(success))
    return np.array(finals), np.array(successes)


def report(label, finals, successes):
    print(f"{label}: n={len(finals)} mean_d={finals.mean():.1f}mm  success={successes.mean()*100:.1f}%  "
          f"@15mm={np.mean(finals<15)*100:.1f}%  @30mm={np.mean(finals<30)*100:.1f}%  "
          f"@50mm={np.mean(finals<50)*100:.1f}%  @100mm={np.mean(finals<100)*100:.1f}%")


def _linear_map_policy(W):
    def policy(adapter):
        e = adapter.target - adapter.arm.fk(adapter.arm.q)
        dq = e @ W
        return np.clip(dq / _Cfg.q_step, -1.0, 1.0)
    return policy


def untrained_policy(adapter):
    return np.random.default_rng().uniform(-1.0, 1.0, size=2)


def oracle_policy(adapter):
    """Sanity ceiling: the TRUE per-step resolved-rate controller
    (q-dependent J(q), not a fixed global map) -- confirms the rollout
    harness itself can reach near-perfect performance given the correct
    map, isolating the fixed map's own failure as real."""
    e = adapter.target - adapter.arm.fk(adapter.arm.q)
    J = adapter.arm.jacobian(adapter.arm.q)
    dq = np.linalg.pinv(J) @ e
    return np.clip(dq / _Cfg.q_step, -1.0, 1.0)


if __name__ == "__main__":
    E, DQ = _sample_oracle_pairs()

    for label, fit_fn in [("raw (misleading)", fit_raw),
                           ("direction-only", fit_direction),
                           ("winsorized-p90 (reported)", fit_winsorized)]:
        W, r2 = fit_fn(E, DQ)
        print(f"--- {label} fit --- R^2={r2:.4f}")
        finals, succ = rollout(_linear_map_policy(W))
        report(f"  fixed-linear-map ({label})", finals, succ)

    finals_u, succ_u = rollout(untrained_policy)
    report("untrained (random action)", finals_u, succ_u)
    finals_o, succ_o = rollout(oracle_policy)
    report("oracle (true q-dependent resolved-rate, sanity ceiling)", finals_o, succ_o)
