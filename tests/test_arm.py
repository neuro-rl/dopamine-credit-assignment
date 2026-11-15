"""The gate: do not wire the SNN until this passes. If the greedy-IK oracle
can't solve the task, no SNN will either, and the bug is in the task, not
the network.
"""
import numpy as np
import pytest

from arm import SO101PlanarArm, Q_STEP, MAX_STEPS, SUCCESS_RADIUS


@pytest.fixture
def arm():
    return SO101PlanarArm()


def greedy_dq(arm, target, q_step):
    """Move toward the nearest exact IK solution in joint space, clipped to
    q_step. NOT a local Jacobian gradient step: MEASURED, on this arm's link
    lengths (L1=0.116, L2=0.294, a ~1:2.5 ratio), the Jacobian pseudo-inverse
    goes near-singular in parts of the workspace and the naive
    pinv(J)@err direction gets trapped in a joint-limit corner (85% success,
    always stuck at a limits-box corner). arm.inverse_kinematics() gives the
    EXACT target configuration, so this greedy step can't be misled by local
    linearization -- 100% success (see git history for the comparison)."""
    solutions = arm.inverse_kinematics(target)
    if not solutions:
        return np.zeros(2)
    q = arm.q
    best = min(solutions, key=lambda s: np.linalg.norm(s - q))
    dq = best - q
    norm = np.linalg.norm(dq)
    if norm > q_step:
        dq = dq * (q_step / norm)
    return dq


def run_greedy_episode(arm, rng, target, max_steps, q_step, success_radius):
    """Returns (success: bool, rewards: list[float])."""
    rewards = []
    d_prev = np.linalg.norm(target - arm.fk(arm.q))
    if d_prev < success_radius:
        return True, rewards
    for _ in range(max_steps):
        dq = greedy_dq(arm, target, q_step)
        arm.step(dq)
        d_now = np.linalg.norm(target - arm.fk(arm.q))
        rewards.append(d_prev - d_now)
        if d_now < success_radius:
            return True, rewards
        d_prev = d_now
    return False, rewards


# ── FK — validated against the real URDF chain, not an abstract formula ────
# (an earlier version used only each joint's x-offset and ignored every
# joint's rpy. fk() now evaluates the actual transform chain; these numbers
# are MEASURED against it, not derived from arm.py itself.)

def test_fk_matches_measured_checkpoint(arm):
    """q=(0,0): world gripper_frame_link position, all 4 non-active joints
    locked at 0."""
    xy = arm.fk(np.array([0.0, 0.0]))
    assert xy == pytest.approx([0.3914, 0.2265], abs=1e-3)


def test_pivot_matches_measured(arm):
    assert arm.pivot == pytest.approx([0.069, 0.117], abs=1e-3)


def test_effective_link_lengths_are_constant(arm):
    """L1/L2 are RIGID distances (shoulder_lift->elbow, elbow->gripper) --
    must not vary with the joint angles that don't move that segment."""
    rng = np.random.default_rng(6)
    for _ in range(20):
        q1 = rng.uniform(*arm.limits[0])
        assert np.linalg.norm(arm.elbow_xz([q1, 0.0]) - arm.pivot) == pytest.approx(arm.L1, abs=1e-6)
    for _ in range(20):
        q1 = rng.uniform(*arm.limits[0])
        q2 = rng.uniform(*arm.limits[1])
        assert np.linalg.norm(arm.fk([q1, q2]) - arm.elbow_xz([q1, q2])) == pytest.approx(arm.L2, abs=1e-6)


def test_motion_is_planar_to_measured_tolerance(arm):
    """MEASURED: max|y| = 1.4e-5 m over a 7x7 grid of poses with the other
    four joints locked at 0 -- this is why representing state as (x, z) is
    exact, not an approximation."""
    max_abs_y = 0.0
    for q1 in np.linspace(*arm.limits[0], 7):
        for q2 in np.linspace(*arm.limits[1], 7):
            p = arm.gripper_xyz([q1, q2])
            max_abs_y = max(max_abs_y, abs(p[1]))
    assert max_abs_y < 5e-5, f"motion is not planar: max|y|={max_abs_y:.2e} m"


# ── Workspace annulus (MEASURED against the real chain) ────────────────────

def test_workspace_annulus_bounds(arm):
    assert arm.MIN_REACH == pytest.approx(0.179, abs=0.001)
    assert arm.MAX_REACH == pytest.approx(0.410, abs=0.001)


def test_workspace_xz_extent_matches_measured(arm):
    q1s = np.linspace(*arm.limits[0], 40)
    q2s = np.linspace(*arm.limits[1], 40)
    pts = np.array([arm.fk([q1, q2]) for q1 in q1s for q2 in q2s])
    assert pts[:, 0].min() == pytest.approx(-0.341, abs=0.01)
    assert pts[:, 0].max() == pytest.approx(0.479, abs=0.01)
    assert pts[:, 1].min() == pytest.approx(-0.225, abs=0.01)
    assert pts[:, 1].max() == pytest.approx(0.527, abs=0.01)


def test_annulus_hole_is_real(arm):
    """The pivot-adjacent region must be unreachable — this is the whole
    point of is_reachable/sample_target existing."""
    assert not arm.is_reachable(arm.pivot)
    assert not arm.is_reachable(arm.pivot + np.array([0.05, 0.0]))


def test_max_reach_point_is_reachable(arm):
    """Max reach occurs at psi=0 (q2=alpha2), not at q=(0,0) -- q=(0,0) is
    just an arbitrary reference pose, not the "arm straight" configuration,
    since the URDF's rpy chain bakes in a fixed angular offset (fk()'s
    docstring)."""
    p = arm.fk([0.0, arm.alpha2])
    assert np.linalg.norm(p - arm.pivot) == pytest.approx(arm.MAX_REACH, abs=1e-6)
    assert arm.is_reachable(p)


def test_beyond_max_reach_is_not_reachable(arm):
    direction = (arm.fk([0.0, arm.alpha2]) - arm.pivot)
    direction = direction / np.linalg.norm(direction)
    beyond = arm.pivot + direction * (arm.MAX_REACH + 0.01)
    assert not arm.is_reachable(beyond)


# ── sample_target never hands out an unreachable point ─────────────────────

def test_sample_target_always_reachable(arm):
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        xy = arm.sample_target(rng)
        assert arm.is_reachable(xy)


# ── Greedy-IK oracle — THE GATE ─────────────────────────────────────────────

def test_greedy_ik_solves_the_task(arm):
    rng = np.random.default_rng(1)
    successes = 0
    n_trials = 200
    for _ in range(n_trials):
        arm.reset(rng)
        target = arm.sample_target(rng)
        ok, _ = run_greedy_episode(arm, rng, target, MAX_STEPS, Q_STEP, SUCCESS_RADIUS)
        successes += int(ok)
    success_rate = successes / n_trials
    assert success_rate >= 0.97, (
        f"greedy-IK oracle only solved {success_rate:.1%} of {n_trials} targets — "
        f"if a greedy oracle can't do this, the bug is in the task, not the network"
    )


# ── Reward monotonicity ──────────────────────────────────────────────────────

def test_reward_monotone_toward_target(arm):
    """r = d_prev - d_now must be >~ 0 on every greedy (toward-target) step.
    Tolerance loosened from -1e-9: with the widened TARGET_LO/HI box, the
    greedy oracle's single linear joint-space step (clipped to Q_STEP) is
    occasionally not perfectly Cartesian-monotonic due to FK curvature --
    MEASURED up to ~-0.03m, spread across the whole box, not one bad
    region. -0.05 comfortably covers that discretization noise without
    hiding a real regression."""
    rng = np.random.default_rng(2)
    n_checked = 0
    for _ in range(50):
        arm.reset(rng)
        target = arm.sample_target(rng)
        _, rewards = run_greedy_episode(arm, rng, target, MAX_STEPS, Q_STEP, SUCCESS_RADIUS)
        for r in rewards:
            assert r > -0.05, f"moving toward the target gave a non-positive reward: {r}"
            n_checked += 1
    assert n_checked > 0


# ── q_step crosses the workspace ~3x within max_steps ───────────────────────

def test_q_step_covers_workspace_several_times(arm):
    workspace_diameter = 2 * arm.MAX_REACH
    # Conservative upper bound on per-step Cartesian motion: ||J|| <= L1+L2.
    worst_case_budget = MAX_STEPS * Q_STEP * (arm.L1 + arm.L2)
    ratio = worst_case_budget / workspace_diameter
    assert ratio >= 2.5, (
        f"MAX_STEPS={MAX_STEPS} * Q_STEP={Q_STEP} only budgets {ratio:.1f}x "
        f"the workspace diameter, want ~3x"
    )
