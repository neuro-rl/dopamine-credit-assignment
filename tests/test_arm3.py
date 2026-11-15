"""The 3-DOF (shoulder_pan-included) gate, mirroring test_arm.py's own
"do not wire the SNN until this passes" role for the 2-DOF arm. Kept in a
SEPARATE file (not merged into test_arm.py) so the existing 2-DOF gate
stays independently green regardless of anything here.
"""
import numpy as np
import pytest

from arm import SO101PlanarArm, Q_STEP, MAX_STEPS, SUCCESS_RADIUS


@pytest.fixture
def arm():
    return SO101PlanarArm(include_shoulder_pan=True)


@pytest.fixture
def arm2():
    """A plain 2-DOF arm, to confirm include_shoulder_pan=True never
    regresses the original, validated behavior."""
    return SO101PlanarArm()


def greedy_dq(arm, target, q_step):
    """Same greedy-IK-oracle strategy as test_arm.py's own greedy_dq,
    generalized to N joints (arm.q's own length) instead of hardcoded 2."""
    solutions = arm.inverse_kinematics(target)
    if not solutions:
        return np.zeros(arm.n_joints)
    q = arm.q
    best = min(solutions, key=lambda s: np.linalg.norm(s - q))
    dq = best - q
    norm = np.linalg.norm(dq)
    if norm > q_step:
        dq = dq * (q_step / norm)
    return dq


def run_greedy_episode(arm, target, max_steps, q_step, success_radius):
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


# ── Backward compatibility: include_shoulder_pan=False stays byte-identical ─

def test_2dof_unaffected(arm2):
    assert arm2.n_joints == 2
    assert arm2.fk(np.zeros(2)) == pytest.approx([0.3914, 0.2265], abs=1e-3)
    assert arm2.target_lo.shape == (2,)


# ── Pan axis: the verticality assumption the whole closed form relies on ───

def test_pan_axis_is_vertical(arm):
    """Already hard-asserted inside __init__ (arm fixture construction
    itself would have raised if this failed) -- this test just makes the
    property independently visible/documented in the test suite."""
    assert hasattr(arm, "pan_sign")
    assert arm.pan_sign in (1.0, -1.0)


def test_3dof_construction_runs_all_self_checks(arm):
    """The arm fixture's mere construction already ran 300 closed-form-vs-
    chain samples, 200 round-trip-IK samples, and 30 jacobian step-size
    checks (_derive_pan_axis) -- if any failed, this test wouldn't even
    reach its body. This test exists to make that gate explicit/named."""
    assert arm.n_joints == 3


# ── FK / closed form ─────────────────────────────────────────────────────────

def test_fk_matches_2dof_at_pan_zero(arm, arm2):
    """At shoulder_pan=0, the 3-DOF fk()'s (x, z) must match the validated
    2-DOF fk() exactly (same underlying 2-link mechanism)."""
    q1, q2 = 0.3, -0.5
    p3 = arm.fk(np.array([q1, q2, 0.0]))
    p2 = arm2.fk(np.array([q1, q2]))
    assert p3[0] == pytest.approx(p2[0], abs=1e-9)
    assert p3[2] == pytest.approx(p2[1], abs=1e-9)
    assert p3[1] == pytest.approx(0.0, abs=2e-5)


def test_pan_sweep_keeps_z_constant(arm):
    """Sweeping pan alone must not change world z at all (to within
    FK-chain floating-point noise)."""
    q1, q2 = 0.3, -0.5
    zs = [arm.fk(np.array([q1, q2, pan]))[2] for pan in np.linspace(-1.5, 1.5, 11)]
    assert max(zs) - min(zs) < 5e-5


def test_closed_form_matches_real_chain(arm):
    rng = np.random.default_rng(7)
    for _ in range(300):
        q = rng.uniform(arm._lo, arm._hi)
        err = np.linalg.norm(arm._fk3_closed(q) - arm.fk(q))
        assert err < 1e-4


# ── Inverse kinematics round-trip (the gap the rho>=0-only construction had)─

def test_round_trip_ik(arm):
    rng = np.random.default_rng(8)
    for _ in range(300):
        q = rng.uniform(arm._lo, arm._hi)
        p = arm.fk(q)
        solutions = arm.inverse_kinematics(p)
        assert len(solutions) > 0, f"no IK solution for reachable q={q}"
        best = min(solutions, key=lambda s: np.linalg.norm(arm.fk(s) - p))
        assert np.linalg.norm(arm.fk(best) - p) < 1e-4


def test_ik_finds_negative_rho_branch(arm):
    """Regression test for the specific bug found during implementation:
    assuming rho=hypot(dx,dy) is always the right branch silently dropped
    solutions where the 2-link mechanism folds back past the pan axis
    (_fk_closed's x2d < pan_x0). Constructs a target known to be in that
    regime (elbow_flex near its most-folded limit) and confirms IK still
    finds it."""
    q = np.array([-0.4555, -1.2501, -0.1659])  # the exact case that failed
    p = arm.fk(q)
    solutions = arm.inverse_kinematics(p)
    assert len(solutions) > 0
    best = min(solutions, key=lambda s: np.linalg.norm(arm.fk(s) - p))
    assert np.linalg.norm(arm.fk(best) - p) < 1e-4


# ── Jacobian ──────────────────────────────────────────────────────────────

def test_jacobian_shape_and_consistency(arm):
    rng = np.random.default_rng(9)
    for _ in range(30):
        q = rng.uniform(arm._lo, arm._hi)
        J = arm.jacobian(q)
        assert J.shape == (3, 3)
        J_alt = arm._jacobian_numeric(q, h=1e-5)
        assert np.max(np.abs(J - J_alt)) < 1e-3


# ── Target sampling ──────────────────────────────────────────────────────

def test_sample_target_always_reachable(arm):
    rng = np.random.default_rng(0)
    for _ in range(2000):
        xyz = arm.sample_target(rng)
        assert xyz.shape == (3,)
        assert arm.is_reachable(xyz)


# ── Greedy-IK oracle -- THE GATE ─────────────────────────────────────────

def test_greedy_ik_solves_the_3dof_task(arm):
    rng = np.random.default_rng(1)
    successes = 0
    n_trials = 200
    for _ in range(n_trials):
        arm.reset(rng)
        target = arm.sample_target(rng)
        ok, _ = run_greedy_episode(arm, target, MAX_STEPS, Q_STEP, SUCCESS_RADIUS)
        successes += int(ok)
    success_rate = successes / n_trials
    assert success_rate >= 0.90, (
        f"greedy-IK oracle only solved {success_rate:.1%} of {n_trials} 3-DOF "
        f"targets -- if a greedy oracle struggles here, the bug is in the "
        f"3-DOF task/reachability math, not the network"
    )


def test_reward_monotone_toward_target(arm):
    rng = np.random.default_rng(2)
    n_checked = 0
    for _ in range(50):
        arm.reset(rng)
        target = arm.sample_target(rng)
        _, rewards = run_greedy_episode(arm, target, MAX_STEPS, Q_STEP, SUCCESS_RADIUS)
        for r in rewards:
            assert r > -0.05, f"moving toward the target gave a non-positive reward: {r}"
            n_checked += 1
    assert n_checked > 0
