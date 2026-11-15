"""SO101PlanarArm — 2-DOF (shoulder_lift + elbow_flex) planar kinematic model.

fk() evaluates the REAL URDF transform chain (rpy_to_R + axis_R composition),
not an abstract L1*cos+L2*cos formula. MEASURED CORRECTION: an earlier version
of this file used only each joint's x-offset and ignored every joint's rpy
(orientation) -- that abstract formula gives [0.2475, 0.0] at q=(0,0), but the
real chain's gripper_frame_link sits at world (x,z) = (0.3914, 0.2265): off by
~140mm of base offsets (shoulder_pan mount) and ~198mm of wrist+gripper that
the abstract model omitted entirely.

The other four joints (shoulder_pan, wrist_flex, wrist_roll, gripper) are
locked at 0 -- this IS the planar arm (shoulder_lift + elbow_flex move;
shoulder_pan, wrist_*, gripper held fixed). MEASURED: even though the full
chain is a general 3D mechanism, with those four locked the remaining 2-DOF
motion stays planar to 1.4e-5 m out-of-plane over a 7x7 grid of poses -- so
representing arm state as (x, z) is exact here, not an approximation.

The chain to gripper_frame_link excludes the 'gripper' (jaw) revolute joint:
gripper_frame_joint's parent is gripper_link, a SIBLING of the jaw branch
(moving_jaw_so101_v1_link), not its descendant -- jaw angle never affects
gripper_frame_link's position.

arm.py and viz.py's renderer both call fk() -- if they ever diverge, the
video shows the target in the wrong place and looks fine doing it.
"""
from pathlib import Path

import numpy as np

import urdf_kin

_ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "so101_new_calib.urdf"

# Chain from base to gripper_frame_link, in kinematic order.
_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper_frame_joint"]

# Tuned so a full episode crosses the workspace diameter ~3x (test_arm.py
# checks this via the worst-case Jacobian-operator-norm bound). Kept here as
# the single source of truth; experiment configs mirror these defaults.
Q_STEP = 0.08
MAX_STEPS = 100
SUCCESS_RADIUS = 0.015

# Target box -- ~2x the area of the original [0.20,0.40]x[0.00,0.25] box,
# NOT guaranteed reachable (overlaps the annulus hole -- MEASURED ~81% of
# uniform samples land inside it -- so sample_target must reject-sample
# against is_reachable, not just draw uniformly). Bigger than this starts
# to matter for test_reward_monotone_toward_target: MEASURED, widening
# exposes small (<=0.01-0.03m) Cartesian non-monotonicity in the greedy-IK
# oracle's single linear joint-space step, spread across the whole
# workspace (not one bad corner) -- see test tolerance below.
TARGET_LO = np.array([0.10, -0.13])
TARGET_HI = np.array([0.48, 0.38])


def _rpy_to_R(rpy):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_R(axis, theta):
    """Rodrigues' rotation formula: rotate by theta about unit vector axis."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class SO101PlanarArm:

    def __init__(self, urdf_path=None, include_shoulder_pan=False):
        chain = urdf_kin.chain(str(urdf_path or _ASSET_PATH))
        self._joints = {j["name"]: j for j in chain}
        self.include_shoulder_pan = include_shoulder_pan
        # Order is (shoulder_lift, elbow_flex[, shoulder_pan]) -- pan
        # APPENDED last, not prepended -- so q[0]/q[1] keep meaning exactly
        # what every existing 2-DOF caller (_thetas, _fk_closed, jacobian,
        # inverse_kinematics) already assumes, and self._q/_lo/_hi stay
        # byte-identical for include_shoulder_pan=False.
        self._active_joint_names = (
            ["shoulder_lift", "elbow_flex"] + (["shoulder_pan"] if include_shoulder_pan else []))
        self._limits = [
            (float(self._joints[name]["lower"]), float(self._joints[name]["upper"]))
            for name in self._active_joint_names
        ]
        self._lo = np.array([lo for lo, hi in self._limits])
        self._hi = np.array([hi for lo, hi in self._limits])

        self._derive_planar_reduction()
        if include_shoulder_pan:
            self._derive_pan_axis()
            # y bound +-0.30: MEASURED (not guessed), same convention as
            # TARGET_LO/HI's own x/z bounds above -- rejection-sampling
            # this box against is_reachable() gives a 68.8% hit rate
            # (3000-sample check), comparable to the existing x/z-only
            # box's own documented ~81%, with accepted-sample y spanning
            # the full +-0.30 range roughly uniformly (not clustered near
            # 0). x/z bounds carried over unchanged from the validated
            # 2-DOF box -- the radial-plane geometry is identical.
            self.target_lo = np.array([TARGET_LO[0], -0.30, TARGET_LO[1]])
            self.target_hi = np.array([TARGET_HI[0], 0.30, TARGET_HI[1]])
        else:
            self.target_lo = TARGET_LO.copy()
            self.target_hi = TARGET_HI.copy()
        self._q = np.zeros(len(self._active_joint_names))

    # ── The real transform chain -- the single source of truth ─────────────

    def _link_position(self, thetas, up_to=None):
        """World (x, y, z) of the `up_to` link's origin (default: the full
        chain, i.e. gripper_frame_link). thetas: {joint_name: angle}; any
        joint not present is locked at 0."""
        R = np.eye(3)
        p = np.zeros(3)
        for name in _JOINT_ORDER:
            j = self._joints[name]
            p = p + R @ j["xyz"]
            R = R @ _rpy_to_R(j["rpy"])
            R = R @ _axis_R(j["axis"], thetas.get(name, 0.0))
            if name == up_to:
                break
        return p

    def _thetas(self, q):
        thetas = {"shoulder_lift": float(q[0]), "elbow_flex": float(q[1])}
        if self.include_shoulder_pan:
            thetas["shoulder_pan"] = float(q[2])
        return thetas

    def fk(self, q) -> np.ndarray:
        """World (x, z) of gripper_frame_link, four locked joints at 0
        (world (x, y, z) when include_shoulder_pan=True, three locked).
        MEASURED planar to 1.4e-5 m out-of-plane -- see module docstring."""
        p = self._link_position(self._thetas(q))
        return p if self.include_shoulder_pan else np.array([p[0], p[2]])

    def link_transforms(self, q) -> dict:
        """World (R, p) of every link in the chain (base_link through
        gripper_frame_link, plus the jaw branch) at joint state q -- for
        mesh rendering (arm_mesh.py), so it stays consistent with fk() and
        doesn't re-derive its own copy of the chain."""
        thetas = self._thetas(q)
        transforms = {"base_link": (np.eye(3), np.zeros(3))}
        R = np.eye(3)
        p = np.zeros(3)
        for name in _JOINT_ORDER:
            j = self._joints[name]
            p = p + R @ j["xyz"]
            R = R @ _rpy_to_R(j["rpy"])
            R = R @ _axis_R(j["axis"], thetas.get(name, 0.0))
            transforms[j["child"]] = (R.copy(), p.copy())
        # jaw branch: off gripper_link (state before gripper_frame_joint's
        # own fixed offset -- both branch from the same parent link).
        gripper_link_R, gripper_link_p = transforms[self._joints["wrist_roll"]["child"]]
        jj = self._joints["gripper"]
        Rj = gripper_link_R @ _rpy_to_R(jj["rpy"]) @ _axis_R(jj["axis"], 0.0)
        pj = gripper_link_p + gripper_link_R @ jj["xyz"]
        transforms[jj["child"]] = (Rj, pj)
        return transforms

    def gripper_xyz(self, q) -> np.ndarray:
        """Full 3D world position (diagnostics/tests -- e.g. verifying the
        planarity claim in the module docstring). fk() is the (x, z)
        projection of this."""
        thetas = {"shoulder_lift": float(q[0]), "elbow_flex": float(q[1])}
        return self._link_position(thetas)

    def elbow_xz(self, q) -> np.ndarray:
        """World (x, z) of the elbow (shoulder_lift->elbow_flex junction) --
        for the video renderer, so it draws the same chain fk() computes
        instead of a separately hand-derived link position. Unchanged by
        include_shoulder_pan (2-DOF callers stay byte-identical) -- see
        elbow_xyz for the 3-DOF/pan-aware version."""
        p = self._link_position({"shoulder_lift": float(q[0])}, up_to="elbow_flex")
        return np.array([p[0], p[2]])

    def pivot_xyz(self, q) -> np.ndarray:
        """World (x, y, z) of the shoulder_lift pivot, pan-aware -- for the
        Axes3D workspace panel (viz.py). Matches self.pivot's own (x, z) at
        pan=0 when include_shoulder_pan=False."""
        thetas = {"shoulder_pan": float(q[2])} if self.include_shoulder_pan else {}
        return self._link_position(thetas, up_to="shoulder_lift")

    def elbow_xyz(self, q) -> np.ndarray:
        """World (x, y, z) of the elbow, pan-aware -- for the new Axes3D
        workspace panel. Only meaningful when include_shoulder_pan=True."""
        thetas = {"shoulder_lift": float(q[0])}
        if self.include_shoulder_pan:
            thetas["shoulder_pan"] = float(q[2])
        return self._link_position(thetas, up_to="elbow_flex")

    # ── Fast analytic reduction, derived FROM fk() at construction ─────────

    def _derive_planar_reduction(self):
        """The 2 active joints reduce to a standard 2-link planar arm (fk()
        matches this closed form to well under a millimetre, self-checked
        below) -- centred on shoulder_lift's own world position (self.pivot,
        NOT the origin), with each joint's raw URDF angle offset/flipped by a
        fixed amount coming from the URDF's rpy chain (fk()'s docstring).
        Deriving these constants FROM fk() itself, instead of hardcoding
        them, means they can't silently drift out of sync with the real
        chain if the URDF ever changes.
        """
        pivot3 = self._link_position({}, up_to="shoulder_lift")
        self.pivot = np.array([pivot3[0], pivot3[2]])

        elbow3 = self._link_position({}, up_to="elbow_flex")
        elbow_xz = np.array([elbow3[0], elbow3[2]])
        self.L1 = float(np.linalg.norm(elbow_xz - self.pivot))
        self.alpha1 = float(np.arctan2(elbow3[2] - pivot3[2], elbow3[0] - pivot3[0]))

        grip3 = self._link_position({})
        grip_xz = np.array([grip3[0], grip3[2]])
        self.L2 = float(np.linalg.norm(grip_xz - elbow_xz))
        self.alpha2 = float(np.arctan2(grip_xz[1] - elbow_xz[1], grip_xz[0] - elbow_xz[0]) - self.alpha1)

        # Explicit [0]/[1] indexing (not tuple-unpack) -- self._limits has
        # a 3rd (pan) entry when include_shoulder_pan; this reach-envelope
        # derivation is purely about the 2-link mechanism.
        q1_lo, q1_hi = self._limits[0]
        q2_lo, q2_hi = self._limits[1]
        psi_lo, psi_hi = self.alpha2 - q2_lo, self.alpha2 - q2_hi
        lo, hi = min(psi_lo, psi_hi), max(psi_lo, psi_hi)
        r_candidates = [self._r_of_psi(psi_lo), self._r_of_psi(psi_hi)]
        if lo <= 0.0 <= hi:
            r_candidates.append(self.L1 + self.L2)
        if lo <= np.pi <= hi or lo <= -np.pi <= hi:
            r_candidates.append(abs(self.L1 - self.L2))
        self.MIN_REACH = min(r_candidates)
        self.MAX_REACH = max(r_candidates)

        # Self-check: the closed form must reproduce the real chain's fk()
        # to well within a millimetre, or this derivation is wrong for this
        # URDF and is_reachable()/sample_target() must not be trusted. Tests
        # ONLY the 2-link (shoulder_lift, elbow_flex) mechanism at pan=0 --
        # _fk_closed() never models pan, by construction (see _fk3_closed) --
        # so this always samples/pads a plain 2-vector, even when
        # include_shoulder_pan=True and self._lo/_hi are length 3.
        rng = np.random.default_rng(0)
        lo2, hi2 = self._lo[:2], self._hi[:2]
        for _ in range(200):
            q2v = rng.uniform(lo2, hi2)
            q_full = np.array([q2v[0], q2v[1], 0.0]) if self.include_shoulder_pan else q2v
            p = self.fk(q_full)
            p_xz = np.array([p[0], p[2]]) if self.include_shoulder_pan else p
            err = np.linalg.norm(self._fk_closed(q2v) - p_xz)
            assert err < 1e-5, (
                f"planar reduction diverges from the real chain by {err:.2e} m "
                f"at q={q2v} -- do not trust is_reachable()/sample_target()")

        # Self-check: the 2-DOF analytic jacobian() (hand-derived) must
        # match a central finite difference of the real chain's fk() -- or
        # env_reach.py's reward_shaping="jacobian" must not be trusted.
        # Only meaningful/run for the 2-DOF path -- the 3-DOF numeric
        # jacobian has its own, separate self-check in _derive_pan_axis().
        if not self.include_shoulder_pan:
            h = 1e-6
            for _ in range(50):
                q = rng.uniform(self._lo, self._hi)
                J_numeric = np.zeros((2, 2))
                for k in range(2):
                    dq = np.zeros(2); dq[k] = h
                    J_numeric[:, k] = (self.fk(q + dq) - self.fk(q - dq)) / (2 * h)
                J_err = np.max(np.abs(self.jacobian(q) - J_numeric))
                assert J_err < 1e-4, (
                    f"jacobian() diverges from a finite-difference check by "
                    f"{J_err:.2e} at q={q} -- do not trust reward_shaping=\"jacobian\"")

    def _r_of_psi(self, psi):
        return float(np.sqrt(self.L1 ** 2 + self.L2 ** 2 + 2 * self.L1 * self.L2 * np.cos(psi)))

    def _fk_closed(self, q):
        phi1 = self.alpha1 - q[0]
        psi = self.alpha2 - q[1]
        return self.pivot + self.L1 * np.array([np.cos(phi1), np.sin(phi1)]) \
                          + self.L2 * np.array([np.cos(phi1 + psi), np.sin(phi1 + psi)])

    # ── 3-DOF (shoulder_pan) reparameterization ─────────────────────────────

    def _derive_pan_axis(self):
        """shoulder_pan is the OUTERMOST joint (chain root) -- everything
        downstream of it (the validated 2-link planar mechanism above) is
        therefore a single rigid body that rotates as one piece about a
        FIXED axis as q[2] (pan) varies. MEASURED directly (diagnostic
        script, this session) to be vertical (world Z) to ~1e-5m precision
        across the full pan range -- confirmed here by hard assertion, not
        assumed. If this assertion ever fails for a different URDF, this
        whole closed-form path is invalid; fall back to a numeric approach
        (rejection-sampled fk() grid for reachability, gradient descent for
        IK) instead of trusting the algebra below."""
        j = self._joints["shoulder_pan"]
        self.pan_x0 = float(j["xyz"][0])
        self.pan_y0 = float(j["xyz"][1])
        axis_world = _rpy_to_R(j["rpy"]) @ np.asarray(j["axis"], dtype=float)
        # Tolerance 1e-4, not 1e-6 -- the URDF's own rpy values carry
        # ~1e-6 to 1e-8 noise (e.g. "4.18253e-17" scale terms), which
        # compounds slightly through the rotation-matrix product; MEASURED
        # actual off-axis components are ~1e-6 to 1e-12, two-plus orders
        # of magnitude under this gate.
        assert abs(axis_world[0]) < 1e-4 and abs(axis_world[1]) < 1e-4, (
            f"shoulder_pan's world-frame rotation axis is {axis_world}, not "
            f"vertical -- the radial/height closed form is invalid for this "
            f"URDF; do not trust inverse_kinematics()/sample_target().")
        self.pan_sign = 1.0 if axis_world[2] > 0 else -1.0

        rng = np.random.default_rng(1)

        # Self-check: closed form vs. real chain over the FULL 3-DOF range
        # (the 2-DOF check above only ever tests pan=0).
        for _ in range(300):
            q = rng.uniform(self._lo, self._hi)
            err = np.linalg.norm(self._fk3_closed(q) - self.fk(q))
            assert err < 1e-4, (
                f"3-DOF closed form diverges from the real chain by "
                f"{err:.2e} m at q={q} -- do not trust "
                f"is_reachable()/sample_target()")

        # Self-check: round-trip IK -- confirms the rho>=0 azimuth
        # construction in inverse_kinematics() doesn't silently drop
        # reachable configurations (see module docstring/plan notes).
        for _ in range(200):
            q = rng.uniform(self._lo, self._hi)
            p = self.fk(q)
            solutions = self.inverse_kinematics(p)
            assert len(solutions) > 0, (
                f"round-trip IK found no solution for reachable q={q}, p={p}")
            best = min(solutions, key=lambda s: np.linalg.norm(self.fk(s) - p))
            err = np.linalg.norm(self.fk(best) - p)
            assert err < 1e-4, f"round-trip IK error {err:.2e} m at q={q}"

        # Self-check: numeric jacobian() step-size consistency -- there is
        # no independent second derivation to cross-validate against for
        # the 3-DOF case (see module docstring), so this only catches
        # gross floating-point/step-size mistakes, not a derivation error.
        for _ in range(30):
            q = rng.uniform(self._lo, self._hi)
            J1 = self._jacobian_numeric(q, h=1e-6)
            J2 = self._jacobian_numeric(q, h=1e-5)
            j_err = np.max(np.abs(J1 - J2))
            assert j_err < 1e-3, (
                f"numeric jacobian unstable across step sizes at q={q}: "
                f"{j_err:.2e}")

    def _fk3_closed(self, q):
        """3D closed form: the validated 2-link (x,z) solution, rotated
        rigidly about the (vertical) pan axis by q[2]."""
        x2d, z = self._fk_closed(q[:2])
        rho = x2d - self.pan_x0
        theta = self.pan_sign * q[2]
        x = self.pan_x0 + rho * np.cos(theta)
        y = self.pan_y0 + rho * np.sin(theta)
        return np.array([x, y, z])

    def _jacobian_numeric(self, q, h=1e-6):
        n = len(q)
        J = np.zeros((3, n))
        for k in range(n):
            dq = np.zeros(n); dq[k] = h
            J[:, k] = (self.fk(q + dq) - self.fk(q - dq)) / (2 * h)
        return J

    def jacobian(self, q) -> np.ndarray:
        """Analytic 2x2 Jacobian d(fk)/dq of the closed-form planar
        reduction (row 0 = d(x)/dq, row 1 = d(z)/dq) -- exact derivative of
        _fk_closed, which self-checks to within 1e-5m of the real chain's
        fk() (see _derive_planar_reduction). Added for reward shaping
        (env_reach.py's reward_shaping="jacobian") -- the raw
        finite-difference reward (d_prev-d_now) has only
        a small fraction of its variance explained by either channel's own current
        action, because the SAME dq produces very different distance
        changes depending on the current arm configuration (this Jacobian
        varies sharply across the workspace). A reward built from THIS
        instead of a finite difference is, by construction, an exact
        linear function of dq at the pre-step configuration.

        include_shoulder_pan=True: no analytic 3-link closed form is
        derived here -- returns the numeric (central finite-difference)
        3xN Jacobian instead (_jacobian_numeric), self-checked for
        step-size consistency in _derive_pan_axis(). The 2-DOF path below
        is completely untouched."""
        if self.include_shoulder_pan:
            return self._jacobian_numeric(q)
        phi1 = self.alpha1 - q[0]
        psi = self.alpha2 - q[1]
        s1, c1 = np.sin(phi1), np.cos(phi1)
        s12, c12 = np.sin(phi1 + psi), np.cos(phi1 + psi)
        # d(phi1)/dq0=-1, d(phi1+psi)/dq0=-1, d(phi1+psi)/dq1=-1, d(phi1)/dq1=0
        return np.array([
            [self.L1 * s1 + self.L2 * s12,  self.L2 * s12],
            [-self.L1 * c1 - self.L2 * c12, -self.L2 * c12],
        ])

    def step(self, dq) -> np.ndarray:
        self._q = np.clip(self._q + np.asarray(dq, dtype=float), self._lo, self._hi)
        return self._q.copy()

    def inverse_kinematics(self, target) -> list:
        """Dispatches to the 2-link closed form directly (include_
        shoulder_pan=False, byte-identical to before) or, when 3-DOF,
        peels off the pan/azimuth component first and delegates the
        remaining radial-height solve to the SAME _inverse_kinematics_
        2link -- one implementation, so the two modes can't drift apart.

        (x,y) relative to the pan axis has a genuinely AMBIGUOUS polar
        decomposition once signed radius is allowed: _fk_closed's own x2d
        can land on either side of pan_x0 (elbow folded back past the
        pivot's own radial line), so the same (x,y) is reachable either as
        (rho=+r, pan_angle=atan2(dy,dx)) or (rho=-r, pan_angle=atan2(dy,dx)
        +pi) -- MEASURED (round-trip IK self-check) that always assuming
        rho>=0 silently drops real, in-limit solutions. Try both branches,
        keep whichever land within pan's joint limits."""
        if not self.include_shoulder_pan:
            return self._inverse_kinematics_2link(target)
        x, y, z = float(target[0]), float(target[1]), float(target[2])
        dx, dy = x - self.pan_x0, y - self.pan_y0
        r = float(np.hypot(dx, dy))
        base_theta = float(np.arctan2(dy, dx))
        q_pan_lo, q_pan_hi = self._limits[2]
        eps = 1e-4
        solutions = []
        for rho_signed, theta in ((r, base_theta), (-r, base_theta + np.pi)):
            q_pan = self.pan_sign * theta
            q_pan = (q_pan + np.pi) % (2 * np.pi) - np.pi
            if not (q_pan_lo - eps <= q_pan <= q_pan_hi + eps):
                continue
            x2d_equiv = self.pan_x0 + rho_signed
            for (q1, q2) in self._inverse_kinematics_2link([x2d_equiv, z]):
                solutions.append(np.array([q1, q2, q_pan]))
        return solutions

    def _inverse_kinematics_2link(self, xy) -> list:
        """Analytic 2-link IK, in the pivot-centred, angle-offset frame
        derived in _derive_planar_reduction(): every elbow-up/elbow-down
        solution (q1, q2) that lands within both joints' limits (0, 1, or 2
        of them). Single source of truth for is_reachable() below AND for
        any greedy/oracle controller that wants to move toward an exact
        target joint configuration instead of a local Jacobian gradient
        (which can get trapped near-singularities on this arm's link-length
        ratio -- see tests/test_arm.py's greedy oracle). Also the radial-
        height solver the 3-DOF inverse_kinematics() dispatches into."""
        x, y = float(xy[0]), float(xy[1])
        dx, dz = x - self.pivot[0], y - self.pivot[1]
        # Margin set by the closed-form-vs-real-chain numerical gap (~1.4e-7 m,
        # see _derive_planar_reduction's self-check), not machine epsilon --
        # points exactly at MAX_REACH/MIN_REACH (e.g. fk() at psi=0) must not
        # be rejected by float noise between the two representations.
        eps = 1e-4
        r2 = dx * dx + dz * dz
        r = np.sqrt(r2)
        if r > self.MAX_REACH + eps or r < self.MIN_REACH - eps:
            return []
        cos_psi = np.clip((r2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2), -1.0, 1.0)
        # Always index [0]/[1] explicitly (not tuple-unpack self._limits)
        # -- self._limits has a 3rd (pan) entry when include_shoulder_pan,
        # which this 2-link-only solver never touches.
        q1_lo, q1_hi = self._limits[0]
        q2_lo, q2_hi = self._limits[1]
        target_angle = np.arctan2(dz, dx)
        solutions = []
        for sign in (1.0, -1.0):
            psi = sign * np.arccos(cos_psi)
            q2 = self.alpha2 - psi
            if not (q2_lo - eps <= q2 <= q2_hi + eps):
                continue
            phi1 = target_angle - np.arctan2(self.L2 * np.sin(psi), self.L1 + self.L2 * np.cos(psi))
            q1 = self.alpha1 - phi1
            q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
            if q1_lo - eps <= q1 <= q1_hi + eps:
                solutions.append(np.array([q1, q2]))
        return solutions

    def is_reachable(self, xy) -> bool:
        return len(self.inverse_kinematics(xy)) > 0

    def sample_target(self, rng, max_tries=10_000) -> np.ndarray:
        for _ in range(max_tries):
            xy = rng.uniform(self.target_lo, self.target_hi)
            if self.is_reachable(xy):
                return xy
        raise RuntimeError("sample_target: no reachable point found in target box")

    def reset(self, rng) -> np.ndarray:
        self._q = rng.uniform(self._lo, self._hi)
        return self._q.copy()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def q(self):
        return self._q.copy()

    @property
    def limits(self):
        return self._limits

    @property
    def n_joints(self):
        return len(self._active_joint_names)
