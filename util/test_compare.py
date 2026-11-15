"""Unit-check for compare.py (spec Order step 1): the engine must reproduce
the known exact-vs-asymptotic divergence on a fully-separated 9-vs-9 sample
-- p~=4e-5 exact, p~=4e-4 asymptotic -- and pick the right method by the
documented rule (small + untied -> exact; otherwise asymptotic), not by
scipy's own conservative method="auto" (<=8/group) or a bare try/except.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # util/, holds compare.py

from compare import compare, compare_paired, _select_method, _has_ties  # noqa: E402


def test_fully_separated_9v9_exact_and_asymptotic_diverge():
    a = list(range(10, 19))   # 9 values, all above b
    b = list(range(0, 9))     # 9 values, all below a

    r = compare(a, b)
    print(f"\nexact:      p={r.p:.6g}  method={r.method}  U={r.U}")

    assert r.n_a == 9 and r.n_b == 9
    assert r.method == "exact"
    assert abs(r.p - 4.113533525298231e-05) < 1e-12

    # Force the asymptotic path the same way reproduce_stats.py would see it
    # if ties were present, to confirm the two methods really do disagree
    # by an order of magnitude on this exact case (not just a rounding blip).
    from scipy.stats import mannwhitneyu
    _, p_asym = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    print(f"asymptotic: p={float(p_asym):.6g}  (same data, forced asymptotic)")

    assert abs(float(p_asym) - 0.00041229480206169127) < 1e-12
    assert p_asym > r.p * 5   # genuinely an order of magnitude apart


def test_method_selection_untied_small_is_exact():
    assert _select_method(list(range(9)), list(range(9, 18))) == "exact"


def test_method_selection_ties_forces_asymptotic():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 5.0]   # shares value 2.0 with a -> tie in the pooled sample
    assert _has_ties(a, b) is True
    assert _select_method(a, b) == "asymptotic"


def test_method_selection_large_group_forces_asymptotic():
    a = list(range(30))
    b = list(range(30, 60))
    assert _select_method(a, b) == "asymptotic"


def test_pairwise_win_fraction_matches_known_case():
    # a entirely above b -> every pair is a win for a
    r = compare(list(range(10, 19)), list(range(0, 9)))
    assert r.pairwise_win_fraction == 1.0


def test_identical_groups_not_significant():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = compare(a, b)
    assert r.p == 1.0
    assert r.pairwise_win_fraction == 0.5


def test_empty_group_raises():
    import pytest
    with pytest.raises(ValueError):
        compare([], [1.0, 2.0])


# ---------------------------------------------------------------------------
# compare_paired() -- Wilcoxon signed-rank. Checked against already-
# published numbers (9 seeds 0-8), computed independently before this
# engine existed -- reproducing them exactly here is the real test, not
# a synthetic case.
# ---------------------------------------------------------------------------

_ARM_T = [169.1, 152.8, 237.5, 214.8, 211.0, 216.2, 149.7, 373.1, 168.8]
_ARM_R = [110.1, 180.6, 204.7, 173.4, 122.8, 154.1, 186.6, 58.7, 194.4]
_ARM_R_SWAP = [219.2, 208.1, 243.1, 212.9, 215.5, 175.1, 241.3, 276.7, 232.6]


def test_paired_reproduces_reachvec_vs_target_p0037():
    # one-sided: does Arm R's distance tend to be LESS than Arm T's?
    r = compare_paired(_ARM_R, _ARM_T, alternative="less")
    assert r.method == "exact"
    assert r.n_zero_diffs == 0
    assert abs(r.p - 0.037109375) < 1e-9
    assert r.n_b_beats_a == 6   # Arm T > Arm R (R wins) on 6/9 seeds


def test_paired_reproduces_swap_control_p0020():
    r = compare_paired(_ARM_R, _ARM_R_SWAP, alternative="less")
    assert r.method == "exact"
    assert abs(r.p - 0.001953125) < 1e-9
    assert r.n_b_beats_a == 9   # swap worse in all 9/9 seeds, no ties


def test_paired_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_all_tied_raises():
    import pytest
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_paired_reports_zero_diffs_without_dropping_n():
    a = [1.0, 2.0, 3.0, 5.0]
    b = [1.0, 4.0, 1.0, 2.0]   # first pair tied (diff=0)
    r = compare_paired(a, b)
    assert r.n == 4            # full pair count, zero diff included
    assert r.n_zero_diffs == 1


def test_paired_float_fraction_ties_match_equivalent_integer_scale():
    # Regression: success@15mm from per_seed_value() comes out
    # as count/20 fractions (0.25, 0.2, ...). 0.25-0.2 != 0.05 exactly in
    # IEEE754, so two conceptually-tied +5pp diffs landed at slightly
    # different floats (0.05 vs 0.049999999999999996) and Wilcoxon ranked
    # them as different magnitudes -- W=14/p=0.461->0.641 on the SAME data,
    # just scaled by 1/100. The percent-integer and fraction forms of the
    # exact same real run data (Arm R vs Arm T, eval_firming9) must now
    # agree exactly.
    armT_pct = [5, 20, 5, 20, 5, 5, 10, 5, 10]
    armR_pct = [0, 25, 0, 10, 25, 10, 15, 35, 10]
    armT_frac = [x / 100 for x in armT_pct]
    armR_frac = [x / 100 for x in armR_pct]

    r_int = compare_paired(armR_pct, armT_pct, alternative="two-sided")
    r_frac = compare_paired(armR_frac, armT_frac, alternative="two-sided")

    assert r_int.W == r_frac.W == 12.0
    assert abs(r_int.p - r_frac.p) < 1e-12
    assert abs(r_int.p - 0.4609375) < 1e-9
