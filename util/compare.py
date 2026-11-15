#!/usr/bin/env python3
"""Mann-Whitney U comparison of two CONDITIONS, each a set of per-seed values.

The test engine behind `reproduce_stats.py`. Moved/rebuilt here from
`experiments/mannwhitney_compare.py` (the original prototype, kept for ad hoc
one-off CSV comparisons -- this module is the one `reproduce_stats.py` and
any other script in the pipeline import).

What it compares
-----------------
NOT two runs. It compares two conditions, each contributing ONE value per
seed (e.g. the greedy-eval score/distance for that seed, from
`per_seed_value.py`). It asks whether one condition's seeds systematically
outrank the other's. The seeds are the samples within each condition.

Method selection (the one change from the prototype that actually matters
for correctness, not just code organization): the prototype picked "exact"
via try/except around a ValueError, and scipy's own `method="auto"` only
chooses exact when a group has <=8 samples (SciPy's own conservative
default) -- both give WRONG/inconsistent results for the project's typical
9-vs-9 comparisons, which fall just outside that cutoff. This module instead
applies an EXPLICIT rule: exact whenever both groups are small enough for
exact computation to be fast (measured: still <2ms at n=50 per group, so
capped generously below that, at group size <=25, far above anything this
project's registry will ever need at n=9/group) AND the pooled combined
sample has no tied values; otherwise the continuity-corrected normal
approximation. The method actually used is always returned explicitly --
never left implicit -- because the two methods really do disagree
numerically: a fully-separated 9-vs-9 is p=4.11e-5 exact vs p=4.12e-4
asymptotic, an order of magnitude apart (see `test_compare.py`).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import mean, median

from scipy.stats import mannwhitneyu, wilcoxon

# Exact is measured fast (<2ms) up to at least n=50/group on this machine;
# capped well below that since nothing in this project's registry needs
# more than n=9/group. Not scipy's own method="auto" threshold (<=8), which
# would silently push this project's n=9 comparisons to asymptotic.
EXACT_MAX_GROUP_SIZE = 25


@dataclass
class CompareResult:
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    mean_a: float
    mean_b: float
    U: float
    p: float
    method: str          # "exact" or "asymptotic (continuity-corrected)"
    alternative: str
    pairwise_win_fraction: float   # fraction of (x in a, y in b) with x > y


def _has_ties(a, b) -> bool:
    pooled = list(a) + list(b)
    return len(set(pooled)) < len(pooled)


def _select_method(a, b) -> str:
    small_enough = max(len(a), len(b)) <= EXACT_MAX_GROUP_SIZE
    untied = not _has_ties(a, b)
    return "exact" if (small_enough and untied) else "asymptotic"


def pairwise_win_fraction(a, b) -> float:
    """Fraction of (x in a, y in b) pairs with x > y (ties count as 0.5)."""
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return wins / (len(a) * len(b))


def compare(a, b, alternative: str = "two-sided") -> CompareResult:
    """The one entry point reproduce_stats.py (and anything else) calls.

    a, b: sequences of per-seed values (floats), already reduced to one
    number per seed (see per_seed_value.py) -- this function does no
    reduction of its own.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b:
        raise ValueError(f"empty group: n_a={len(a)} n_b={len(b)}")

    method = _select_method(a, b)
    if method == "exact":
        U, p = mannwhitneyu(a, b, alternative=alternative, method="exact")
        method_label = "exact"
    else:
        U, p = mannwhitneyu(a, b, alternative=alternative, method="asymptotic")
        method_label = "asymptotic (continuity-corrected)"

    return CompareResult(
        n_a=len(a), n_b=len(b),
        median_a=median(a), median_b=median(b),
        mean_a=mean(a), mean_b=mean(b),
        U=float(U), p=float(p),
        method=method_label,
        alternative=alternative,
        pairwise_win_fraction=pairwise_win_fraction(a, b),
    )


@dataclass
class PairedCompareResult:
    n: int                  # pairs (seeds) compared
    n_zero_diffs: int       # pairs with a==b, dropped from the test itself (scipy's zero_method="wilcox")
    median_diff: float      # median(a - b), over ALL n pairs including zero diffs
    mean_diff: float        # mean(a - b), over ALL n pairs including zero diffs
    n_a_beats_b: int        # a > b
    n_b_beats_a: int        # b > a
    W: float
    p: float
    method: str             # "exact" or "approx (normal approximation)"
    alternative: str


def _select_paired_method(n_nonzero: int) -> str:
    return "exact" if n_nonzero <= EXACT_MAX_GROUP_SIZE else "approx"


def compare_paired(a, b, alternative: str = "two-sided") -> PairedCompareResult:
    """Wilcoxon signed-rank test: a and b are the SAME seeds' values under
    two conditions (e.g. a trained network evaluated normally vs. under a
    swap control) -- paired by position, a[i] and b[i] must be the same
    seed. Use this instead of compare() whenever the same seed means the
    same starting network/weights across both conditions; use compare()
    (unpaired Mann-Whitney) when the two groups are independent samples.

    Ties (a[i] == b[i], a "zero difference") are dropped from the test
    itself (scipy's default zero_method="wilcox") but never silently --
    n_zero_diffs reports how many, and n (and median_diff/mean_diff) still
    reflect the full pair count given in.

    Diffs are rounded to 9 decimal places before any tie/zero/rank logic --
    two conceptually-equal diffs computed from
    float fractions, e.g. a rate metric stored as count/20, can differ by
    ~5e-17 in raw IEEE754 subtraction -- e.g. 0.25-0.2 != 0.05 exactly --
    which Wilcoxon's rank test treats as genuinely different magnitudes
    instead of a tie, silently changing W and p. 9 decimals is far below
    any real effect size this project reports and far above float64
    subtraction noise, so it only removes representation artifacts, never
    real signal.
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal-length sequences: n_a={len(a)} n_b={len(b)}")
    if not a:
        raise ValueError("empty group")

    diffs = [round(x - y, 9) for x, y in zip(a, b)]
    n_zero = sum(1 for d in diffs if d == 0)
    n_nonzero = len(diffs) - n_zero
    if n_nonzero == 0:
        raise ValueError("every pair is tied (a[i] == b[i] for all i) -- Wilcoxon is undefined")

    method = _select_paired_method(n_nonzero)
    scipy_method = "exact" if method == "exact" else "approx"
    W, p = wilcoxon(diffs, alternative=alternative, method=scipy_method)

    return PairedCompareResult(
        n=len(a),
        n_zero_diffs=n_zero,
        median_diff=median(diffs),
        mean_diff=mean(diffs),
        n_a_beats_b=sum(1 for d in diffs if d > 0),
        n_b_beats_a=sum(1 for d in diffs if d < 0),
        W=float(W), p=float(p),
        method="exact" if method == "exact" else "approx (normal approximation)",
        alternative=alternative,
    )


# ---------------------------------------------------------------------------
# Ad hoc CLI (unchanged use case from the original prototype): compare two
# conditions straight from a tidy CSV, without going through the
# comparisons.yaml registry / per_seed_value reducer. Useful for one-off
# checks; reproduce_stats.py does NOT use this CLI path, it imports
# `compare()` directly.
# ---------------------------------------------------------------------------

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract(rows, value_col, group_col=None, group=None, seed_col=None, reduce="mean"):
    """Return the list of per-seed values for one condition."""
    if group_col is not None:
        rows = [r for r in rows if r[group_col] == group]
    if not rows:
        raise SystemExit(f"No rows for group={group!r}. Check --group-col/--a/--b.")
    if seed_col:  # several rows per seed -> collapse to one value per seed
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[r[seed_col]].append(float(r[value_col]))
        agg = {"mean": mean, "median": median}[reduce]
        return [agg(v) for v in by_seed.values()]
    return [float(r[value_col]) for r in rows]


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="tidy CSV (with --group-col)")
    ap.add_argument("--group-col")
    ap.add_argument("--a", help="name of condition A in the group column")
    ap.add_argument("--b", help="name of condition B in the group column")
    ap.add_argument("--a-csv", help="alternative: CSV holding only condition A")
    ap.add_argument("--b-csv", help="alternative: CSV holding only condition B")
    ap.add_argument("--value-col", required=True, help="column with the metric")
    ap.add_argument("--seed-col", help="collapse multiple rows per seed first")
    ap.add_argument("--reduce", choices=["mean", "median"], default="mean")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"],
                    default="two-sided")
    args = ap.parse_args()

    if args.a_csv and args.b_csv:
        a = extract(read_rows(args.a_csv), args.value_col,
                    seed_col=args.seed_col, reduce=args.reduce)
        b = extract(read_rows(args.b_csv), args.value_col,
                    seed_col=args.seed_col, reduce=args.reduce)
        name_a, name_b = args.a_csv, args.b_csv
    else:
        if not (args.csv and args.group_col and args.a and args.b):
            raise SystemExit("Give either a tidy CSV with --group-col/--a/--b, "
                             "or --a-csv and --b-csv.")
        rows = read_rows(args.csv)
        a = extract(rows, args.value_col, args.group_col, args.a,
                    args.seed_col, args.reduce)
        b = extract(rows, args.value_col, args.group_col, args.b,
                    args.seed_col, args.reduce)
        name_a, name_b = args.a, args.b

    r = compare(a, b, alternative=args.alternative)
    print(f"Condition A: {name_a!r}  n={r.n_a}  median={r.median_a:.3f}  mean={r.mean_a:.3f}")
    print(f"Condition B: {name_b!r}  n={r.n_b}  median={r.median_b:.3f}  mean={r.mean_b:.3f}")
    print(f"Mann-Whitney U = {r.U:.1f}   p = {r.p:.4g}   ({r.alternative}, {r.method})")
    print(f"Pairwise: A beats B in {r.pairwise_win_fraction*100:.1f}% "
          f"of {r.n_a*r.n_b} seed-pairs  (50% = no difference)")
    print("Significant at 0.05." if r.p < 0.05 else "Not significant at 0.05.")


if __name__ == "__main__":
    _cli()
