#!/usr/bin/env python3
"""Compare two conditions straight from a runs folder + name prefixes --
no hand-built CSV, no registry file. Finds every seed matching each prefix,
reduces each to its per-seed number via the locked greedy-eval convention
(per_seed_value.py), and runs the comparison (compare.py) -- Wilcoxon
signed-rank, paired by matching seed number, by default. Every comparison
in this paper is paired (see Section 4's "paired versus unpaired testing"
rule: same seed means the same starting network in both conditions, which
is always true here by construction). Pass --unpaired only for the
hypothetical case of two genuinely independent groups with no shared seed
-- not used by anything in the paper, kept for completeness.

This is the "give me a run folder and two prefixes" tool -- the quick,
interactive way to check a comparison. The registry-driven pipeline
(comparisons.yaml + reproduce_stats.py, not built yet) is for the frozen,
audited set of numbers that actually go in the paper; this is for poking
at data while it's still being generated.

Matching convention
--------------------
Given --runs-dir DIR and a prefix P, this looks for:
  - directories  DIR/P_seed<N>        (arm-style eval runs, contain stats.csv)
  - files        DIR/P_seed<N>_result.txt   (CartPole-style)
and sorts whatever it finds by the seed number <N>. Both forms can't match
the same prefix in the same DIR (if both exist, that's almost certainly a
naming collision worth looking at by hand, so it's treated as an error).

Goal-set metrics
----------------
"selectivity", "mean_d_commanded" and "mean_d_centroid" aren't in
per_seed_value()'s ARM_METRICS dispatch -- they need an external goal set
(a discrete goal-selection run isn't self-describing about which goals it
was choosing between). Pass --goals-mm "x1,y1;x2,y2;..." (mm) to use one of
these; per_seed_value.py's selectivity_per_seed_value() computes it.

Training-tail metrics
----------------------
"d1_weight_std" reduces a TRAINING run (not an eval run) instead -- the
mean of its d1_ch0_w_std column over the last --n-tail episodes (default
20), e.g. the settled D1 weight spread once learning has plateaued.
Training run directories don't follow the eval convention's clean
<prefix>_seed<N> naming (they're TIMESTAMP_<config>_seed<S>_seed<S+1>,
logical seed then NEST-offset seed) -- --prefix-a/-b are matched as a
plain substring against the whole directory name instead, same as
make_figures.py's own _find_9seed_dirs, and exactly 9 matches are
required. Pass --exclude "sub1,sub2" to rule out directories whose name
would otherwise also match the substring (e.g. prefix "p3_reach_
baseline_7x7" also matches "..._nocenter_7x7_...", "..._reachvec_7x7_...").

Example
-------
    python compare_runs.py --runs-dir ../runs/eval_firming9_armT \\
        --prefix-a armT --metric mean_distance \\
        --runs-dir-b ../runs/eval_firming9_armR --prefix-b armR

    python compare_runs.py --runs-dir ../runs --prefix-a multigoal3_eval9 \\
        --prefix-b multigoal3cg_eval9 --metric mean_d_commanded \\
        --goals-mm "260,50;360,200;300,300"

    python compare_runs.py --runs-dir ../runs --prefix-a p3_reach_baseline_7x7 \\
        --prefix-b p3_reach_baseline_nocenter_7x7 --metric d1_weight_std \\
        --exclude "reachvec,multigoal,nocenter"
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "util"))
from compare import compare, compare_paired  # noqa: E402
from per_seed_value import (per_seed_value, selectivity_per_seed_value, training_tail_value,
                             ARM_METRICS, CARTPOLE_METRICS)

GOAL_SET_METRICS = ("selectivity", "selectivity_decoy", "selectivity_other",
                     "mean_d_commanded", "mean_d_centroid")
TRAINING_TAIL_METRICS = ("d1_weight_std",)
_TRAINING_TAIL_COLUMN = {"d1_weight_std": "d1_ch0_w_std"}

_SEED_RE = re.compile(r"_seed(\d+)(?:_result\.txt)?$")
_TRAINING_DIR_SEED_RE = re.compile(r"_seed(\d+)_seed\d+$")


def parse_goals_mm(s: str):
    """'x1,y1;x2,y2;...' -> [(x1,y1), (x2,y2), ...], floats, mm."""
    return [tuple(float(v) for v in pair.split(",")) for pair in s.split(";")]


def _goal_set_value(path: Path, metric: str, goals_mm, n_warmup: int, force: bool) -> float:
    selectivity_per_seed_value(path, goals_mm, n_warmup=n_warmup, force=force)  # populates the cache
    with open(path / "greedy_eval_summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] == metric:
                return float(row["value"])
    raise RuntimeError(f"{path}: {metric!r} missing from greedy_eval_summary.csv after computing it")


def find_training_seed_dirs(runs_dir: Path, pattern_substr: str, exclude_substrs=()):
    """Return [(seed:int, path:Path), ...] sorted by seed, for training
    run directories under runs_dir whose name CONTAINS pattern_substr
    (not an exact prefix match -- training dirs are TIMESTAMP_<config>_
    seed<S>_seed<S+1>, so there's no clean prefix to match against) and
    contains none of exclude_substrs. Requires exactly 9 matches (this
    project's locked seed count), same as make_figures.py's own
    _find_9seed_dirs -- raises if more or fewer are found, rather than
    silently using a partial or ambiguous set.
    """
    cands = sorted(p for p in runs_dir.glob("*_seed*_seed*") if p.is_dir()
                    and pattern_substr in p.name
                    and not any(x in p.name for x in exclude_substrs))
    if len(cands) != 9:
        raise SystemExit(
            f"expected 9 training dirs matching {pattern_substr!r} (excluding {exclude_substrs}) "
            f"under {runs_dir}, found {len(cands)}: {[p.name for p in cands]}")
    found = []
    for p in cands:
        m = _TRAINING_DIR_SEED_RE.search(p.name)
        if not m:
            raise SystemExit(f"{p.name}: doesn't match the TIMESTAMP_..._seed<S>_seed<S+1> convention")
        found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def find_seed_runs(runs_dir: Path, prefix: str):
    """Return [(seed:int, path:Path), ...] sorted by seed, for everything
    under runs_dir matching '<prefix>_seed<N>' (dir) or
    '<prefix>_seed<N>_result.txt' (file)."""
    dirs = sorted(runs_dir.glob(f"{prefix}_seed*"))
    found = []
    for p in dirs:
        m = _SEED_RE.search(p.name)
        if not m:
            continue
        if p.is_dir() or p.name.endswith("_result.txt"):
            found.append((int(m.group(1)), p))
    if not found:
        raise SystemExit(
            f"No runs matching '{prefix}_seed*' under {runs_dir} "
            f"(looked for directories or *_result.txt files).")
    seeds = [s for s, _ in found]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"Duplicate seed numbers found for prefix {prefix!r} in {runs_dir} -- "
                          f"a directory AND a result.txt both matched the same seed? Resolve by hand.")
    found.sort(key=lambda t: t[0])
    return found


def per_seed_values_for_prefix(runs_dir: Path, prefix: str, metric: str, n_warmup: int, force: bool,
                                goals_mm=None, exclude=(), n_tail=20):
    if metric in TRAINING_TAIL_METRICS:
        runs = find_training_seed_dirs(runs_dir, prefix, exclude)
        column = _TRAINING_TAIL_COLUMN[metric]
        values = []
        for seed, path in runs:
            v = training_tail_value(path, column, n_tail=n_tail, force=force)
            values.append(v)
            print(f"  seed{seed}: {path}  ->  {v:.4f}")
        return runs, values
    runs = find_seed_runs(runs_dir, prefix)
    values = []
    for seed, path in runs:
        if metric in GOAL_SET_METRICS:
            v = _goal_set_value(path, metric, goals_mm, n_warmup, force)
        else:
            v = per_seed_value(path, metric, n_warmup=n_warmup, force=force)
        values.append(v)
        print(f"  seed{seed}: {path}  ->  {v:.4f}")
    return runs, values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, type=Path,
                     help="directory to search for group A (and group B, unless --runs-dir-b is given)")
    ap.add_argument("--runs-dir-b", type=Path, default=None,
                     help="directory to search for group B, if different from --runs-dir")
    ap.add_argument("--prefix-a", required=True)
    ap.add_argument("--prefix-b", required=True)
    ap.add_argument("--metric", required=True,
                     help=f"arm: {ARM_METRICS}  |  cartpole: {CARTPOLE_METRICS}  |  goal-set: "
                          f"{GOAL_SET_METRICS}  |  training-tail: {TRAINING_TAIL_METRICS}")
    ap.add_argument("--goals-mm", type=str, default=None,
                     help='required for goal-set metrics: "x1,y1;x2,y2;..." in mm')
    ap.add_argument("--n-warmup", type=int, default=1,
                     help="episodes discarded as warmup for arm-style runs (default 1, the locked convention)")
    ap.add_argument("--exclude", type=str, default=None,
                     help='training-tail metrics only: "sub1,sub2" -- directory names containing any '
                          'of these are excluded from the substring match')
    ap.add_argument("--n-tail", type=int, default=20,
                     help="training-tail metrics only: episodes averaged from the end of training (default 20)")
    ap.add_argument("--unpaired", action="store_true",
                     help="Mann-Whitney U instead of the default paired Wilcoxon signed-rank -- only "
                          "correct when the two groups do NOT share seed numbers (no comparison in "
                          "this paper qualifies)")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"], default="two-sided")
    ap.add_argument("--force", action="store_true",
                     help="recompute per_seed_value, ignoring any cached greedy_eval_summary.csv")
    args = ap.parse_args()

    if args.metric in GOAL_SET_METRICS and not args.goals_mm:
        raise SystemExit(f"--metric {args.metric!r} needs --goals-mm")
    goals_mm = parse_goals_mm(args.goals_mm) if args.goals_mm else None
    exclude = tuple(s.strip() for s in args.exclude.split(",")) if args.exclude else ()

    runs_dir_b = args.runs_dir_b or args.runs_dir

    print(f"Group A ({args.prefix_a!r} in {args.runs_dir}):")
    runs_a, values_a = per_seed_values_for_prefix(
        args.runs_dir, args.prefix_a, args.metric, args.n_warmup, args.force, goals_mm,
        exclude, args.n_tail)
    print(f"Group B ({args.prefix_b!r} in {runs_dir_b}):")
    runs_b, values_b = per_seed_values_for_prefix(
        runs_dir_b, args.prefix_b, args.metric, args.n_warmup, args.force, goals_mm,
        exclude, args.n_tail)

    if args.unpaired:
        r = compare(values_a, values_b, alternative=args.alternative)
        print()
        print(f"Metric: {args.metric}")
        print(f"A ({args.prefix_a}): n={r.n_a}  median={r.median_a:.3f}  mean={r.mean_a:.3f}")
        print(f"B ({args.prefix_b}): n={r.n_b}  median={r.median_b:.3f}  mean={r.mean_b:.3f}")
        print(f"Mann-Whitney U = {r.U:.1f}   p = {r.p:.4g}   ({r.alternative}, {r.method})")
        print(f"Pairwise: A beats B in {r.pairwise_win_fraction*100:.1f}% "
              f"of {r.n_a*r.n_b} seed-pairs  (50% = no difference)")
    else:
        seeds_a = [s for s, _ in runs_a]
        seeds_b = [s for s, _ in runs_b]
        if seeds_a != seeds_b:
            raise SystemExit(f"paired comparison requires identical, identically-ordered seed "
                              f"lists -- got A={seeds_a} B={seeds_b}. If these two groups really "
                              f"don't share seeds, pass --unpaired.")
        r = compare_paired(values_a, values_b, alternative=args.alternative)

        def _mean_std(values):
            m = sum(values) / len(values)
            var = sum((v - m) ** 2 for v in values) / len(values)
            return m, var ** 0.5

        mean_a, std_a = _mean_std(values_a)
        mean_b, std_b = _mean_std(values_b)
        print()
        print(f"Metric: {args.metric}  (paired by seed)")
        print(f"A ({args.prefix_a}): mean={mean_a:.3f}  cross-seed std={std_a:.3f}")
        print(f"B ({args.prefix_b}): mean={mean_b:.3f}  cross-seed std={std_b:.3f}")
        print(f"n={r.n}  median diff={r.median_diff:.3f}  mean diff={r.mean_diff:.3f}")
        print(f"A beats B in {r.n_a_beats_b}/{r.n}, B beats A in {r.n_b_beats_a}/{r.n}, "
              f"{r.n_zero_diffs} tied")
        print(f"Wilcoxon signed-rank W = {r.W:.1f}   p = {r.p:.4g}   ({r.alternative}, {r.method})")
    print("Significant at 0.05." if r.p < 0.05 else "Not significant at 0.05.")
    print()
    print("Run dirs/files used:")
    print(f"  A: {[str(p) for _, p in runs_a]}")
    print(f"  B: {[str(p) for _, p in runs_b]}")


if __name__ == "__main__":
    main()
