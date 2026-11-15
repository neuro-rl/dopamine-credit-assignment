#!/usr/bin/env python3
"""cart_pole/stats_report.py -- the one command that produces every number
in the paper's CartPole section (sec4). No number there should be hand-typed:
this script reads experiments_cartpole/ directly and writes stats_report.md
+ stats_report.json; those files are the only source for sec4.

Engine: imports compare_paired() from util/compare.py (Wilcoxon
signed-rank, by seed). Every comparison here is paired: build_network(seed=
SEED) draws cortex/D1/D2 initial weights from NEST's seed-determined RNG
stream before any condition-specific flag is read, and run.py pins the
Gymnasium env's RNG to the same SEED -- so the same seed means the same
starting network AND the same environment RNG stream across every
condition, verified empirically (W_d1_init_std matches across all 5
conditions, every seed; see EXPERIMENTS.md). Mann-Whitney (unpaired) was
used for the main chain until 2026-11-07, before this pairing had been
verified project-wide rather than just for the mutual_inhibition extra.
This script does not reimplement the test, only the reduction (run dir ->
value) and the grouping/reporting specific to CartPole's run-folder layout.

Reducer: one run dir's per-seed value is the `return` row in its own
`greedy_eval_summary.csv` (written by run_io.write_greedy_eval_summary at
training time -- mean over all greedy-eval episodes, see that function's
docstring for why CartPole has no warmup-episode discard, unlike the arm).

Bimodality: each seed lands near the ~10-step floor or well above it, so a
single mean+-std understates the real spread. Per condition this script
also reports `learn_fraction`: the share of seeds with return above a
fixed threshold T, plus every per-seed value so the paper's figure can
plot seeds as dots, not just a bar. T is set once, in THRESHOLD below, and
is not tuned to the result -- see the comment there.

Usage:
    cd cart_pole
    micromamba run -n snn310 python stats_report.py
    micromamba run -n snn310 python stats_report.py --runs-dir ../experiments_cartpole \
        --threshold 30 --out-md stats_report.md --out-json stats_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent / "util"))
from compare import compare_paired  # noqa: E402

DEFAULT_RUNS_DIR = HERE.parent / "experiments_cartpole"

# T=30: set once, before reading any condition's result -- well above
# the ~10-step floor (an untrained/non-learning policy
# balances the pole for only a handful of steps) and below any genuine
# learner. Not adjusted after seeing results; do not retune this to change
# which seeds count as "learned".
THRESHOLD = 30.0

SEEDS = tuple(range(1, 10))  # the locked 9-seed convention

# Condition order matches the paper's Table 2 (5-condition chain): neither
# -> +raw action -> +action-centered -> +presyn.-centered -> +population
# (vec2d_proj). uncentered_rawaction is the intermediate condition added
# after the original 4-condition chain (isolates the tanh(g/g_scale)
# surrogate-centering term's own effect, see configs/uncentered_rawaction.py).
CONDITIONS = ("noCentNoAction", "uncentered_rawaction", "uncentered", "centered", "vec2d_proj")

# Every adjacent-row comparison in the chain, paired by seed (see the
# module docstring for why pairing holds across all 5 conditions).
CHAIN_COMPARISONS = (
    ("noCentNoAction", "uncentered_rawaction", "neither vs +raw action"),
    ("uncentered_rawaction", "uncentered", "+raw action vs +action-centered"),
    ("uncentered", "centered", "+action-centered vs +presyn.-centered"),
    ("centered", "vec2d_proj", "+presyn.-centered vs +population"),
)

# mutual_inhibition: a real, static, bidirectional D1<->D2
# lateral inhibition circuit, built from "centered" + mutual_inhibition=True
# (see cart_pole/configs/mutual_inhibition.py). Not part of the orthogonal
# 5-condition chain above (it isn't one of the locked ablation
# factors) -- reported separately, but paired against "centered" the same
# way: build_network(seed=SEED) draws cortex/D1/D2 initial weights from
# NEST's seed-determined RNG stream BEFORE the fixed-weight mutual-
# inhibition connections are added, so the same seed gives the same
# starting network in both conditions (verified empirically: seed=1's
# W_d1_init_std is 28.8 in both run logs).
EXTRA_CONDITIONS = ("mutual_inhibition", "d1_only")
PAIRED_COMPARISONS = CHAIN_COMPARISONS + (
    ("centered", "mutual_inhibition", "+centering vs +centering+mutual-inhibition"),
    ("centered", "d1_only", "+centering vs +centering, D2 removed (d1_only)"),
)

_DIR_RE = re.compile(r"^(?P<ts>\d{8}-\d{6})_(?P<label>.+)_seed(?P<seed>\d+)$")


def find_run_dirs(runs_dir: Path) -> dict[tuple[str, int], Path]:
    """Map (label, seed) -> run dir, for every <ts>_<label>_seed<N> directory
    under runs_dir. Errors loudly if the same (label, seed) is matched by
    more than one directory (e.g. an old run left behind after a re-run) --
    this script never silently picks "the latest"; ambiguity must be
    resolved by deleting the stale directory before running this."""
    found: dict[tuple[str, int], list[Path]] = {}
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        m = _DIR_RE.match(p.name)
        if not m:
            continue
        key = (m.group("label"), int(m.group("seed")))
        found.setdefault(key, []).append(p)

    dupes = {k: v for k, v in found.items() if len(v) > 1}
    if dupes:
        lines = [f"  {label} seed{seed}: {[str(p) for p in paths]}"
                 for (label, seed), paths in sorted(dupes.items())]
        raise SystemExit(
            "Ambiguous run directories (same condition+seed matched more "
            "than once) -- delete the stale one(s) before running this:\n"
            + "\n".join(lines))
    return {k: v[0] for k, v in found.items()}


def run_dir_value(run_dir: Path) -> float:
    """The reducer: one run dir -> its per-seed `return` value, read from
    the run's own greedy_eval_summary.csv (metric column == 'return')."""
    summary_path = run_dir / "greedy_eval_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"{run_dir}: missing greedy_eval_summary.csv")
    import csv
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] == "return":
                return float(row["value"])
    raise SystemExit(f"{summary_path}: no 'return' row found")


@dataclass
class ConditionStats:
    label: str
    n: int
    median: float
    mean: float
    std: float
    learn_fraction: float
    threshold: float
    values: list  # per-seed, in seed order 1..9
    run_dirs: list  # str, same order as values


def collect_condition(label: str, runs_dir: Path, run_dir_map: dict, threshold: float) -> ConditionStats:
    values, dirs = [], []
    missing = []
    for seed in SEEDS:
        key = (label, seed)
        if key not in run_dir_map:
            missing.append(seed)
            continue
        run_dir = run_dir_map[key]
        values.append(run_dir_value(run_dir))
        dirs.append(os.path.relpath(run_dir, REPO_ROOT))
    if missing:
        raise SystemExit(
            f"Condition '{label}': missing seed(s) {missing} under {runs_dir} "
            f"-- need all of {list(SEEDS)}.")
    return ConditionStats(
        label=label,
        n=len(values),
        median=statistics.median(values),
        mean=statistics.mean(values),
        std=statistics.pstdev(values),
        learn_fraction=sum(1 for v in values if v > threshold) / len(values),
        threshold=threshold,
        values=values,
        run_dirs=dirs,
    )


@dataclass
class PairedComparisonReport:
    name: str
    label_a: str
    label_b: str
    n: int
    n_zero_diffs: int
    median_diff: float
    n_a_beats_b: int
    n_b_beats_a: int
    W: float
    p: float
    method: str
    alternative: str
    run_dirs_a: list
    run_dirs_b: list


def collect_paired_comparison(name: str, cond_a: ConditionStats, cond_b: ConditionStats) -> PairedComparisonReport:
    r = compare_paired(cond_a.values, cond_b.values, alternative="two-sided")
    return PairedComparisonReport(
        name=name,
        label_a=cond_a.label, label_b=cond_b.label,
        n=r.n, n_zero_diffs=r.n_zero_diffs, median_diff=r.median_diff,
        n_a_beats_b=r.n_a_beats_b, n_b_beats_a=r.n_b_beats_a,
        W=r.W, p=r.p, method=r.method, alternative=r.alternative,
        run_dirs_a=cond_a.run_dirs, run_dirs_b=cond_b.run_dirs,
    )


def render_markdown(conditions: dict, threshold: float,
                     extra_conditions: dict | None = None,
                     paired_comparisons: list | None = None) -> str:
    lines = []
    lines.append("# CartPole statistics report (sec4)")
    lines.append("")
    lines.append("Regenerated by `cart_pole/stats_report.py` directly from "
                  "`experiments_cartpole/`. This file (and `stats_report.json`) "
                  "is the only source for the numbers in the paper's Section 4.")
    lines.append("")
    lines.append(f"Threshold T={threshold:.0f} (fixed before inspecting any "
                  "condition's results): `learn_fraction` = share of the 9 "
                  "seeds with return above T.")
    lines.append("")
    lines.append("## Per-condition")
    lines.append("")
    lines.append("| condition | n | median | mean | std | learn_fraction (>T) | per-seed values |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in CONDITIONS:
        c = conditions[label]
        vals = ", ".join(f"{v:.1f}" for v in c.values)
        lines.append(f"| {c.label} | {c.n} | {c.median:.1f} | {c.mean:.1f} | "
                      f"{c.std:.1f} | {c.learn_fraction:.3f} | {vals} |")
    lines.append("")

    if extra_conditions:
        lines.append("## Extra conditions (not part of the 5-condition chain)")
        lines.append("")
        lines.append("| condition | n | median | mean | std | learn_fraction (>T) | per-seed values |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, c in extra_conditions.items():
            vals = ", ".join(f"{v:.1f}" for v in c.values)
            lines.append(f"| {c.label} | {c.n} | {c.median:.1f} | {c.mean:.1f} | "
                          f"{c.std:.1f} | {c.learn_fraction:.3f} | {vals} |")
        lines.append("")

    if paired_comparisons:
        lines.append("## Paired comparisons (Wilcoxon signed-rank, two-sided, by seed)")
        lines.append("")
        lines.append("| comparison | n | median diff (a-b) | a>b | b>a | W | p | method |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in paired_comparisons:
            lines.append(f"| {r.name} ({r.label_a} vs {r.label_b}) | {r.n} | "
                          f"{r.median_diff:+.1f} | {r.n_a_beats_b} | {r.n_b_beats_a} | "
                          f"{r.W:.1f} | {r.p:.4g} | {r.method} |")
        lines.append("")

    lines.append("## Provenance (run dirs used)")
    lines.append("")
    for label in CONDITIONS:
        c = conditions[label]
        lines.append(f"**{label}**:")
        for seed, d in zip(SEEDS, c.run_dirs):
            lines.append(f"  - seed{seed}: `{d}`")
    if extra_conditions:
        for label, c in extra_conditions.items():
            lines.append(f"**{label}**:")
            for seed, d in zip(SEEDS, c.run_dirs):
                lines.append(f"  - seed{seed}: `{d}`")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out-md", type=Path, default=HERE / "stats_report.md")
    ap.add_argument("--out-json", type=Path, default=HERE / "stats_report.json")
    args = ap.parse_args()
    threshold = args.threshold

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    run_dir_map = find_run_dirs(runs_dir)

    conditions = {label: collect_condition(label, runs_dir, run_dir_map, threshold) for label in CONDITIONS}

    extra_conditions = {label: collect_condition(label, runs_dir, run_dir_map, threshold)
                         for label in EXTRA_CONDITIONS
                         if all((label, seed) in run_dir_map for seed in SEEDS)}
    all_conditions = {**conditions, **extra_conditions}
    paired_comparisons = [collect_paired_comparison(name, all_conditions[a], all_conditions[b])
                           for a, b, name in PAIRED_COMPARISONS
                           if a in all_conditions and b in all_conditions]

    md = render_markdown(conditions, threshold,
                          extra_conditions=extra_conditions,
                          paired_comparisons=paired_comparisons)
    args.out_md.write_text(md)

    payload = {
        "threshold": threshold,
        "seeds": list(SEEDS),
        "runs_dir": os.path.relpath(runs_dir, REPO_ROOT),
        "conditions": {label: asdict(c) for label, c in conditions.items()},
        "extra_conditions": {label: asdict(c) for label, c in extra_conditions.items()},
        "paired_comparisons": [asdict(r) for r in paired_comparisons],
    }
    args.out_json.write_text(json.dumps(payload, indent=2))

    print(md)
    print(f"\nWrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
