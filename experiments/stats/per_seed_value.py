#!/usr/bin/env python3
"""Reduce ONE eval run to its single per-seed number, using the project's
locked greedy-eval convention: frozen weights, discard the warmup episode,
average the clean eval episodes.

Handles the two raw-data formats this project actually has:

- **Arm (reaching task)**: a run DIRECTORY containing `stats.csv`, one row
  per episode (columns include `episode`, `final_d_mm`, `success`).
  Episode 0 is the warmup episode (frozen/greedy from the start, but still
  discarded per the locked convention) -- everything from episode 1 onward
  is a clean eval episode.
- **CartPole**: a single FILE, `{LABEL}_seed{N}_result.txt`, already a
  one-line summary written by `train_and_eval_*.py` with a
  `GREEDY_EVAL_mean=...` field -- the reduction already happened at
  training time, this just reads it back out.

`per_seed_value()` detects which format it was given (directory vs file)
and dispatches accordingly. Either way it caches the computed numbers into
a small `greedy_eval_summary.csv` next to the raw data (inside the run
directory for the arm case, next to the result.txt for CartPole) so the
reduction is auditable without re-running code, and reproducible (delete
the summary, re-run, get the same file back) -- not a hand-maintained
ledger that can drift from the raw data it's derived from.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ARM_METRICS = ("mean_distance", "success_rate",
               "success@15mm", "success@30mm", "success@50mm",
               "success@100mm", "success@150mm", "mean_decoy_distance")
CARTPOLE_METRICS = ("return",)

_SUCCESS_THRESHOLDS = {
    "success@15mm": 15.0, "success@30mm": 30.0, "success@50mm": 50.0,
    "success@100mm": 100.0, "success@150mm": 150.0,
}


def _read_arm_clean_episodes(run_dir: Path, n_warmup: int = 1):
    stats_path = run_dir / "stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found -- is {run_dir} a real eval run directory?")
    with open(stats_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(float(r["episode"])) >= n_warmup]
    if not rows:
        raise ValueError(f"{stats_path}: no episodes left after discarding the first {n_warmup} as warmup")
    return rows


def _compute_arm_metric(rows, metric: str) -> float:
    if metric == "mean_distance":
        return sum(float(r["final_d_mm"]) for r in rows) / len(rows)
    if metric == "mean_decoy_distance":
        # swap-control sanity check: distance from the final hand position
        # to the DECOY target fed to perception (see main.py's decoy_d_mm).
        # NaN on runs that don't use target_continuous_swap_eval, and on
        # older stats.csv files predating this column entirely.
        if "decoy_d_mm" not in rows[0]:
            return float("nan")
        return sum(float(r["decoy_d_mm"]) for r in rows) / len(rows)
    if metric == "success_rate":
        return sum(float(r["success"]) for r in rows) / len(rows)
    if metric in _SUCCESS_THRESHOLDS:
        thr = _SUCCESS_THRESHOLDS[metric]
        return sum(1.0 for r in rows if float(r["final_d_mm"]) < thr) / len(rows)
    raise ValueError(f"unknown arm metric {metric!r}; supported: {ARM_METRICS}")


def _write_arm_summary(run_dir: Path, rows, n_warmup: int):
    summary_path = run_dir / "greedy_eval_summary.csv"
    values = {m: _compute_arm_metric(rows, m) for m in ARM_METRICS}
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "n_clean_episodes", "n_discarded_warmup"])
        for m in ARM_METRICS:
            w.writerow([m, f"{values[m]:.6f}", len(rows), n_warmup])
    return values


def _arm_per_seed_value(run_dir: Path, metric: str, n_warmup: int = 1, force: bool = False) -> float:
    if metric not in ARM_METRICS:
        raise ValueError(f"unknown arm metric {metric!r}; supported: {ARM_METRICS}")
    summary_path = run_dir / "greedy_eval_summary.csv"
    if summary_path.exists() and not force:
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["metric"] == metric:
                    return float(row["value"])
        # summary exists but is missing this metric (e.g. an older/partial
        # summary) -- fall through and recompute rather than error.
    rows = _read_arm_clean_episodes(run_dir, n_warmup)
    values = _write_arm_summary(run_dir, rows, n_warmup)
    return values[metric]


_GREEDY_EVAL_RE = re.compile(r"GREEDY_EVAL_mean=([-\d.eE]+)")


def _cartpole_per_seed_value(result_file: Path, metric: str, force: bool = False) -> float:
    if metric != "return":
        raise ValueError(f"unknown CartPole metric {metric!r}; supported: {CARTPOLE_METRICS}")
    summary_path = result_file.with_name(result_file.stem + "_greedy_eval_summary.csv")
    if summary_path.exists() and not force:
        with open(summary_path, newline="") as f:
            row = next(csv.DictReader(f))
            return float(row["value"])
    text = result_file.read_text()
    m = _GREEDY_EVAL_RE.search(text)
    if not m:
        raise ValueError(f"{result_file}: no GREEDY_EVAL_mean= field found")
    value = float(m.group(1))
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "source"])
        w.writerow(["return", f"{value:.6f}", result_file.name])
    return value


def _compute_selectivity(rows, goals_mm):
    """goals_mm: sequence of (x_mm, y_mm). Fraction of rows where the final
    hand position is nearest the COMMANDED goal (vs. any other goal in the
    set) -- see scratchpad/compute_selectivity.py, the original prototype
    this reimplements per-seed (that script only pools across paths).

    Also splits the misses under a swap control into two buckets:
    "selectivity_decoy" (nearest the SPECIFIC decoy goal main.py's
    target_swap_eval would have shown -- (commanded_idx+1) % n_goals,
    the same "next goal in rotation" rule) vs. "selectivity_other"
    (nearest neither the true nor that specific decoy goal). This
    distinguishes "the policy follows whatever wrong goal it's shown"
    from "the policy just stops using the goal signal and lands
    somewhere among the rest" -- only a meaningful distinction at 3+
    goals (at 2 goals "not the true one" and "the decoy" are the same
    goal by construction, so selectivity_other is always 0). Computed
    unconditionally, even on non-swap runs, where it's just a
    descriptive fact about where the hand lands relative to the goal
    numbering, not evidence of anything.
    """
    import numpy as np
    goals = np.asarray(goals_mm, dtype=float)
    n_goals = len(goals)
    hits = hits_decoy = hits_other = 0
    d_commanded, d_centroid = [], []
    centroid = goals.mean(axis=0)
    for r in rows:
        ee = np.array([float(r["ee_x_mm"]), float(r["ee_y_mm"])])
        tgt = np.array([float(r["target_x_mm"]), float(r["target_y_mm"])])
        nearest_idx = int(np.argmin(np.linalg.norm(goals - ee, axis=1)))
        commanded_idx = int(np.argmin(np.linalg.norm(goals - tgt, axis=1)))
        decoy_idx = (commanded_idx + 1) % n_goals
        if nearest_idx == commanded_idx:
            hits += 1
        elif nearest_idx == decoy_idx:
            hits_decoy += 1
        else:
            hits_other += 1
        d_commanded.append(float(np.linalg.norm(ee - tgt)))
        d_centroid.append(float(np.linalg.norm(ee - centroid)))
    n = len(rows)
    return {
        "selectivity": hits / n,
        "selectivity_decoy": hits_decoy / n,
        "selectivity_other": hits_other / n,
        "mean_d_commanded": sum(d_commanded) / n,
        "mean_d_centroid": sum(d_centroid) / n,
    }


def selectivity_per_seed_value(run_dir, goals_mm, n_warmup: int = 1, force: bool = False) -> float:
    """Like _arm_per_seed_value() but for the "selectivity" metric, which
    (unlike ARM_METRICS) needs an external goal set -- not auto-detected
    from the run, so not folded into per_seed_value()'s dispatch. Each
    run_dir is always evaluated under one fixed goal set by construction
    (a 2-goal run is never re-scored against the 3-goal set), so caching
    keyed by the bare metric name "selectivity" is unambiguous.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "greedy_eval_summary.csv"
    if summary_path.exists() and not force:
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["metric"] == "selectivity":
                    return float(row["value"])
    rows = _read_arm_clean_episodes(run_dir, n_warmup)
    values = _compute_selectivity(rows, goals_mm)
    # append to (or create) the shared summary cache, same file ARM_METRICS uses
    existing = {}
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["metric"]] = row
    for k, v in values.items():
        existing[k] = {"metric": k, "value": f"{v:.6f}",
                        "n_clean_episodes": len(rows), "n_discarded_warmup": n_warmup}
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "n_clean_episodes", "n_discarded_warmup"])
        w.writeheader()
        for row in existing.values():
            w.writerow(row)
    return values["selectivity"]


def training_tail_value(run_dir, column: str, n_tail: int = 20, force: bool = False) -> float:
    """Mean of a stats.csv column over the LAST n_tail episodes of a
    TRAINING run (not an eval run) -- e.g. d1_ch0_w_std's settled value
    once learning has plateaued. No warmup discard here (that convention
    is for eval runs); this reduces a live training curve's tail instead.
    Cached into training_tail_summary.csv next to stats.csv, keyed by
    "column:n_tail" since the same run can be reduced under different
    columns/tail lengths.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "training_tail_summary.csv"
    key = f"{column}:{n_tail}"
    if summary_path.exists() and not force:
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["key"] == key:
                    return float(row["value"])
    stats_path = run_dir / "stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found -- is {run_dir} a real training run directory?")
    with open(stats_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r["episode"])))
    if len(rows) < n_tail:
        raise ValueError(f"{stats_path}: only {len(rows)} episodes, need at least n_tail={n_tail}")
    vals = [float(r[column]) for r in rows[-n_tail:]]
    value = sum(vals) / len(vals)
    existing = {}
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["key"]] = row
    existing[key] = {"key": key, "value": f"{value:.6f}", "n_tail": n_tail, "n_total_episodes": len(rows)}
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value", "n_tail", "n_total_episodes"])
        w.writeheader()
        for row in existing.values():
            w.writerow(row)
    return value


def per_seed_value(run_path, metric: str, n_warmup: int = 1, force: bool = False) -> float:
    """The one function the rest of the pipeline calls.

    run_path: a directory (arm-style eval run, must contain stats.csv) or a
    file (CartPole-style {LABEL}_seed{N}_result.txt).
    metric: one of ARM_METRICS for a directory, or "return" for a file.
    n_warmup: episodes discarded as warmup (arm only; default 1, the
    locked convention -- don't change this without a reason, it's what
    every reported number in this project uses).
    force: recompute and overwrite the cached greedy_eval_summary.csv even
    if one already exists (use after changing the reduction logic itself).
    """
    p = Path(run_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.is_dir():
        return _arm_per_seed_value(p, metric, n_warmup=n_warmup, force=force)
    return _cartpole_per_seed_value(p, metric, force=force)


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_path", help="eval run directory (arm) or result.txt (CartPole)")
    ap.add_argument("--metric", required=True)
    ap.add_argument("--n-warmup", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="recompute, ignore cached summary")
    args = ap.parse_args()
    value = per_seed_value(args.run_path, args.metric, n_warmup=args.n_warmup, force=args.force)
    print(f"{args.run_path}  {args.metric} = {value:.6f}")


if __name__ == "__main__":
    _cli()
