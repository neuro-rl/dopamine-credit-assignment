"""Unit tests for per_seed_value.py: the arm-style (stats.csv directory) and
CartPole-style (result.txt file) reduction paths, the warmup-episode
discard rule, and the greedy_eval_summary.csv caching behavior (written
once, read back on subsequent calls, overwritten only with --force).
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from per_seed_value import per_seed_value, ARM_METRICS  # noqa: E402


def _write_arm_stats_csv(path, rows):
    """rows: list of dicts with at least episode/final_d_mm/success."""
    fieldnames = ["episode", "final_d_mm", "success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.fixture
def arm_run_dir(tmp_path):
    """A synthetic eval run: episode 0 is a warmup OUTLIER that must be
    discarded (if it weren't, these numbers would be very different),
    episodes 1-4 are the clean eval episodes with known, hand-computable
    values."""
    run_dir = tmp_path / "fake_eval_run"
    run_dir.mkdir()
    rows = [
        {"episode": 0, "final_d_mm": 999.0, "success": 0},   # warmup, must be discarded
        {"episode": 1, "final_d_mm": 10.0, "success": 1},    # <15mm
        {"episode": 2, "final_d_mm": 20.0, "success": 0},    # <30mm, not <15mm
        {"episode": 3, "final_d_mm": 40.0, "success": 0},    # <50mm
        {"episode": 4, "final_d_mm": 80.0, "success": 0},    # <100mm
    ]
    _write_arm_stats_csv(run_dir / "stats.csv", rows)
    return run_dir


def test_arm_mean_distance_excludes_warmup(arm_run_dir):
    # clean episodes only: (10+20+40+80)/4 = 37.5 -- NOT (999+10+20+40+80)/5
    v = per_seed_value(arm_run_dir, "mean_distance")
    assert v == pytest.approx(37.5)


def test_arm_success_rate(arm_run_dir):
    # 1 success out of 4 clean episodes
    v = per_seed_value(arm_run_dir, "success_rate")
    assert v == pytest.approx(0.25)


def test_arm_success_thresholds(arm_run_dir):
    assert per_seed_value(arm_run_dir, "success@15mm") == pytest.approx(1 / 4)   # only 10mm
    assert per_seed_value(arm_run_dir, "success@30mm") == pytest.approx(2 / 4)   # 10, 20
    assert per_seed_value(arm_run_dir, "success@50mm") == pytest.approx(3 / 4)   # 10, 20, 40
    assert per_seed_value(arm_run_dir, "success@100mm") == pytest.approx(4 / 4)  # all


def test_arm_writes_summary_csv_with_all_metrics(arm_run_dir):
    per_seed_value(arm_run_dir, "mean_distance")
    summary = arm_run_dir / "greedy_eval_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    assert set(rows) == set(ARM_METRICS)
    assert float(rows["mean_distance"]["value"]) == pytest.approx(37.5)
    assert int(rows["mean_distance"]["n_clean_episodes"]) == 4
    assert int(rows["mean_distance"]["n_discarded_warmup"]) == 1


def test_arm_summary_is_cached_not_recomputed(arm_run_dir):
    per_seed_value(arm_run_dir, "mean_distance")
    summary = arm_run_dir / "greedy_eval_summary.csv"
    # Corrupt stats.csv so a real recompute would give a different (wrong) answer
    _write_arm_stats_csv(arm_run_dir / "stats.csv", [
        {"episode": 1, "final_d_mm": 0.0, "success": 1},
    ])
    v = per_seed_value(arm_run_dir, "mean_distance")   # should read the CACHED summary
    assert v == pytest.approx(37.5)
    # --force must recompute and pick up the corrupted (new) stats.csv
    v_forced = per_seed_value(arm_run_dir, "mean_distance", force=True)
    assert v_forced == pytest.approx(0.0)


def test_arm_missing_stats_csv_raises(tmp_path):
    empty_dir = tmp_path / "no_stats_here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        per_seed_value(empty_dir, "mean_distance")


def test_arm_unknown_metric_raises(arm_run_dir):
    with pytest.raises(ValueError):
        per_seed_value(arm_run_dir, "not_a_real_metric")


def test_arm_n_warmup_zero_keeps_episode_0(arm_run_dir):
    # With n_warmup=0, episode 0 (the 999.0 outlier) is included -- confirms
    # the discard is actually driven by n_warmup, not hardcoded.
    v = per_seed_value(arm_run_dir, "mean_distance", n_warmup=0, force=True)
    assert v == pytest.approx((999.0 + 10.0 + 20.0 + 40.0 + 80.0) / 5)


@pytest.fixture
def cartpole_result_file(tmp_path):
    f = tmp_path / "noCentNoAction_seed4_result.txt"
    f.write_text(
        "[noCentNoAction] seed=4 eps=0.3 lr=0.1 WARMUP=50 CENTERED: "
        "train_ep0-250=17.5 GREEDY_EVAL_mean=9.6 GREEDY_EVAL_std=0.6 "
        "GREEDY_EVAL_max=11 elapsed=274.7s\n"
    )
    return f


def test_cartpole_reads_greedy_eval_mean(cartpole_result_file):
    v = per_seed_value(cartpole_result_file, "return")
    assert v == pytest.approx(9.6)


def test_cartpole_writes_summary_file(cartpole_result_file):
    per_seed_value(cartpole_result_file, "return")
    summary = cartpole_result_file.with_name(
        cartpole_result_file.stem + "_greedy_eval_summary.csv")
    assert summary.exists()
    with open(summary, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["metric"] == "return"
    assert float(row["value"]) == pytest.approx(9.6)


def test_cartpole_unknown_metric_raises(cartpole_result_file):
    with pytest.raises(ValueError):
        per_seed_value(cartpole_result_file, "mean_distance")


def test_cartpole_no_greedy_field_raises(tmp_path):
    f = tmp_path / "broken_seed1_result.txt"
    f.write_text("[broken] seed=1 no greedy field here\n")
    with pytest.raises(ValueError):
        per_seed_value(f, "return")


def test_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        per_seed_value("/no/such/path/anywhere", "mean_distance")
