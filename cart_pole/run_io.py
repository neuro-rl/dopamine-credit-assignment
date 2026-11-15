"""Shared run-folder I/O for cart_pole/run.py, mirroring the arm's
experiments/runs/ layout (model/file_utils.py + model/viz.py) under a
parallel experiments_cartpole/ root:

  experiments_cartpole/<timestamp>_<label>_seed<seed>/
    experiment.txt            -- experiment_id + run params (arm-style)
    config_used.py            -- copy of the nest_cartpole_<label>_test.py
                                  module actually imported (the "config")
    stats.csv                 -- one row per episode: episode, phase
                                  (warmup/train/greedy_eval), score, epsilon
    summary.pdf               -- training curve + greedy-eval mean (vector PDF)
    final_checkpoint.pkl      -- extract_state(net) at the end of training
    greedy_eval_summary.csv   -- arm-style metric cache: mean over ALL
                                  N_EVAL_EPISODES greedy episodes, NO
                                  warmup-episode discard -- NOT the arm's
                                  discard-episode-0 convention, which
                                  doesn't apply here (every CartPole
                                  episode, train or eval, already starts
                                  from a fresh build_network() restore;
                                  there's no analogous "first episode
                                  after reload" transient to discard).
    debug/                    -- unused today (CartPole produces no PDFs
                                  or per-episode NPZ dumps), created for
                                  structural parity with the arm.
"""
import csv
import pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "experiments_cartpole"


def create_run_dir(label, seed):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / f"{ts}_{label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "debug").mkdir(exist_ok=True)
    return run_dir


def save_experiment_params(run_dir, params):
    experiment_id = uuid.uuid4()
    with open(run_dir / "experiment.txt", "w", encoding="utf-8") as f:
        f.write(f"experiment_id={experiment_id}\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")
    return experiment_id


def save_config_used(run_dir, module_file):
    shutil.copy(module_file, run_dir / "config_used.py")


def write_stats_csv(run_dir, rows, filename="stats.csv"):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(run_dir / filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_checkpoint(run_dir, restore, filename="final_checkpoint.pkl"):
    """Pickles `restore` (a plain dict of numpy arrays: W_d1, W_d2, x_bar,
    w_critic) -- same filename convention as the arm's final_checkpoint.pkl,
    though the arm's is a much larger NEST-projection-tracking structure
    (NestCheckpointer); CartPole's is just this dict."""
    with open(run_dir / filename, "wb") as f:
        pickle.dump(restore, f)


def write_greedy_eval_summary(run_dir, eval_scores):
    """Arm-style summary CSV, but no warmup-episode discard -- see the
    module docstring for why. value == mean(eval_scores), i.e. exactly
    GREEDY_EVAL_mean as already reported everywhere else."""
    if not eval_scores:
        raise ValueError("no greedy-eval episodes to summarize")
    value = float(np.mean(eval_scores))
    with open(run_dir / "greedy_eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "n_clean_episodes", "n_discarded_warmup"])
        w.writerow(["return", f"{value:.6f}", len(eval_scores), 0])
    return value


def read_stats_csv(run_dir, filename="stats.csv"):
    """Read stats.csv back into the same row-dict shape write_stats_csv
    took (episode: int, phase: str, score/epsilon: float) -- so a
    summary can be (re)built from a run folder alone, without needing
    the in-memory rows from the training run that produced it."""
    with open(Path(run_dir) / filename, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["episode"] = int(r["episode"])
        r["score"] = float(r["score"])
        r["epsilon"] = float(r["epsilon"])
    return rows


def render_summary_pdf(run_dir, warmup_episodes=0, rolling_window=20):
    """Reads stats.csv itself -- the PDF is strictly reproducible from
    the run folder's own persisted data, not from any in-memory state.
    Callable standalone on any existing run_dir, e.g. to regenerate
    summary.pdf after the fact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = read_stats_csv(run_dir)
    train_rows = [r for r in rows if r["phase"] in ("warmup", "train")]
    eval_rows = [r for r in rows if r["phase"] == "greedy_eval"]
    if not train_rows:
        return

    ep = [r["episode"] for r in train_rows]
    score = [r["score"] for r in train_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, score, alpha=0.3, label="train score")
    if len(score) >= rolling_window:
        kernel = np.ones(rolling_window) / rolling_window
        rolled = np.convolve(score, kernel, mode="valid")
        ax.plot(ep[rolling_window - 1:], rolled, label=f"rolling mean ({rolling_window})")
    if warmup_episodes > 0:
        ax.axvline(warmup_episodes, color="gray", linestyle="--", linewidth=0.7,
                   label="end of critic warmup")
    if eval_rows:
        eval_mean = np.mean([r["score"] for r in eval_rows])
        ax.axhline(eval_mean, color="red", linestyle=":", label=f"greedy eval mean ({eval_mean:.1f})")
    ax.set_xlabel("episode")
    ax.set_ylabel("score (steps balanced)")
    ax.set_title("CartPole training + greedy eval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.pdf")   # vector output -- pure line plot, no raster-size concern
    plt.close(fig)
