#!/usr/bin/env python3
"""experiments/stats/make_figures.py -- per-figure CSV data export, regenerated
from the run folders by one command, never drawn by hand. Each figure reuses
the SAME run reader / registry as its section's stats_report.py (imported,
never copied), so the exported numbers and the p-value quoted next to them in
the paper always come from the exact same run directories.

Output:
  - paper/figures/data/*.csv -- the real deliverable. This is the second
    source of truth bound to the paper: committed here AND copied into the
    paper repo (event-based-snn), where a separate, dependency-free script
    (paper/figures/make_figures.py there) reads these same CSVs and renders
    the actual camera-ready PDFs used by the manuscript.
  - paper/figures/*.pdf -- still produced here too, but now just a local
    debug/preview rendering (quick sanity check that the exported numbers
    look right). These PDFs are NOT copied into the paper repo any more and
    are not the paper's source of truth.

Usage:
    cd experiments/stats
    micromamba run -n snn310 python make_figures.py            # all implemented figures
    micromamba run -n snn310 python make_figures.py --only fig2_cartpole
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_DIR = REPO_ROOT / "paper" / "figures"
DATA_DIR = OUT_DIR / "data"

sys.path.insert(0, str(REPO_ROOT / "cart_pole"))
import stats_report as cartpole_stats  # noqa: E402 -- the CartPole registry/reader, imported not copied

sys.path.insert(0, str(REPO_ROOT / "util"))
sys.path.insert(0, str(HERE))
from compare import compare_paired  # noqa: E402
from per_seed_value import per_seed_value  # noqa: E402

import csv  # noqa: E402

# ---------------------------------------------------------------------------
# Style rules (apply to every figure in this file -- see the spec):
#   - vector PDF, no raster
#   - single column ~3.4in wide, full width ~7.0in, height in proportion
#   - 8-9pt font, ~1.2pt line width
#   - colour-blind-safe (Okabe-Ito) palette, same colour for the same thing
#     across figures
#   - panel labels a/b/c, no in-figure title (the caption in the tex does that)
#   - tight layout, white background, trimmed margins
#   - curves are mean over seeds with a shaded band; THE BAND IS 1 STANDARD
#     DEVIATION across seeds, not IQR -- stated once here, used everywhere
#     in this file. Do not mix bands within one figure set.
# ---------------------------------------------------------------------------
# Measured from the actual compiled sn-jnl class (pdflatex,twocolumn,iicol,
# sn-mathphys-ay), not guessed: \the\textwidth=455.24pt=6.30in,
# \the\columnwidth=(455.24-22.76)/2=216.24pt=3.00in. Using the
# real values means \includegraphics[width=\columnwidth/\textwidth] never
# has to rescale the PDF, so the fonts drawn here at FONT_SIZE actually
# render at FONT_SIZE in the paper instead of being shrunk by a mismatch.
COL_WIDTH_IN = 3.00
FULL_WIDTH_IN = 6.30
FONT_SIZE = 8.5
LINE_WIDTH = 1.2
BAND_IS_STD = True  # see note above -- the single source of truth for this choice

# Okabe-Ito colour-blind-safe palette. Roles are assigned per-figure below
# but reuse these same hex values for the same kind of thing (e.g. "the
# floor/no-mechanism condition" is always black, "swap/control" is always
# vermillion) so a reader sees the same colour mean the same thing across
# figures.
OI_BLACK = "#000000"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_VERMILLION = "#D55E00"
OI_BLUE = "#0072B2"

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.linewidth": LINE_WIDTH * 0.8,
    "lines.linewidth": LINE_WIDTH,
    "xtick.major.width": LINE_WIDTH * 0.8,
    "ytick.major.width": LINE_WIDTH * 0.8,
    "pdf.fonttype": 42,  # editable text in the PDF, not outlined
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_path} (debug preview only, not the paper's source)")


def _save_csv(rows, fieldnames, name):
    """The real deliverable: one tidy CSV per dataset, committed here and
    copied into the paper repo's paper/figures/data/ as the frozen source
    the paper's own renderer reads from."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / name
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# fig2_cartpole.pdf -- Section 4. x = 4 conditions, y = return, every seed a
# point (not mean+-std -- the task is bimodal, see sec4), a short bar at the
# median per condition. Reuses cart_pole/stats_report.py's own registry
# (CONDITIONS, SEEDS, THRESHOLD, find_run_dirs, collect_condition) directly.
# ---------------------------------------------------------------------------
_CARTPOLE_LABELS = {
    "noCentNoAction": "Blind",
    "uncentered_rawaction": "Raw",
    "uncentered": "Action ctr.",
    "centered": "Presyn. ctr.",
    "vec2d_proj": "Pop. RO",
}
_CARTPOLE_COLORS = {
    "noCentNoAction": OI_BLACK,
    "uncentered_rawaction": OI_VERMILLION,
    "uncentered": OI_ORANGE,
    "centered": OI_SKY,
    "vec2d_proj": OI_GREEN,
}


def fig2_cartpole(runs_dir=None, threshold=None):
    runs_dir = (runs_dir or cartpole_stats.DEFAULT_RUNS_DIR).resolve()
    threshold = threshold if threshold is not None else cartpole_stats.THRESHOLD
    run_dir_map = cartpole_stats.find_run_dirs(runs_dir)
    conditions = [
        cartpole_stats.collect_condition(label, runs_dir, run_dir_map, threshold)
        for label in cartpole_stats.CONDITIONS
    ]

    rows = [{"condition": c.label, "seed": s, "return": v}
            for c in conditions for s, v in zip(cartpole_stats.SEEDS, c.values)]
    _save_csv(rows, ["condition", "seed", "return"], "fig2_cartpole.csv")

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN * 0.65))
    rng = np.random.default_rng(0)  # jitter only, never affects reported values
    for i, c in enumerate(conditions):
        color = _CARTPOLE_COLORS[c.label]
        jitter = rng.uniform(-0.12, 0.12, size=len(c.values))
        ax.scatter(np.full(len(c.values), i) + jitter, c.values,
                   s=14, color=color, edgecolor="none", alpha=0.85, zorder=3)
        ax.plot([i - 0.18, i + 0.18], [c.median, c.median],
                color=color, linewidth=LINE_WIDTH * 1.6, zorder=4)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([_CARTPOLE_LABELS[c.label] for c in conditions],
                        fontsize=FONT_SIZE - 1.5, rotation=20, ha="right",
                        rotation_mode="anchor")
    ax.set_ylabel("return (greedy eval)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig2_cartpole.pdf")


# ---------------------------------------------------------------------------
# fig3_weights.pdf -- Section 5d (centering). A fresh 9-seed
# p3_reach_baseline_nocenter.py run, matched against the existing
# centered Arm T batch (p3_reach_baseline_7x7). (a) D1 weight std vs episode,
# mean +- 1 s.d. across seeds, centered vs uncentered. (b) final distance
# vs episode, same convention. (c) per-seed final greedy-eval distance,
# paired by matching seed -- shows the two uncentered seeds that diverge.
# ---------------------------------------------------------------------------
_CENTERED_COLOR = OI_SKY
_UNCENTERED_COLOR = OI_ORANGE


def fig3_weights(runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir else REPO_ROOT / "experiments" / "runs"

    cen_dirs = _find_9seed_dirs(runs_dir, "p3_reach_baseline_7x7",
                                 exclude_substrs=("reachvec", "multigoal", "nocenter"))
    noc_dirs = _find_9seed_dirs(runs_dir, "p3_reach_baseline_nocenter_7x7")

    ep, cen_dist, _ = _load_training_curves(cen_dirs)
    _, noc_dist, _ = _load_training_curves(noc_dirs)

    def _load_d1_std(dirs):
        out = []
        for d in dirs:
            with open(d / "stats.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            rows.sort(key=lambda r: int(float(r["episode"])))
            out.append([float(r["d1_ch0_w_std"]) for r in rows])
        return np.array(out)

    cen_std = _load_d1_std(cen_dirs)
    noc_std = _load_d1_std(noc_dirs)

    # Panel (c) and every pooled number below it are a true restored,
    # frozen-weight, exploration-off greedy eval (eval_firming9_armT /
    # eval_firming9_nocenter), NOT the training run's own last-20-episode
    # eps-greedy tail -- panels (a)/(b) above stay on the raw training
    # curves (cen_dist/noc_dist, loaded above) since those are legitimately
    # showing the learning trajectory, not a final-performance claim.
    cen_final_dist = np.array([per_seed_value(runs_dir / "eval_firming9_armT" / f"armT_seed{s}",
                                               "mean_distance") for s in range(9)])
    noc_final_dist = np.array([per_seed_value(runs_dir / "eval_firming9_nocenter" / f"nocenter_seed{s}",
                                               "mean_distance") for s in range(9)])
    dist_result = compare_paired(list(cen_final_dist), list(noc_final_dist))

    ts_rows = []
    for condition, dist_arr, std_arr in (("centered", cen_dist, cen_std), ("uncentered", noc_dist, noc_std)):
        for seed in range(dist_arr.shape[0]):
            for e, episode in enumerate(ep):
                ts_rows.append({"condition": condition, "seed": seed, "episode": int(episode),
                                 "d1_weight_std": std_arr[seed, e], "final_distance_mm": dist_arr[seed, e]})
    _save_csv(ts_rows, ["condition", "seed", "episode", "d1_weight_std", "final_distance_mm"],
              "fig3_weights_timeseries.csv")

    paired_rows = (
        [{"condition": "centered", "seed": s, "distance_mm": v} for s, v in enumerate(cen_final_dist)]
        + [{"condition": "uncentered", "seed": s, "distance_mm": v} for s, v in enumerate(noc_final_dist)]
    )
    _save_csv(paired_rows, ["condition", "seed", "distance_mm"], "fig3_weights_paired.csv")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.32))

    ax = axes[0]
    ax.plot(ep, cen_std.mean(axis=0), color=_CENTERED_COLOR, label="centered")
    ax.fill_between(ep, cen_std.mean(0) - cen_std.std(0), cen_std.mean(0) + cen_std.std(0),
                     color=_CENTERED_COLOR, alpha=0.15, linewidth=0)
    ax.plot(ep, noc_std.mean(axis=0), color=_UNCENTERED_COLOR, label="uncentered")
    ax.fill_between(ep, noc_std.mean(0) - noc_std.std(0), noc_std.mean(0) + noc_std.std(0),
                     color=_UNCENTERED_COLOR, alpha=0.15, linewidth=0)
    ax.set_xlabel("episode")
    ax.set_ylabel("D1 weight std")
    ax.legend(fontsize=FONT_SIZE - 2, frameon=False, loc="lower right")
    ax.text(-0.18, 1.0, "a", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    ax = axes[1]
    ax.plot(ep, cen_dist.mean(axis=0), color=_CENTERED_COLOR)
    ax.fill_between(ep, cen_dist.mean(0) - cen_dist.std(0), cen_dist.mean(0) + cen_dist.std(0),
                     color=_CENTERED_COLOR, alpha=0.15, linewidth=0)
    ax.plot(ep, noc_dist.mean(axis=0), color=_UNCENTERED_COLOR)
    ax.fill_between(ep, noc_dist.mean(0) - noc_dist.std(0), noc_dist.mean(0) + noc_dist.std(0),
                     color=_UNCENTERED_COLOR, alpha=0.15, linewidth=0)
    ax.set_xlabel("episode")
    ax.set_ylabel("final distance (mm)")
    ax.text(-0.18, 1.0, "b", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    ax = axes[2]
    for s in range(9):
        ax.plot([0, 1], [cen_final_dist[s], noc_final_dist[s]], color="grey",
                linewidth=LINE_WIDTH * 0.6, zorder=1, alpha=0.7)
    ax.scatter(np.zeros(9), cen_final_dist, color=_CENTERED_COLOR, s=16, zorder=3)
    ax.scatter(np.ones(9), noc_final_dist, color=_UNCENTERED_COLOR, s=16, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["centered", "uncentered"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("final distance (mm)")
    ax.text(1.0, 1.02, f"$p={dist_result.p:.4f}$", transform=ax.transAxes,
            fontsize=FONT_SIZE - 1, ha="right", va="bottom")
    ax.text(-0.18, 1.0, "c", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    _save(fig, "fig3_weights.pdf")


# ---------------------------------------------------------------------------
# fig4_reach.pdf -- Section 5a (reach-vec vs raw-target + swap control).
# Section 5a p-values (two-sided, p=0.074 distance / p=0.46 success@15mm
# for Arm R vs Arm T; p=0.0039 for the swap control) must stay consistent
# with the figure's caption.
# (a) final distance vs episode, mean +- 1 s.d. across the 9 training seeds,
# Arm T vs Arm R. (b) success@15mm vs episode, same seeds/convention. (c)
# per-seed greedy-eval mean distance, Arm R normal vs swapped-target
# evaluation of the SAME trained network (the paired swap control) -- lines
# connect each seed's own pair, matching compare_paired()'s pairing.
# ---------------------------------------------------------------------------
_ARMT_COLOR = OI_BLACK
_ARMR_COLOR = OI_BLUE
_ARMR_SWAP_COLOR = OI_VERMILLION


def _load_training_curves(run_dirs):
    """run_dirs: 9 training run directories (400-episode stats.csv each).
    Returns (episodes, dist[9,E], succ[9,E]) arrays, sorted by episode."""
    all_dist, all_succ = [], []
    for d in run_dirs:
        with open(d / "stats.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: int(float(r["episode"])))
        all_dist.append([float(r["final_d_mm"]) for r in rows])
        all_succ.append([float(r["success"]) for r in rows])
    all_dist = np.array(all_dist)
    all_succ = np.array(all_succ)
    episodes = np.arange(all_dist.shape[1])
    return episodes, all_dist, all_succ


def _find_9seed_dirs(runs_dir, pattern_substr, exclude_substrs=()):
    cands = sorted(p for p in runs_dir.glob("*_seed*_seed*") if p.is_dir()
                    and pattern_substr in p.name
                    and not any(x in p.name for x in exclude_substrs))
    if len(cands) != 9:
        raise RuntimeError(
            f"expected 9 run dirs matching '{pattern_substr}' "
            f"(excluding {exclude_substrs}) under {runs_dir}, found "
            f"{len(cands)}: {[p.name for p in cands]}")
    return cands


def fig4_reach(runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir else REPO_ROOT / "experiments" / "runs"

    armT_dirs = _find_9seed_dirs(runs_dir, "p3_reach_baseline_7x7",
                                  exclude_substrs=("reachvec", "multigoal"))
    armR_dirs = _find_9seed_dirs(runs_dir, "p3_reach_baseline_reachvec_7x7")

    ep, distT, succT = _load_training_curves(armT_dirs)
    _, distR, succR = _load_training_curves(armR_dirs)

    armR_normal = [per_seed_value(runs_dir / "eval_firming9_armR" / f"armR_seed{s}",
                                   "mean_distance") for s in range(9)]
    armR_swap = [per_seed_value(runs_dir / "eval_firming9_armRswap" / f"armRswap_seed{s}",
                                 "mean_distance") for s in range(9)]
    swap_result = compare_paired(armR_normal, armR_swap)

    ts_rows = []
    for condition, dist_arr, succ_arr in (("ArmT", distT, succT), ("ArmR", distR, succR)):
        for seed in range(dist_arr.shape[0]):
            for e, episode in enumerate(ep):
                ts_rows.append({"condition": condition, "seed": seed, "episode": int(episode),
                                 "final_distance_mm": dist_arr[seed, e], "success": succ_arr[seed, e]})
    _save_csv(ts_rows, ["condition", "seed", "episode", "final_distance_mm", "success"],
              "fig4_reach_timeseries.csv")

    paired_rows = (
        [{"condition": "normal", "seed": s, "distance_mm": v} for s, v in enumerate(armR_normal)]
        + [{"condition": "swapped", "seed": s, "distance_mm": v} for s, v in enumerate(armR_swap)]
    )
    _save_csv(paired_rows, ["condition", "seed", "distance_mm"], "fig4_reach_paired.csv")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, FULL_WIDTH_IN * 0.32))

    ax = axes[0]
    ax.plot(ep, distT.mean(axis=0), color=_ARMT_COLOR, label="Arm T (raw-target)")
    ax.fill_between(ep, distT.mean(0) - distT.std(0), distT.mean(0) + distT.std(0),
                     color=_ARMT_COLOR, alpha=0.15, linewidth=0)
    ax.plot(ep, distR.mean(axis=0), color=_ARMR_COLOR, label="Arm R (reach-vec)")
    ax.fill_between(ep, distR.mean(0) - distR.std(0), distR.mean(0) + distR.std(0),
                     color=_ARMR_COLOR, alpha=0.15, linewidth=0)
    ax.set_xlabel("episode")
    ax.set_ylabel("final distance (mm)")
    ax.legend(fontsize=FONT_SIZE - 2, frameon=False, loc="upper right")
    ax.text(-0.18, 1.0, "a", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    ax = axes[1]
    ax.plot(ep, succT.mean(axis=0), color=_ARMT_COLOR)
    ax.fill_between(ep, succT.mean(0) - succT.std(0), succT.mean(0) + succT.std(0),
                     color=_ARMT_COLOR, alpha=0.15, linewidth=0)
    ax.plot(ep, succR.mean(axis=0), color=_ARMR_COLOR)
    ax.fill_between(ep, succR.mean(0) - succR.std(0), succR.mean(0) + succR.std(0),
                     color=_ARMR_COLOR, alpha=0.15, linewidth=0)
    ax.set_xlabel("episode")
    ax.set_ylabel("success@15mm")
    ax.text(-0.18, 1.0, "b", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    ax = axes[2]
    for s in range(9):
        ax.plot([0, 1], [armR_normal[s], armR_swap[s]], color="grey",
                linewidth=LINE_WIDTH * 0.6, zorder=1, alpha=0.7)
    ax.scatter(np.zeros(9), armR_normal, color=_ARMR_COLOR, s=16, zorder=3, label="normal")
    ax.scatter(np.ones(9), armR_swap, color=_ARMR_SWAP_COLOR, s=16, zorder=3, label="swapped")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["normal", "swapped"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("mean distance (mm)")
    ax.text(1.0, 1.02, f"$p={swap_result.p:.4f}$", transform=ax.transAxes,
            fontsize=FONT_SIZE - 1, ha="right", va="bottom")
    ax.text(-0.18, 1.0, "c", transform=ax.transAxes, fontsize=FONT_SIZE + 1, fontweight="bold")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    _save(fig, "fig4_reach.pdf")


# ---------------------------------------------------------------------------
# fig7_mutual_inhibition.pdf -- Section 4, "centered" vs "centered" + real
# D1<->D2 lateral inhibition. Paired per-seed scatter, same convention as
# fig4/fig5/fig6's panel-c swap/opponent comparisons. Reuses
# cart_pole/stats_report.py's own collect_condition() directly.
# ---------------------------------------------------------------------------
def fig7_mutual_inhibition(runs_dir=None, threshold=None):
    runs_dir = (runs_dir or cartpole_stats.DEFAULT_RUNS_DIR).resolve()
    threshold = threshold if threshold is not None else cartpole_stats.THRESHOLD
    run_dir_map = cartpole_stats.find_run_dirs(runs_dir)
    c_centered = cartpole_stats.collect_condition("centered", runs_dir, run_dir_map, threshold)
    c_mi = cartpole_stats.collect_condition("mutual_inhibition", runs_dir, run_dir_map, threshold)
    result = compare_paired(c_centered.values, c_mi.values)

    rows = (
        [{"condition": "centered", "seed": s, "return": v} for s, v in zip(cartpole_stats.SEEDS, c_centered.values)]
        + [{"condition": "mutual_inhibition", "seed": s, "return": v} for s, v in zip(cartpole_stats.SEEDS, c_mi.values)]
    )
    _save_csv(rows, ["condition", "seed", "return"], "fig7_mutual_inhibition.csv")

    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN * 0.75))
    n = len(c_centered.values)
    for a, b in zip(c_centered.values, c_mi.values):
        ax.plot([0, 1], [a, b], color="grey", linewidth=LINE_WIDTH * 0.6, zorder=1, alpha=0.7)
    ax.scatter(np.zeros(n), c_centered.values, color=OI_SKY, s=20, zorder=3)
    ax.scatter(np.ones(n), c_mi.values, color=OI_ORANGE, s=20, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["centered\n(no D1$\\leftrightarrow$D2 link)", "+mutual\ninhibition"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("return (greedy eval)")
    ax.text(1.0, 1.02, f"$p={result.p:.3f}$", transform=ax.transAxes,
            fontsize=FONT_SIZE - 1, ha="right", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig7_mutual_inhibition.pdf")


FIGURES = {
    "fig2_cartpole": fig2_cartpole,
    "fig3_weights": fig3_weights,
    "fig4_reach": fig4_reach,
    "fig7_mutual_inhibition": fig7_mutual_inhibition,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FIGURES), default=None,
                     help="build just one figure (default: all implemented ones)")
    args = ap.parse_args()

    targets = [args.only] if args.only else list(FIGURES)
    failed = []
    for name in targets:
        try:
            FIGURES[name]()
        except NotImplementedError as e:
            print(f"SKIPPED {name}: {e}")
            failed.append(name)
    if failed and not args.only:
        print(f"\n{len(failed)} figure(s) not yet buildable: {failed}")


if __name__ == "__main__":
    main()
