# experiments/stats — reproducible statistics for the paper

Every p-value quoted in the paper is regenerated from
`experiments/runs/` by one command, never hand-typed, and traceable back
to the exact run directories it came from.

## Pipeline

- `../../util/compare.py` — the test engine, shared with
  `cart_pole/stats_report.py` (it's shared infrastructure, not
  arm-specific). Implements both `compare()` (Mann-Whitney U, unpaired)
  and `compare_paired()` (Wilcoxon signed-rank, paired by seed).
- `per_seed_value.py` — the reducer: turns one eval run
  (directory or file) into its single per-seed number.
- `compare_runs.py` — a prefix-based CLI on top of both: point
  it at a runs folder and two name prefixes, it finds the matching
  seeds, reduces each, and runs the comparison. This is the tool every
  reported arm p-value traces back to.

CartPole has its own analog of this pipeline, built directly on
`util/compare.py`: see `cart_pole/stats_report.py`.

### Choosing paired vs. unpaired

Use `compare_paired()` (Wilcoxon signed-rank) when both conditions were
trained on matching seed numbers — it asks "does condition A
beat condition B seed-by-seed." Use `compare()` (Mann-Whitney U) when the
two groups are independent samples — it asks "are these two
unpaired sets of numbers different." The two tests can disagree on the
same data (e.g. a paired comparison significant at p=0.037 while the
unpaired comparison on the same numbers reads p=0.11): that is not a
contradiction, it is two different questions. The paired-vs-unpaired rule
used for each reported number is stated once in the paper, Section 4,
right after Table 2.

## Running it yourself

### 1. Run the unit tests (confirms the engine + reducer are correct)

```bash
cd /path/to/dopamine-credit-assignment
micromamba run -n snn310 python -m pytest util/test_compare.py experiments/stats/test_per_seed_value.py -v -s
```

20 tests total (7 in `util/test_compare.py`, 13 in `test_per_seed_value.py`),
all should pass. The `-s` flag matters: pytest hides `print()` output by
default, and `test_fully_separated_9v9_exact_and_asymptotic_diverge`
prints the actual p-values it computed:

```
exact:      p=4.11353e-05  method=exact  U=81.0
asymptotic: p=0.000412295  (same data, forced asymptotic)
```

### 2. Compare two conditions straight from a runs folder (no CSV needed)

This is the main entry point now. Point it at a runs folder and two name
prefixes — it finds every `<prefix>_seed<N>` directory (arm-style,
contains `stats.csv`) or `<prefix>_seed<N>_result.txt` file
(CartPole-style) under that folder, reduces each seed via the locked
greedy-eval convention, and runs the comparison:

```bash
cd experiments/stats
python compare_runs.py \
    --runs-dir ../runs/eval_firming9_armT --prefix-a armT \
    --runs-dir-b ../runs/eval_firming9_armR --prefix-b armR \
    --metric mean_distance
```

(`--runs-dir-b` is optional — omit it if both prefixes live in the same
folder.) It prints every seed's value as it finds it, then the comparison:
n, median, mean per group, U, p, method, and the pairwise win fraction —
plus the exact list of run paths used, for provenance.

Supported `--metric` values: for arm-style runs, `mean_distance`,
`success_rate`, `success@15mm`, `success@30mm`, `success@50mm`,
`success@100mm`, `success@150mm`; for CartPole, `return`.

### 3. Lower-level building blocks

```python
from per_seed_value import per_seed_value
from compare import compare

v = per_seed_value("../runs/eval_firming9_armT/armT_seed0", "mean_distance")
r = compare(group_a_values, group_b_values, alternative="two-sided")
```

`per_seed_value()` does the reduction (and caches it — see below);
`compare()` does the test on numbers you already have. `compare_runs.py`
is just these two wired together with prefix-matching glue.

### 4. Ad hoc CLI on a hand-built CSV

For one-off checks that aren't tied to a real run folder:

```bash
python ../../util/compare.py data.csv --group-col condition --value-col return \
    --a no_action --b with_action
```

## How `per_seed_value()` reduces a run, and the `greedy_eval_summary.csv` cache

**Arm-style** (a directory with `stats.csv`, one row per episode): episode
0 is the warmup episode and is always discarded (`n_warmup=1`, the locked
convention — don't change this without a reason); every metric is computed
from episode 1 onward. **CartPole-style** (a `{LABEL}_seed{N}_result.txt`
file): the reduction already happened at training time, it just reads the
`GREEDY_EVAL_mean=` field back out.

Either way, the computed value(s) get cached into a small
`greedy_eval_summary.csv` next to the raw data — inside the run directory
for arm-style runs, or `<name>_greedy_eval_summary.csv` next to the
`result.txt` for CartPole. This is a cache/audit trail, not a hand-edited
ledger: delete it and it's regenerated identically from the raw data;
pass `--force` (CLI) or `force=True` (Python) to recompute and overwrite
even if a cached summary already exists (do this after changing the
reduction logic itself, e.g. if `per_seed_value.py` changes).

## Method selection in `compare()` (why this isn't just `scipy.stats.mannwhitneyu`)

`compare()` picks the Mann-Whitney method explicitly, not via scipy's own
`method="auto"` (which only chooses exact when a group has <=8 samples) and
not via a bare try/except, which would silently land on the wrong method
for this project's typical 9-vs-9 comparisons. The rule, in
`_select_method()`:

- **exact** when both groups have <=25 samples AND the pooled sample has no
  tied values (measured: exact computation stays under 2ms even at 50
  samples/group on this machine, so 25 leaves real headroom over anything
  this project needs — nothing here goes past n=9/group).
- **asymptotic (continuity-corrected)** otherwise.

The method actually used is always returned (`result.method`), never left
implicit — the two methods can disagree by an order of magnitude on the
same data, so which one was used is part of what makes a reported p-value
meaningful.

On a fully-separated 9-vs-9 comparison, scipy's own default behavior
picks the asymptotic method and reports p=0.0004; this module's explicit
rule picks exact and reports p~=4.1e-5 for the identical data — same
conclusion, different digit. Every p-value quoted in the paper is
produced by this module's explicit rule, not scipy's default.

## Files

- `../../util/compare.py` — the test engine, shared with `cart_pole/`.
  `compare(a, b, alternative=...)` and `compare_paired(a, b, alternative=...)`
  are the functions everything else imports. Also has a standalone CLI
  for ad hoc CSV checks.
- `../../util/test_compare.py` — unit tests for the engine (method
  selection, the exact/asymptotic divergence, edge cases).
- `per_seed_value.py` — the reducer. `per_seed_value(run_path, metric, ...)`
  handles both the arm's directory/`stats.csv` runs and CartPole's flat
  `result.txt` files, with `greedy_eval_summary.csv` caching.
- `test_per_seed_value.py` — unit tests for the reducer (warmup-episode
  discard, both input formats, summary-file caching/force-recompute, error
  cases).
- `compare_runs.py` — the prefix-based CLI: runs folder + two prefixes in,
  full comparison out. No registry, no YAML — for interactive use while
  data is still being generated.
- `linear_baseline_control.py` — the linear-baseline control (paper
  Section 5.1, "Control 1" in the Methods). Run standalone, before any
  training: fits one global linear map `dq = W @ e` against a
  resolved-rate oracle and checks whether that trivial fixed map alone
  already solves the continuous reach. It doesn't — 3.3% success,
  indistinguishable from untrained (1.7%), versus 86.7% for the true
  `q`-dependent oracle — which is what rules out a disguised linear fit
  as the explanation for Arm R's result. Pure numpy, no NEST; run it with
  `micromamba run -n snn310 python linear_baseline_control.py`.

Run the tests after touching `compare.py` or `per_seed_value.py`, before
trusting anything downstream of them:

```bash
micromamba run -n snn310 python -m pytest util/test_compare.py experiments/stats/ -v -s
```

## Figures

`make_figures.py` turns run data into each figure's underlying numbers —
same rule as the stats: nothing drawn by hand, each figure reuses the
*same* run reader/registry as its section's stats script, so the exported
numbers and the p-value quoted next to them always come from the identical
run directories.

Output is now split in two:
- `paper/figures/data/*.csv` — the real deliverable, one tidy CSV per
  dataset. This is the second source of truth bound to the paper: it's
  committed here **and** copied by hand into the paper repo
  (`event-based-snn/paper/figures/data/`), where a separate,
  dependency-free script (`paper/figures/make_figures.py` there) reads
  these same CSVs and renders the actual camera-ready PDFs the manuscript
  uses. That keeps the paper repo self-contained — it can regenerate every
  data-driven figure without this repo, its run directories, or its stats
  tooling.
- `paper/figures/*.pdf` — still produced here too, but now just a local
  debug/preview rendering (a quick sanity check that the exported numbers
  look right). These are **not** copied into the paper repo any more and
  are not the paper's source of truth.

```bash
cd experiments/stats
micromamba run -n snn310 python make_figures.py            # all implemented figures
micromamba run -n snn310 python make_figures.py --only fig2_cartpole
```

Style, fixed once and used everywhere in that file: vector PDF, ~3.4in
single-column / ~7.0in full-width, 8.5pt font, ~1.2pt lines, Okabe-Ito
colour-blind-safe palette (same colour for the same thing across figures),
panel labels a/b/c with no in-figure title, tight layout, white background.
Curves are the mean over seeds with a **1-standard-deviation** shaded band
(not IQR — stated once, applied everywhere, see `BAND_IS_STD` in the file).

- `fig2_cartpole.pdf` (Section 4) — imports `cart_pole/stats_report.py`'s own
  `CONDITIONS`/`find_run_dirs`/`collect_condition` directly (not copied), so
  it can never drift from `stats_report.md`'s numbers.
- `fig4_reach.pdf` (Section 5a, reach-vec vs. raw-target + swap) —
  panels (a)/(b) read the raw 400-episode training curves from the
  9-seed `p3_reach_baseline_7x7`/`p3_reach_baseline_reachvec_7x7` run
  dirs directly; panel (c) calls `per_seed_value()` + `compare_paired()`
  on `experiments/runs/eval_firming9_arm{R,Rswap}`, the same calls that
  produce Table 3's numbers (two-sided, p=0.074 distance / p=0.46
  success@15mm for Arm R vs Arm T; p=0.0039 for the swap control), so the
  figure's own `p=0.0039` annotation can never drift from the table.
- `fig3_weights.pdf` (Section 5d, centering) — built from the
  9-seed `p3_reach_baseline_nocenter.py` run matched against the centered
  `p3_reach_baseline_7x7` batch. D1 weight std doesn't collapse without
  centering, but its cross-seed spread is ~8x tighter when centered
  (±0.14 vs ±1.04), and 2 of 9 uncentered seeds diverge to
  catastrophic final distance (648mm, 455mm) that no centered seed
  matches. Paired distance comparison is not significant (p=0.570).

## What the run-folder names mean

Every condition name under `experiments/runs/` is built compositionally out
of a small set of tokens. This section explains the tokens once, then gives
one line per full condition and per eval-folder group — "which folder is
which experiment," traced from each config's own docstring, not guessed.

### Naming tokens (shared across condition names)

| Token | Meaning |
|---|---|
| `p3_reach_baseline` | The P2/P3-locked raw-target baseline: full-workspace continuous reach, cortex straight to D1/D2, Python (non-spiking) TD critic. "Arm T" in the paper. |
| `reachvec` | Swaps the raw-target input stream for the precomputed reach vector `fk(joints)-target`, instead of making cortex learn that subtraction itself. "Arm R" in the paper. |
| `multigoal2` / `multigoal3` | Discrete goal-selection variant: 2 or 3 fixed, far-apart goals instead of one continuous target, so homing to a single fixed point can't solve the task. |
| `contextgated` | Per-goal baseline and eligibility accumulator in `BatchedD1Learning`, instead of one baseline/accumulator pooled across every goal — only applies to the `multigoal3` conditions. |
| `nocenter` | Presynaptic centering (`actor_batched_d1_presyn_baseline_correction`) turned off — the Section 5d ablation. |
| `smalltarget5mm` / `smalltarget25mm` / `smalltarget50mm` | Target confined to a small fixed box of that half-width, instead of the full workspace. |
| `batched` | Trains D1 with `BatchedD1Learning` (a Python reimplementation of dopamine-modulated STDP, committed per episode batch) instead of the native NEST `stdp_dopamine_synapse`. |
| `gainfield` | A fixed conjunctive gain-field basis (`place(q) + place(q)*target`) feeds D1/D2 directly, bypassing the recurrent cortex. |
| `bypass_cortex` | The combined-encoder signal feeds D1/D2 directly, skipping the recurrent cortex layer entirely (the cortex's own representation was found to collapse this signal). |
| `fixedhidden` | A frozen random hidden layer (N=750) sits between the encoder and D1/D2 — depth with no plasticity in that layer. |
| `hidden` (as a suffix, e.g. `*_smalltarget5mm_hidden`) | Adds that same kind of frozen fixed-hidden layer on top of whichever base path (cortex or gain-field) the rest of the name describes. |
| `burst` | The burst-dependent-plasticity hidden layer (cortex → H → D1/D2, Izhikevich bursting neurons + feedback-alignment broadcast) — a *trained*, not frozen, hidden layer. Section 7.2. |
| `nativestdp` | Uses the native NEST `stdp_dopamine_synapse` instead of `BatchedD1Learning` — the validation control for the `batched` conditions. |
| `spikingcritic` | Replaces the Python TD critic with a genuinely spiking value critic. Section 7.1. |
| `spikingcritic_probe` | Trains the spiking critic *passively* first (the actor still learns from the Python critic's signal) to check the spiking critic's own fidelity before the real closed-loop test (the plain `spikingcritic` config). |

### The 24 training conditions

| Condition | Config | What it tests |
|---|---|---|
| `p3_reach_baseline_7x7` | `p3_reach_baseline.py` | The locked baseline itself — Arm T, raw-target input, full workspace. |
| `p3_reach_baseline_reachvec_7x7` | `p3_reach_baseline_reachvec.py` | Arm R — does feeding the precomputed reach vector (instead of raw target) help, and does the actor genuinely follow it (swap control)? |
| `p3_reach_baseline_nocenter_7x7` | `p3_reach_baseline_nocenter.py` | Presynaptic-centering ablation (Section 5d) — everything else held identical to the baseline. |
| `p3_reach_baseline_multigoal2_batched_7x7` | `p3_reach_baseline_multigoal2_batched.py` | First discrete-goal test: does the actor use the target when a fixed point can no longer solve the task (2 goals)? |
| `p3_reach_baseline_multigoal3_batched_7x7` | `p3_reach_baseline_multigoal3_batched.py` | 3-goal follow-up to multigoal2, the stronger claim (chance = 33% instead of 50%) — Table 4b's "Baseline" row. |
| `p3_reach_baseline_multigoal3_batched_contextgated_7x7` | `p3_reach_baseline_multigoal3_batched_contextgated.py` | Does giving each goal its own baseline/eligibility accumulator (instead of one pooled across goals) improve the 3-goal result? Table 4b's "Context-gated" row. |
| `p3_reach_baseline_smalltarget5mm_7x7` | `p3_reach_baseline_smalltarget5mm.py` | Does the *standard* architecture (real cortex, not bypassed, no gain-field, native STDP) learn a small (±5mm) fixed-region reach on its own? |
| `p3_reach_baseline_smalltarget5mm_batched_7x7` | `p3_reach_baseline_smalltarget5mm_batched.py` | Same question, with `BatchedD1Learning` instead of native STDP — the single best small-target result in the project (64.7mm/45%). |
| `p3_reach_baseline_smalltarget25mm_batched_7x7` | `p3_reach_baseline_smalltarget25mm_batched.py` | Degradation-curve point 2 of 3: does the box need to be small enough for fixed-point homing to work, or does the actor still learn at ±25mm? Backs the Section 5.3 target-ablation control. |
| `p3_reach_baseline_smalltarget50mm_batched_7x7` | `p3_reach_baseline_smalltarget50mm_batched.py` | Degradation-curve point 3 of 3, ±50mm. |
| `p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe_7x7` | `p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py` | Phase 1: trains the spiking critic passively alongside Cell 1's actor (still on the Python critic's signal) to check its own fidelity first. |
| `p3_reach_baseline_smalltarget5mm_batched_spikingcritic_7x7` | `p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py` | Phase 2: Cell 1's actor now trained from the spiking critic's own δ — the real closed-loop test. Table 5's "Cell 1, spiking" row. |
| `p3_reach_baseline_smalltarget5mm_hidden_batched_7x7` | `p3_reach_baseline_smalltarget5mm_hidden_batched.py` | Cell 2: a frozen fixed-hidden layer after the real cortex, trained with `BatchedD1Learning` — does added depth help the small-target task? |
| `p3_reach_bypass_cortex_7x7` | `p3_reach_bypass_cortex.py` | Skips the recurrent cortex entirely for the actor's pathway (the cortex representation was found to collapse the place-coded signal). |
| `p3_reach_fixedhidden_7x7` | `p3_reach_fixedhidden.py` | A frozen random hidden layer (N=750) between the encoder and D1/D2, cortex bypassed, measuring distance directly. |
| `p3_reach_gainfield_7x7` | `p3_reach_gainfield.py` | Fixed gain-field basis feeding D1/D2 directly — the architecture that gave the best offline decode fidelity before this training config was built. |
| `p3_reach_gainfield_nativestdp_7x7` | `p3_reach_gainfield_nativestdp.py` | Validates the gain-field result against the *native* NEST STDP synapse (every arm result so far had only used the batched Python reimplementation). |
| `p3_reach_gainfield_smalltarget5mm_7x7` | `p3_reach_gainfield_smalltarget5mm.py` | Does shrinking the target (not the representation) let the local rule learn, on the gain-field/native-STDP path — Arm B's native-STDP reference. |
| `p3_reach_gainfield_smalltarget5mm_batched_7x7` | `p3_reach_gainfield_smalltarget5mm_batched.py` | Arm A — same small-target task, `BatchedD1Learning` instead of native STDP (122.1mm/10%, the other positive small-target result). |
| `p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe_7x7` | `p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py` | The spiking-critic-hardening fallback's Phase 1 (passive), for Arm A instead of Cell 1. |
| `p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_7x7` | `p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py` | The fallback's Phase 2 (real closed-loop test) for Arm A. Table 5's "Arm A, spiking" row. |
| `p3_reach_gainfield_smalltarget5mm_hidden_7x7` | `p3_reach_gainfield_smalltarget5mm_hidden.py` | Arm C — a frozen fixed-hidden layer stacked after the gain-field relay, native STDP. |
| `p3_reach_gainfield_smalltarget5mm_hidden_batched_7x7` | `p3_reach_gainfield_smalltarget5mm_hidden_batched.py` | Arm C with `BatchedD1Learning` instead of native STDP — completes the {cortex/bypass} × {hidden/no-hidden} × {native/batched} grid. |
| `p3_reach_burst_7x7` | `p3_reach_burst.py` | Path B — the burst-dependent-plasticity hidden layer (a *trained* hidden layer, not frozen), replacing the flat cortex→D1/D2 projection. Section 7.2. |

### The 7 eval-folder groups

These are second-pass, restore-and-evaluate runs (see `README.md`'s "Running
evaluations" section), not training runs — each one re-evaluates an
already-trained checkpoint from one of the 24 conditions above.

| Folder prefix | Meaning |
|---|---|
| `eval_firming9_armT` | 9-seed greedy-eval confirmation of Arm T (`p3_reach_baseline_7x7`'s checkpoints) — backs Table 3's "Arm T" row. |
| `eval_firming9_armR` | Same, for Arm R (`p3_reach_baseline_reachvec_7x7`'s checkpoints) — backs Table 3's "Arm R" row. |
| `eval_firming9_armRswap` | Arm R's same 9 checkpoints, re-evaluated with a decoy target fed to perception while scoring against the real one — backs Table 3's "Arm R (swap)" row and the p=0.0039 swap-control result. |
| `multigoal3_eval9_seed*` | 9-seed greedy eval of the 3-goal baseline (`p3_reach_baseline_multigoal3_batched_7x7`) — backs Table 4b's "Baseline" row. |
| `multigoal3_swap9_seed*` | Same checkpoints, with the next goal in rotation fed to perception instead of the commanded one (`cfg.target_swap_eval=True`) — backs Table 4b's swap column for "Baseline." |
| `multigoal3cg_eval9_seed*` | 9-seed greedy eval of the 3-goal context-gated condition — backs Table 4b's "Context-gated" row. |
| `multigoal3cg_swap9_seed*` | Same, swap control — backs Table 4b's swap column for "Context-gated." |

## Exact reproduction commands

Every number and figure in this pipeline traces back to a specific
`experiments/runs/` folder (the mission stated at the top of this file).
These two tables are that trace made literal: the exact command that
produces each run currently on disk, generated (not hand-typed) by
resolving each condition/seed against the actual folder it lives in.

### Exact commands for every existing training run (`p3_reach_*`)

All 24 `p3_reach_*` ("P2/P3 locked") conditions under `experiments/runs/`
ultimately call `p3_reach_baseline.py`'s own `Config()` (directly or through
a chain of `importlib`-loaded parents that just rename `cfg.name`), which is
where the `P12_SEED` env var is read — same convention and same
`--seed`-vs-`P12_SEED` caveat as the eval commands below (see also
`README.md`'s own "Running evaluations" section for the general pattern
and the `--seed`-vs-`P12_SEED` gotcha explained in prose). Every condition
here trains for `reach_state.py`'s default `num_episodes = 400` (none of
them override it), so no `--episodes` flag is needed; training itself (not
eval) never takes `--restore-checkpoint` either. This table was generated by
resolving, for every seed of every condition, the exact
`experiments/runs/` folder it currently lives in — where a seed was trained
more than once (the same 3 seeds flagged below, from the superseded n=3
multigoal investigations), the folder listed is the one the eval table's
checkpoints point to.

<details>
<summary>114 commands, one per (condition, seed) — click to expand</summary>

**`p3_reach_baseline_7x7`** (`experiments/configs/p3_reach_baseline.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 1 | `experiments/runs/20261022-122939_p3_reach_baseline_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 2 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 3 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 4 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 5 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 6 | `experiments/runs/20261022-122939_p3_reach_baseline_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 7 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |
| 8 | `experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline.py` |

**`p3_reach_baseline_multigoal2_batched_7x7`** (`experiments/configs/p3_reach_baseline_multigoal2_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-144041_p3_reach_baseline_multigoal2_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal2_batched.py` |
| 1 | `experiments/runs/20261021-144041_p3_reach_baseline_multigoal2_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal2_batched.py` |
| 5 | `experiments/runs/20261021-144041_p3_reach_baseline_multigoal2_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal2_batched.py` |

**`p3_reach_baseline_multigoal3_batched_7x7`** (`experiments/configs/p3_reach_baseline_multigoal3_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 1 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 2 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 3 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 4 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 5 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 6 | `experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 7 | `experiments/runs/20261024-025742_p3_reach_baseline_multigoal3_batched_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |
| 8 | `experiments/runs/20261024-025759_p3_reach_baseline_multigoal3_batched_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched.py` |

**`p3_reach_baseline_multigoal3_batched_contextgated_7x7`** (`experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 1 | `experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 2 | `experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 3 | `experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 4 | `experiments/runs/20261024-025806_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 5 | `experiments/runs/20261024-025829_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 6 | `experiments/runs/20261024-025851_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 7 | `experiments/runs/20261024-025929_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |
| 8 | `experiments/runs/20261024-025945_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated.py` |

**`p3_reach_baseline_nocenter_7x7`** (`experiments/configs/p3_reach_baseline_nocenter.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 1 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 2 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 3 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 4 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 5 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 6 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 7 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |
| 8 | `experiments/runs/20261024-034111_p3_reach_baseline_nocenter_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_nocenter.py` |

**`p3_reach_baseline_reachvec_7x7`** (`experiments/configs/p3_reach_baseline_reachvec.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 1 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 2 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 3 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 4 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 5 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 6 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 7 | `experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |
| 8 | `experiments/runs/20261022-122939_p3_reach_baseline_reachvec_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec.py` |

**`p3_reach_baseline_smalltarget25mm_batched_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget25mm_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-120942_p3_reach_baseline_smalltarget25mm_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget25mm_batched.py` |
| 1 | `experiments/runs/20261021-120942_p3_reach_baseline_smalltarget25mm_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget25mm_batched.py` |
| 5 | `experiments/runs/20261021-120942_p3_reach_baseline_smalltarget25mm_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget25mm_batched.py` |

**`p3_reach_baseline_smalltarget50mm_batched_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget50mm_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-124509_p3_reach_baseline_smalltarget50mm_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget50mm_batched.py` |
| 1 | `experiments/runs/20261021-124509_p3_reach_baseline_smalltarget50mm_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget50mm_batched.py` |
| 5 | `experiments/runs/20261021-124509_p3_reach_baseline_smalltarget50mm_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget50mm_batched.py` |

**`p3_reach_baseline_smalltarget5mm_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget5mm.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-145314_p3_reach_baseline_smalltarget5mm_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm.py` |
| 1 | `experiments/runs/20261020-145314_p3_reach_baseline_smalltarget5mm_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm.py` |
| 5 | `experiments/runs/20261020-145314_p3_reach_baseline_smalltarget5mm_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm.py` |

**`p3_reach_baseline_smalltarget5mm_batched_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget5mm_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-203423_p3_reach_baseline_smalltarget5mm_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched.py` |
| 1 | `experiments/runs/20261020-203423_p3_reach_baseline_smalltarget5mm_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched.py` |
| 5 | `experiments/runs/20261020-203423_p3_reach_baseline_smalltarget5mm_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched.py` |

**`p3_reach_baseline_smalltarget5mm_batched_spikingcritic_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-012249_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py` |
| 1 | `experiments/runs/20261021-012249_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py` |
| 5 | `experiments/runs/20261021-012249_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic.py` |

**`p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-004802_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py` |
| 1 | `experiments/runs/20261021-004802_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py` |
| 5 | `experiments/runs/20261021-004802_p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_batched_spikingcritic_probe.py` |

**`p3_reach_baseline_smalltarget5mm_hidden_batched_7x7`** (`experiments/configs/p3_reach_baseline_smalltarget5mm_hidden_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-210627_p3_reach_baseline_smalltarget5mm_hidden_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_hidden_batched.py` |
| 1 | `experiments/runs/20261020-210627_p3_reach_baseline_smalltarget5mm_hidden_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_hidden_batched.py` |
| 5 | `experiments/runs/20261020-210627_p3_reach_baseline_smalltarget5mm_hidden_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_smalltarget5mm_hidden_batched.py` |

**`p3_reach_burst_7x7`** (`experiments/configs/p3_reach_burst.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261017-112135_p3_reach_burst_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_burst.py` |
| 1 | `experiments/runs/20261017-112135_p3_reach_burst_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_burst.py` |
| 2 | `experiments/runs/20261017-112135_p3_reach_burst_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_burst.py` |

**`p3_reach_bypass_cortex_7x7`** (`experiments/configs/p3_reach_bypass_cortex.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261017-225053_p3_reach_bypass_cortex_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 1 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 2 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 3 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 4 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 5 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 6 | `experiments/runs/20261017-225052_p3_reach_bypass_cortex_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 7 | `experiments/runs/20261017-225053_p3_reach_bypass_cortex_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |
| 8 | `experiments/runs/20261017-225053_p3_reach_bypass_cortex_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_bypass_cortex.py` |

**`p3_reach_fixedhidden_7x7`** (`experiments/configs/p3_reach_fixedhidden.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 1 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 2 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed2_seed3` | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 3 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed3_seed4` | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 4 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed4_seed5` | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 5 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 6 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed6_seed7` | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 7 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed7_seed8` | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |
| 8 | `experiments/runs/20261018-091026_p3_reach_fixedhidden_7x7_seed8_seed9` | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_fixedhidden.py` |

**`p3_reach_gainfield_7x7`** (`experiments/configs/p3_reach_gainfield.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261019-173654_p3_reach_gainfield_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield.py` |
| 1 | `experiments/runs/20261019-173655_p3_reach_gainfield_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield.py` |
| 5 | `experiments/runs/20261019-173656_p3_reach_gainfield_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield.py` |

**`p3_reach_gainfield_nativestdp_7x7`** (`experiments/configs/p3_reach_gainfield_nativestdp.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261019-190719_p3_reach_gainfield_nativestdp_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_nativestdp.py` |
| 1 | `experiments/runs/20261019-190720_p3_reach_gainfield_nativestdp_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_nativestdp.py` |
| 5 | `experiments/runs/20261019-190721_p3_reach_gainfield_nativestdp_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_nativestdp.py` |

**`p3_reach_gainfield_smalltarget5mm_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-114146_p3_reach_gainfield_smalltarget5mm_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm.py` |
| 1 | `experiments/runs/20261020-114146_p3_reach_gainfield_smalltarget5mm_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm.py` |
| 5 | `experiments/runs/20261020-114146_p3_reach_gainfield_smalltarget5mm_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm.py` |

**`p3_reach_gainfield_smalltarget5mm_batched_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched.py` |
| 1 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched.py` |
| 5 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched.py` |

**`p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-030029_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py` |
| 1 | `experiments/runs/20261021-030029_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py` |
| 5 | `experiments/runs/20261021-030029_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic.py` |

**`p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261021-015754_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py` |
| 1 | `experiments/runs/20261021-015754_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py` |
| 5 | `experiments/runs/20261021-015754_p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_batched_spikingcritic_probe.py` |

**`p3_reach_gainfield_smalltarget5mm_hidden_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_hidden_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden.py` |
| 1 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_hidden_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden.py` |
| 5 | `experiments/runs/20261020-145347_p3_reach_gainfield_smalltarget5mm_hidden_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden.py` |

**`p3_reach_gainfield_smalltarget5mm_hidden_batched_7x7`** (`experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden_batched.py`)

| Seed | Output folder | Command |
|---|---|---|
| 0 | `experiments/runs/20261020-203423_p3_reach_gainfield_smalltarget5mm_hidden_batched_7x7_seed0_seed1` | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden_batched.py` |
| 1 | `experiments/runs/20261020-203423_p3_reach_gainfield_smalltarget5mm_hidden_batched_7x7_seed1_seed2` | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden_batched.py` |
| 5 | `experiments/runs/20261020-203423_p3_reach_gainfield_smalltarget5mm_hidden_batched_7x7_seed5_seed6` | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_gainfield_smalltarget5mm_hidden_batched.py` |

</details>

### Exact commands for every existing eval folder

The table below gives the literal, already-verified command that reproduces
each of the 63 per-seed eval folders currently under `experiments/runs/`
(the ones in the "must keep" category — hardcoded into
`experiments/stats/make_figures.py` or cited by name in this project's
internal investigation notes). Each command was checked by
actually re-running it and diffing the output against the existing folder —
`eval_firming9_armT/armT_seed0` reproduced bit-for-bit identical
`final_d_mm` values across all 21 episodes. Where a seed's training run was
re-launched more than once (seeds 0, 1, 5 of the two `multigoal3*` batches,
left over from an earlier, superseded n=3 investigation that was re-run from
scratch at full 9-seed power), the checkpoint below is
confirmed identical either way: both candidate runs converge to the same
weights (this project uses fixed, not random, seeds per logical seed
number), so it doesn't matter which one is cited.

<details>
<summary>63 commands, one per eval folder (click to expand)</summary>

**`eval_firming9_armT`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122939_p3_reach_baseline_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122939_p3_reach_baseline_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armT/armT_seed8` |

**`eval_firming9_armR`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval.py --restore-checkpoint experiments/runs/20261022-122939_p3_reach_baseline_reachvec_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armR/armR_seed8` |

**`eval_firming9_armRswap`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122938_p3_reach_baseline_reachvec_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_reachvec_eval_swap.py --restore-checkpoint experiments/runs/20261022-122939_p3_reach_baseline_reachvec_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/eval_firming9_armRswap/armRswap_seed8` |

**`multigoal3_eval9`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-025742_p3_reach_baseline_multigoal3_batched_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval.py --restore-checkpoint experiments/runs/20261024-025759_p3_reach_baseline_multigoal3_batched_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_eval9_seed8` |

**`multigoal3_swap9`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-022509_p3_reach_baseline_multigoal3_batched_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-025742_p3_reach_baseline_multigoal3_batched_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_eval_swap.py --restore-checkpoint experiments/runs/20261024-025759_p3_reach_baseline_multigoal3_batched_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3_swap9_seed8` |

**`multigoal3cg_eval9`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-025806_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-025829_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-025851_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-025929_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval.py --restore-checkpoint experiments/runs/20261024-025945_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_eval9_seed8` |

**`multigoal3cg_swap9`**

| Seed | Command |
|---|---|
| 0 | `P12_SEED=0 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed0_seed1/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed0` |
| 1 | `P12_SEED=1 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-015216_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed1_seed2/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed1` |
| 2 | `P12_SEED=2 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed2_seed3/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed2` |
| 3 | `P12_SEED=3 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-015217_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed3_seed4/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed3` |
| 4 | `P12_SEED=4 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-025806_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed4_seed5/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed4` |
| 5 | `P12_SEED=5 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-025829_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed5_seed6/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed5` |
| 6 | `P12_SEED=6 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-025851_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed6_seed7/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed6` |
| 7 | `P12_SEED=7 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-025929_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed7_seed8/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed7` |
| 8 | `P12_SEED=8 micromamba run -n snn310 python model/main.py --config experiments/configs/p3_reach_baseline_multigoal3_batched_contextgated_eval_swap.py --restore-checkpoint experiments/runs/20261024-025945_p3_reach_baseline_multigoal3_batched_contextgated_7x7_seed8_seed9/final_checkpoint.pkl --episodes 21 --out-dir experiments/runs/multigoal3cg_swap9_seed8` |

</details>
