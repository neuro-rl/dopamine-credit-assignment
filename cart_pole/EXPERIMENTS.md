# CartPole experiments

A companion/validation task to the main reaching-arm project (see the
top-level `README.md`): a small NEST network (36-neuron cortex, 10 D1 + 10
D2 striatal neurons, a 14-feature linear TD(0) critic) learns CartPole-v1
via dopamine-modulated, presynaptically-centered plasticity with
action-identity-aware credit. Full mechanism writeup and the raw
investigation history are in `findings_new.md`; this file documents only
the five **locked** conditions actually run and reported, and how to
reproduce them.

## The five locked conditions (the "E1 ablation")

All five share one network/critic (`model.py`, `state_encoder.py`,
`td_critic.py`) and differ only in four orthogonal flags, set in each
condition's `configs/<label>.py`. This is the paper's Table 2 (Section 4):

| condition | config | action_identity | action_centered | centered | population_readout |
|---|---|---|---|---|---|
| neither | `configs/noCentNoAction.py` | False | — | False | False |
| +raw action | `configs/uncentered_rawaction.py` | True | False | False | False |
| +action-centered | `configs/uncentered.py` | True | True | False | False |
| +presyn.-centered | `configs/centered.py` | True | True | True | False |
| +population readout | `configs/vec2d_proj.py` | True | True | True | True |

- **action_identity**: credit = `dev * eps` (eps carries which action was
  actually taken vs. the network's own lean) instead of `dev` alone.
- **action_centered**: when action_identity is on, subtract a smooth
  surrogate of the expected action, `eps = a - tanh(g/g_scale)`, instead
  of using the raw action `a` on its own (`action_centered` defaults to
  `True` in `run.py`/`model.py`; `uncentered_rawaction` is the one config
  that turns it off, isolating this term's own effect against
  `uncentered`, which has everything else held fixed).
- **centered**: the presynaptic term in the weight update is
  `x_j - x_bar_j` (activity relative to each cortex neuron's own running
  baseline) instead of raw `x_j` — the fix for D1's weight-std collapse
  during training.
- **population_readout**: per-D1/D2-neuron preferred-direction decode and
  credit (a population-vector projection) instead of one scalar
  total-count decode/credit shared by every neuron.

## Extra condition: real D1<->D2 inhibition

All five conditions above realise the D1$-$D2 "opponent" push purely
computationally (no connection between the two populations at all; the
arm's actor has always had a real one). `configs/mutual_inhibition.py` is
"centered" plus real, static, bidirectional, all-to-all lateral
inhibition between D1 and D2 (weight `-30`, calibrated against a
bistability cliff found at `~-200` — see
`findings/MUTUAL_INHIBITION_FINDING.md`). Not part of the orthogonal
3-flag chain above; `stats_report.py` reports it separately, paired
against "centered" (same seed = same starting network, verified). Result:
null (p=0.844, learn_fraction identical 6/9 both) — see the finding doc
and paper Section 4.1 for the full 9-seed numbers and the 3-seed quick
gate that did not survive.

## Locked 9-seed results

**The numbers below are never hand-typed.** `stats_report.py` reads every
run directly from `experiments_cartpole/` and writes `stats_report.md` +
`stats_report.json`; those two files are the only source for the numbers
quoted in the paper's Section 4 (and this section — regenerate, then copy
from `stats_report.md`, never type a p-value by hand):

```bash
cd cart_pole
micromamba run -n snn310 python stats_report.py
```

It reports, per condition: n, median, mean, std, and `learn_fraction` (the
share of the 9 seeds with return above a fixed threshold T=30 — the task
is bimodal, each seed lands near the ~10-step floor or well above it, so
median/learn_fraction carry more information than mean±std alone). Every
adjacent-row comparison in the chain (neither-vs-raw-action,
raw-action-vs-action-centered, action-centered-vs-presyn-centered,
presyn-centered-vs-population) is reported as a **paired** Wilcoxon
signed-rank test, by seed — not unpaired Mann-Whitney, since pairing holds
across the whole chain: `build_network(seed=SEED)` draws cortex/D1/D2
initial weights before any condition-specific flag is read, and `run.py`
pins the Gymnasium env's RNG to the same `SEED` too, so the same seed
means the same starting network and environment stream in every
condition (verified empirically: `W_d1_init_std` matches across all 5
conditions, every seed). Every number is traceable back to its exact run
directories (listed in the report's provenance section).

Supersedes the raw `findings_new.md` 2x2-ablation table: that table mixed
the warmup confound described above (see "The five locked conditions") and
pre-dates this script. `stats_report.md` (regenerate it — do not read
stale numbers out of this file) is authoritative going forward.

## Running them

Each condition, one seed at a time, through the single shared engine:

```bash
cd cart_pole
micromamba run -n snn310 python run.py --config configs/noCentNoAction.py     --seed 1
micromamba run -n snn310 python run.py --config configs/uncentered_rawaction.py --seed 1
micromamba run -n snn310 python run.py --config configs/uncentered.py         --seed 1
micromamba run -n snn310 python run.py --config configs/centered.py          --seed 1
micromamba run -n snn310 python run.py --config configs/vec2d_proj.py        --seed 1
```

The locked 9-seed sweep uses seeds 1-9 (matching `findings_new.md`'s own
convention), 1000 training episodes + 50-episode critic warm-up (matched
across all five conditions, each config's own default — `centered`
previously used warmup=0, which confounded the uncentered-vs-centered
comparison, fixed 2026-10-23) + 30-episode greedy eval.

**Training-time scores are not comparable to the reported numbers.**
`anneal=False` by default, so `epsilon` stays flat at `EPSILON_EXPLORE=0.3`
for every training episode, including the last one — 30% of actions are
random, forever. CartPole is physically fragile, so that constant noise
caps training scores well below what the same weights can do once
exploration is switched off. Confirmed directly on one trained checkpoint,
same weights, only epsilon varied: mean return 340.6 at epsilon=0.0 (greedy
eval's setting), 202.2 at epsilon=0.1, 51.1 at epsilon=0.3 (training's own
setting) — matching that run's own training-time quartile means (18.6,
17.4, 29.8, 31.9). Every number in `stats_report.md`/this file is the
30-episode **greedy eval** (`epsilon=0.0`), never a training-time score.

Useful overrides: `--seed N`, `--episodes N`, `--warmup N`,
`--eval-episodes N`, `--wmax X` (only meaningful for `vec2d_proj`, see its
config's docstring).

## Where runs land

Each run writes `experiments_cartpole/<timestamp>_<label>_seed<N>/`,
containing:

- `stats.csv` — one row per episode (`episode, phase, score, epsilon`),
  `phase` is `warmup`/`train`/`greedy_eval`
- `summary.pdf` — training curve + greedy-eval mean (vector PDF)
- `experiment.txt` — `experiment_id` + every run parameter
- `config_used.py` — a copy of the config file the run was launched with
- `final_checkpoint.pkl` — the trained `W_d1`/`W_d2`/`x_bar`/`w_critic`
- `greedy_eval_summary.csv` — the GREEDY_EVAL_mean, as a small CSV (same
  value as the `GREEDY_EVAL_mean=` field in the printed summary line — no
  warmup-episode discard here, unlike the arm's own convention; see
  `run_io.py`'s docstring for why that doesn't apply to CartPole)
- `debug/` — unused today (CartPole produces no PDFs/per-episode NPZ
  dumps), kept for structural parity with the arm's run folders

`experiments_cartpole/` is git-tracked — run dirs here are the reported,
locked results, not scratch output.

## Files

- `model.py` — network builder + flag-parameterized training/eval loop
- `state_encoder.py` — CartPole state -> spike-rate encoding
- `td_critic.py` — the 14-feature linear TD(0) critic
- `run.py` — the CLI engine (what you actually run)
- `run_io.py` — writes the run folder
- `configs/*.py` — the five locked conditions (see table above), plus
  `mutual_inhibition.py` (see "Extra condition" above)
- `findings_new.md` — full investigation history and statistical analysis
