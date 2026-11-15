#!/usr/bin/env python3
"""cart_pole/run.py -- single generic train+eval engine for the CartPole
dopamine-STDP credit-assignment conditions, mirroring the arm's own
experiments/run.py -> model/main.py split: this script is the engine,
cart_pole/model.py is the (single, shared) network+loop, and
cart_pole/configs/<label>.py files are the configs -- which of
model.py's three flags (action_identity/centered/population_readout) +
hyperparams make up each locked condition.

Replaces the four near-duplicate nest_cartpole_{noCentNoAction,
uncentered,centered,vec2d_proj}_test.py modules (each a full copy of the
network/critic/loop with one or two lines different) with one model.py +
four ~20-line config files -- a new condition is now a new config file,
not a 300-line copy-paste.

Output: experiments_cartpole/<timestamp>_<label>_seed<N>/ (via run_io.py),
plus the same GREEDY_EVAL summary line printed to stdout.

Usage (mirrors experiments/run.py):
    python run.py --config configs/uncentered.py --seed 4
    python run.py --config configs/vec2d_proj.py --seed 1 --episodes 200 --wmax 800
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import nest

import model as mod
import run_io


def load_config(config_path):
    # equivalent to import config_path_as_module; return module.Config(),
    # but working with an arbitrary file path instead of
    # needing the file to be on sys.path/have a package name.
    spec = importlib.util.spec_from_file_location("cartpole_run_config", config_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.Config()


def eps_schedule(cfg, ep, n_ep):
    if not cfg.anneal:
        return mod.EPSILON_EXPLORE
    anneal_eps = n_ep // 2
    if ep >= anneal_eps: # annealing is done, half of the training is passed
        return cfg.eps_end
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (ep / anneal_eps)


def measure_drive(net, env, n_windows=30):
    d1_ids = net["pop_D1"].tolist()
    d2_ids = net["pop_D2"].tolist()
    state, _ = env.reset()
    rates_d1, rates_d2 = [], []
    for _ in range(n_windows):
        obs6 = mod.extended_obs(mod.normalize_state(state))
        mod.set_state_rates(net["pg_vars"], obs6)
        nest.Simulate(mod.DECISION_WINDOW_MS)
        d1c = mod.read_and_clear(net["rec_D1"], d1_ids)
        d2c = mod.read_and_clear(net["rec_D2"], d2_ids)
        mod.read_and_clear(net["cortex"].recorder_exc, net["exc_ids"])
        net["cortex"].clear_rates()
        rates_d1.append(d1c.sum() / mod.N_D1 / (mod.DECISION_WINDOW_MS / 1000))
        rates_d2.append(d2c.sum() / mod.N_D2 / (mod.DECISION_WINDOW_MS / 1000))
        action = 1 if (d1c.sum() - d2c.sum()) > 0 else 0
        new_state, r, term, trunc, _ = env.step(action)
        state = new_state if not (term or trunc) else env.reset()[0]
    return float(np.mean(rates_d1)), float(np.mean(rates_d2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="path to a cart_pole/configs/*.py file")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=None, help="override config.n_episodes")
    ap.add_argument("--eval-episodes", type=int, default=None, help="override config.n_eval_episodes")
    ap.add_argument("--warmup", type=int, default=None, help="override config.warmup_episodes")
    ap.add_argument("--wmax", type=float, default=None, help="override config.wmax_override")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.n_episodes = args.episodes
    if args.eval_episodes is not None:
        cfg.n_eval_episodes = args.eval_episodes
    if args.warmup is not None:
        cfg.warmup_episodes = args.warmup
    if args.wmax is not None:
        cfg.wmax_override = args.wmax
    SEED = args.seed
    # mutual_inhibition: off unless a config opts in --
    # getattr-guarded so every existing config stays byte-identical.
    mi_kwargs = dict(
        mutual_inhibition=getattr(cfg, "mutual_inhibition", False),
        mutual_inhibition_weight=getattr(cfg, "mutual_inhibition_weight", -30.0),
    )

    if cfg.wmax_override is not None:
        mod.WMAX = cfg.wmax_override

    env = gym.make("CartPole-v1")
    # Pin the environment's own internal RNG (pole start angle/position
    # each reset) to SEED -- --seed previously only
    # controlled NEST's weight-init RNG and the exploration-flip RNG
    # below, never the env itself, so "same seed" was never actually
    # reproducible run-to-run. This single call seeds the stream once;
    # every later env.reset() inside run_episode() (no seed arg) advances
    # it deterministically from here. Going forward only -- every
    # already-reported number (old harnesses and this pipeline's own
    # first 36-run batch) was run without this and isn't retroactively
    # affected or invalidated by it.
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    restore = None
    train_scores = []
    run_rows = []
    t0 = time.time()
    W_init_d1 = W_init_d2 = None
    mod.FLIP_COUNT[0] = 0
    mod.STEP_COUNT[0] = 0

    for ep in range(cfg.n_episodes):
        net = mod.build_network(seed=SEED, restore=restore, **mi_kwargs)
        is_warmup = ep < cfg.warmup_episodes
        if ep == cfg.warmup_episodes:   # first LEARNING episode's pre-update weights
            W_init_d1 = net["W_d1"].copy()
            W_init_d2 = net["W_d2"].copy()
        eps_this_ep = eps_schedule(cfg, ep, cfg.n_episodes)
        score = mod.run_episode(net, env, rng=rng, learn=(not is_warmup), epsilon=eps_this_ep,
                                 action_identity=cfg.action_identity, centered=cfg.centered,
                                 population_readout=cfg.population_readout,
                                 action_centered=getattr(cfg, "action_centered", True),
                                 d1_only=getattr(cfg, "d1_only", False))
        restore = mod.extract_state(net)
        train_scores.append(score)
        run_rows.append({"episode": ep, "phase": "warmup" if is_warmup else "train",
                          "score": score, "epsilon": eps_this_ep})
        if (ep + 1) % 200 == 0 or ep == cfg.warmup_episodes - 1:
            tag = " [WARMUP]" if is_warmup else ""
            print(f"[{cfg.label} seed={SEED}] ep={ep+1}/{cfg.n_episodes} "
                  f"recent_mean={np.mean(train_scores[-200:]):.1f} "
                  f"W_d1_std={net['W_d1'].std():.1f} W_d2_std={net['W_d2'].std():.1f} "
                  f"elapsed={time.time()-t0:.1f}s{tag}", flush=True)

    if W_init_d1 is None:   # warmup_episodes==0 edge case
        net0 = mod.build_network(seed=SEED, **mi_kwargs)
        W_init_d1, W_init_d2 = net0["W_d1"].copy(), net0["W_d2"].copy()

    train_scores = np.array(train_scores)
    qsize = cfg.n_episodes // 4
    qs = [train_scores[q * qsize:(q + 1) * qsize].mean() for q in range(4)]

    W_trained_d1 = restore["W_d1"]
    W_trained_d2 = restore["W_d2"]

    eval_scores = []
    for i in range(cfg.n_eval_episodes):
        net = mod.build_network(seed=SEED, restore=restore, **mi_kwargs)
        s = mod.run_episode(net, env, rng=rng, learn=False, epsilon=0.0,
                             action_identity=cfg.action_identity, centered=cfg.centered,
                             population_readout=cfg.population_readout,
                             action_centered=getattr(cfg, "action_centered", True),
                             d1_only=getattr(cfg, "d1_only", False))
        eval_scores.append(s)
        run_rows.append({"episode": cfg.n_episodes + i, "phase": "greedy_eval", "score": s, "epsilon": 0.0})
    eval_scores = np.array(eval_scores)

    net_final = mod.build_network(seed=SEED, restore=restore, **mi_kwargs)
    d1_rate, d2_rate = measure_drive(net_final, env)

    flip_rate = mod.FLIP_COUNT[0] / max(1, mod.STEP_COUNT[0])
    anneal_tag = f"ANNEAL({cfg.eps_start}->{cfg.eps_end})" if cfg.anneal else f"eps={mod.EPSILON_EXPLORE}"
    line = (f"[{cfg.label}] seed={SEED} {anneal_tag} lr={mod.PLASTICITY_LR} "
            f"WARMUP={cfg.warmup_episodes} CENTERED: "
            f"train_ep0-{qsize}={qs[0]:.1f} ep{qsize}-{2*qsize}={qs[1]:.1f} "
            f"ep{2*qsize}-{3*qsize}={qs[2]:.1f} ep{3*qsize}-{cfg.n_episodes}={qs[3]:.1f} "
            f"train_max={train_scores.max():.0f} flip_rate={flip_rate:.3f} "
            f"GREEDY_EVAL_mean={eval_scores.mean():.1f} GREEDY_EVAL_std={eval_scores.std():.1f} "
            f"GREEDY_EVAL_max={eval_scores.max():.0f} "
            f"W_d1_std={W_trained_d1.std():.1f} W_d2_std={W_trained_d2.std():.1f} "
            f"W_d1_init_std={W_init_d1.std():.1f} "
            f"d1_range=[{W_trained_d1.min():.1f},{W_trained_d1.max():.1f}] "
            f"d2_range=[{W_trained_d2.min():.1f},{W_trained_d2.max():.1f}] "
            f"d1_rate={d1_rate:.1f} d2_rate={d2_rate:.1f} "
            f"elapsed={time.time()-t0:.1f}s")
    print(line, flush=True)

    run_dir = run_io.create_run_dir(cfg.label, SEED)
    run_io.save_config_used(run_dir, args.config)
    run_io.write_stats_csv(run_dir, run_rows)
    run_io.render_summary_pdf(run_dir, warmup_episodes=cfg.warmup_episodes)
    run_io.save_checkpoint(run_dir, restore)
    run_io.write_greedy_eval_summary(run_dir, eval_scores.tolist())
    run_io.save_experiment_params(run_dir, {
        "label": cfg.label, "seed": SEED,
        "action_identity": cfg.action_identity, "centered": cfg.centered,
        "population_readout": cfg.population_readout,
        "action_centered": getattr(cfg, "action_centered", True),
        "d1_only": getattr(cfg, "d1_only", False),
        "n_episodes": cfg.n_episodes, "n_eval_episodes": cfg.n_eval_episodes,
        "warmup_episodes": cfg.warmup_episodes, "epsilon_explore": mod.EPSILON_EXPLORE,
        "plasticity_lr": mod.PLASTICITY_LR, "anneal": cfg.anneal,
        "wmax_override": cfg.wmax_override, "elapsed_s": f"{time.time() - t0:.1f}",
        "summary_line": line,
    })
    print(f"[run_io] saved to {run_dir}")


if __name__ == "__main__":
    main()
