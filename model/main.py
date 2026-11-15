"""main.py — training loop, CLI entry.

Loop shape copied from the acrobot source (activate -> measure -> act ->
reward -> rest, RunManager/Run not Simulate()) but generalized: no
gymnasium -- ReachAdapter owns the environment directly. The mode branch
below (state vs events) is the one place main.py knows about the vision
stream's rendering mechanics (resync/observe/OU jitter); everything
task-specific otherwise (kinematics, reward, action space) lives in
env_reach.py, not here.
"""
import argparse
import dataclasses
import datetime
import importlib.util
import os
import shutil
import uuid
from collections import deque
from pathlib import Path

import nest
import numpy as np

import debug_util as du
import file_utils as futils
import recorder_tracking
import viz
from base_util import Base
from encoders import (LinearRateEncoder, TopographicPlaceFieldEncoder,
                      SensoryInput, Grid2DPlaceFieldEncoder, GainFieldEncoder)
from env_reach import ReachAdapter
from arm import SO101PlanarArm
from batched_actor_learning import BatchedD1Learning
from burst_hidden import BurstHiddenTeacher
from nest_checkpoint import NestCheckpointer
from network import Network
from python_critic import PythonCritic

np.set_printoptions(precision=3)


def _calibrate_reach_vec_range(cfg, n_samples=300):
    """lo/hi for Grid2DPlaceFieldEncoder's reach_vec=fk(q)-target bins --
    same percentile-of-random-sample convention as the Step 0 gate probe
    (_cortex_decode_probe) that validated this population's decodability
    before it was wired in here. A FIXED local seed (not cfg.seed/rng) --
    this calibration must be deterministic and must not perturb the
    training RNG's own draw sequence, run to run."""
    arm = SO101PlanarArm(include_shoulder_pan=getattr(cfg, "arm_include_shoulder_pan", False))
    calib_rng = np.random.default_rng(20261015)
    samples = np.array([arm.fk(calib_rng.uniform(arm._lo, arm._hi)) - arm.sample_target(calib_rng)
                        for _ in range(n_samples)])
    lo = np.percentile(samples, 1, axis=0)
    hi = np.percentile(samples, 99, axis=0)
    return lo, hi


def _build_encoders(cfg) -> SensoryInput:
    # encoder_type: "topo_both" (default) -- joints and object both use
    # TopographicPlaceFieldEncoder, spanning the full cortex extent so some
    # rows coincide across streams (a crude but real way for one cortex
    # neuron to see both a joint and a target feature at once -- needed
    # because value depends on their RELATIONSHIP, not either alone).
    # "grid2d_joints_lineartarget" -- joints uses Grid2DPlaceFieldEncoder
    # (a true 2-D Gaussian field), object uses LinearRateEncoder; measured
    # reach_vec R^2=0.99 offline vs ~0.76 for topo_both.
    encoder_type = getattr(cfg, "encoder_type", "topo_both")
    # cortex_extent is needed whenever a TopographicPlaceFieldEncoder gets built.
    if encoder_type == "topo_both" or getattr(cfg, "velocity_input", False):
        cortex_extent = getattr(cfg, "cortex_extent", None) or float(
            max(cfg.cortex_exc_shape[0], cfg.cortex_inh_shape[0]))
    # n_joints/n_object: 3 for the 3-DOF arm (arm_include_shoulder_pan), else 2.
    n_joints = 3 if getattr(cfg, "arm_include_shoulder_pan", False) else 2
    n_object = n_joints
    if encoder_type == "topo_both":
        joints_enc = TopographicPlaceFieldEncoder(
            n_joints, extent=cortex_extent, **cfg.joints_encoder_kw)
    else:
        joints_enc = Grid2DPlaceFieldEncoder(**cfg.joints_encoder_kw)
    if encoder_type == "topo_both":
        object_enc = TopographicPlaceFieldEncoder(
            n_object, extent=cortex_extent, **cfg.object_encoder_kw)
    else:
        object_enc = LinearRateEncoder(n_object, **cfg.object_raw_encoder_kw)
    encoders = {"joints": joints_enc, "object": object_enc}
    # velocity_input: a third topographic stream carrying joint velocity instead of position.
    if getattr(cfg, "velocity_input", False):
        if encoder_type not in ("topo_both", "grid2d_joints_lineartarget"):
            raise NotImplementedError(
                "velocity_input requires encoder_type='topo_both' or "
                "'grid2d_joints_lineartarget'")
        encoders["joints_vel"] = TopographicPlaceFieldEncoder(
            n_joints, extent=cortex_extent, **cfg.joints_vel_encoder_kw)
    # critic/actor_reach_vec_input: a dedicated reach-vector population feeding
    # the critic and/or actor directly (network.py), not cortex.
    if getattr(cfg, "critic_reach_vec_input", False) or getattr(cfg, "actor_reach_vec_input", False):
        rv_lo, rv_hi = _calibrate_reach_vec_range(cfg)
        encoders["reach_vec"] = Grid2DPlaceFieldEncoder(
            lo=rv_lo, hi=rv_hi, **cfg.reach_vec_encoder_kw)
    # actor_gainfield_hidden: a gain-field population observing (q, target) jointly.
    if getattr(cfg, "actor_gainfield_hidden", False):
        encoders["gainfield"] = GainFieldEncoder(**cfg.gainfield_encoder_kw)
    return SensoryInput(encoders)


def decay_stdp(model: str, factor: float):
    d = nest.GetDefaults(model)
    nest.SetDefaults(model, {'A_plus': d["A_plus"] * factor,
                             'A_minus': d["A_minus"] * factor})


def apply_freeze(episode: int, cfg, model_names) -> float:
    for _m in model_names:
        decay_stdp(_m, 0.0)
    print(f"[FREEZE] episode {episode+1}: STDP zeroed, policy greedy")
    return 0.0




def _critic_static_probe(net, enc, cfg, adapter, python_critic, rng, n_states=8, n_repeats=15):
    """Static (frozen-readout) probe: does V_spiking = critic.diagnose_value_
    signal() carry real state-dependent information, or is it flat noise
    drowned by decode variance? Freezes all dopamine-gated plasticity and
    never calls adapter.step()/update(), so repeated presentations of the
    same (joints, target) isolate decode noise alone. Curated states span
    the trained python_critic's own value range, not raw distance.
    """
    if python_critic is None:
        print("[critic_probe] skipped -- no python_critic reference available")
        return
    net.dopamine.control_volume_transmitter(learning_enabled=False, warmup_until_ms=0.0)
    signed = net.critic.signed_pools

    arm = adapter.arm
    pool = []
    for _ in range(300):
        q = rng.uniform(arm._lo, arm._hi)
        target = arm.sample_target(rng)
        obs = {"joints": q, "object": target}
        pool.append((obs, python_critic.value(obs)))
    pool.sort(key=lambda kv: kv[1])
    idxs = np.linspace(0, len(pool) - 1, n_states).astype(int)
    curated = [pool[i] for i in idxs]

    print(f"\n{'='*70}")
    print(f"[critic_probe] static frozen-readout probe: {n_states} states x {n_repeats} repeats  "
          f"(signed_pools={signed}, decode window={cfg.activate_time:.0f}ms, "
          f"N_pos={len(net.critic.layer1_pos) if signed else None}, "
          f"N_neg={len(net.critic.layer1_neg) if signed else None})")
    print(f"{'='*70}")

    per_state_means, per_state_stds, v_pythons = [], [], []
    per_state_pos_mean, per_state_pos_std = [], []
    per_state_neg_mean, per_state_neg_std = [], []
    for si, (obs, v_py) in enumerate(curated):
        samples, pos_samples, neg_samples = [], [], []
        for _ in range(n_repeats):
            enc.reset()
            enc.observe("joints", obs["joints"])
            enc.observe("object", obs["object"])
            if "reach_vec" in enc.enc:
                enc.observe("reach_vec", arm.fk(obs["joints"]) - obs["object"])
            enc.commit()
            t0 = nest.GetKernelStatus("biological_time")
            with nest.RunManager():
                nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            if signed:
                r_pos, r_neg = net.critic.diagnose_pool_rates(t0, t1)
                pos_samples.append(r_pos)
                neg_samples.append(r_neg)
                samples.append(r_pos - r_neg)
            else:
                samples.append(net.critic.diagnose_value_signal(t0, t1))
        samples = np.array(samples)
        per_state_means.append(samples.mean())
        per_state_stds.append(samples.std())
        v_pythons.append(v_py)
        if signed:
            pos_samples, neg_samples = np.array(pos_samples), np.array(neg_samples)
            per_state_pos_mean.append(pos_samples.mean())
            per_state_pos_std.append(pos_samples.std())
            per_state_neg_mean.append(neg_samples.mean())
            per_state_neg_std.append(neg_samples.std())
            print(f"  state {si}: v_python={v_py:+.4f}  rate_pos={pos_samples.mean():6.2f}+-{pos_samples.std():.2f}Hz  "
                  f"rate_neg={neg_samples.mean():6.2f}+-{neg_samples.std():.2f}Hz  "
                  f"diff mean={samples.mean():+.4f} std={samples.std():.4f}")
        else:
            print(f"  state {si}: v_python={v_py:+.4f}  v_spiking mean={samples.mean():+.4f} "
                  f"std={samples.std():.4f}  range=[{samples.min():+.4f}, {samples.max():+.4f}]")

    per_state_means = np.array(per_state_means)
    per_state_stds = np.array(per_state_stds)
    v_pythons = np.array(v_pythons)

    def _f_ratio(means, stds):
        bv = float(np.var(means))
        wv = float(np.mean(stds ** 2))
        return bv, wv, (bv / wv if wv > 0 else float("inf"))

    between_var, within_var, f_ratio = _f_ratio(per_state_means, per_state_stds)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(per_state_means, v_pythons)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0

    print(f"{'-'*70}")
    print(f"[critic_probe] (pos-neg) between-state variance = {between_var:.6f}")
    print(f"[critic_probe] (pos-neg) within-state variance (decode noise, mean) = {within_var:.6f}")
    print(f"[critic_probe] (pos-neg) F-ratio (between/within) = {f_ratio:.3f}")
    print(f"[critic_probe] corr(v_spiking_mean, v_python) across states = {corr:+.4f}")

    pool_result = {}
    if signed:
        per_state_pos_mean = np.array(per_state_pos_mean)
        per_state_pos_std = np.array(per_state_pos_std)
        per_state_neg_mean = np.array(per_state_neg_mean)
        per_state_neg_std = np.array(per_state_neg_std)
        bv_pos, wv_pos, f_pos = _f_ratio(per_state_pos_mean, per_state_pos_std)
        bv_neg, wv_neg, f_neg = _f_ratio(per_state_neg_mean, per_state_neg_std)
        with np.errstate(invalid="ignore"):
            corr_pos_neg = np.corrcoef(per_state_pos_mean, per_state_neg_mean)[0, 1]
        if not np.isfinite(corr_pos_neg):
            corr_pos_neg = 0.0
        print(f"[critic_probe] rate_pos: F-ratio={f_pos:.3f}  between_var={bv_pos:.4f}  within_var={wv_pos:.4f}  "
              f"mean_rate={per_state_pos_mean.mean():.2f}Hz")
        print(f"[critic_probe] rate_neg: F-ratio={f_neg:.3f}  between_var={bv_neg:.4f}  within_var={wv_neg:.4f}  "
              f"mean_rate={per_state_neg_mean.mean():.2f}Hz")
        print(f"[critic_probe] corr(rate_pos, rate_neg) across states = {corr_pos_neg:+.4f}")
        if f_pos < 2.0 and f_neg < 2.0:
            print("[critic_probe] VERDICT: BOTH pools individually flat -- the problem is upstream "
                  "of the subtraction: cortex->L1 never learned state-discriminative weights. "
                  "Signed pools cannot rescue what neither pool encodes. Points to cortex->L1 "
                  "learning/features, not the opponent trick.")
        elif corr_pos_neg > 0.7:
            print("[critic_probe] VERDICT: each pool DOES discriminate states individually, but they "
                  "move together (corr>0.7) -- CANCELLATION. The opponent readout (pos-neg) subtracts "
                  "away real signal both pools independently carry. Design flaw in the opponent "
                  "scheme, not a learning failure.")
        else:
            print("[critic_probe] VERDICT: pools discriminate states and are not strongly redundant -- "
                  "the flat pos-neg readout is not simply explained by cancellation or upstream "
                  "flatness; needs a closer look.")
        pool_result = {
            "rate_pos_mean": per_state_pos_mean.tolist(), "rate_pos_std": per_state_pos_std.tolist(),
            "rate_neg_mean": per_state_neg_mean.tolist(), "rate_neg_std": per_state_neg_std.tolist(),
            "f_ratio_pos": None if not np.isfinite(f_pos) else float(f_pos),
            "f_ratio_neg": None if not np.isfinite(f_neg) else float(f_neg),
            "corr_pos_neg": float(corr_pos_neg),
            "n_pos": len(net.critic.layer1_pos), "n_neg": len(net.critic.layer1_neg),
        }
    elif f_ratio < 2.0:
        print("[critic_probe] VERDICT: readout is FLAT -- between-state variance is drowned "
              "by decode noise. V carries little/no state information; a learning-rule tweak "
              "will not fix this.")
    elif abs(corr) < 0.3:
        print("[critic_probe] VERDICT: readout DOES distinguish states (between-state variance "
              "exceeds noise) but is NOT aligned with true value -- a learning-signal problem "
              "(check delta_spiking vs delta_python next), not a flat-readout problem.")
    else:
        print("[critic_probe] VERDICT: readout distinguishes states AND tracks true value "
              "reasonably well -- the flat episode-level corr is likely a speed/lr issue, not "
              "a structural one.")
    print(f"{'='*70}\n")

    import json
    probe_path = os.path.join(cfg.output_folder, "critic_static_probe.json")
    with open(probe_path, "w") as f:
        json.dump({
            "signed_pools": signed,
            "decode_window_ms": float(cfg.activate_time),
            "n_repeats": n_repeats,
            "v_python": v_pythons.tolist(),
            "v_spiking_mean": per_state_means.tolist(),
            "v_spiking_std": per_state_stds.tolist(),
            "between_var": between_var,
            "within_var": within_var,
            "f_ratio": None if not np.isfinite(f_ratio) else float(f_ratio),
            "corr": float(corr),
            **pool_result,
        }, f, indent=2)
    print(f"[critic_probe] saved to {probe_path}")


def _critic_vstar_probe(net, enc, cfg, adapter, python_critic, rng, n_states=8, n_repeats=15, n_rollout_repeats=5):
    """Sign check: _critic_static_probe above compares v_spiking against
    v_python, which is itself only an approximation and could share a bias
    with v_spiking or a sign mismatch. This probe compares v_spiking
    instead against V* = the true Monte-Carlo return-to-go, computed by
    rolling the policy out greedily from each curated state to termination
    and backward-recursing g[t]=r[t]+gamma*g[t+1], the same methodology
    _cortex_decode_probe uses. Curated-state selection matches
    _critic_static_probe's so the two probes' states are comparable.

    All NEST STDP stays frozen throughout, same as the other probes --
    V* rollouts must not perturb the weights being diagnosed. No
    dopamine/reward injection into NEST either; V* is accounted in plain
    Python from adapter.step()'s reward.
    """
    if python_critic is None:
        print("[vstar_probe] skipped -- no python_critic reference available")
        return None
    net.dopamine.control_volume_transmitter(learning_enabled=False, warmup_until_ms=0.0)
    gamma = net.critic.gamma()
    arm = adapter.arm

    pool = []
    for _ in range(300):
        q = rng.uniform(arm._lo, arm._hi)
        target = arm.sample_target(rng)
        obs = {"joints": q, "object": target}
        pool.append((obs, python_critic.value(obs)))
    pool.sort(key=lambda kv: kv[1])
    idxs = np.linspace(0, len(pool) - 1, n_states).astype(int)
    curated = [pool[i] for i in idxs]

    print(f"\n{'='*70}")
    print(f"[vstar_probe] P1.1 sign check: {n_states} states, {n_repeats} v_spiking repeats + "
          f"{n_rollout_repeats} V* rollouts/state (greedy, max_steps={cfg.max_steps}), gamma={gamma:.4f}")
    print(f"{'='*70}")

    v_spiking_means, v_star_means, v_star_stds, v_pythons = [], [], [], []
    for si, (obs0, v_py) in enumerate(curated):
        samples = []
        for _ in range(n_repeats):
            enc.reset()
            enc.observe("joints", obs0["joints"])
            enc.observe("object", obs0["object"])
            if "reach_vec" in enc.enc:
                enc.observe("reach_vec", arm.fk(obs0["joints"]) - obs0["object"])
            enc.commit()
            t0 = nest.GetKernelStatus("biological_time")
            with nest.RunManager():
                nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            samples.append(net.critic.diagnose_value_signal(t0, t1))
        v_spiking_means.append(float(np.mean(samples)))

        g0_samples = []
        for _ in range(n_rollout_repeats):
            adapter.reset(rng)   # any adapter-internal side state; overwritten below
            adapter.arm._q = obs0["joints"].copy()   # arm.q is a read-only property (returns _q.copy())
            adapter.target = obs0["object"].copy()
            adapter.d_prev = float(np.linalg.norm(adapter.target - adapter.arm.fk(adapter.arm.q)))
            adapter.step_count = 0
            adapter.q_vel = np.zeros(adapter.arm.n_joints)
            obs = adapter._obs()
            rs = []
            for step in range(cfg.max_steps):
                net.clear_rates()
                enc.reset()
                enc.observe("joints", obs["joints"])
                if getattr(cfg, "velocity_input", False):
                    enc.observe("joints_vel", obs["joints_vel"])
                enc.observe("object", obs["object"])
                if "reach_vec" in enc.enc:
                    enc.observe("reach_vec", arm.fk(obs["joints"]) - obs["object"])
                enc.commit()
                t0r = nest.GetKernelStatus("biological_time")
                with nest.RunManager():
                    nest.Run(cfg.activate_time)
                t1r = nest.GetKernelStatus("biological_time")
                action, _ = adapter.select_action(net.actor, cfg, t0r, t1r, 0.0)
                new_obs, reward, terminated, success = adapter.step(action)
                rs.append(reward)
                obs = new_obs
                if terminated:
                    break
            g = 0.0
            for r in reversed(rs):
                g = r + gamma * g
            g0_samples.append(g)
        v_star_means.append(float(np.mean(g0_samples)))
        v_star_stds.append(float(np.std(g0_samples)))
        v_pythons.append(v_py)
        print(f"  state {si}: v_python={v_py:+.4f}  v_spiking={v_spiking_means[-1]:+.4f}  "
              f"V*={v_star_means[-1]:+.4f} (+-{v_star_stds[-1]:.4f}, n={n_rollout_repeats})")

    v_spiking_arr = np.array(v_spiking_means)
    v_star_arr = np.array(v_star_means)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(v_spiking_arr, v_star_arr)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0

    print(f"{'-'*70}")
    print(f"[vstar_probe] corr(v_spiking, V*) across {n_states} states = {corr:+.4f}")
    if corr > 0:
        print("[vstar_probe] P1.1 GATE: PASS -- correct-signed vs true return-to-go. "
              "Proceed to P1.2 (seed robustness).")
    else:
        print("[vstar_probe] P1.1 GATE: FAIL -- sign flip vs true return-to-go. Needs a "
              "delta-sign/one-step-lag fix in SpikingCriticLearning, re-probe before P1.2.")
    print(f"{'='*70}\n")

    import json
    probe_path = os.path.join(cfg.output_folder, "p1_1_vstar_probe.json")
    with open(probe_path, "w") as f:
        json.dump({
            "n_states": n_states, "n_repeats": n_repeats, "n_rollout_repeats": n_rollout_repeats,
            "gamma": gamma,
            "v_python": [float(x) for x in v_pythons],
            "v_spiking": v_spiking_arr.tolist(),
            "v_star": v_star_arr.tolist(),
            "v_star_std": v_star_stds,
            "corr_vspiking_vstar": float(corr),
        }, f, indent=2)
    print(f"[vstar_probe] saved to {probe_path}")
    return corr


def _cortex_decode_probe(net, enc, cfg, adapter, python_critic, rng, n_episodes=40):
    """Offline cortex-rate decodability probe: does the raw spiking cortex
    rate vector x carry linearly decodable value information? One NEST
    rollout pass (frozen policy, no learning), then four offline fits of
    x -> V* (realized return-to-go): A ridge, B NNLS (mirrors the real
    critic's non-negative, no-bias readout), C TD(0), D TD(0) with a short
    vs long integration window. Separates a representation ceiling
    (A fails) from a mechanism-specific gap (A works, B/C/D narrow it
    down) -- see the printed verdict below.
    """
    import json
    from sklearn.linear_model import RidgeCV
    from scipy.optimize import nnls

    if python_critic is None:
        print("[decode_probe] skipped -- no python_critic reference available")
        return

    net.dopamine.control_volume_transmitter(learning_enabled=False, warmup_until_ms=0.0)
    gamma = net.critic.gamma()
    cortex = net.cortex.output()
    id_to_idx = {gid: i for i, gid in enumerate(cortex.tolist())}
    n_cortex = len(cortex)
    short_frac = 0.5   # "short" window = first half of activate_time

    # ── Step 0 gate: a dedicated 2-D Gaussian place-field population over
    # reach_vec = fk(q)-target, built fresh here (not wired to
    # cortex/critic), to test whether the numeric reach_vec R2 survives
    # actually being encoded as spikes. True 2D
    # receptive fields (a center in (dx,dy), not PlaceFieldEncoder's
    # separable per-axis tuning) since distance is a joint function of
    # both components. Range calibrated from a random (q,target) sample,
    # same convention as the rest of this codebase's encoder calibration
    # (measured, not guessed).
    arm = adapter.arm
    _calib = np.array([arm.fk(rng.uniform(arm._lo, arm._hi)) - arm.sample_target(rng)
                        for _ in range(300)])
    pf_bins = 12
    pf_lo = np.percentile(_calib, 1, axis=0)
    pf_hi = np.percentile(_calib, 99, axis=0)
    pf_sigma = (pf_hi - pf_lo) / (pf_bins - 1)
    _cx, _cy = np.meshgrid(np.linspace(pf_lo[0], pf_hi[0], pf_bins),
                            np.linspace(pf_lo[1], pf_hi[1], pf_bins))
    pf_centers = np.stack([_cx.ravel(), _cy.ravel()], axis=1)
    pf_max_rate = 800.0   # matches PlaceFieldEncoder's own default
    n_pf = pf_bins * pf_bins
    pf_gen = nest.Create("poisson_generator", n_pf, {"rate": 0.0})
    pf_par = nest.Create("parrot_neuron", n_pf)
    nest.Connect(pf_gen, pf_par, "one_to_one")
    pf_rec = recorder_tracking.create()
    nest.Connect(pf_par, pf_rec)
    pf_id_to_idx = {gid: i for i, gid in enumerate(pf_par.tolist())}

    def _pf_set_rate(rv):
        d2 = ((rv[0] - pf_centers[:, 0]) / pf_sigma[0]) ** 2 + \
             ((rv[1] - pf_centers[:, 1]) / pf_sigma[1]) ** 2
        pf_gen.rate = (pf_max_rate * np.exp(-d2 / 2.0)).tolist()

    def _pf_measure(t0, t1):
        events = nest.GetStatus(pf_rec, "events")[0]
        times, senders = events["times"], events["senders"]
        counts = np.zeros(n_pf)
        for s in senders[(times >= t0) & (times < t1)]:
            counts[pf_id_to_idx[int(s)]] += 1
        return counts / ((t1 - t0) / 1000.0)

    print(f"\n{'='*70}")
    print(f"[decode_probe] offline cortex-rate decodability probe: "
          f"{n_episodes} frozen-policy episodes, n_cortex={n_cortex}, gamma={gamma:.4f}")
    print(f"{'='*70}")

    def _measure(t0, t1, t_mid):
        events = nest.GetStatus(net._cortex_diag_rec, "events")[0]
        times, senders = events["times"], events["senders"]
        x_long = np.zeros(n_cortex)
        x_short = np.zeros(n_cortex)
        for s in senders[(times >= t0) & (times < t1)]:
            x_long[id_to_idx[int(s)]] += 1
        for s in senders[(times >= t0) & (times < t_mid)]:
            x_short[id_to_idx[int(s)]] += 1
        x_long /= (t1 - t0) / 1000.0
        x_short /= (t_mid - t0) / 1000.0
        return x_long, x_short

    ep_x_long, ep_x_short, ep_r, ep_vpy, ep_ee, ep_target, ep_pf, ep_q = [], [], [], [], [], [], [], []
    for ep in range(n_episodes):
        obs = adapter.reset(rng)
        xs_long, xs_short, rs, vpys, ees, targets, pfs, qs = [], [], [], [], [], [], [], []
        for step in range(cfg.max_steps):
            net.clear_rates()
            Base.clear_spikes(pf_rec)
            # ee_xy/target: logged alongside x purely for the
            # augmented-feature offline fits below -- obs itself is joint
            # angles + task-space target with no target-relative term, so
            # a linear readout of cortex rates alone structurally cannot
            # form distance
            # (requires forward kinematics, a nonlinearity, PLUS a
            # cross-space subtraction). fk(obs["joints"]) is the SAME
            # pre-step hand position env_reach.py's own step() computes
            # as p_before. Computed BEFORE commit/Run (not after, as the
            # first version of this probe did) so the Step-0 place-field
            # population's rate can be set in the SAME RunManager window
            # as cortex's own encoders -- main.py's own documented NEST
            # constraint: a generator's rate must be set before Run(),
            # never spliced in mid-block.
            ee_xy = arm.fk(obs["joints"])
            _pf_set_rate(ee_xy - obs["object"])
            enc.reset()
            enc.observe("joints", obs["joints"])
            enc.observe("object", obs["object"])
            enc.commit()
            t0 = nest.GetKernelStatus("biological_time")
            with nest.RunManager():
                nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            x_long, x_short = _measure(t0, t1, t0 + short_frac * cfg.activate_time)
            pf_rate = _pf_measure(t0, t1)

            action, _ = adapter.select_action(net.actor, cfg, t0, t1, 0.0)
            v_py = python_critic.value(obs)
            new_obs, reward, terminated, success = adapter.step(action)

            xs_long.append(x_long)
            xs_short.append(x_short)
            rs.append(reward)
            vpys.append(v_py)
            ees.append(ee_xy)
            targets.append(obs["object"].copy())
            pfs.append(pf_rate)
            qs.append(obs["joints"].copy())
            obs = new_obs
            if terminated:
                break
        ep_x_long.append(np.array(xs_long))
        ep_x_short.append(np.array(xs_short))
        ep_pf.append(np.array(pfs))
        ep_r.append(np.array(rs))
        ep_vpy.append(np.array(vpys))
        ep_ee.append(np.array(ees))
        ep_target.append(np.array(targets))
        ep_q.append(np.array(qs))
        if (ep + 1) % 10 == 0:
            print(f"  logged episode {ep+1}/{n_episodes}  (T={len(rs)} steps)")

    # ── V* (return-to-go) per episode + TD transition pairs ────────────────
    X_long_l, X_short_l, Vstar_l, Vpy_l, EE_l, TGT_l, PF_l, Q_l = [], [], [], [], [], [], [], []
    Xtd_long_l, Xnext_long_l, Xtd_short_l, Xnext_short_l, Rtd_l = [], [], [], [], []
    for xs_l, xs_s, rs, vpys, ees, targets, pfs, qs in zip(
            ep_x_long, ep_x_short, ep_r, ep_vpy, ep_ee, ep_target, ep_pf, ep_q):
        T = len(rs)
        g = np.zeros(T)
        g[-1] = rs[-1]
        for t in range(T - 2, -1, -1):
            g[t] = rs[t] + gamma * g[t + 1]
        X_long_l.append(xs_l); X_short_l.append(xs_s); Vstar_l.append(g); Vpy_l.append(vpys)
        EE_l.append(ees); TGT_l.append(targets); PF_l.append(pfs); Q_l.append(qs)
        if T >= 2:
            Xtd_long_l.append(xs_l[:-1]); Xnext_long_l.append(xs_l[1:])
            Xtd_short_l.append(xs_s[:-1]); Xnext_short_l.append(xs_s[1:])
            Rtd_l.append(rs[:-1])

    X_long = np.concatenate(X_long_l); X_short = np.concatenate(X_short_l)
    Vstar = np.concatenate(Vstar_l); Vpy = np.concatenate(Vpy_l)
    Xtd_long = np.concatenate(Xtd_long_l); Xnext_long = np.concatenate(Xnext_long_l)
    Xtd_short = np.concatenate(Xtd_short_l); Xnext_short = np.concatenate(Xnext_short_l)
    Rtd = np.concatenate(Rtd_l)
    EE_all = np.concatenate(EE_l); TGT_all = np.concatenate(TGT_l)
    REACH_all = EE_all - TGT_all
    DIST_all = np.linalg.norm(REACH_all, axis=1, keepdims=True)
    PF_all = np.concatenate(PF_l)
    Q_all = np.concatenate(Q_l)

    n_total = len(Vstar)
    print(f"[decode_probe] logged {n_total} transitions "
          f"({'OK, >=3000' if n_total >= 3000 else 'WARNING: <3000, noisier fits'})")
    with np.errstate(invalid="ignore"):
        corr_py_vstar = np.corrcoef(Vpy, Vstar)[0, 1]
    print(f"[decode_probe] corr(v_python, V*) = {corr_py_vstar:+.3f}  "
          f"(sanity check: python_critic should track realized return)")

    split_rng = np.random.default_rng(12345)
    perm = split_rng.permutation(n_total)
    n_train = int(0.7 * n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    n_td = len(Rtd)
    perm_td = split_rng.permutation(n_td)
    train_idx_td = perm_td[:int(0.7 * n_td)]

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    results = {}
    _alpha_grid = np.logspace(-2, 8, 40)   # wide enough that the optimum lands inside
                                             # the grid, not pinned at an edge -- a narrower
                                             # grid silently caps regularization at its max,
                                             # making added features look like zero effect

    # ── Step 0 gate (run first): does reach_vec's R2 (0.41-0.43
    # on the clean numeric feature) SURVIVE being encoded as actual spikes
    # through a dedicated 144-neuron 2-D place-field population, before
    # committing to building this into the real critic (Steps 1-3)? Same
    # train/test split as everything else below, same no-intercept NNLS
    # convention.
    ridge_pf = RidgeCV(alphas=_alpha_grid)
    ridge_pf.fit(PF_all[train_idx], Vstar[train_idx])
    r2_pf_ridge = r2(Vstar[test_idx], ridge_pf.predict(PF_all[test_idx]))
    w_nnls_pf, _ = nnls(PF_all[train_idx], Vstar[train_idx])
    r2_pf_nnls = r2(Vstar[test_idx], PF_all[test_idx] @ w_nnls_pf)
    results["STEP0_reach_vec_spiking_pf"] = {
        "R2_ridge": r2_pf_ridge, "R2_nnls": r2_pf_nnls, "alpha": float(ridge_pf.alpha_),
        "n_pf_neurons": int(n_pf), "pf_bins": pf_bins,
    }
    print(f"[decode_probe] STEP 0 GATE -- reach_vec, spiking 2D place-field ({n_pf} neurons): "
          f"ridge R2={r2_pf_ridge:.4f}  NNLS R2={r2_pf_nnls:.4f}  alpha={ridge_pf.alpha_:.3g}")
    if r2_pf_ridge >= 0.3:
        print("  -> GATE PASSED (R2>=0.3): spiking decode noise has NOT crushed the signal. "
              "Proceed to Steps 1-3 (dedicated critic input, signed readout, measure).")
    elif r2_pf_ridge < 0.2:
        print("  -> GATE FAILED (R2<0.2): decode noise alone crushes reach_vec's spiking "
              "representation below what Steps 1-3 could plausibly recover. Stop here.")
    else:
        print("  -> GATE AMBIGUOUS (0.2<=R2<0.3): borderline -- note this explicitly before "
              "deciding whether to proceed.")

    # A: ridge, signed weights, x_long -> V*
    ridge = RidgeCV(alphas=_alpha_grid)
    ridge.fit(X_long[train_idx], Vstar[train_idx])
    r2_a = r2(Vstar[test_idx], ridge.predict(X_long[test_idx]))
    results["A_ridge_signed"] = {"R2": r2_a, "alpha": float(ridge.alpha_)}
    _pinned = " ** PINNED AT GRID EDGE **" if ridge.alpha_ in (_alpha_grid[0], _alpha_grid[-1]) else ""
    print(f"[decode_probe] A (ridge, signed, long window): R2={r2_a:.4f}  alpha={ridge.alpha_:.3g}{_pinned}")

    # B: NNLS, non-negative weights, NO intercept -- matches the real
    # single-pool critic's own architecture (value = nonneg-weighted sum
    # of nonneg rates, no separate bias population).
    w_nnls, _ = nnls(X_long[train_idx], Vstar[train_idx])
    r2_b = r2(Vstar[test_idx], X_long[test_idx] @ w_nnls)
    results["B_nnls_nonneg"] = {"R2": r2_b}
    print(f"[decode_probe] B (NNLS, non-negative, no intercept, long window): R2={r2_b:.4f}")

    # C/D: TD(0), signed, bootstrap on logged transitions (standardized
    # features -- raw Hz-scale TD with a fixed small lr is ill-conditioned).
    def fit_td(X_cur, X_next, R, tr_idx, epochs=100):
        # alpha scaled by 1/n_features: per-sample semi-gradient TD(0) on
        # this many correlated features has no contraction guarantee and
        # can diverge to overflow at a fixed alpha, so there's a hard
        # delta clip and an early-abort-on-divergence guard, so a bad run
        # reports "diverged" instead of a silent nan R2.
        d = X_cur.shape[1]
        alpha = 0.02 / d
        mu, sd = X_cur[tr_idx].mean(axis=0), X_cur[tr_idx].std(axis=0) + 1e-6
        Phi = np.concatenate([(X_cur - mu) / sd, np.ones((len(X_cur), 1))], axis=1)
        Phi_next = np.concatenate([(X_next - mu) / sd, np.ones((len(X_next), 1))], axis=1)
        w = np.zeros(Phi.shape[1])
        idx = tr_idx.copy()
        mean_abs_delta = None
        diverged = False
        for _ in range(epochs):
            split_rng.shuffle(idx)
            deltas = []
            for i in idx:
                delta = R[i] + gamma * (w @ Phi_next[i]) - (w @ Phi[i])
                delta = float(np.clip(delta, -50.0, 50.0))
                w += alpha * delta * Phi[i]
                deltas.append(abs(delta))
            mean_abs_delta = float(np.mean(deltas))
            if not np.all(np.isfinite(w)):
                diverged = True
                w = np.zeros_like(w)
                break
        if diverged:
            print("[decode_probe] WARNING: TD(0) fit diverged -- returning zero weights (R2 will be <=0)")
        return w, (mu, sd), mean_abs_delta

    def eval_td(w, stats, X_eval, y_eval):
        mu, sd = stats
        Phi = np.concatenate([(X_eval - mu) / sd, np.ones((len(X_eval), 1))], axis=1)
        return r2(y_eval, Phi @ w)

    w_c, stats_c, mean_abs_delta_c = fit_td(Xtd_long, Xnext_long, Rtd, train_idx_td)
    r2_c = eval_td(w_c, stats_c, X_long[test_idx], Vstar[test_idx])
    results["C_td_signed"] = {"R2": r2_c, "mean_abs_td_error_final_epoch": mean_abs_delta_c}
    print(f"[decode_probe] C (TD(0), signed, long window): R2={r2_c:.4f}  "
          f"mean|TD error| (final epoch, train)={mean_abs_delta_c:.4f}")

    w_d_short, stats_d_short, _ = fit_td(Xtd_short, Xnext_short, Rtd, train_idx_td)
    r2_d_short = eval_td(w_d_short, stats_d_short, X_short[test_idx], Vstar[test_idx])
    r2_d_long = r2_c   # same fit as C -- long window is C's own window
    results["D_td_long"] = {"R2": r2_d_long, "window_ms": float(cfg.activate_time)}
    results["D_td_short"] = {"R2": r2_d_short, "window_ms": float(short_frac * cfg.activate_time)}
    print(f"[decode_probe] D (TD(0), window-length): "
          f"R2_short={r2_d_short:.4f} (window={short_frac*cfg.activate_time:.0f}ms)  "
          f"R2_long={r2_d_long:.4f} (window={cfg.activate_time:.0f}ms)")

    # ── Augmented-feature probe: obs = {joints, target_xy, joints_vel} has
    # no target-relative term, so value requires forward kinematics plus a
    # cross-space subtraction (ee_xy - target), neither of which a linear
    # readout of cortex rates alone can form. Tests whether adding an
    # explicit target-relative feature lifts A's ceiling: ee_xy (hand
    # position), reach_vec (ee-target, parietal-cortex-style coding),
    # distance (||ee-target||, an upper-bound ablation, not a candidate
    # fix). Same train/test split and NNLS convention as A/B above.
    augmented_variants = {"ee_xy": EE_all, "reach_vec": REACH_all, "distance": DIST_all}
    for vname, extra in augmented_variants.items():
        Xv = np.concatenate([X_long, extra], axis=1)
        ridge_v = RidgeCV(alphas=_alpha_grid)
        ridge_v.fit(Xv[train_idx], Vstar[train_idx])
        r2_ridge_v = r2(Vstar[test_idx], ridge_v.predict(Xv[test_idx]))
        w_nnls_v, _ = nnls(Xv[train_idx], Vstar[train_idx])
        r2_nnls_v = r2(Vstar[test_idx], Xv[test_idx] @ w_nnls_v)
        results[f"A_ridge_signed+{vname}"] = {"R2": r2_ridge_v, "alpha": float(ridge_v.alpha_)}
        results[f"B_nnls_nonneg+{vname}"] = {"R2": r2_nnls_v}
        _pinned_v = " ** PINNED AT GRID EDGE **" if ridge_v.alpha_ in (_alpha_grid[0], _alpha_grid[-1]) else ""
        print(f"[decode_probe] augmented[+{vname}]: ridge(signed) R2={r2_ridge_v:.4f} "
              f"alpha={ridge_v.alpha_:.3g}{_pinned_v}  NNLS(nonneg) R2={r2_nnls_v:.4f}")

    print(f"[decode_probe] AUGMENTED-FEATURE VERDICT (baseline ridge R2={r2_a:.3f}, "
          f"baseline NNLS R2={r2_b:.3f}):")
    jumped = {v: results[f"A_ridge_signed+{v}"]["R2"] - r2_a for v in augmented_variants}
    if any(jumped[v] > 0.15 and results[f"B_nnls_nonneg+{v}"]["R2"] > 0 for v in ("ee_xy", "reach_vec")):
        print("  -> ee_xy and/or reach_vec jumps signed R2 AND NNLS stays positive: the missing "
              "target-relative term was the representation cause, and the fix is sign-compatible. "
              "A spiking critic with this feature added becomes viable to build.")
    elif jumped.get("distance", 0.0) > 0.15 and jumped.get("ee_xy", 0.0) <= 0.15 and jumped.get("reach_vec", 0.0) <= 0.15:
        print("  -> only the distance ablation jumps: confirms value ~ distance, but the honest, "
              "biologically-defensible features (ee_xy/reach_vec) give substantially less -- "
              "report this gap carefully rather than building on the ablation number.")
    else:
        print("  -> nothing jumps meaningfully: the missing target-relative term is not the "
              "(or not the whole) representation limit -- a deeper ceiling, per the earlier "
              "cortex-only result. Do not build a spiking critic on this basis.")

    # ── Trig-feature ceiling test: numeric (cos th1, sin th1,
    # cos(th1+th2), sin(th1+th2), target_x, target_y), a lossless
    # cumulative-angle encoding that linearizes a serial 2-link chain's
    # hand position, unlike raw q which needs a nonlinear trig product.
    # Compared against the spiking reach_vec place-field gate to decide
    # which ceiling is higher. arm.fk() is the real URDF chain (rpy
    # offsets per joint), not an abstract serial-link formula, so this
    # trig encoding isn't exactly equivalent to fk(q) here -- tested
    # empirically anyway rather than assuming the mismatch kills it.
    th1, th2 = Q_all[:, 0], Q_all[:, 1]
    TRIG_all = np.stack([np.cos(th1), np.sin(th1), np.cos(th1 + th2), np.sin(th1 + th2)],
                         axis=1)
    TRIG_all = np.concatenate([TRIG_all, TGT_all], axis=1)

    def _fit_ridge_nnls(Xf, tag):
        ridge_f = RidgeCV(alphas=_alpha_grid)
        ridge_f.fit(Xf[train_idx], Vstar[train_idx])
        r2_ridge_f = r2(Vstar[test_idx], ridge_f.predict(Xf[test_idx]))
        w_nnls_f, _ = nnls(Xf[train_idx], Vstar[train_idx])
        r2_nnls_f = r2(Vstar[test_idx], Xf[test_idx] @ w_nnls_f)
        _pinned_f = " ** PINNED AT GRID EDGE **" if ridge_f.alpha_ in (_alpha_grid[0], _alpha_grid[-1]) else ""
        results[tag] = {"R2_ridge": r2_ridge_f, "R2_nnls": r2_nnls_f, "alpha": float(ridge_f.alpha_)}
        print(f"[decode_probe] TRIG[{tag}]: ridge R2={r2_ridge_f:.4f} alpha={ridge_f.alpha_:.3g}"
              f"{_pinned_f}  NNLS R2={r2_nnls_f:.4f}")
        return r2_ridge_f, r2_nnls_f

    print(f"\n[decode_probe] TRIG-FEATURE CEILING TEST (vs. STEP 0 spiking place-field: "
          f"ridge={r2_pf_ridge:.4f}  NNLS={r2_pf_nnls:.4f}):")
    print(f"[decode_probe] TRIG[baseline_cortex_only]: ridge R2={r2_a:.4f}  NNLS R2={r2_b:.4f}  "
          f"(reproduced from A/B above)")
    r2_trig_ridge, r2_trig_nnls = _fit_ridge_nnls(TRIG_all, "TRIG_numeric_only")
    r2_trigcx_ridge, r2_trigcx_nnls = _fit_ridge_nnls(
        np.concatenate([X_long, TRIG_all], axis=1), "TRIG_plus_cortex")

    print(f"[decode_probe] TRIG TABLE  "
          f"(ridge / NNLS -- all vs. True V*, held-out 70/30 split):")
    print(f"  baseline (cortex-rates only):      {r2_a:.4f} / {r2_b:.4f}")
    print(f"  trig (numeric, no cortex):         {r2_trig_ridge:.4f} / {r2_trig_nnls:.4f}")
    print(f"  trig + cortex:                     {r2_trigcx_ridge:.4f} / {r2_trigcx_nnls:.4f}")
    print(f"  reach_vec, spiking place-field:     {r2_pf_ridge:.4f} / {r2_pf_nnls:.4f}")
    if r2_trig_ridge >= 0.5 and r2_trig_nnls >= 0.25:
        print("  -> TRIG ceiling matches/beats the place-field population: the clean "
              "re-encoding is good enough -- build the trig re-encoding (more defensible, "
              "but requires a full retrain).")
    elif r2_trig_ridge <= r2_pf_ridge - 0.1:
        print("  -> TRIG ceiling clearly below the spiking place-field population's: "
              "the place-field population wins -- proceed with Steps 1-3 on that basis "
              "instead of a trig re-encoding.")
    else:
        print("  -> TRIG and place-field ceilings are close -- no clean winner from R2 alone; "
              "weigh against the trig encoding's larger engineering cost (full retrain).")

    # ── F-ratio probe: same curated-state methodology as _critic_static_
    # probe, applied to each offline readout's predictions ────────────────
    arm = adapter.arm
    pool = []
    for _ in range(300):
        q = rng.uniform(arm._lo, arm._hi)
        target = arm.sample_target(rng)
        obs_c = {"joints": q, "object": target}
        pool.append((obs_c, python_critic.value(obs_c)))
    pool.sort(key=lambda kv: kv[1])
    n_states, n_repeats = 8, 15
    idxs = np.linspace(0, len(pool) - 1, n_states).astype(int)
    curated = [pool[i] for i in idxs]

    def _f_ratio(means, stds):
        bv = float(np.var(means))
        wv = float(np.mean(np.asarray(stds) ** 2))
        return bv, wv, (bv / wv if wv > 0 else float("inf"))

    per_state_long, per_state_short, v_py_curated = [], [], []
    for obs_c, v_py in curated:
        samples_long, samples_short = [], []
        for _ in range(n_repeats):
            net.clear_rates()
            enc.reset()
            enc.observe("joints", obs_c["joints"])
            enc.observe("object", obs_c["object"])
            enc.commit()
            t0 = nest.GetKernelStatus("biological_time")
            with nest.RunManager():
                nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            x_long, x_short = _measure(t0, t1, t0 + short_frac * cfg.activate_time)
            samples_long.append(x_long)
            samples_short.append(x_short)
        per_state_long.append(np.array(samples_long))
        per_state_short.append(np.array(samples_short))
        v_py_curated.append(v_py)
    v_py_curated = np.array(v_py_curated)

    def _predict_all(readout_fn, per_state_x):
        means, stds = [], []
        for x_samples in per_state_x:
            preds = np.array([readout_fn(x) for x in x_samples])
            means.append(preds.mean())
            stds.append(preds.std())
        return np.array(means), np.array(stds)

    def _pred_td(x, w, stats):
        mu, sd = stats
        return float(np.concatenate([(x - mu) / sd, [1.0]]) @ w)

    probes = [
        ("A_ridge_signed", lambda x: float(ridge.predict(x.reshape(1, -1))[0]), per_state_long),
        ("B_nnls_nonneg", lambda x: float(x @ w_nnls), per_state_long),
        ("C_td_signed", lambda x: _pred_td(x, w_c, stats_c), per_state_long),
        ("D_td_long", lambda x: _pred_td(x, w_c, stats_c), per_state_long),
        ("D_td_short", lambda x: _pred_td(x, w_d_short, stats_d_short), per_state_short),
    ]
    for name, fn, per_state_x in probes:
        means, stds = _predict_all(fn, per_state_x)
        bv, wv, f_ratio = _f_ratio(means, stds)
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(means, v_py_curated)[0, 1]
        corr = 0.0 if not np.isfinite(corr) else float(corr)
        results[name]["f_ratio"] = None if not np.isfinite(f_ratio) else float(f_ratio)
        results[name]["corr_vs_python_curated"] = corr
        print(f"[decode_probe] F-ratio[{name}] = {f_ratio:.3f}  corr(pred,v_python)={corr:+.3f}")

    # ── Decision table (pasted spec, Step 4) ────────────────────────────────
    print(f"\n{'-'*70}")
    print("[decode_probe] DECISION TABLE:")
    f_a = results["A_ridge_signed"]["f_ratio"] or 0.0
    if r2_a > 0.5 and f_a > 2.0:
        print(f"  A: value IS linearly decodable from cortex rates (R2={r2_a:.3f}, F={f_a:.2f}). "
              "Representation is fine -- failure is MECHANISM-side:")
        if r2_b < 0.3 * max(r2_a, 1e-9):
            print(f"  -> B fails (R2={r2_b:.3f}) while A works: the sign/non-negative-weight "
                  "constraint is a PRIME CULPRIT. Actionable: give the readout effective "
                  "negative weights (stronger opponent/inhibitory design).")
        else:
            print(f"  -> B also works reasonably (R2={r2_b:.3f}): the sign constraint alone "
                  "does not explain the spiking critic's failure.")
        if r2_c > 0.3 * max(r2_a, 1e-9):
            print(f"  -> C (TD, signed) works (R2={r2_c:.3f}): bootstrap/bounded-weight "
                  "dynamics are not the blocker on these features.")
        else:
            print(f"  -> C (TD, signed) ALSO fails (R2={r2_c:.3f}) despite A working: "
                  "bootstrap + bounded-weight collapse dynamics are a real, separate culprit.")
        if r2_d_short < 0.5 * max(r2_d_long, 1e-9):
            print(f"  -> D: short window much worse (R2={r2_d_short:.3f} vs {r2_d_long:.3f}): "
                  "decode noise IS a real contributor; longer integration/more neurons would help.")
        else:
            print(f"  -> D: short vs long window similar (R2={r2_d_short:.3f} vs {r2_d_long:.3f}): "
                  "decode noise/window length is not the main limiting factor.")
    else:
        print(f"  A: R2={r2_a:.3f}, F-ratio={f_a:.3f} (<=2/0.5 threshold) -- value is NOT reliably "
              "linearly decodable from cortex rates even with signed weights and a long window.")
        print("  => REPRESENTATION/DECODE limit. No readout-side fix (including distributional "
              "coding) will help here -- clean hard stop.")
    print(f"{'='*70}\n")

    probe_path = os.path.join(cfg.output_folder, "cortex_decode_probe.json")
    with open(probe_path, "w") as f:
        json.dump({
            "n_transitions": int(n_total),
            "n_episodes": n_episodes,
            "gamma": float(gamma),
            "corr_vpython_vstar": float(corr_py_vstar),
            "results": results,
        }, f, indent=2)
    print(f"[decode_probe] saved to {probe_path}")


# ── Main training function ────────────────────────────────────────────────────

def train(cfg, episode_callback=None, step_callback=None, restore_checkpoint_path=None):
    # episode_callback(net, episode, enc, adapter): diagnostic hook called
    # after each episode's bookkeeping but before the checkpoint/rebuild
    # decision -- lets an external probe inspect the network and drive its
    # own mini-rollout through the same live enc/adapter. None by default,
    # zero behavior change.
    #
    # step_callback(delta_python, nb_value, delta_python_vec): diagnostic
    # hook called once per control step, right after sample_nb() -- lets a
    # diagnostic compare python_critic's clean TD-error against nb_value
    # (what the real spiking DA trace actually delivers after Poisson-rate
    # injection and spiking). delta_python_vec is the per-joint vector from
    # python_critics_per_joint, or None otherwise. None by default, zero
    # overhead.
    #
    # restore_checkpoint_path (--restore-checkpoint CLI flag): path to a
    # final_checkpoint.pkl from a prior run, loaded once right after the
    # first network build, before the training loop starts. Lets this run
    # continue training, or do a pure greedy eval/video pass on an
    # already-trained network with no re-training. None by default.
    print("[CONFIG] " + "-" * 50)
    for f in dataclasses.fields(cfg):
        print(f"[CONFIG]  {f.name:<28} = {getattr(cfg, f.name)}")
    print("[CONFIG] " + "-" * 50)

    os.makedirs(cfg.output_folder, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_folder, "videos"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_folder, "debug"), exist_ok=True)
    if cfg.clean_output:
        [f.unlink() for f in Path(cfg.output_folder).glob("*") if f.is_file()]

    experiment_id = uuid.uuid4()
    futils.save_experiment_params(experiment_id, cfg.output_folder,
                                  {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)})

    # ── NEST kernel ───────────────────────────────────────────────────────────
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.resolution = cfg.nest_resolution
    seed = cfg.seed if cfg.seed >= 0 else int(np.random.randint(0, 1_000_000))
    # Persisted back onto cfg (not just the local `seed`) so the adapter's
    # own exploration-noise RNG (env_reach.py/env_1d.py select_action --
    # previously np.random.uniform's UNSEEDED global state, a real
    # reproducibility bug: exploration noise is both part of the actual
    # env dynamics and, under use_action_perturbation, the credit signal
    # itself, so two "same seed" runs could silently diverge from the
    # first step) can read the SAME resolved value cfg.seed<0 would
    # otherwise hide.
    cfg.seed = seed
    print(f"NEST seed={seed}  threads={cfg.num_threads}  mode={cfg.mode}")

    rng = np.random.default_rng(seed)
    # _decoy_rng: dedicated stream for target_continuous_swap_eval's
    # per-episode decoy target, so it doesn't perturb the main
    # training/eval RNG's draw sequence (same as env_reach.py's _explore_rng).
    _decoy_rng = np.random.default_rng(seed + 424242)
    name_prefix = cfg.name + "_"

    # ── Encoders + network (Dopamine built first inside Network) ─────────────
    # Wrapped in a rebuildable closure -- NestCheckpointer (nest_checkpoint.py)
    # calls this again after a periodic nest.ResetKernel() to work around a
    # NEST memory-growth characteristic in stdp_dopamine_synapse +
    # volume_transmitter (confirmed present in NEST 3.8.0/3.10.0, not fixable
    # at the application level otherwise). enc/net are reassigned via
    # nonlocal so every other closure/reference in train() sees the rebuilt
    # objects without needing its own plumbing.
    enc = net = pool_rec = pool_shape = None
    _rebuild_count = 0
    batched_d1_list = []   # set below, after the first build, if cfg.actor_batched_d1 -- one entry per D1/D2 channel
    batched_d1 = None       # backward-compat alias for batched_d1_list[0] (unified/single-channel case)
    batched_d2_list = []   # set below, after the first build, if cfg.actor_batched_d2 -- one entry per channel
    critic_learners = []  # set below, after the first build -- [learner] normally, [learner_pos, learner_neg] when cfg.critic_signed_pools
    # nest.ResetKernel() (checkpoint_every's periodic rebuild) resets
    # NEST's own biological_time clock to 0, but cfg.warmup_time is
    # cumulative across the whole run, not per-segment -- without this
    # offset, warmup_until_ms compared against the reset clock could never
    # be satisfied once checkpoint_every * episode_duration < warmup_time.
    # Incremented by each segment's final biological_time right before
    # checkpoint_and_rebuild() resets it.
    _time_offset = 0.0

    def _apply_stdp_ab_override():
        # Optional, additive: override a plastic model's post-warmup A_plus/
        # A_minus without touching dopamine.py's source (getattr default is
        # a no-op for every existing config). Exists for the adopsyn
        # A_plus/A_minus sweep -- dopsyn's ratio was empirically
        # recalibrated for this task's negative-skewed reward (dopamine.py's
        # CopyModel comments); adopsyn's never was, and its weight was
        # measured drifting monotonically (D1 up, D2 down) over a full
        # 100-episode run even from a symmetric init, recreating the same
        # D1/D2 imbalance that caused q1/q2 to saturate at their joint
        # limits every episode. Dopamine.__init__ zeroes A_plus/A_minus
        # during warmup and restores them from self._a_init when learning
        # activates (control_volume_transmitter) -- so the override must
        # land in _a_init, not nest.SetDefaults directly, or warmup's own
        # restore would clobber it. Reapplied on every rebuild too, since
        # Network() constructs a fresh Dopamine (with fresh _a_init) each
        # time.
        stdp_ab_override = getattr(cfg, "stdp_ab_override", None)
        if stdp_ab_override:
            for _model, _vals in stdp_ab_override.items():
                net.dopamine._a_init[_model] = dict(_vals)

    def _build_network_and_encoders(load_plastic=True):
        nonlocal enc, net, pool_rec, pool_shape, _rebuild_count
        recorder_tracking.clear_tracked_recorders()
        recorder_tracking.plugin_recorder_listeners()
        nest.resolution = cfg.nest_resolution
        # Derived, not reused verbatim, across rebuilds -- reusing the exact
        # same seed after ResetKernel() would replay an identical stochastic
        # stream (exploration noise, Poisson generators) every segment
        # instead of a fresh one. NestCheckpointer's own docstring: there is
        # no API to serialize/restore NEST's internal RNG state, so a reseed
        # on every rebuild is unavoidable -- treat it like ordinary
        # run-to-run seed variance, not a reproducibility guarantee across
        # a checkpoint boundary.
        nest.SetKernelStatus({
            "local_num_threads": cfg.num_threads,
            "rng_seed": seed + 1_000_003 * _rebuild_count,
        })
        _rebuild_count += 1
        enc = _build_encoders(cfg)
        net = Network(enc, cfg, name_prefix=name_prefix, load_plastic=load_plastic)
        _apply_stdp_ab_override()
        pool_rec = None
        pool_shape = None

    _build_network_and_encoders(load_plastic=True)

    # Constructed ONCE here (not inside Network.__init__, which gets
    # rebuilt fresh on every checkpoint_every cycle -- see network.py's
    # own comment) so self.W/self.x_bar survive a nest.ResetKernel() the
    # same way batched_d1_list's elig/credit arrays do. Re-attached to
    # each rebuilt net.critic in checkpoint_and_rebuild() below, never
    # recreated mid-run.
    # critic_signed_pools: TWO learners instead of one -- layer1_neg's
    # gets a NEGATED lr (same class, same delta, opposite update sign),
    # mirroring dopsyn(D1, +)/adopsyn(D2, -)'s cooperation under one
    # shared delta. Both called every step with the SAME v_s/reward/done.
    from spiking_critic_learning import SpikingCriticLearning
    if getattr(cfg, "critic_signed_pools", False):
        learner_pos = SpikingCriticLearning(
            net.critic_input_pop, net.critic.layer1_pos, net.critic.input_synapse_model_pos,
            lr=cfg.critic_centered_lr, xbar_alpha=cfg.critic_centered_xbar_alpha,
            wmin=cfg.critic_centered_wmin, wmax=cfg.critic_centered_wmax,
            gamma=net.critic.gamma(),
            delta_bar_alpha=cfg.critic_centered_delta_bar_alpha,
            center_delta=getattr(cfg, "critic_centered_delta_centering", True),
        )
        learner_neg = SpikingCriticLearning(
            net.critic_input_pop, net.critic.layer1_neg, net.critic.input_synapse_model_neg,
            lr=-cfg.critic_centered_lr, xbar_alpha=cfg.critic_centered_xbar_alpha,
            wmin=cfg.critic_centered_wmin, wmax=cfg.critic_centered_wmax,
            gamma=net.critic.gamma(),
            delta_bar_alpha=cfg.critic_centered_delta_bar_alpha,
            center_delta=getattr(cfg, "critic_centered_delta_centering", True),
        )
        critic_learners = [learner_pos, learner_neg]
    else:
        critic_learners = [SpikingCriticLearning(
            net.critic_input_pop, net.critic.layer1, net.critic.input_synapse_model,
            lr=cfg.critic_centered_lr, xbar_alpha=cfg.critic_centered_xbar_alpha,
            wmin=cfg.critic_centered_wmin, wmax=cfg.critic_centered_wmax,
            gamma=net.critic.gamma(),
            delta_bar_alpha=cfg.critic_centered_delta_bar_alpha,
            center_delta=getattr(cfg, "critic_centered_delta_centering", True),
        )]
    net.critic_learners = critic_learners

    # actor_batched_d1: holds cortex->D1 weights fixed across a batch of
    # several episodes on different targets, accumulating eligibility x
    # dopamine-deviation credit in Python the whole time
    # (BatchedD1Learning, batched_actor_learning.py), then applies one
    # combined update per batch. dopsyn's A_plus/A_minus are forced to 0
    # every step below, right after control_volume_transmitter -- this is
    # what makes cortex->D1's weight a plain settable property NEST's own
    # online mechanism never touches. adopsyn (D2) and critic_dopsyn
    # (critic) are untouched. Default False, every existing config's
    # behavior is unchanged unless it explicitly opts in.
    actor_batched_d1 = getattr(cfg, "actor_batched_d1", False)
    # actor_batched_d2: the same presynaptic-rate-centering treatment
    # actor_batched_d1_presyn_baseline_correction gives cortex->D1,
    # extended to cortex->D2, which otherwise always learns via NEST's
    # native online adopsyn STDP with raw, uncentered presynaptic
    # activity. Deliberately minimal vs. actor_batched_d1's full
    # machinery: no pref-vector support, no use_action_perturbation, no
    # d1_hidden_layer-style chain. Just the same nb_value signal D1's
    # default path uses, optionally presynaptic-baseline-centered.
    # Default False, every existing config unchanged.
    actor_batched_d2 = getattr(cfg, "actor_batched_d2", False)
    # actor_unified_readout: when False, actor.py builds cfg.n_actions
    # separate D1/D2 population pairs, each with its own scalar preferred
    # value, instead of one shared vector-readout population (see
    # actor.py's own unified_readout comment). Every D1 neuron's fixed
    # random preference in a shared population entangles its contribution
    # to both action dimensions, which no signal-side fix can remove;
    # separate populations avoid that cross-talk structurally.
    # _n_bd1_channels controls how many independent BatchedD1Learning
    # instances get built -- 1 (unified) or cfg.n_actions (one per
    # D1/D2 channel).
    _n_bd1_channels = cfg.n_actions if (
        actor_batched_d1 and not getattr(cfg, "actor_unified_readout", False)) else 1

    def _bd1_populations(ch, kind):
        """(source_pop, target_pop, source_rec, target_rec) for one
        batched_d1_list entry. kind: "single" (plain cortex->d1, every
        existing config), "l1" (cortex->d1_l1, actor_d1_hidden_layer=True's
        first hop), "l2" (d1_l1->d1, its second hop). Re-derived from the
        CURRENT net/net.actor each call (not cached) so this is safe to
        call again after a checkpoint rebuild, same convention as
        _set_batched_d1_pref below."""
        if kind == "single":
            # actor_burst_hidden (Path B): H sits between cortex and D1/
            # D2 -- BatchedD1Learning's presynaptic source becomes H
            # (its own DECISION-window activity, via rec_H_bd1, read
            # non-destructively) instead of cortex. H's conns to D1 were
            # already built from H (network.py passed H.output() as the
            # Actor's "cortex" arg), so this is purely a bookkeeping
            # redirect, not a new wiring decision.
            if getattr(cfg, "actor_bypass_cortex_joints", False):
                return (net.actor_bypass_pop, net.actor.d1(ch),
                        net.actor_bypass_rec, net.actor.get_recorder(ch, "d1"))
            if getattr(cfg, "actor_burst_hidden", False):
                return (net.burst_h_pop, net.actor.d1(ch),
                        net.burst_h_rec_bd1, net.actor.get_recorder(ch, "d1"))
            if getattr(cfg, "actor_cortex_fixed_hidden", False):
                # Cell 2 (small-target grid completion): the frozen hidden
                # layer, not cortex itself, is what's actually wired to
                # D1/D2 here -- BatchedD1Learning's presynaptic source
                # must be redirected to match (same bookkeeping-only
                # redirect as the burst_hidden branch above).
                return (net.actor_cortex_hidden_pop, net.actor.d1(ch),
                        net._actor_cortex_hidden_diag_rec, net.actor.get_recorder(ch, "d1"))
            return (net.cortex.output(), net.actor.d1(ch),
                    net._cortex_diag_rec, net.actor.get_recorder(ch, "d1"))
        elif kind == "l1":
            return (net.cortex.output(), net.actor.d1_l1(ch),
                    net._cortex_diag_rec, net.actor.get_recorder(ch, "d1_l1"))
        else:
            return (net.actor.d1_l1(ch), net.actor.d1(ch),
                    net.actor.get_recorder(ch, "d1_l1"), net.actor.get_recorder(ch, "d1"))

    def _set_batched_d1_pref(ch, bd1):
        """(Re)derive bd1's per-neuron pref array in the CURRENT
        net.actor.d1(ch) id ordering -- see BatchedD1Learning.set_pref()'s
        docstring. Called once right after construction and again after
        every checkpoint rebuild (ids renumber; the cached pref vectors
        themselves don't). No-op (bd1.pref stays None) in per-channel mode
        (_n_bd1_channels>1) -- there, each population's own pref is a
        SCALAR (one joint only, actor.py's non-unified branch), not the 2D
        vector debias_pref's regression machinery requires; debias_pref is
        also meaningless there (no cross-dimension entanglement to
        regress out of a population that only ever represents one joint),
        so it's simplest to skip pref-setting entirely rather than half-
        support a incompatible shape."""
        if _n_bd1_channels > 1:
            return
        pref = net.actor.channels[ch]["d1_pref"]
        pref_id_to_idx = net.actor.channels[ch]["d1_id_to_idx"]
        d1_ids = net.actor.d1(ch).tolist()
        pref_ordered = np.array([pref[pref_id_to_idx[nid]] for nid in d1_ids])
        bd1.set_pref(pref_ordered)

    batched_d1_layers = []   # parallel to batched_d1_list: (ch, kind) -- see _bd1_populations

    if actor_batched_d1:
        if _n_bd1_channels > 1 and getattr(cfg, "actor_batched_d1_debias_pref", False):
            print("[batched_d1] NOTE: actor_batched_d1_debias_pref=True is "
                  "meaningless in per-channel mode (actor_unified_readout=False) "
                  "-- forced off for every channel.")
        _bd1_common_kwargs = dict(
            tau_c=cfg.da_tau_c, a_plus=cfg.dopsyn_a_plus,
            wmin=getattr(cfg, "dopsyn_wmin", 10.0), wmax=getattr(cfg, "dopsyn_wmax", 40.0),
            batch_episodes=getattr(cfg, "actor_batched_d1_batch_episodes", 10),
            lr_scale=getattr(cfg, "actor_batched_d1_lr_scale", 3.0),
            center_nb_per_batch=getattr(cfg, "actor_batched_d1_center_nb", False),
            debias_include_intercept=getattr(cfg, "actor_batched_d1_debias_include_intercept", True),
            postsyn_baseline_correction=getattr(cfg, "actor_batched_d1_postsyn_baseline_correction", False),
            postsyn_baseline_tau=getattr(cfg, "actor_batched_d1_postsyn_baseline_tau", 5000.0),
            use_action_perturbation=getattr(cfg, "actor_batched_d1_use_action_perturbation", False),
            action_perturbation_scale=getattr(cfg, "actor_batched_d1_action_perturbation_scale", 1.0),
            coincidence_bin_ms=getattr(cfg, "actor_batched_d1_coincidence_bin_ms", None),
            source_novelty_gate=getattr(cfg, "actor_batched_d1_source_novelty_gate", False),
            source_novelty_tau=getattr(cfg, "actor_batched_d1_source_novelty_tau", 5000.0),
            source_novelty_gain=getattr(cfg, "actor_batched_d1_source_novelty_gain", 1.0),
            presyn_baseline_correction=getattr(cfg, "actor_batched_d1_presyn_baseline_correction", False),
            presyn_baseline_tau=getattr(cfg, "actor_batched_d1_presyn_baseline_tau", 50000.0),
            reset_elig_per_episode=getattr(cfg, "actor_batched_d1_reset_elig_per_episode", False),
            n_goals=(len(cfg.target_goal_set)
                     if getattr(cfg, "actor_batched_d1_context_gated", False) else None))
        _debias_pref = lambda: getattr(cfg, "actor_batched_d1_debias_pref", False) if _n_bd1_channels == 1 else False
        for ch in range(_n_bd1_channels):
            _has_hidden = net.actor.channels[ch].get("d1_l1") is not None
            if _has_hidden:
                # cortex->D1_L1: no pref vector exists for D1_L1 (it's a
                # relay, not a readout population) -- debias_pref forced
                # off regardless of config, same rationale as per-channel
                # mode's own forced-off case above. use_action_perturbation
                # also forced off here: it projects the
                # known action-noise vector through the TARGET population's
                # own pref (step()'s post_signal), which is structurally
                # meaningless (and crashes, pref is None) for a relay
                # layer -- falls back to postsyn_baseline_correction/raw
                # counts, same as every non-action-perturbation config.
                src1, tgt1, _, _ = _bd1_populations(ch, "l1")
                _bd1_l1_kwargs = dict(_bd1_common_kwargs, use_action_perturbation=False)
                bd1_l1 = BatchedD1Learning(src1, tgt1, debias_pref=False, **_bd1_l1_kwargs)
                batched_d1_list.append(bd1_l1)
                batched_d1_layers.append((ch, "l1"))
                # D1_L1->D1_L2(=d1): the readout-connected hop -- gets the
                # SAME debias_pref/pref-setting treatment the plain single-
                # layer case would have gotten.
                src2, tgt2, _, _ = _bd1_populations(ch, "l2")
                bd1_l2 = BatchedD1Learning(src2, tgt2, debias_pref=_debias_pref(), **_bd1_common_kwargs)
                _set_batched_d1_pref(ch, bd1_l2)
                batched_d1_list.append(bd1_l2)
                batched_d1_layers.append((ch, "l2"))
                print(f"[batched_d1] channel={ch}  D1_L1 (cortex->D1_L1): N_PARAMS={len(bd1_l1.conns)}  "
                      f"batch_episodes={bd1_l1.batch_episodes}  lr_scale={bd1_l1.lr_scale}")
                print(f"[batched_d1] channel={ch}  D1_L2 (D1_L1->D1, readout): N_PARAMS={len(bd1_l2.conns)}  "
                      f"batch_episodes={bd1_l2.batch_episodes}  lr_scale={bd1_l2.lr_scale}  "
                      f"debias_pref={bd1_l2.debias_pref}")
            else:
                src, tgt, _, _ = _bd1_populations(ch, "single")
                bd1 = BatchedD1Learning(src, tgt, debias_pref=_debias_pref(), **_bd1_common_kwargs)
                # Always derived (not just when debias_pref=True) -- commit()
                # always computes last_residual_credit for diagnostics
                # regardless of whether the correction is actually applied to
                # weights. No-op in per-channel mode, see _set_batched_d1_pref.
                _set_batched_d1_pref(ch, bd1)
                batched_d1_list.append(bd1)
                batched_d1_layers.append((ch, "single"))
                print(f"[batched_d1] channel={ch}  enabled: N_PARAMS={len(bd1.conns)}  "
                      f"batch_episodes={bd1.batch_episodes}  lr_scale={bd1.lr_scale}  "
                      f"debias_pref={bd1.debias_pref}  context_gated_n_goals={bd1._n_goals}")
        # Exposed on net (not just the local `batched_d1_list` var) so
        # external diagnostics driving train() via episode_callback can
        # reach the REAL, live instances -- e.g. to snapshot credit/elig
        # state -- without reimplementing checkpoint_every's rebuild
        # machinery themselves (that duplication is exactly what caused a
        # severe, documented NEST memory-growth slowdown in an earlier
        # diagnostic this session that ran its own manual training loop
        # with checkpoint_every=0). Re-set after every rebuild below too,
        # since Network()/net itself is a fresh object each time.
        # net.batched_d1 (singular) stays a backward-compat alias to
        # channel 0's instance -- correct/complete on its own in the
        # unified (_n_bd1_channels==1) case; per-channel diagnostics should
        # use net.batched_d1_list instead.
        net.batched_d1_list = batched_d1_list
        net.batched_d1 = batched_d1_list[0]
        batched_d1 = batched_d1_list[0]

    def _bd2_populations(ch):
        """(source_pop, target_pop, source_rec, target_rec) for one
        batched_d2_list entry -- always the plain cortex->d2 hop, no
        hidden-layer/l1 variant (D2 has none). Re-derived from the CURRENT
        net/net.actor each call, same rebuild-safety convention as
        _bd1_populations."""
        if getattr(cfg, "actor_bypass_cortex_joints", False):
            return (net.actor_bypass_pop, net.actor.d2(ch),
                    net.actor_bypass_rec, net.actor.get_recorder(ch, "d2"))
        return (net.cortex.output(), net.actor.d2(ch),
                net._cortex_diag_rec, net.actor.get_recorder(ch, "d2"))

    if actor_batched_d2:
        _bd2_common_kwargs = dict(
            tau_c=cfg.da_tau_c_adopsyn, a_plus=getattr(cfg, "adopsyn_a_plus", -0.0035),
            wmin=getattr(cfg, "adopsyn_wmin", 8.0), wmax=getattr(cfg, "adopsyn_wmax", 30.0),
            batch_episodes=getattr(cfg, "actor_batched_d2_batch_episodes", 10),
            lr_scale=getattr(cfg, "actor_batched_d2_lr_scale", 3.0),
            presyn_baseline_correction=getattr(cfg, "actor_batched_d2_presyn_baseline_correction", False),
            presyn_baseline_tau=getattr(cfg, "actor_batched_d2_presyn_baseline_tau", 50000.0),
            postsyn_baseline_correction=getattr(cfg, "actor_batched_d2_postsyn_baseline_correction", False),
            postsyn_baseline_tau=getattr(cfg, "actor_batched_d2_postsyn_baseline_tau", 5000.0))
        for ch in range(_n_bd1_channels):
            src, tgt, _, _ = _bd2_populations(ch)
            bd2 = BatchedD1Learning(src, tgt, **_bd2_common_kwargs)
            batched_d2_list.append(bd2)
            print(f"[batched_d2] channel={ch}  enabled: N_PARAMS={len(bd2.conns)}  "
                  f"batch_episodes={bd2.batch_episodes}  lr_scale={bd2.lr_scale}  "
                  f"presyn_baseline_correction={bd2.presyn_baseline_correction}")
        net.batched_d2_list = batched_d2_list

    # actor_burst_hidden: the persistent Python-side piece
    # (BurstHiddenTeacher, model/burst_hidden.py), constructed once here
    # like batched_d1_list/critic_learners above, holding fb_weight/x_bar
    # state that must survive every nest.ResetKernel() rebuild. A local
    # variable rather than just net.burst_teacher, since net itself is a
    # fresh object every rebuild, so the persistent object gets re-bolted
    # onto each new net instance in _reattach_after_restore below.
    burst_teacher = None
    if getattr(cfg, "actor_burst_hidden", False):
        burst_teacher = BurstHiddenTeacher(
            h_size=len(net.burst_h_pop), seed=seed,
            isi_ms=cfg.burst_isi_ms, base_std=cfg.burst_base_std,
            std_min=cfg.burst_std_min, std_max=cfg.burst_std_max,
            gain_td=cfg.burst_gain_td, p_baseline=cfg.burst_p_baseline,
            lr=cfg.burst_lr, fb_weight_lo=cfg.burst_fb_weight_lo,
            fb_weight_hi=cfg.burst_fb_weight_hi,
            wmin_ch=cfg.burst_ch_wmin, wmax_ch=cfg.burst_ch_wmax)
        burst_teacher.attach(
            net.cortex.output(), net.burst_h_pop,
            nest.GetConnections(net.cortex.output(), net.burst_h_pop,
                                 synapse_model="static_ch_burst"))
    net.burst_teacher = burst_teacher

    # ── NEST checkpoint/rebuild (see nest_checkpoint.py) ────────────────────
    # The checkpointer itself (and its tracked-projection registry) is now
    # ALWAYS constructed -- it has THREE independent uses, not just one:
    #   1. cfg.checkpoint_every>0: periodic in-memory save/ResetKernel/
    #      rebuild/restore (unchanged behavior, still gated below at its
    #      own call site -- every 5 plastic cortex->{critic L1, D1 ch0/1,
    #      D2 ch0/1} projections' LIVE weights plus the Python-side scalars
    #      that don't live in NEST-restorable form).
    #   2. Always: dump the SAME snapshot to disk once, at the very end of
    #      training (final_checkpoint.pkl in cfg.output_folder -- every
    #      run persists its trained weights).
    #   3. restore_checkpoint_path (--restore-checkpoint CLI flag): load
    #      a prior run's disk snapshot before training starts.
    # All three need the identical track_projection() registry regardless
    # of whether periodic rebuilding is itself enabled, so constructing it
    # is now unconditional (cheap -- just registers getters, no NEST calls).
    checkpoint_every = getattr(cfg, "checkpoint_every", 0)
    # checkpoint_every (default 3) is not a multiple of
    # actor_batched_d1/d2_batch_episodes (default 10) in this project's
    # standing configs, so most periodic rebuilds land mid-batch.
    # BatchedD1Learning.attach() remaps elig/credit/elig_sum through a
    # rebuild-stable (cortex_idx, d1_idx) key, so correctness doesn't
    # depend on alignment -- logged once rather than asserted, since
    # misalignment is just a mild inefficiency now, not a correctness
    # problem.
    if checkpoint_every > 0 and (actor_batched_d1 or actor_batched_d2):
        for _attr in ("actor_batched_d1_batch_episodes", "actor_batched_d2_batch_episodes"):
            _be = getattr(cfg, _attr, 10)
            if checkpoint_every % _be != 0 and _be % checkpoint_every != 0:
                print(f"[batched_learning] NOTE: checkpoint_every={checkpoint_every} is not "
                      f"a multiple of {_attr}={_be} -- rebuilds will land mid-batch; "
                      "attach()'s accumulator remap handles this correctly.")
    _checkpointer = NestCheckpointer(build_fn=lambda: _build_network_and_encoders(load_plastic=False))
    if getattr(cfg, "critic_signed_pools", False):
        _checkpointer.track_projection("input_critic_l1_pos", lambda: net.critic_input_pop,
                                       lambda: net.critic.layer1_pos)
        _checkpointer.track_projection("input_critic_l1_neg", lambda: net.critic_input_pop,
                                       lambda: net.critic.layer1_neg)
    else:
        _checkpointer.track_projection("input_critic_l1", lambda: net.critic_input_pop, lambda: net.critic.layer1)
    for _ch in range(net.actor._n_built_channels):
        if net.actor.channels[_ch].get("d1_l1") is not None:
            # actor_d1_hidden_layer=True: cortex->d1 no longer exists
            # directly -- track the two hops that replaced it instead,
            # or their live weights would silently reset to their
            # INITIAL random values every checkpoint_every episodes.
            _checkpointer.track_projection(f"cortex_d1l1_ch{_ch}", lambda: net.cortex.output(),
                                           (lambda ch=_ch: net.actor.d1_l1(ch)))
            _checkpointer.track_projection(f"d1l1_d1_ch{_ch}",
                                           (lambda ch=_ch: net.actor.d1_l1(ch)),
                                           (lambda ch=_ch: net.actor.d1(ch)))
        else:
            # actor_burst_hidden (Path B): H sits between cortex and D1/
            # D2 at the NEST level too (network.py passed H.output() as
            # the Actor's "cortex" arg) -- the LIVE projection that needs
            # tracking is H->D1/D2, not cortex->D1/D2, or a rebuild would
            # silently try to replay saved weights onto a nonexistent
            # (empty) cortex->D1 connection while H->D1's real weights
            # vanish back to their initial random draw.
            _ctx_src = ((lambda: net.actor_bypass_pop) if getattr(cfg, "actor_bypass_cortex_joints", False)
                        else (lambda: net.burst_h_pop) if getattr(cfg, "actor_burst_hidden", False)
                        else (lambda: net.actor_cortex_hidden_pop) if getattr(cfg, "actor_cortex_fixed_hidden", False)
                        else (lambda: net.cortex.output()))
            _checkpointer.track_projection(f"cortex_d1_ch{_ch}", _ctx_src,
                                           (lambda ch=_ch: net.actor.d1(ch)))
        _ctx_src_d2 = ((lambda: net.actor_bypass_pop) if getattr(cfg, "actor_bypass_cortex_joints", False)
                       else (lambda: net.burst_h_pop) if getattr(cfg, "actor_burst_hidden", False)
                       else (lambda: net.actor_cortex_hidden_pop) if getattr(cfg, "actor_cortex_fixed_hidden", False)
                       else (lambda: net.cortex.output()))
        _checkpointer.track_projection(f"cortex_d2_ch{_ch}", _ctx_src_d2,
                                       (lambda ch=_ch: net.actor.d2(ch)))
    if getattr(cfg, "actor_burst_hidden", False):
        _checkpointer.track_projection("cortex_h", lambda: net.cortex.output(),
                                       lambda: net.burst_h_pop, synapse_model="static_ch_burst")

    def _capture_checkpoint_extra_state():
        state = {
            "da_baseline": net.dopamine.da_baseline,
            "da_b_smoothed": dict(net.dopamine._da_b_smoothed),
            "stdp_a": {m: dict(v) for m, v in net.dopamine._a_init.items()},
            "learning_active": net.dopamine.learning_active,
        }
        for _ch in range(net.actor._n_built_channels):
            state[f"d1_ch{_ch}_v_th"] = [float(v) for v in nest.GetStatus(net.actor.d1(_ch), "V_th")]
            state[f"d2_ch{_ch}_v_th"] = [float(v) for v in nest.GetStatus(net.actor.d2(_ch), "V_th")]
            if getattr(cfg, "actor_readout_baseline_correction", False):
                # readout_baseline_correction (actor.py): a slow (~10-
                # episode) EMA meant to track TONIC rate across many
                # episodes -- MUST survive checkpoint_every's rebuild
                # (every 3-5 episodes by default) or it would never
                # accumulate past a few episodes' worth of history,
                # same rationale as V_th's own save/restore above.
                for _name in ("d1", "d2"):
                    _baseline = net.actor.get_readout_baseline(_ch, _name)
                    if _baseline is not None:
                        state[f"{_name}_ch{_ch}_readout_baseline"] = _baseline.tolist()
        if burst_teacher is not None:
            # fb_weight/x_bar have no NEST equivalent (pure Python) --
            # must be explicitly captured for process-boundary (disk)
            # persistence, same rationale as readout_baseline above.
            # Across an in-memory checkpoint_and_rebuild(), the live
            # burst_teacher object's own attributes already survive
            # untouched (same as batched_d1_list's elig/credit) -- this
            # capture only matters for save_checkpoint_to_disk/restore.
            # Uses the LOCAL `burst_teacher` closure variable, not net.
            # burst_teacher -- this can be called (from _reattach_after_
            # restore's caller, checkpoint_and_rebuild) with the OLD net
            # still live, but ALSO needs to work consistently once net
            # has been reassigned by a rebuild, before the re-bolt below.
            _bstate = burst_teacher.extract_state()
            state["burst_fb_weight"] = _bstate["fb_weight"].tolist()
            state["burst_x_bar"] = _bstate["x_bar"].tolist()
        return state

    def _apply_checkpoint_extra_state(state):
        net.dopamine.da_baseline = state["da_baseline"]
        net.dopamine._da_b_smoothed = dict(state["da_b_smoothed"])
        for m, b_val in state["da_b_smoothed"].items():
            nest.SetDefaults(m, {"b": float(b_val)})
        for m, vals in state["stdp_a"].items():
            nest.SetDefaults(m, {"A_plus": vals["A_plus"], "A_minus": vals["A_minus"]})
        net.dopamine._a_init = {m: dict(v) for m, v in state["stdp_a"].items()}
        # Latch state: must survive a rebuild, or warm-up would re-trigger
        # every time biological_time (reset to 0 by nest.ResetKernel())
        # passes cfg.warmup_time again after a mid-training checkpoint.
        net.dopamine.learning_active = state["learning_active"]
        for _ch in range(net.actor._n_built_channels):
            nest.SetStatus(net.actor.d1(_ch), "V_th", state[f"d1_ch{_ch}_v_th"])
            nest.SetStatus(net.actor.d2(_ch), "V_th", state[f"d2_ch{_ch}_v_th"])
            if getattr(cfg, "actor_readout_baseline_correction", False):
                for _name in ("d1", "d2"):
                    _key = f"{_name}_ch{_ch}_readout_baseline"
                    if _key in state:
                        net.actor.set_readout_baseline(_ch, _name, state[_key])
        if burst_teacher is not None and "burst_fb_weight" in state:
            # Redundant (no-op) on a periodic in-memory rebuild --
            # burst_teacher is the SAME persistent object, untouched by
            # nest.ResetKernel(), same as batched_d1_list. NECESSARY on a
            # disk restore (--restore-checkpoint): that path constructs a
            # BRAND NEW BurstHiddenTeacher with a fresh random fb_weight,
            # which must be overwritten with the saved one or the H->D1/
            # D2 weights (trained against the ORIGINAL fb_weight mapping)
            # would be paired with the wrong feedback-alignment signal.
            # Uses the LOCAL `burst_teacher` closure variable -- `net` may
            # already be a freshly-rebuilt object with no .burst_teacher
            # attribute yet at this point (re-bolted later, in
            # _reattach_after_restore's own tail).
            burst_teacher.restore_state({
                "fb_weight": np.array(state["burst_fb_weight"]),
                "x_bar": np.array(state["burst_x_bar"]),
            })

    def _reattach_after_restore(extra):
        """Shared tail for BOTH checkpoint_and_rebuild() (periodic, in-
        memory) and restore_from_disk() (once, at startup, from a file) --
        the restore itself (ResetKernel + build_fn + replay) only touches
        raw NEST connections; every Python-side learner object (batched_
        d1/d2, critic_learners) holds its OWN numpy state (elig/credit
        accumulators, W/x_bar) that survives a kernel reset untouched, but
        its connection HANDLES go stale and must be re-fetched against the
        just-rebuilt population, via each object's own .attach()."""
        _apply_checkpoint_extra_state(extra)
        if batched_d1_list:
            for (_ch, _kind), _bd1 in zip(batched_d1_layers, batched_d1_list):
                _src, _tgt, _, _ = _bd1_populations(_ch, _kind)
                _bd1.attach(_src, _tgt)
                if _kind != "l1":   # D1_L1 has no pref vector -- see construction site
                    _set_batched_d1_pref(_ch, _bd1)
            net.batched_d1_list = batched_d1_list
            net.batched_d1 = batched_d1_list[0]
            batched_d1 = batched_d1_list[0]
        if batched_d2_list:
            for _ch, _bd2 in enumerate(batched_d2_list):
                _src, _tgt, _, _ = _bd2_populations(_ch)
                _bd2.attach(_src, _tgt)
            net.batched_d2_list = batched_d2_list
        if critic_learners:
            if getattr(cfg, "critic_signed_pools", False):
                critic_learners[0].attach(net.critic_input_pop, net.critic.layer1_pos)
                critic_learners[1].attach(net.critic_input_pop, net.critic.layer1_neg)
            else:
                critic_learners[0].attach(net.critic_input_pop, net.critic.layer1)
            net.critic_learners = critic_learners
        if burst_teacher is not None:
            burst_teacher.attach(
                net.cortex.output(), net.burst_h_pop,
                nest.GetConnections(net.cortex.output(), net.burst_h_pop,
                                     synapse_model="static_ch_burst"))
        net.burst_teacher = burst_teacher   # re-bolt onto the (possibly fresh) net object

    def checkpoint_and_rebuild(episode):
        nonlocal _time_offset
        print(f"[CHECKPOINT] episode={episode}: saving and rebuilding network")
        _time_offset += nest.GetKernelStatus("biological_time")
        _checkpointer.save_checkpoint(extra_state=_capture_checkpoint_extra_state())
        extra = _checkpointer.restore_from_checkpoint()
        _reattach_after_restore(extra)
        print(f"[CHECKPOINT] episode={episode}: rebuild complete")

    def restore_from_disk(path):
        # No LIVE state to save first (unlike checkpoint_and_rebuild) --
        # this runs once, at startup, in what may be a brand-new process
        # that never trained anything itself.
        print(f"[checkpoint] restoring from {path}")
        _checkpointer.load_checkpoint_from_disk(path)
        extra = _checkpointer.restore_from_checkpoint()
        _reattach_after_restore(extra)
        print(f"[checkpoint] restore complete")

    if restore_checkpoint_path:
        restore_from_disk(restore_checkpoint_path)

    # cfg.task: pluggable adapter, default "reach". "reach1d" is a
    # deliberately minimal sanity-check task (env_1d.py): same
    # cortex/actor/critic/dopamine machinery and real training loop, but
    # a 1-DOF abstract reach with no 2D kinematics or two-joint
    # coordination, to test whether the architecture can learn any task
    # at all through this exact pipeline.
    if getattr(cfg, "task", "reach") == "reach1d":
        from env_1d import Simple1DReachAdapter
        adapter = Simple1DReachAdapter(cfg)
        python_critic = None   # PythonCritic couples to the real 2-joint arm's _lo/_hi
    else:
        adapter = ReachAdapter(cfg)
        python_critic = PythonCritic(adapter.arm,
                                      feature_set=getattr(cfg, "python_critic_feature_set", "linear"))

    # python_critics_per_joint: N independent clones of PythonCritic, one
    # per action dimension, mirroring dopamine.py's n_reward_channels
    # pattern but applied to the critic side. Every signal tried so far
    # for BatchedD1Learning delivers one identical scalar to every D1
    # neuron regardless of its own preferred direction, so credit can't
    # tell which neuron's direction actually caused the outcome.
    # env_reach.py's ReachAdapter always computes last_per_joint_reward,
    # an exact additive decomposition of the reward into each joint's own
    # causal contribution, and these clones turn that into a properly
    # value-baselined per-joint delta. Only built/used when
    # actor_batched_d1_signal_source is "delta_python_directional" or
    # "delta_python_per_joint"; otherwise unused. reach1d has no
    # per-joint decomposition (1 DOF), so this is reach-task-only.
    python_critics_per_joint = None
    if (getattr(cfg, "task", "reach") != "reach1d"
            and getattr(cfg, "actor_batched_d1_signal_source", "nb_value")
            in ("delta_python_directional", "delta_python_per_joint")):
        python_critics_per_joint = [
            PythonCritic(adapter.arm, feature_set=getattr(cfg, "python_critic_feature_set", "linear"))
            for _ in range(cfg.n_actions)
        ]
    # teacher: PythonCritic is always built and trained online every
    # step, regardless of this setting -- it's a passive or active
    # observer either way. cfg.teacher only controls which signal
    # actually gets injected as what dopamine responds to. "spiking"
    # (default): raw reward injected, the spiking critic's own
    # value-population feeds DA via connect_dopamine(). "python":
    # delta_python is injected instead of raw reward -- pair with
    # cfg.critic_da_factor=0.0 in the config so the spiking critic's
    # output doesn't also blend into DA.
    teacher = getattr(cfg, "teacher", "spiking")

    print(f"Network built: cortex + {cfg.n_actions}-channel D1/D2 actor + 2-layer critic")

    def clear_rates_all():
        net.clear_rates()
        if pool_rec is not None:
            Base.clear_spikes(pool_rec)

    # ── Multi-panel spike-raster video: critic+DA, D1+D2, cortex spikes ──────
    # A sliding-window raster: each frame covers the last
    # cfg.spike_panel_window_ms of biological time, not just this step's
    # own narrow slice -- a per-step-only window made consecutive frames
    # near-disjoint and the raster flickered instead of scrolling.
    # Querying a wider, overlapping window off the recorders'
    # accumulated history gives each spike several frames of on-screen
    # life before it scrolls off the left edge.
    def _combined_id_to_idx(pops):
        """Merge several populations' GIDs into one 0-based row index,
        channel 0's neurons first, so D1/D2 panels can stack both actor
        channels in a single scatter plot."""
        m, offset = {}, 0
        for pop in pops:
            for i, gid in enumerate(pop.tolist()):
                m[gid] = offset + i
            offset += len(pop)
        return m, offset

    def _spike_raster(recs, id_to_idx, t_lo, t_hi):
        """recs: a recorder or list of recorders sharing one id_to_idx (e.g.
        per-channel D1 recorders merged into one panel). Returns (times
        relative to t_lo, row indices) covering [t_lo, t_hi)."""
        if not isinstance(recs, (list, tuple)):
            recs = [recs]
        all_t, all_r = [], []
        for rec in recs:
            ev = nest.GetStatus(rec, "events")[0]
            times, senders = np.asarray(ev["times"]), np.asarray(ev["senders"])
            mask = (times >= t_lo) & (times < t_hi)
            all_t.append(times[mask] - t_lo)
            all_r.append(np.array([id_to_idx[int(s)] for s in senders[mask]], dtype=int))
        return np.concatenate(all_t), np.concatenate(all_r)

    # _spk_hist: clear_rates_all() wipes the same recorders the
    # sliding-window raster above reads from, every cfg.clear_rates_every
    # (default 20) steps -- so any spike older than the most recent clear
    # is gone from the recorder even if it's still nominally inside the
    # sliding window, producing a periodic "blank scrolling through"
    # artifact right after each clear
    # (old spikes don't scroll off smoothly -- they vanish all at once,
    # then the window looks empty until new spikes refill it). Fix: right
    # before each clear, siphon that recorder's about-to-be-wiped events
    # into a small persistent Python-side buffer (independent of the
    # recorder, survives the clear), trimmed to the sliding-window length.
    # _spike_raster_with_history merges this buffer with the live
    # recorder's own (since-last-clear) events -- only built/touched
    # during video-recording episodes (video_frames is not None), zero
    # cost otherwise.
    _spk_hist = {}

    def _snapshot_spike_history_before_clear(panels):
        """panels: dict of key -> (recs, id_to_idx). Call once, right
        before clear_rates_all(), only when recording video this
        episode."""
        t_now = nest.GetKernelStatus("biological_time")
        window_ms = getattr(cfg, "spike_panel_window_ms", 600.0)
        for key, (recs, id_to_idx) in panels.items():
            new_t, new_r = _spike_raster(recs, id_to_idx, 0.0, t_now + 1.0)  # absolute times
            old_t, old_r = _spk_hist.get(key, (np.zeros(0), np.zeros(0, dtype=int)))
            keep = old_t >= (t_now - window_ms)
            _spk_hist[key] = (np.concatenate([old_t[keep], new_t]),
                               np.concatenate([old_r[keep], new_r]))

    def _spike_raster_with_history(key, recs, id_to_idx, t_lo, t_hi):
        """Same contract as _spike_raster, but also pulls in anything
        already-cleared-from-the-recorder-but-still-in-window from
        _spk_hist[key] (populated by _snapshot_spike_history_before_clear
        right before each clear)."""
        live_t, live_r = _spike_raster(recs, id_to_idx, t_lo, t_hi)
        hist_t, hist_r = _spk_hist.get(key, (np.zeros(0), np.zeros(0, dtype=int)))
        mask = (hist_t >= t_lo) & (hist_t < t_hi)
        return (np.concatenate([hist_t[mask] - t_lo, live_t]),
                np.concatenate([hist_r[mask], live_r]))

    # ── Training loop ─────────────────────────────────────────────────────────
    stats = {}
    recent_success = deque(maxlen=cfg.solved_history_len)
    recent_final_d = deque(maxlen=cfg.solved_history_len)
    exploration_rate = cfg.exploration_start
    t_us = [0.0]   # mutable box -- persists across control steps within an episode

    # critic_sign / signflip probe: only matters when
    # actor_batched_d1_signal_source="delta_spiking" -- the spiking
    # critic's own TD delta can converge to either +V or -V depending on
    # seed, so the sign driving the actor must be pinned once, early,
    # from data, not assumed. Frozen permanently after the probe window.
    #
    # critic_sign_override: the online probe below (corr(v_spiking,
    # episode-local realized return) over the first N episodes) is
    # confounded -- this task's reward is negative-skewed, so
    # episode-local return-to-go is systematically more negative earlier
    # in an episode regardless of true state value, and v_spiking can
    # drift with elapsed-episode-time for unrelated reasons. A
    # rollout-based V* probe over curated fixed states (no
    # time-in-episode confound) is more trustworthy: cfg.critic_sign_
    # override, when set, is used directly and the probe below is
    # skipped entirely.
    _sign_override = getattr(cfg, "critic_sign_override", None)
    critic_sign = 1.0 if _sign_override is None else float(_sign_override)
    critic_sign_frozen = _sign_override is not None
    _signflip_vs, _signflip_returns = [], []

    for episode in range(cfg.num_episodes):
        # critic_gamma_warmup_episodes (P1.2b Arm A): force gamma=0 (reward-
        # only, no bootstrap) for the first N episodes to anchor the value
        # sign, then revert to the critic's real gamma. Re-applied every
        # episode (cheap) since net.critic_learners is rebuilt on checkpoint
        # cycles and this keeps it correct without extra bookkeeping.
        if net.critic_learners and getattr(cfg, "critic_gamma_warmup_episodes", 0) > 0:
            _warm_gamma = 0.0 if episode < cfg.critic_gamma_warmup_episodes else net.critic.gamma()
            for _cl in net.critic_learners:
                _cl.gamma = _warm_gamma

        obs = adapter.reset(rng)
        # _goal_idx: this episode's commanded goal's index into
        # cfg.target_goal_set, recomputed fresh every episode. Only
        # meaningful when actor_batched_d1_context_gated is on; None
        # otherwise.
        _goal_idx = None
        if getattr(cfg, "actor_batched_d1_context_gated", False):
            _goal_set_arr = np.asarray(cfg.target_goal_set, dtype=float)
            _goal_idx = int(np.argmin(np.linalg.norm(_goal_set_arr - adapter.target, axis=1)))
        # _decoy_target: one fresh random target per episode, fed to
        # perception in place of the true commanded target when
        # target_continuous_swap_eval is on; adapter.target (reward/
        # success scoring) is untouched.
        _decoy_target = None
        if getattr(cfg, "target_continuous_swap_eval", False):
            _decoy_target = adapter.arm.sample_target(_decoy_rng)
        episode_actions, episode_rewards = [], []
        ep_v_python, ep_v_spiking = [], []
        ep_delta_python, ep_delta_spiking = [], []
        ep_burst_frac, ep_burst_h_rate = [], []
        _last_burst_w_ch_mean = _last_burst_w_ch_std = 0.0
        # cfg.freeze_weights_episode freezes NEST's own STDP rates
        # (apply_freeze -> decay_stdp), but burst_teacher's cortex->H
        # update is a separate, Python-managed weight write every step
        # (not gated by any NEST synapse rate) -- without this, a
        # freeze_weights_episode=1 greedy-eval pass would keep drifting
        # cortex->H weights throughout, defeating the point of a frozen
        # eval video.
        _burst_frozen = (getattr(cfg, "freeze_weights_episode", 0) > 0
                         and episode + 1 >= cfg.freeze_weights_episode)
        video_frames = [] if (cfg.video_interval > 0 and episode % cfg.video_interval == 0) else None

        # Spike-panel id->row maps, rebuilt once per recorded episode (cheap;
        # population identity is stable within an episode -- checkpoint_
        # and_rebuild only ever runs between episodes, never mid-episode).
        if video_frames is not None:
            # Reset every video-recording episode (not just once) -- a
            # checkpoint rebuild resets biological_time to 0 between
            # episodes, and stale absolute times from a prior episode
            # would otherwise misdate themselves against the new clock.
            _spk_hist.clear()
            _map_cortex_exc, _n_cortex_exc = _combined_id_to_idx([net.cortex.exc])
            _map_cortex_inh, _n_cortex_inh = _combined_id_to_idx([net.cortex.inh])
            _map_critic, _n_critic = _combined_id_to_idx([net.critic.features()])
            _map_da, _n_da = _combined_id_to_idx([net.dopamine.DA_neurons])
            # len(net.actor.channels), not cfg.n_actions: actor_unified_
            # readout builds exactly one D1/D2 pair with a vector
            # preference, not one pair per action dimension -- cfg.n_actions
            # channels would IndexError here.
            _n_built_channels = len(net.actor.channels)
            _map_d1, _n_d1 = _combined_id_to_idx([net.actor.d1(ch) for ch in range(_n_built_channels)])
            _map_d2, _n_d2 = _combined_id_to_idx([net.actor.d2(ch) for ch in range(_n_built_channels)])
            _d1_recs = [net.actor.get_recorder(ch, "d1") for ch in range(_n_built_channels)]
            _d2_recs = [net.actor.get_recorder(ch, "d2") for ch in range(_n_built_channels)]

        step = -1
        cum_reward = 0.0
        success = False
        for step in range(cfg.max_steps):
            net.dopamine.control_volume_transmitter(
                learning_enabled=True, warmup_until_ms=cfg.warmup_time - _time_offset)
            if actor_batched_d1:
                # Reasserted every step -- control_volume_transmitter above
                # re-enables A_plus/A_minus for ALL dopsyn-family models
                # once warmup elapses, since it operates on them together.
                nest.SetDefaults("dopsyn", {"A_plus": 0.0, "A_minus": 0.0})
            if actor_batched_d2:
                nest.SetDefaults("adopsyn", {"A_plus": 0.0, "A_minus": 0.0})
            if step % getattr(cfg, "clear_rates_every", 20) == 0:
                if video_frames is not None:
                    _snapshot_spike_history_before_clear({
                        "cortex_exc": (net.cortex.recorder_exc, _map_cortex_exc),
                        "cortex_inh": (net.cortex.recorder_inh, _map_cortex_inh),
                        "critic": (net.critic.recorders["features"], _map_critic),
                        "da": (net.dopamine.recorders["da"], _map_da),
                        "d1": (_d1_recs, _map_d1),
                        "d2": (_d2_recs, _map_d2),
                    })
                clear_rates_all()

            # ── Sensory input for this control step ─────────────────────────
            enc.reset()
            enc.observe("joints", obs["joints"])
            # target-ablation control: feed a fixed point (the target
            # box's own center) to every perception channel instead of
            # the true sampled target -- reward/distance below still use
            # the real obs["object"] unchanged, so this isolates whether
            # the policy actually reads the target vs. servos to a fixed
            # point.
            _object_in = obs["object"]
            if getattr(cfg, "target_ablation", False):
                _object_in = np.asarray(cfg.target_box_center, dtype=float)
            elif getattr(cfg, "target_swap_eval", False):
                # multi-goal swap control: feed the next goal in
                # cfg.target_goal_set (wrapping) instead of the true
                # commanded one, while reward/distance below still score
                # against the real commanded target -- a
                # target-conditioned policy follows the swapped input
                # and gets worse; a centroid-homing policy is roughly
                # unaffected.
                _goal_set = np.asarray(cfg.target_goal_set, dtype=float)
                _idx = int(np.argmin(np.linalg.norm(_goal_set - obs["object"], axis=1)))
                _object_in = _goal_set[(_idx + 1) % len(_goal_set)]
            elif getattr(cfg, "target_continuous_swap_eval", False):
                # reach-vector swap control: same idea as
                # target_swap_eval above, generalized to a continuous
                # target space -- feed a fixed-per-episode random decoy
                # target instead of the true commanded one; reward/
                # distance below still score against the real target.
                _object_in = _decoy_target
            if "gainfield" in enc.enc:
                enc.observe("gainfield", (obs["joints"], _object_in))
            if getattr(cfg, "velocity_input", False):
                enc.observe("joints_vel", obs["joints_vel"])
            if "reach_vec" in enc.enc:
                enc.observe("reach_vec", adapter.arm.fk(obs["joints"]) - _object_in)
            camera_frame = None
            camera_frame_right = None   # always None (no camera pipeline) -- kept for viz's panel dict
            enc.observe("object", _object_in)
            enc.commit()

            # nest.RunManager() calls Prepare() once on entry, and a
            # stimulation device's event schedule is computed against
            # whatever rate was in effect at that point -- changing .rate
            # mid-block is silently never picked up for the rest of that
            # block. So start_reward()/stop_reward() (both change
            # generator rates) must sit outside any active RunManager --
            # activate_time and reward_time need their own separate blocks.
            t0 = nest.GetKernelStatus("biological_time")
            with nest.RunManager():
                nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            net.dopamine.measure_da_activity(t0 + 5, t1)
            action, stat_action = adapter.select_action(
                net.actor, cfg, t0, t1, exploration_rate)
            episode_actions.append(stat_action)
            # action_deviation: env_reach.py's select_action() returns
            # stat_action as the network's raw, pre-noise readout ("mu"),
            # while `action` is the realized, clipped, post-noise action
            # -- their difference is the exploration perturbation
            # actually injected this step. Only meaningful for adapters
            # whose stat_action is the pre-noise readout (currently just
            # ReachAdapter), read only when
            # actor_batched_d1_use_action_perturbation is on.
            action_deviation = action - stat_action

            new_obs, reward, terminated, success = adapter.step(action)
            episode_rewards.append(reward)
            cum_reward += reward

            # PythonCritic: value() before update() -- update() mutates
            # self.w, so value() called after would no longer reflect the
            # weights that produced this step's delta.
            # diagnose_value_signal(t0,t1) reads the real spiking critic's
            # L2 rate over the same activation window obs was drawn from,
            # so their correlation is a fair check of whether the spiking
            # critic is learning anything that lines up with a
            # properly-converging reference.
            if python_critic is not None:
                v_python_s = python_critic.value(obs)
                v_spiking_s = net.critic.diagnose_value_signal(t0, t1)
                ep_v_python.append(v_python_s)
                ep_v_spiking.append(v_spiking_s)
                delta_python = python_critic.update(obs, reward, new_obs)
                ep_delta_python.append(delta_python)
            else:
                v_spiking_s = None
                delta_python = 0.0

            # cortex->critic-L1's own centered, delta-gated Python update --
            # independent of `teacher` (only changes how the critic itself
            # learns, not which signal drives the actor's dopamine).
            delta_spiking = None   # P1.3: resolved, centered, one-step-lagged TD delta (None = nothing resolved this step)
            if net.critic_learners:
                _v_s = v_spiking_s if v_spiking_s is not None else net.critic.diagnose_value_signal(t0, t1)
                for _cl in net.critic_learners:
                    _d = _cl.step(net.critic_input_rec, t0, t1, _v_s, reward, terminated)
                    if _d is not None:
                        delta_spiking = _d
                if delta_spiking is not None:
                    ep_delta_spiking.append(delta_spiking)

            # delta_python_vec: per-joint delta from the
            # python_critics_per_joint clones, each fed that joint's own
            # causal reward contribution instead of the pooled scalar.
            # Independent of delta_python/injected_reward above; only
            # feeds BatchedD1Learning when
            # actor_batched_d1_signal_source="delta_python_directional".
            if python_critics_per_joint is not None:
                per_joint_reward = adapter.last_per_joint_reward
                delta_python_vec = np.array([
                    python_critics_per_joint[i].update(obs, per_joint_reward[i], new_obs)
                    for i in range(len(python_critics_per_joint))
                ])
            else:
                delta_python_vec = None

            injected_reward = delta_python if (teacher == "python" and python_critic is not None) else reward
            net.dopamine.start_reward(injected_reward, cfg.pos_reward_scale, cfg.neg_reward_scale, t1)
            if getattr(cfg, "actor_per_channel_reward", False):
                # Per-channel injection is ADDITIVE to the pooled call above
                # (drives disjoint generator pools -- the pooled call still
                # feeds the critic's global value pathway unchanged).
                net.dopamine.start_reward_per_channel(
                    adapter.last_per_joint_reward, cfg.pos_reward_scale, cfg.neg_reward_scale, t1)
            # actor_burst_hidden (Path B): teaching window setup -- the
            # scalar TD error (delta_python, this step's `dev`) is
            # broadcast to each H neuron's top-down noise amplitude via
            # the fixed random feedback weight, BEFORE the reward_time
            # window below actually runs (so the modulated noise is live
            # throughout it). rec_H_burst is cleared here (not earlier)
            # so it starts the window empty -- it may have accumulated
            # decision-window spikes too, since it's wired to the same H
            # population as rec_H_bd1.
            if burst_teacher is not None and not _burst_frozen:
                Base.clear_spikes(net.burst_h_rec_burst)
                burst_teacher.set_teaching_std(delta_python, net.burst_h_ngs)
            with nest.RunManager():
                nest.Run(cfg.reward_time)
            net.dopamine.stop_reward()
            if burst_teacher is not None and not _burst_frozen:
                _t_teach_end = nest.GetKernelStatus("biological_time")
                _burst_stats = burst_teacher.apply_burst_update(
                    net._cortex_diag_rec, t0, t1, net.burst_h_rec_burst, t1, _t_teach_end)
                burst_teacher.reset_decision_std(net.burst_h_ngs)
                # Accumulated per-step, reduced to ONE value per episode
                # below (near final_d_mm/critic_w_mean) -- stats is a
                # per-EPISODE dict (one row per episode in stats.csv);
                # appending a per-step value directly here would silently
                # desync every column's row count.
                ep_burst_frac.append(_burst_stats["burst_frac_mean"])
                ep_burst_h_rate.append(_burst_stats["h_rate_hz_mean"])
                _last_burst_w_ch_mean = _burst_stats["w_ch_mean"]
                _last_burst_w_ch_std = _burst_stats["w_ch_std"]

            # ── Video-only extras (spike-raster panels, events frame, pool
            # heatmap) ───────────────────────────────────────────────────
            spk = None
            if video_frames is not None:
                t2 = nest.GetKernelStatus("biological_time")   # end of THIS step's reward window
                # Sliding window, not just this step's own [t0,t2) slice --
                # see cfg.spike_panel_window_ms's own comment (reach_state.py)
                # and the module-level comment above _combined_id_to_idx.
                t_lo = max(0.0, t2 - getattr(cfg, "spike_panel_window_ms", 600.0))
                spk = {
                    "window_ms": t2 - t_lo,
                    "cortex_exc": _spike_raster_with_history(
                        "cortex_exc", net.cortex.recorder_exc, _map_cortex_exc, t_lo, t2),
                    "cortex_inh": _spike_raster_with_history(
                        "cortex_inh", net.cortex.recorder_inh, _map_cortex_inh, t_lo, t2),
                    "n_cortex_exc": _n_cortex_exc, "n_cortex_inh": _n_cortex_inh,
                    "critic": _spike_raster_with_history(
                        "critic", net.critic.recorders["features"], _map_critic, t_lo, t2),
                    "da": _spike_raster_with_history(
                        "da", net.dopamine.recorders["da"], _map_da, t_lo, t2),
                    "n_critic": _n_critic, "n_da": _n_da,
                    "d1": _spike_raster_with_history("d1", _d1_recs, _map_d1, t_lo, t2),
                    "d2": _spike_raster_with_history("d2", _d2_recs, _map_d2, t_lo, t2),
                    "n_d1": _n_d1, "n_d2": _n_d2,
                }

            # No camera/event pipeline -- these panels are always empty,
            # kept only so viz's per-step dict (below) has the keys.
            events_frame = pool_frame = None
            events_frame_right = pool_frame_right = None

            net.dopamine.sample_nb()
            if step % 20 == 0:
                for _m in net.dopamine.baseline_model_names():
                    net.dopamine.update_da_baseline(_m)

            nb_value = net.dopamine._nb_samples[-1] if net.dopamine._nb_samples else 0.0
            if step_callback is not None:
                step_callback(delta_python, nb_value, delta_python_vec)

            if actor_batched_d1:
                # actor_batched_d1_signal_source: "nb_value" (default)
                # feeds BatchedD1Learning.step() the real spiking-DA-
                # population-derived (n-b) sample. "delta_python" instead
                # feeds it python_critic's own clean TD-error directly,
                # bypassing the spiking DA population's Poisson-injection
                # noise -- corr(delta_python, nb_value)=0.88 per-step, so
                # they mostly agree, this mainly changes noise
                # characteristics. delta_python's std is much smaller than
                # nb_value's, so actor_batched_d1_delta_python_scale
                # compensates to keep lr_scale comparable across signal
                # sources. Default "nb_value": existing behavior
                # unchanged.
                # "delta_python_directional": feeds BatchedD1Learning.step()
                # the 2-vector [delta_1, delta_2] from
                # python_critics_per_joint instead of a single scalar
                # shared by every D1 neuron -- BatchedD1Learning projects
                # this vector through each synapse's own target neuron's
                # preferred-direction vector, so different neurons receive
                # genuinely different credit.
                # "delta_python_per_joint": per-channel mode
                # (actor_unified_readout=False) -- channel ch gets its own
                # joint's delta as a plain scalar, no projection needed
                # since separate populations have no shared neurons to
                # entangle.
                _signal_source = getattr(cfg, "actor_batched_d1_signal_source", "nb_value")
                _scale = getattr(cfg, "actor_batched_d1_delta_python_scale", 236.5)
                _scale_spiking = getattr(cfg, "actor_batched_d1_delta_spiking_scale", 236.5)
                for (_ch, _kind), _bd1 in zip(batched_d1_layers, batched_d1_list):
                    if _signal_source == "delta_python_directional" and _kind == "l1":
                        # D1_L1 has no pref vector (relay, not readout --
                        # see _bd1_populations/_set_batched_d1_pref) so the
                        # per-joint VECTOR can't be projected through it
                        # (step() would crash, self.pref is None). Falls
                        # back to the POOLED scalar TD-error -- the vector's
                        # own components are an exact additive decomposition
                        # of this same scalar, so this is a real, if non-
                        # directional, credit signal, not an arbitrary
                        # substitute.
                        _bd1_signal = delta_python * _scale
                    elif _signal_source == "delta_python_directional":
                        _bd1_signal = delta_python_vec * _scale
                    elif _signal_source == "delta_python_per_joint":
                        _bd1_signal = delta_python_vec[_ch] * _scale
                    elif _signal_source == "delta_python":
                        _bd1_signal = delta_python * _scale
                    elif _signal_source == "raw_reward":
                        # raw_reward: this step's raw reward, not a TD(0)
                        # error -- no value function subtracted at all.
                        # Reuses delta_python's scale, since reward
                        # dominates delta_python early on when the
                        # critic's own V is small.
                        _bd1_signal = reward * _scale
                    elif _signal_source == "delta_spiking":
                        # delta_spiking: the reach-vec spiking critic's
                        # own delta drives the actor instead of
                        # delta_python -- critic_sign corrects seeds that
                        # converge to -V (frozen early, never re-decided).
                        # None (no transition resolved yet this step) ->
                        # 0.0, a true no-signal step, not a guess.
                        _bd1_signal = (critic_sign * delta_spiking * _scale_spiking
                                       if delta_spiking is not None else 0.0)
                    else:
                        _bd1_signal = nb_value
                    _, _, _src_rec, _tgt_rec = _bd1_populations(_ch, _kind)
                    _bd1.step(_src_rec, _tgt_rec, t0, t1, _bd1_signal,
                              action_deviation=action_deviation, goal_idx=_goal_idx)

            if actor_batched_d2:
                for _ch, _bd2 in enumerate(batched_d2_list):
                    _, _, _src_rec, _tgt_rec = _bd2_populations(_ch)
                    _bd2.step(_src_rec, _tgt_rec, t0, t1, nb_value)

            if video_frames is not None:
                d_mm = float(np.linalg.norm(adapter.target - adapter.arm.fk(adapter.arm.q))) * 1000.0
                video_frames.append({
                    "q": adapter.arm.q, "d_mm": d_mm, "cum_reward": cum_reward,
                    "ep": episode, "step": step,
                    "camera": camera_frame, "events": events_frame, "pool": pool_frame,
                    "camera_right": camera_frame_right,
                    "events_right": events_frame_right, "pool_right": pool_frame_right,
                    "spk": spk,
                })

            print(f"  [{episode:3d}|{step:3d}]  da={net.dopamine.da_rate_hz:5.1f}Hz "
                  f"base={net.dopamine.da_baseline:5.1f}  action={np.array2string(np.atleast_1d(action), precision=2)}  "
                  f"r={reward:+.4f}  cum={cum_reward:+.3f}")

            obs = new_obs
            if terminated:
                nest.Simulate(100.0)
                print(f"Episode {episode:4d}  steps={step+1:4d}  cum_reward={cum_reward:+.3f}  "
                      f"success={success}  explore={exploration_rate:.3f}")
                break

        net.dopamine.log_nb_episode()
        net.dopamine.reset_episode_samples()

        # critic_signflip probe: accumulate (v_spiking, short-horizon
        # realized return) pairs over the first
        # critic_signflip_probe_episodes episodes; freeze critic_sign
        # from their pooled correlation once that window closes.
        _signflip_n = getattr(cfg, "critic_signflip_probe_episodes", 20)
        if (not critic_sign_frozen and _signflip_n > 0
                and getattr(cfg, "actor_batched_d1_signal_source", "nb_value") == "delta_spiking"
                and len(ep_v_spiking) == len(episode_rewards) and len(ep_v_spiking) > 0):
            _g = net.critic.gamma()
            _returns = [0.0] * len(episode_rewards)
            _running = 0.0
            for _t in range(len(episode_rewards) - 1, -1, -1):
                _running = episode_rewards[_t] + _g * _running
                _returns[_t] = _running
            _signflip_vs.extend(ep_v_spiking)
            _signflip_returns.extend(_returns)
            if episode == _signflip_n - 1:
                with np.errstate(invalid="ignore"):
                    _probe_corr = np.corrcoef(_signflip_vs, _signflip_returns)[0, 1]
                critic_sign = -1.0 if (np.isfinite(_probe_corr) and _probe_corr < 0) else 1.0
                critic_sign_frozen = True
                print(f"[critic_signflip] probe corr={_probe_corr:+.4f} over "
                      f"{len(_signflip_vs)} samples ({_signflip_n} episodes) "
                      f"-> critic_sign={critic_sign:+.0f} (frozen for rest of run)")
                import json
                with open(os.path.join(cfg.output_folder, "p1_3_signflip_probe.json"), "w") as _f:
                    json.dump({
                        "probe_episodes": _signflip_n,
                        "n_samples": len(_signflip_vs),
                        "probe_corr": float(_probe_corr) if np.isfinite(_probe_corr) else None,
                        "critic_sign": critic_sign,
                    }, _f, indent=2)

        # ── Per-episode stats ─────────────────────────────────────────────────
        t_end = nest.GetKernelStatus("biological_time")
        Base.append_stat(stats, "episode", float(episode))
        Base.append_stat(stats, "cum_reward", cum_reward)
        Base.append_stat(stats, "episode_length", step + 1)
        Base.append_stat(stats, "success", float(success))
        Base.append_stat(stats, "final_d_mm",
                         float(np.linalg.norm(adapter.target - adapter.arm.fk(adapter.arm.q))) * 1000.0)
        # ee_x/y_mm + target_x/y_mm: raw final hand position + commanded
        # target, in mm -- lets post-processing compute goal selectivity
        # and distance-to-centroid without new per-config logic.
        _ee_xy_mm = adapter.arm.fk(adapter.arm.q) * 1000.0
        Base.append_stat(stats, "ee_x_mm", float(_ee_xy_mm[0]))
        Base.append_stat(stats, "ee_y_mm", float(_ee_xy_mm[1]))
        Base.append_stat(stats, "target_x_mm", float(adapter.target[0]) * 1000.0)
        Base.append_stat(stats, "target_y_mm", float(adapter.target[1]) * 1000.0)
        # decoy_x/y_mm + decoy_d_mm: the per-episode decoy target itself
        # (target_continuous_swap_eval's perception-only substitute) and
        # the final hand position's distance to it. NaN when the decoy
        # wasn't used this run. Rules out a frozen/target-blind arm
        # producing the same "distance to true target gets worse" pattern.
        _decoy_xy_mm = _decoy_target * 1000.0 if _decoy_target is not None else np.full(2, np.nan)
        Base.append_stat(stats, "decoy_x_mm", float(_decoy_xy_mm[0]))
        Base.append_stat(stats, "decoy_y_mm", float(_decoy_xy_mm[1]))
        Base.append_stat(stats, "decoy_d_mm",
                         float(np.linalg.norm(_decoy_target - adapter.arm.fk(adapter.arm.q)) * 1000.0)
                         if _decoy_target is not None else float("nan"))
        Base.append_stat(stats, "cortex_active_fraction",
                         net.cortex_active_fraction(max(t_end - 200.0, 0.0), t_end))
        if getattr(cfg, "actor_bypass_cortex_joints", False) and net.actor_bypass_pop is not None:
            # Reservoir-regime diagnostic for the fixed-hidden-layer
            # configs: reports where H sits on the silent/graded/saturated
            # spectrum so weight-sweep numbers can be read against the
            # actual operating point, not assumed from the raw weight value.
            _bp_t0 = max(t_end - 200.0, 0.0)
            events = nest.GetStatus(net.actor_bypass_rec, "events")[0]
            _bp_mask = (events["times"] >= _bp_t0) & (events["times"] < t_end)
            _bp_active = len(set(events["senders"][_bp_mask].tolist()))
            Base.append_stat(stats, "bypass_pop_active_fraction",
                             _bp_active / len(net.actor_bypass_pop))
            Base.append_stat(stats, "bypass_pop_rate_hz",
                             Base.rate_in_window(net.actor_bypass_rec, len(net.actor_bypass_pop), _bp_t0, t_end))
        Base.append_stat(stats, "critic_sign", critic_sign)
        Base.append_stat(stats, "nb_mean", net.dopamine._nb_mean)
        Base.append_stat(stats, "nb_pos_frac", net.dopamine._nb_pos_frac)

        # critic_python_corr: per-episode corr(spiking critic's L2 rate,
        # PythonCritic's V(s)), both sampled at the same per-step
        # windows, printed to check whether the critic is actually
        # learning. NaN (common early in training) is logged as 0.0, not
        # dropped, so a flat run shows up as 0.0 rather than missing data.
        if python_critic is not None and len(ep_v_spiking) >= 2:
            with np.errstate(invalid="ignore"):
                critic_corr = np.corrcoef(ep_v_spiking, ep_v_python)[0, 1]
            if not np.isfinite(critic_corr):
                critic_corr = 0.0
        else:
            critic_corr = 0.0
        Base.append_stat(stats, "critic_python_corr", critic_corr)
        print(f"[critic] spiking-vs-python V(s) corr this episode = {critic_corr:+.4f}  "
              f"(teacher={teacher})")
        # delta_python_std / delta_spiking_std: per-episode std of the
        # two raw TD-error streams, logged unconditionally so
        # actor_batched_d1_delta_spiking_scale can be calibrated directly
        # from stats.csv instead of a one-off calibration script.
        Base.append_stat(stats, "delta_python_std", float(np.std(ep_delta_python)) if len(ep_delta_python) >= 2 else 0.0)
        Base.append_stat(stats, "delta_spiking_std", float(np.std(ep_delta_spiking)) if len(ep_delta_spiking) >= 2 else 0.0)
        net.dopamine.collect_stats(stats)   # da_base, da_phasic, da_eff_mean/min/max; also resets episode_da_effective

        adapter.record_action_stats(episode_actions, stats)

        for ch_i in range(net.actor._n_built_channels):
            # d1_hidden_layer: cortex doesn't connect to d1 directly --
            # d1_l1->d1 is the pathway that actually feeds it (same
            # convention as Actor.collect_stats' own fix for this).
            _d1_l1 = net.actor.channels[ch_i].get("d1_l1")
            # actor_burst_hidden: H replaced cortex as the Actor's own
            # "cortex" input (network.py) -- the real D1/D2 connections
            # are H->D1/D2, not cortex->D1/D2, so this diagnostic read
            # must follow the same substitution or it silently reads an
            # empty (nonexistent) projection and logs 0.0/0.0 forever.
            # actor_bypass_cortex_joints: same hazard, same fix -- D1/D2
            # connect from the bypass encoder population, not cortex.
            _ctx_or_h = (net.actor_bypass_pop if getattr(cfg, "actor_bypass_cortex_joints", False)
                         else net.burst_h_pop if getattr(cfg, "actor_burst_hidden", False)
                         else net.cortex.output())
            d1_source = _d1_l1 if _d1_l1 is not None else _ctx_or_h
            conns_d1 = nest.GetConnections(d1_source, net.actor.d1(ch_i))
            conns_d2 = nest.GetConnections(_ctx_or_h, net.actor.d2(ch_i))
            w_d1 = np.array(nest.GetStatus(conns_d1, "weight")) if len(conns_d1) else np.array([0.])
            w_d2 = np.array(nest.GetStatus(conns_d2, "weight")) if len(conns_d2) else np.array([0.])
            Base.append_stat(stats, f"d1_ch{ch_i}_w_mean", float(np.mean(w_d1)))
            Base.append_stat(stats, f"d1_ch{ch_i}_w_std",  float(np.std(w_d1)))
            Base.append_stat(stats, f"d2_ch{ch_i}_w_mean", float(np.mean(w_d2)))
            Base.append_stat(stats, f"d2_ch{ch_i}_w_std",  float(np.std(w_d2)))  # log std, not just mean
            if _d1_l1 is not None:
                conns_d1l1 = nest.GetConnections(net.cortex.output(), _d1_l1)
                w_d1l1 = np.array(nest.GetStatus(conns_d1l1, "weight")) if len(conns_d1l1) else np.array([0.])
                Base.append_stat(stats, f"d1l1_ch{ch_i}_w_mean", float(np.mean(w_d1l1)))
                Base.append_stat(stats, f"d1l1_ch{ch_i}_w_std",  float(np.std(w_d1l1)))

        w_mean, w_std, w_min, w_max = net.critic.diagnose_weight_changes(net.critic_input_pop)
        Base.append_stat(stats, "critic_w_mean", w_mean)
        Base.append_stat(stats, "critic_w_std",  w_std)
        Base.append_stat(stats, "critic_w_min",  w_min)
        Base.append_stat(stats, "critic_w_max",  w_max)

        if burst_teacher is not None:
            Base.append_stat(stats, "burst_frac_mean",
                              float(np.mean(ep_burst_frac)) if ep_burst_frac else 0.0)
            Base.append_stat(stats, "burst_h_rate_hz_mean",
                              float(np.mean(ep_burst_h_rate)) if ep_burst_h_rate else 0.0)
            Base.append_stat(stats, "burst_w_ch_mean", _last_burst_w_ch_mean)
            Base.append_stat(stats, "burst_w_ch_std", _last_burst_w_ch_std)

        recent_success.append(float(success))
        success_rate = float(np.mean(recent_success))
        final_d_mm = stats["final_d_mm"][-1]
        recent_final_d.append(final_d_mm)
        final_d_rolling_mean = float(np.mean(recent_final_d))

        exploration_rate = max(cfg.exploration_min, exploration_rate * cfg.exploration_decay)
        for _m in net.dopamine.baseline_model_names():
            if actor_batched_d1 and _m == "dopsyn":
                continue   # frozen -- decay_stdp on a permanently-zeroed A_plus/A_minus would be a no-op anyway, skip for clarity
            if actor_batched_d2 and _m == "adopsyn":
                continue   # same as dopsyn's skip above, for D2
            decay_stdp(_m, cfg.decay_learning_critic if _m == "critic_dopsyn" else cfg.decay_learning)

        if actor_batched_d1:
            for (_ch, _kind), _bd1 in zip(batched_d1_layers, batched_d1_list):
                committed = _bd1.end_episode()
                if committed:
                    _w = np.array(nest.GetStatus(_bd1.conns, "weight"))
                    print(f"[batched_d1] channel={_ch} ({_kind})  episode={episode}  commit #{_bd1.n_commits}  "
                          f"w_mean={_w.mean():.2f} w_std={_w.std():.2f}")

        if actor_batched_d2:
            for _ch, _bd2 in enumerate(batched_d2_list):
                committed = _bd2.end_episode()
                if committed:
                    _w = np.array(nest.GetStatus(_bd2.conns, "weight"))
                    print(f"[batched_d2] channel={_ch}  episode={episode}  commit #{_bd2.n_commits}  "
                          f"w_mean={_w.mean():.2f} w_std={_w.std():.2f}")

        if cfg.freeze_weights_episode > 0 and episode + 1 == cfg.freeze_weights_episode:
            exploration_rate = apply_freeze(episode, cfg, net.dopamine.baseline_model_names())

        print(f"{'-'*70}")
        # final_d_mm (rolling mean over cfg.solved_history_len episodes, same
        # window success_rate used) tracked as the PRIMARY trend metric
        # instead of success_rate -- success is a binary/rare event (tight
        # success_radius=0.015m within max_steps=50), so a genuinely
        # improving-but-not-yet-successful policy can look completely flat
        # on success_rate while final distance is actually shrinking.
        # success/success_rate still computed and written to stats.csv
        # (cheap, still useful), just no longer the printed headline.
        print(f"  Episode {episode:4d}  cum_reward={cum_reward:+7.3f}  success={success}  "
              f"final_d_mm={final_d_mm:7.2f}  final_d_mm({cfg.solved_history_len})={final_d_rolling_mean:7.2f}  "
              f"steps={step+1:4d}")
        print(f"  cortex_active_fraction={stats['cortex_active_fraction'][-1]:.2f}  "
              f"critic w_mean={w_mean:.3f} w_std={w_std:.3f}  success_rate({cfg.solved_history_len})={success_rate:.2f}")
        _ab = "  ".join(
            f"{_m}: A+={nest.GetDefaults(_m)['A_plus']:+.5f} A-={nest.GetDefaults(_m)['A_minus']:+.5f}"
            for _m in net.dopamine.baseline_model_names()
        )
        print(f"  {_ab}")
        print(f"{'-'*70}\n")

        futils.write_stats_csv(stats, cfg.output_folder)

        if video_frames:
            video_path = os.path.join(cfg.output_folder, "videos", f"ep_{episode:04d}.mp4")
            viz.render_episode_video(video_path, adapter.arm, cfg.mode, video_frames,
                                     adapter.target, success, cfg.success_radius)

        if cfg.spike_pdf_interval > 0 and episode % cfg.spike_pdf_interval == 0:
            du.print_epoch_spikes(episode, net, os.path.join(cfg.output_folder, "debug"))

        if episode_callback is not None:
            episode_callback(net, episode, enc, adapter)

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            checkpoint_and_rebuild(episode)

    futils.write_stats_csv(stats, cfg.output_folder)
    viz.render_summary(stats, cfg.output_folder)

    if getattr(cfg, "critic_probe_at_end", False):
        _critic_static_probe(net, enc, cfg, adapter, python_critic, rng)

    if getattr(cfg, "critic_vstar_probe_at_end", False):
        _critic_vstar_probe(net, enc, cfg, adapter, python_critic, rng)

    if getattr(cfg, "cortex_decode_probe_at_end", False):
        _cortex_decode_probe(net, enc, cfg, adapter, python_critic, rng)

    _ckpt_path = os.path.join(cfg.output_folder, "final_checkpoint.pkl")
    _checkpointer.save_checkpoint_to_disk(_ckpt_path, extra_state=_capture_checkpoint_extra_state())
    print(f"[checkpoint] saved final trained state to {_ckpt_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Config()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spiking reach trainer")
    parser.add_argument("--config",   type=str, required=True,
                        help="config file, e.g. experiments/configs/reach_state.py")
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out-dir",  type=str, default=None,
                        help="override the default experiments/runs/<timestamp>_<name>_seed<N> folder")
    parser.add_argument("--restore-checkpoint", type=str, default=None,
                        help="path to a final_checkpoint.pkl (every run saves one, "
                             "unconditionally) from a prior run -- loaded before training "
                             "starts, e.g. to continue training or (with --episodes N and "
                             "cfg.freeze_weights_episode=1) run a pure greedy eval/video "
                             "pass with no retraining")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.verbose:
        cfg.verbose = True

    _seed = cfg.seed if cfg.seed >= 0 else "rnd"
    _stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    _run_id = f"{_stamp}_{cfg.name}_seed{_seed}"
    if args.out_dir is not None:
        cfg.output_folder = os.path.normpath(args.out_dir)
    else:
        cfg.output_folder = os.path.join("experiments", "runs", _run_id)
    cfg.clean_output = False
    os.makedirs(cfg.output_folder, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.output_folder, "config_used.py"))
    print(f"[Run] {_run_id}")
    print(f"[Run] output -> {cfg.output_folder}")

    train(cfg, restore_checkpoint_path=args.restore_checkpoint)
