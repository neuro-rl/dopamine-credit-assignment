"""CartPole dopamine-STDP credit-assignment model: single shared network
builder + flag-parameterized training/eval loop, replacing four
near-identical nest_cartpole_*_test.py modules (noCentNoAction/
uncentered/centered/vec2d_proj). Diffing all four pairwise
showed their network construction (cortex via state_encoder.py, D1/D2,
critic via td_critic.py) was already byte-identical; only run_episode's
credit/decode logic varied, along three orthogonal flags:

  - action_identity: credit = td0*eps (True) vs td0 alone (False).
    eps = a - tanh(g/g_scale) carries which action was actually taken
    (a = +-1) vs the network's own continuous lean (g).
  - centered: the presynaptic term used in the weight update is
    (x_j - x_bar_j), the cortex neuron's activity relative to its own
    running-mean baseline (True), vs raw x_j (False) -- the fix for D1's
    weight-std collapse during training.
  - population_readout: per-D1/D2-neuron preferred-direction decode and
    credit (vec2d_proj's population-vector projection) vs a single
    scalar total-count decode/credit shared by every D1/D2 neuron
    (every other condition).

See cart_pole/configs/*.py for which combination each locked condition
uses, and cart_pole/run.py for how a config drives this module.

Dropped in this consolidation: nest_cartpole_centered_test.py's optional
hidden-layer scaffold (hidden_layer/hidden_rule="dopamine"/"static"/
"burst"). It defaulted to off (hidden_layer=False) in every config ever
actually used -- none of the four locked E1 conditions exercise it -- so
it was unreachable dead weight here. Available in git history if needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import nest
import numpy as np

from state_encoder import (N_EXC, build_input_and_cortex, normalize_state,
                            extended_obs, set_state_rates)
from td_critic import critic_features, critic_update, N_CRITIC_FEATURES

N_D1 = 10
N_D2 = 10

# population_readout decode (vec2d_proj): each D1/D2 neuron's preferred
# direction spread over a HALF circle (0..pi), so its x-component
# (cos(angle)) spans the full [-1,1] range but DENSER near +-1 (cos's
# derivative is 0 at the extremes) -- the opposite distribution from
# linspace(-1,1,N_D1), which clustered several neurons near zero and
# starved them of credit in an earlier (abandoned) raw-magnitude variant.
PREF_ANGLE = np.linspace(0, np.pi, N_D1)
PREF_VEC = np.stack([np.cos(PREF_ANGLE), np.sin(PREF_ANGLE)], axis=1)   # (N_D1, 2)
ALIGN_AXIS = PREF_VEC[:, 0]   # shared credit-weighting axis for D1 and D2

DECISION_WINDOW_MS = 60.0
EPSILON_EXPLORE = 0.3
XBAR_ALPHA = 0.05     # EMA rate for each cortex neuron's running-mean activity
PLASTICITY_LR = 0.1   # scales credit*(presynaptic term) into a weight delta
WMIN, WMAX = 50.0, 800.0

# one-element lists so run_episode can mutate them in place, no `global` needed
FLIP_COUNT = [0]
STEP_COUNT = [0]


def build_network(init_lo=100.0, init_hi=200.0, seed=1, restore=None,
                   mutual_inhibition=False, mutual_inhibition_weight=-30.0):
    nest.ResetKernel()
    # local_num_threads pinned to 1 explicitly (2026-11-07): NEST's RNG
    # stream assignment depends on the number of virtual processes, so
    # leaving this to an unset/environment-dependent default would make
    # "same seed" reproducibility depend on whatever OMP_NUM_THREADS
    # happens to be set to on a given machine, rather than being
    # guaranteed by the seed alone.
    nest.SetKernelStatus({"rng_seed": seed, "resolution": 0.1,
                           "local_num_threads": 1})

    pg_vars, cortex = build_input_and_cortex()
    all_cortex = cortex.output()

    pop_D1 = nest.Create("iaf_psc_alpha", N_D1)
    pop_D2 = nest.Create("iaf_psc_alpha", N_D2)
    rec_D1 = nest.Create("spike_recorder")
    rec_D2 = nest.Create("spike_recorder")
    nest.Connect(pop_D1, rec_D1)
    nest.Connect(pop_D2, rec_D2)

    # mutual_inhibition: real, static, bidirectional D1<->D2
    # lateral inhibition -- the arm's actor already has this on by default
    # (model/actor.py's disable_cross_inhibition); CartPole's D1/D2 had
    # NO connection to each other at all (the "opponent" push was purely
    # the Python sign of their spike-count difference). Flat populations
    # here (no grid/topology), so all-to-all static inhibition is the
    # direct analogue of the arm's topographic Gaussian cross-inhibition.
    # Weight is NOT yet calibrated against a target D1/D2 firing-rate
    # regime the way the arm's was -- tune mutual_inhibition_weight and
    # check rates before trusting any result built on this.
    if mutual_inhibition:
        nest.CopyModel("static_synapse", "static_d1_to_d2")
        nest.CopyModel("static_synapse", "static_d2_to_d1")
        nest.Connect(pop_D1, pop_D2, "all_to_all",
                     syn_spec={"synapse_model": "static_d1_to_d2",
                               "weight": mutual_inhibition_weight, "delay": 1.0})
        nest.Connect(pop_D2, pop_D1, "all_to_all",
                     syn_spec={"synapse_model": "static_d2_to_d1",
                               "weight": mutual_inhibition_weight, "delay": 1.0})

    # plain static synapses -- NEST does NO automatic plasticity; weights
    # are set explicitly from Python every step.
    nest.CopyModel("static_synapse", "static_d1")
    nest.CopyModel("static_synapse", "static_d2")
    nest.Connect(all_cortex, pop_D1, "all_to_all",
                 syn_spec={"synapse_model": "static_d1",
                           "weight": nest.random.uniform(init_lo, init_hi), "delay": 1.0})
    nest.Connect(all_cortex, pop_D2, "all_to_all",
                 syn_spec={"synapse_model": "static_d2",
                           "weight": nest.random.uniform(init_lo, init_hi), "delay": 1.0})

    exc_ids = cortex.exc.tolist()
    d1_ids = pop_D1.tolist()
    d2_ids = pop_D2.tolist()
    exc_idx = {nid: i for i, nid in enumerate(exc_ids)}
    d1_idx = {nid: i for i, nid in enumerate(d1_ids)}
    d2_idx = {nid: i for i, nid in enumerate(d2_ids)}

    conns_d1 = nest.GetConnections(synapse_model="static_d1")
    conns_d2 = nest.GetConnections(synapse_model="static_d2")
    # cache source/target -> row/col index once (topology is fixed given seed)
    d1_src = np.array([exc_idx[int(s)] for s in conns_d1.get("source")])
    d1_tgt = np.array([d1_idx[int(t)] for t in conns_d1.get("target")])
    d2_src = np.array([exc_idx[int(s)] for s in conns_d2.get("source")])
    d2_tgt = np.array([d2_idx[int(t)] for t in conns_d2.get("target")])

    if restore is not None:
        W_d1 = restore["W_d1"].copy()
        W_d2 = restore["W_d2"].copy()
        x_bar = restore["x_bar"].copy()
        w_critic = restore["w_critic"].copy()
        conns_d1.set(weight=W_d1[d1_src, d1_tgt].tolist())
        conns_d2.set(weight=W_d2[d2_src, d2_tgt].tolist())
    else:
        W_d1 = np.zeros((N_EXC, N_D1))
        W_d2 = np.zeros((N_EXC, N_D2))
        # element-wise fancy-index assignment
        W_d1[d1_src, d1_tgt] = conns_d1.get("weight")
        W_d2[d2_src, d2_tgt] = conns_d2.get("weight")
        x_bar = np.zeros(N_EXC)
        w_critic = np.zeros(N_CRITIC_FEATURES)

    # g_bar: scalar running mean of D1's own total spike count, used only
    # by d1_only mode (see run_episode) as D1's self-referential decision
    # baseline in place of subtracting D2 -- tracked unconditionally
    # (cheap, one float) so every other condition's state dict shape is
    # unaffected and restore() is backward-compatible with older
    # checkpoints that never had this key.
    g_bar = restore.get("g_bar", 0.0) if restore is not None else 0.0

    return dict(pg_vars=pg_vars, cortex=cortex, exc_ids=exc_ids,
                pop_D1=pop_D1, pop_D2=pop_D2, rec_D1=rec_D1, rec_D2=rec_D2,
                conns_d1=conns_d1, conns_d2=conns_d2,
                d1_src=d1_src, d1_tgt=d1_tgt, d2_src=d2_src, d2_tgt=d2_tgt,
                W_d1=W_d1, W_d2=W_d2, x_bar=x_bar, g_bar=g_bar,
                w_critic=w_critic, seed=seed)


def extract_state(net):
    return dict(W_d1=net["W_d1"].copy(), W_d2=net["W_d2"].copy(),
                x_bar=net["x_bar"].copy(), g_bar=net["g_bar"],
                w_critic=net["w_critic"].copy())


def read_and_clear(rec, ids):
    events = rec.get("events")
    senders = np.asarray(events["senders"])
    counts = np.array([np.sum(senders == nid) for nid in ids], dtype=float)
    rec.set({"n_events": 0})
    return counts


def run_episode(net, env, max_steps=500, verbose=False, rng=None, learn=True, epsilon=None,
                 action_identity=True, centered=True, population_readout=False,
                 action_centered=True, d1_only=False):
    # d1_only: removes D2 from the decision and from learning entirely --
    # tests whether the opponent D1/D2 architecture is actually necessary,
    # or whether a single feedforward population does just as well. D2 is
    # still built and simulated (run.py/run_io.py's shared logging assumes
    # it exists), but its spikes are read and discarded, its weights are
    # never touched, and it has zero influence on g or on D1's credit.
    # g becomes D1's own count minus g_bar, a scalar running mean of that
    # same count (the self-referential analogue of D1-D2, replacing "the
    # other population's current count" with "my own recent average").
    # No opponent sign-mirroring: only W_d1 is updated, by itself.
    #
    # action_centered: when
    # action_identity=True, selects which action-dependent factor is used
    # -- eps = a - tanh(g/g_scale) (True, the existing "surrogate-centered"
    # rule, unchanged default) vs eps = a alone (False, "action-dependent,
    # uncentered" -- isolates whether the tanh(g/g_scale) subtraction
    # itself matters, as opposed to the mere presence of action
    # information, which the original action_identity on/off flag
    # conflates). No effect when action_identity=False (credit=td0 either
    # way) or population_readout=True (not wired there; not needed for
    # this ablation, which targets the scalar-decode conditions).
    rng = rng or np.random.default_rng()
    d1_ids = net["pop_D1"].tolist()
    d2_ids = net["pop_D2"].tolist()
    exc_ids = net["exc_ids"]
    # vec2d_proj (population_readout) recalibrated G_SCALE for its very
    # different decode range (measured std~1.35, range [-2.97,4.61] vs
    # the scalar total-count decode's own range) -- every other
    # condition used 1.0.
    g_scale = 2.0 if population_readout else 1.0

    state, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        state_norm = normalize_state(state)
        obs6 = extended_obs(state_norm)
        set_state_rates(net["pg_vars"], obs6)

        nest.Simulate(DECISION_WINDOW_MS)
        d1_counts = read_and_clear(net["rec_D1"], d1_ids)
        # d2_counts still read and cleared even in d1_only mode (keeps the
        # recorder from accumulating unboundedly); just never used below.
        d2_counts = read_and_clear(net["rec_D2"], d2_ids)
        x_j = read_and_clear(net["cortex"].recorder_exc, exc_ids)
        net["cortex"].clear_rates()   # clears inh recorder too (exc already read above)
        d1_sum = d1_counts.sum()

        if population_readout:
            # population-vector projection: REQUIRES within-D1/D2
            # heterogeneity to carry information (see credit_vec below).
            R_D1 = d1_counts @ PREF_VEC
            R_D2 = d2_counts @ PREF_VEC
            g = float(R_D1[0] - R_D2[0])
        elif d1_only:
            # self-referential decision: D1's own count relative to its
            # own recent average, instead of relative to D2's count.
            g = d1_sum - net["g_bar"]
        else:
            # total-count decode (NOT decode_axis/population-vector,
            # which is blind to the across-neuron structure centering
            # builds -- confirmed directly: frozen target weights scored
            # 11.5 via decode_axis vs 183.4 via total-count on the
            # identical network).
            g = d1_sum - d2_counts.sum()
        greedy_action = 1 if g > 0 else 0
        eps_this_step = EPSILON_EXPLORE if epsilon is None else epsilon
        explore = rng.random() < eps_this_step
        action = (1 - greedy_action) if explore else greedy_action
        STEP_COUNT[0] += 1
        # flip = exploration overrode the network's own greedy choice
        if action != greedy_action:
            FLIP_COUNT[0] += 1

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        feat_s = critic_features(obs6)
        feat_next = critic_features(extended_obs(normalize_state(new_state)))
        td0 = critic_update(net["w_critic"], feat_s, feat_next, reward, terminated)

        # x_bar tracked unconditionally (even during a learn=False critic
        # warm-up) so it's already a real running mean, not cold-started
        # at zero, by the time plasticity actually turns on. `pre_x_j` uses
        # the OLD x_bar (pre-update), matching the original per-condition
        # modules exactly.
        pre_x_j = (x_j - net["x_bar"]) if centered else x_j
        net["x_bar"] = (1.0 - XBAR_ALPHA) * net["x_bar"] + XBAR_ALPHA * x_j
        # g_bar tracked unconditionally too (cheap, harmless when unused),
        # same alpha as x_bar, so it is already settled by the time
        # d1_only's first learning step runs.
        net["g_bar"] = (1.0 - XBAR_ALPHA) * net["g_bar"] + XBAR_ALPHA * d1_sum

        if learn:
            if population_readout:
                # PER-NEURON credit: generalizes action-identity from a
                # single population-wide scalar to one value per D1/D2
                # neuron, weighted by how well that neuron's own
                # preferred direction matches the action actually taken
                # (classic preferred-direction/cosine-tuning credit).
                a = 1.0 if action == 1 else -1.0
                eps = a - np.tanh(g / g_scale)
                credit_vec = td0 * eps * ALIGN_AXIS   # (N_D1,) -- same axis reused for D2
                delta_d1 = PLASTICITY_LR * np.outer(pre_x_j, credit_vec)
                delta_d2 = -PLASTICITY_LR * np.outer(pre_x_j, credit_vec)
            else:
                if action_identity:
                    a = 1.0 if action == 1 else -1.0
                    eps = (a - np.tanh(g / g_scale)) if action_centered else a
                    credit = td0 * eps
                else:
                    credit = td0
                delta_d1 = (PLASTICITY_LR * credit * pre_x_j)[:, None]
                delta_d2 = (-PLASTICITY_LR * credit * pre_x_j)[:, None]
            net["W_d1"] += delta_d1
            np.clip(net["W_d1"], WMIN, WMAX, out=net["W_d1"])
            net["conns_d1"].set(weight=net["W_d1"][net["d1_src"], net["d1_tgt"]].tolist())
            # d1_only: D2 is never updated -- it sits frozen at its
            # initial random weights for the whole run, and has already
            # been excluded from g/credit above. This is what makes the
            # ablation genuine: not just "ignore D2's output," but "D2
            # learns nothing and could be deleted with no behavioral
            # change."
            if not d1_only:
                net["W_d2"] += delta_d2
                np.clip(net["W_d2"], WMIN, WMAX, out=net["W_d2"])
                net["conns_d2"].set(weight=net["W_d2"][net["d2_src"], net["d2_tgt"]].tolist())

        if verbose:
            _credit = locals().get("credit", float("nan"))
            print(f"  step {step:3d} action={action} g={g:+.3f} reward={reward:.1f} "
                  f"td0={td0:+.3f} credit={_credit:+.3f}")

        if done:
            break
        state = new_state

    return total_reward
