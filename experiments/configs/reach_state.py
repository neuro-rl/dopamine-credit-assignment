"""reach_state.py — canonical config, fully self-contained (no import chain).
Base fields carried over from reach_state_topo2.py; overrides/new fields
for this session's setup (6x6 unified-vector D1 actor, D2 stabilizer,
python-teacher critic) are noted inline, briefly.
"""
import dataclasses


@dataclasses.dataclass
class Config:
    # ── Identity ──────────────────────────────────────────────────────────
    name:                   str   = "reach_state"
    mode:                   str   = "state"
    encoder_type:           str   = "topo_both"

    # ── NEST kernel ──────────────────────────────────────────────────────
    nest_resolution:        float = 1.0
    # Single-threaded: multi-threaded NEST gave non-reproducible results
    # across runs with the same seed, likely scheduling nondeterminism in
    # spike delivery order. Slower per episode, but reproducible.
    num_threads:            int   = 1
    seed:                   int   = -1

    # ── Training schedule ──────────────────────────────────────────────────
    num_episodes:           int   = 400
    max_steps:               int  = 100
    solved_history_len:     int   = 20

    # ── Timing (ms) ─────────────────────────────────────────────────────────
    activate_time:           int  = 50
    reward_time:              int = 25
    warmup_time:            float = 20000

    # ── Task ────────────────────────────────────────────────────────────────
    q_step:                 float = 0.08
    success_radius:         float = 0.015
    n_actions:               int  = 2
    reward_shaping:          str  = "delta"

    # ── Encoders ────────────────────────────────────────────────────────────
    joints_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        bins=15, lo=(-1.745, -1.690), hi=(1.745, 1.690), max_rate=400.0, sigma_bins=1.0))
    object_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        bins=15, lo=(0.20, 0.00), hi=(0.40, 0.25), max_rate=400.0, sigma_bins=1.0))
    # object_raw_encoder_kw: only used when
    # encoder_type="grid2d_joints_lineartarget" -- same
    # lo/hi range as object_encoder_kw (the target's real sample range),
    # but for LinearRateEncoder (2 raw linear features, no place field;
    # bins/sigma_bins don't apply to that class).
    object_raw_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        lo=(0.20, 0.00), hi=(0.40, 0.25), min_rate=0.0, max_rate=400.0))
    # weight/indegree calibrated for cortex_active_fraction in the 10-30Hz
    # band with spatially-restricted fixed_indegree.
    joints_connect_kw: dict = dataclasses.field(default_factory=lambda: dict(
        weight=18.0, delay=0.5, radius=4.8, indegree=10, spatial_indegree=True))
    object_connect_kw: dict = dataclasses.field(default_factory=lambda: dict(
        weight=18.0, delay=0.5, radius=4.8, indegree=10, spatial_indegree=True))
    # velocity_input: adds a THIRD topographic stream (joint velocity,
    # actual_dq from the just-completed step) alongside joints/object,
    # same place-field style as joints_encoder_kw, re-centred on
    # velocity's own +/-q_step range. Default off -- joints/object are
    # unchanged either way (see main.py's _build_encoders).
    velocity_input:           bool = False
    # actor_bypass_cortex_joints: D1/D2 read the joints+object encoder
    # output directly, skipping the recurrent cortex entirely for the
    # actor's pathway -- the recurrent cortex was measured to collapse the
    # combined encoder's reach_vec signal. Cortex itself is still built
    # and still receives joints/object normally; only what the actor reads
    # is redirected (network.py's _actor_cortex_input).
    actor_bypass_cortex_joints: bool = False
    # actor_bypass_fixed_hidden: only meaningful alongside
    # actor_bypass_cortex_joints=True -- inserts a frozen random
    # feedforward hidden layer (iaf_psc_alpha) between the bypass encoder
    # and D1/D2, cached so it survives checkpoint rebuilds unchanged.
    # Only the H->D1/D2 readout trains. See network.py.
    actor_bypass_fixed_hidden: bool = False
    actor_fixed_hidden_size:      int = 750
    actor_fixed_hidden_indegree:  int = 30
    actor_fixed_hidden_weight:  float = 300.0
    # actor_gainfield_hidden: mutually exclusive alternative to
    # actor_bypass_fixed_hidden -- a fixed conjunctive basis (place(q)
    # plus place(q)*target, see
    # encoders.py's GainFieldEncoder) spanning the q-dependent-linear-
    # map-of-target function class the resolved-rate IK action actually
    # needs, instead of a random basis hoping to span it by luck. D1/D2
    # reads a real iaf_psc_alpha relay (one_to_one, fixed weight, no RNG
    # -- deterministic, so no connectivity caching needed) fed from the
    # encoder's 3K parrot units; efference targets that relay, not the
    # bare parrots. See network.py/main.py.
    actor_gainfield_hidden: bool = False
    actor_gainfield_ff_weight: float = 300.0
    gainfield_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        bins=12, lo_q=(-1.745, -1.690), hi_q=(1.745, 1.690),
        lo_t=(0.20, 0.00), hi_t=(0.40, 0.25), max_rate=400.0, sigma_bins=1.0))
    joints_vel_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        bins=15, lo=(-0.08, -0.08), hi=(0.08, 0.08), max_rate=400.0, sigma_bins=1.0))
    joints_vel_connect_kw: dict = dataclasses.field(default_factory=lambda: dict(
        weight=18.0, delay=0.5, radius=4.8, indegree=10, spatial_indegree=True))

    # ── Network sizes ─────────────────────────────────────────────────────
    cortex_extent:            float = None
    cortex_exc_shape:        tuple = (12, 12)
    cortex_inh_shape:        tuple = (6, 6)
    cortex_bg_n_sources:      int  = 50
    cortex_bg_rate_hz:       float = 300.0
    cortex_bg_p:             float = 0.10
    cortex_bg_w_exc:         float = 1.0
    cortex_bg_w_inh:         float = 1.0
    cortex_bg_shared:        bool  = False
    cortex_bg_one_to_one:    bool  = True
    cortex_w_ei:              float = 100.0

    # 6x6 D1/D2 (36 neurons) -- small enough for fast iteration, big enough
    # to keep chance correlation in the vector readout small (scales ~1/sqrt(N)).
    actor_grid_shape:        tuple = (6, 6)
    # indegree=12 at N=36 -- calibrated (indegree_sweep_6x6.py) to keep D1
    # active_fraction in the healthy 0.28-0.48 band (measured mean 0.375).
    actor_k_ctx_d1_scale:   float = 1.152
    actor_k_ctx_d2_scale:   float = 1.152

    # Single merged D1 population reads out the full 2D action vector
    # (population-vector / weighted-spike-sum decode) instead of one D1/D2
    # pair per joint. D2 stays cortex-driven but purely regulatory (see
    # actor_d2_stabilizer below), not a second vote.
    actor_unified_readout:    bool = True
    actor_vector_readout:     bool = True
    # axis_aligned_pref: split D1 into pure-q1/pure-q2 sub-populations
    # (pref on one axis only) instead of random continuous 2D directions --
    # see actor.py's axis_aligned_pref comment.
    actor_axis_aligned_pref:  bool = False
    # axis_pref_graded: within each axis_aligned_pref group, spread
    # preference MAGNITUDE (cos(linspace(0,pi,m)), denser near +-1) instead
    # of a plain +-1 split -- see actor.py's _draw_axis_aligned_pref
    # docstring. No-op unless actor_axis_aligned_pref is also True.
    actor_axis_pref_graded:   bool = False
    # readout_baseline_correction: centers the ACTUAL behavioral readout
    # (weighted_rate_in_window/compute_heads), not just the learning
    # signal (that's actor_batched_d1_presyn_baseline_correction, a
    # separate mechanism) -- see actor.py's own comment. Default False:
    # unchanged.
    actor_readout_baseline_correction: bool = False
    actor_readout_baseline_tau: float = 50000.0
    # NOT arbitrary: swept 8 seeds at this grid/indegree, this is the only
    # one where corr(D1 spike_count, pref) < 0.05 on BOTH action dims.
    # Old default (12345) had corr=+0.29 on q2 -> that joint saturated
    # against its own limit for an entire 400-episode run. Re-sweep with
    # check_pref_firing_correlation.py before changing this.
    actor_vector_readout_seed: int  = 8675309
    actor_per_channel_reward: bool = False   # doesn't compose with unified_readout
    # D2 doesn't vote on the action; instead does wide, asymmetric
    # suppression of specific D1 neurons. weight_mult=3.0 -> ~34% D1
    # reduction (calibrated, d2_stabilizer_calibrate.py).
    actor_d2_stabilizer:      bool = True
    actor_d2_stabilizer_weight_mult:  float = 3.0
    actor_d2_stabilizer_radius_frac:  float = 1.0

    # Context-gated dopamine: per-zone DA population/volume transmitter.
    actor_context_gated_da:  bool  = True
    actor_baseline_zone_grid: tuple = (1, 1)
    actor_da_zone_v_th:     float  = -52.0
    actor_da_zone_tonic_weight: float = 1.0

    critic_l1_shape:         tuple = (10, 10)

    # ── Actor neuron bias ──────────────────────────────────────────────────
    ie_d1:                   float = 115.0
    ie_d2:                   float = 160.0
    el_d1:                    float = -65.0
    el_d2:                    float = -65.0
    v_reset_d1:                float = -65.0
    v_reset_d2:                float = -65.0
    tau_syn_ex_actor:          float = 5.0
    # D1-only spike-frequency adaptation: swaps D1 to aeif_psc_alpha with
    # a spike-triggered adaptation current when b>0 (see actor.py's
    # d1_adapt_b comment). Default 0.0 = plain iaf_psc_alpha, unchanged.
    actor_d1_adapt_b:          float = 0.0
    actor_d1_adapt_a:          float = 0.0
    actor_d1_adapt_tau_w:      float = 200.0
    tau_syn_ex_critic_l1:      float = None

    # ── Dopamine / STDP ─────────────────────────────────────────────────────
    da_neurons:               int   = 200
    da_convert_hz_to_nest:    float = 9.7e-4
    da_tau_plus:              float = 20.0
    da_tau_c:                 float = 200.0
    da_tau_c_adopsyn:         float = 150.0
    da_tau_n:                 float = 20.0
    tau_syn_ex_da:            float = 5.0
    da_baseline_init:         float = 17.1
    da_tau_m:                 float = 10.
    da_tonic_rate_hz:         float = 600.0
    b_meas_correction_factor: float = 1.
    # da_baseline_fixed: when True, `b` is pinned at its initial value for
    # the whole run instead of continuously EMA-tracked (see dopamine.py's
    # da_baseline_fixed comment). Default False = unchanged behavior.
    da_baseline_fixed:        bool  = False
    # 0.0: real spiking critic doesn't drive DA (see `teacher` below). It
    # still trains passively via critic_dopsyn the whole time.
    critic_da_factor:         float = 0.0
    critic_d:                 float = 30.0
    critic_tau_r:             float = 1000.0
    # The critic's one plastic layer (L1) trains via a Python-computed,
    # centered-presynaptic, delta-gated update -- the actor's own
    # working-rule pattern (see batched_actor_learning.py) -- instead of
    # cortex->L1's native critic_dopsyn STDP. Independent of `teacher`:
    # this only changes HOW the spiking critic's own cortex->L1 weights
    # update, not which signal drives the actor's dopamine.
    # delta has a real mean bias and large noise; a higher learning rate
    # drove w_mean toward Wmin within a handful of episodes, so this is
    # reduced well below the naive value.
    critic_centered_lr:       float = 0.0005
    critic_centered_xbar_alpha: float = 0.05  # same EMA convention as the actor's presyn centering
    critic_centered_wmin:     float = 5.0     # matches critic_dopsyn's own Wmin/Wmax
    critic_centered_wmax:     float = 15.0
    # critic_centered_delta_bar_alpha: delta itself has a persistent mean
    # bias, which centering the presynaptic term alone doesn't fix -- an
    # EMA baseline of delta is subtracted before the update too (see
    # spiking_critic_learning.py), same idea as batched_actor_learning.py's
    # `center_nb_per_batch` for the actor's own dopamine signal.
    critic_centered_delta_bar_alpha: float = 0.05
    # critic_centered_delta_centering: set False to skip the delta-bar
    # subtraction above and use raw delta in the update instead.
    critic_centered_delta_centering: bool = True
    # critic_gamma_warmup_episodes: for the first N episodes,
    # SpikingCriticLearning's gamma is forced to 0 (reward-only, no
    # bootstrap) to anchor the value sign before TD's bootstrap term can
    # lock onto -V; reverts to net.critic.gamma() after.
    critic_gamma_warmup_episodes: int = 0
    # critic_signed_pools: a single L1 population's firing RATE can only
    # represent V(s)>=0 -- a real ceiling, since this task's reward is
    # negative-skewed (plenty of states have negative true value). Splits
    # L1 into two half-size opponent pools (layer1_pos/layer1_neg),
    # mirroring the actor's own D1/D2 convention -- NOT one-to-one wiring,
    # each pool stays DENSELY connected to DA (same pooling/robustness as
    # a single population). V(s) = rate(pos) - rate(neg). layer1_pos
    # trains with +critic_centered_lr, layer1_neg with the SAME rule
    # negated (-critic_centered_lr) -- when delta>0 pos grows AND neg
    # shrinks, both raising V(s) together, exactly mirroring dopsyn(D1)/
    # adopsyn(D2)'s cooperation under one shared dopamine signal.
    # connect_dopamine() wires the two pools with opposite sign too, so
    # the real-time network-computed TD signal is ALSO signed, not just
    # the Python-side diagnostic. Default False: unchanged behavior.
    critic_signed_pools:      bool  = False
    # critic_probe_at_end: after training finishes, run a static
    # frozen-readout probe (main.py's _critic_static_probe) -- holds all
    # plasticity off and presents curated (joints, target) states spanning
    # the trained python_critic's value range, to check whether V_spiking
    # carries real between-state information or is flat decode noise.
    # Default False: unchanged behavior/runtime for every other run.
    critic_probe_at_end:      bool  = False
    # cortex_decode_probe_at_end: after training finishes, run the offline
    # cortex-rate decodability probe (main.py's _cortex_decode_probe) --
    # one frozen-policy rollout logging the raw cortex rate vector, then
    # offline fits (ridge, NNLS, TD(0)) to separate a representation-
    # ceiling failure from a critic-mechanism-specific one. Default False.
    cortex_decode_probe_at_end: bool = False
    # critic_vstar_probe_at_end: after training finishes, run main.py's
    # _critic_vstar_probe -- compares v_spiking against V* (true
    # Monte-Carlo return-to-go from greedy rollouts), not v_python.
    # Default False: unchanged behavior/runtime for every other run.
    critic_vstar_probe_at_end: bool = False
    # critic_reach_vec_input: when True, the critic's L1 input is a
    # dedicated 144-neuron (12x12) 2D Gaussian place-field population over
    # reach_vec = fk(joints) - target (Grid2DPlaceFieldEncoder), replacing
    # cortex as critic input -- offline measurement showed this population
    # decodes V* better alone than cortex does. Trained the same way
    # cortex->L1 would be (SpikingCriticLearning); only the source
    # population changes. Default False: unchanged for every existing
    # config.
    critic_reach_vec_input:  bool = False
    reach_vec_encoder_kw: dict = dataclasses.field(default_factory=lambda: dict(
        bins=12, max_rate=800.0, sigma_bins=1.0))
    # actor_reach_vec_input: when True, cortex's "object" (raw target) stream is
    # REPLACED by the SAME reach_vec = fk(joints)-target population this
    # project already built for the critic (critic_reach_vec_input) --
    # "joints" stays, carrying q (the error-vector-to-joint-motion map is
    # q-dependent, so q is still required). Independent of critic_reach_
    # vec_input -- either, both, or neither may be set; the encoder itself
    # (model/main.py's _build_encoders) is built whenever EITHER flag is
    # on. See network.py's cortex-wiring block.
    actor_reach_vec_input: bool = False
    # reach_vec_cortex_connect_kw: Grid2DPlaceFieldEncoder.connect_to()'s
    # own signature (weight/delay/indegree -- NO radius/spatial_indegree,
    # unlike object_connect_kw/joints_connect_kw: this encoder's parrots
    # aren't laid out on cortex's spatial grid, see encoders.py's own
    # connect_to -- passing object_connect_kw's spatial kwargs here would
    # land in **kw and crash as invalid synapse params). weight=40.0
    # MEASURED (not guessed): a short weight sweep (3 episodes/point,
    # seed0) matched cortex_active_fraction against the UNMODIFIED
    # p3_reach_baseline.py's own value (mean 0.657 over the same 3
    # episodes) -- weight=40 gave 0.637, within this metric's own
    # episode-to-episode noise band (individual episodes swung
    # 0.52-0.74 at every weight tried). Matching the EXISTING baseline's
    # own operating point, not a fixed target band (an earlier draft of
    # this comment cited "10-30%" from a different context -- actor_
    # burst_hidden's own docstring measured the real baseline at
    # 0.13-0.45 in ITS OWN runs, and this project's own live baseline
    # measured 0.52-0.75 here -- cortex's drive-matched operating point
    # is itself noisy/context-dependent, so this calibrates against a
    # same-protocol side-by-side measurement, not an absolute number).
    reach_vec_cortex_connect_kw: dict = dataclasses.field(default_factory=lambda: dict(
        weight=40.0, delay=0.5, indegree=10))
    # dopsyn (cortex->D1) A_plus/A_minus ratio biased toward potentiation
    # (4.5:1), which raised the success count in testing; adopsyn (D2)
    # left at the equalized 1:1 ratio, which zeroes a baseline
    # eligibility-trace bias.
    dopsyn_a_plus:            float = 0.0045
    dopsyn_a_minus:           float = 0.0010
    adopsyn_a_plus:           float = -0.00275
    adopsyn_a_minus:          float = -0.00275
    critic_l1_ie:             float = 240.
    reward_da_weight:         float = 5.0

    pos_reward_scale:        float = 2000.0
    neg_reward_scale:        float = 2000.0

    # ── Exploration ─────────────────────────────────────────────────────────
    exploration_start:        float = 0.5
    exploration_min:          float = 0.05
    exploration_decay:        float = 0.991

    # ── Learning decay ──────────────────────────────────────────────────────
    decay_learning:           float = 0.9998
    decay_learning_critic:    float = 0.9990
    freeze_weights_episode:     int = 0

    # Periodic kernel reset + connectivity reload; works around a NEST
    # memory-growth issue in stdp_dopamine_synapse/volume_transmitter.
    checkpoint_every:           int  = 3

    # "python": model/python_critic.py's ground-truth-state TD critic
    # drives dopamine via start_reward() instead of raw reward. The real
    # spiking critic is always built/trained regardless of this setting
    # (see critic_python_corr stat) -- "spiking" would use it directly.
    # Intent: switch back to "spiking" once its value estimate converges.
    teacher:                    str  = "python"
    # "rich" adds trig/cross-term features to python_critic's linear basis
    # -- see python_critic.py's feature_set comment. Default "linear":
    # unchanged behavior.
    python_critic_feature_set: str  = "linear"

    # Holds cortex->D1 weights fixed across a batch of episodes,
    # accumulating eligibility x dopamine-deviation credit in Python the
    # whole time, then applies one combined update per batch, instead of
    # continuous online STDP. Keeps D1's weight stable/growing instead of
    # the decline plain online STDP showed, at a lower learning rate.
    actor_batched_d1:            bool  = True
    actor_batched_d1_batch_episodes: int = 10
    actor_batched_d1_lr_scale:   float = 0.8
    # Projects out the component of each commit's per-D1-neuron credit
    # that's linearly predictable from that neuron's preferred direction
    # (pref_q1/pref_q2) before applying it -- see batched_actor_learning.py's
    # debias_pref comment. Default False pending a real comparison run.
    actor_batched_d1_debias_pref: bool = False
    # Subtracts each batch's own mean nb_value before it multiplies into
    # credit -- see batched_actor_learning.py's center_nb_per_batch
    # comment. Recovers eligibility's own batch-to-batch consistency
    # without hurting success, unlike other bias-correction mechanisms
    # tried.
    actor_batched_d1_center_nb: bool = True
    # actor_batched_d1_context_gated: generalizes center_nb_per_batch
    # above from one pooled baseline/accumulator shared across whatever
    # goals land in a batch into a separate baseline + eligibility
    # accumulator per goal (BatchedD1Learning's n_goals param) -- see its
    # own comment for why this reduces cross-goal interference. Requires
    # cfg.target_goal_set to be set (n_goals = len(target_goal_set)) and
    # main.py to supply this episode's goal_idx to every bd1.step() call.
    # Default False: every existing config (including every target_goal_
    # set one so far) unchanged.
    actor_batched_d1_context_gated: bool = False
    # False = the debias_pref regression fits [pref_q1, pref_q2] only, no
    # intercept -- preserves the batch's overall mean credit level. See
    # batched_actor_learning.py's debias_include_intercept comment.
    # Default True (debias_pref itself defaults to False, so this is a
    # no-op either way until debias_pref is also turned on).
    actor_batched_d1_debias_include_intercept: bool = True
    # Replaces raw D1 spike count with (count - slow per-neuron EMA
    # baseline) in the eligibility coincidence product. See
    # batched_actor_learning.py's own comment. Default False: unchanged.
    actor_batched_d1_postsyn_baseline_correction: bool = False
    actor_batched_d1_postsyn_baseline_tau: float = 5000.0
    # Replaces D1's own noisy spike-count post factor with the exact,
    # noise-free exploration perturbation env_reach.py's select_action()
    # already injects every step. Requires actor_unified_readout=True. See
    # batched_actor_learning.py's own comment.
    actor_batched_d1_use_action_perturbation: bool = False
    actor_batched_d1_action_perturbation_scale: float = 1.0
    # Narrows the coincidence window used for eligibility from the whole
    # activate_time down to fine sub-bins, so timing resolution is finer
    # within a step, not only across steps. See batched_actor_learning.py's
    # own comment. None = disabled (one bin spanning the whole window).
    actor_batched_d1_coincidence_bin_ms: float = None
    # Multiplies each cortex source's contribution to coincidence by a
    # novelty factor so tonically-active sources get down-weighted and
    # recently-quiet sources get full weight. See batched_actor_learning.py's
    # own comment. Default False: unchanged.
    actor_batched_d1_source_novelty_gate: bool = False
    actor_batched_d1_source_novelty_tau: float = 5000.0
    actor_batched_d1_source_novelty_gain: float = 1.0
    # Subtracts a slow EMA of each cortex neuron's own mean rate before
    # forming the coincidence signal, since raw cortex activity ill-
    # conditions the credit signal. See batched_actor_learning.py's own
    # comment. Default False: unchanged.
    actor_batched_d1_presyn_baseline_correction: bool = False
    actor_batched_d1_presyn_baseline_tau: float = 50000.0
    # actor_batched_d2: the same batched-learning + optional presynaptic-
    # centering treatment for cortex->D2, which otherwise always learns via
    # native online adopsyn STDP regardless of actor_batched_d1 -- see
    # main.py's own comment. Default False: unchanged (D2 keeps learning
    # via native adopsyn).
    actor_batched_d2: bool = False
    actor_batched_d2_batch_episodes: int = 10
    actor_batched_d2_lr_scale: float = 3.0
    actor_batched_d2_presyn_baseline_correction: bool = False
    actor_batched_d2_presyn_baseline_tau: float = 50000.0
    # Centers D2's own raw spike count too, not just the presynaptic
    # cortex side above -- D2 has no pref vector, so it can't use
    # action-perturbation the way D1's post factor does.
    actor_batched_d2_postsyn_baseline_correction: bool = False
    actor_batched_d2_postsyn_baseline_tau: float = 5000.0
    # Inserts a relay population cortex->D1_L1->D1_L2(=D1) instead of
    # cortex->D1 directly, each hop independently plastic. D2 stays
    # single-layer regardless. See actor.py's own comment. Default False.
    actor_d1_hidden_layer: bool = False
    # compute_heads() divides both action dims by the same shared scalar
    # before tanh, correcting a per-dimension gain mismatch a small
    # population's random pref draw can produce. See actor.py's own
    # comment. Default False: unchanged.
    actor_per_dim_normalize: bool = False
    # Forces D1's V_th to a constant instead of its usual jittered
    # distribution -- tests whether V_th-excitability/pref correlation
    # drives a readout bias. See actor.py's own comment. Diagnostic only,
    # risks within-population locked spiking. Default False: unchanged.
    actor_d1_vth_uniform: bool = False
    actor_d1_vth_uniform_value: float = -53.0
    # The opposite direction from actor_d1_vth_uniform -- widens (or
    # narrows) V_th's normal-distribution spread instead of collapsing it
    # to zero. Default 1.0 is the original hardcoded value.
    actor_d1_vth_jitter_std: float = 1.0
    # Resamples D1's pref draw against V_th/grid-position until the worst
    # chance-correlation is below tolerance, instead of trusting one
    # independent draw to land there by luck. See actor.py's own comment.
    # Default False: unchanged.
    actor_pref_decorrelate: bool = False
    # "delta_python" feeds BatchedD1Learning python_critic's own clean
    # TD-error directly instead of the spiking-DA-population-derived
    # nb_value -- see main.py's actor_batched_d1_signal_source comment.
    # Works best together with center_nb_per_batch.
    # "delta_python_directional" feeds a per-joint python_critic vector,
    # projected through each D1 neuron's own preferred-direction vector
    # instead of one scalar shared by every neuron. See main.py's and
    # batched_actor_learning.py's comments. Not yet validated -- kept at
    # "delta_python" as the confirmed default until tested.
    actor_batched_d1_signal_source: str = "delta_python"
    # Compensates delta_python's much smaller magnitude so lr_scale stays
    # roughly comparable across signal sources. Only applies when
    # actor_batched_d1_signal_source="delta_python".
    actor_batched_d1_delta_python_scale: float = 236.5
    # Same compensation idea as delta_python_scale above, but for
    # "delta_spiking" (SpikingCriticLearning's own resolved delta -- see
    # spiking_critic_learning.py's step() and main.py's dispatch).
    # Calibrate per-config via a short smoke run; this is only a
    # placeholder default, not a measured value for this signal source.
    actor_batched_d1_delta_spiking_scale: float = 236.5
    # critic_signflip_probe_episodes: only used when
    # actor_batched_d1_signal_source="delta_spiking". Over episodes
    # [0, this), accumulates (v_spiking, episode-local realized return)
    # pairs; at the end of that window their pooled correlation decides
    # critic_sign once (+1 or -1), frozen for the rest of the run,
    # correcting the small fraction of seeds whose spiking critic
    # converges to -V. 0 disables the probe (critic_sign stays +1).
    critic_signflip_probe_episodes: int = 20

    # ── Burst-plasticity hidden layer: inserts an Izhikevich intrinsically-bursting
    # population H between cortex and D1/D2 (replacing the flat cortex->
    # D1/D2 projection), trained via burst-dependent plasticity -- ported
    # from cart_pole/nest_cartpole_centered_burst_test.py. See model/
    # burst_hidden.py. Default False: every existing config unchanged.
    actor_burst_hidden:            bool  = False
    # None -> defaults to len(cortex.output()) at build time (mirrors
    # the CartPole reference's HIDDEN_SIZE=N_EXC convention).
    actor_burst_hidden_size:        int  = None
    burst_isi_ms:                 float  = 8.0
    burst_base_std:               float  = 35.0
    burst_std_min:                float  = 5.0
    burst_std_max:                float  = 80.0
    burst_gain_td:                float  = 40.0
    burst_p_baseline:             float  = 0.40
    burst_lr:                     float  = 0.05
    burst_fb_weight_lo:           float  = -1.0
    burst_fb_weight_hi:           float  = 1.0
    burst_ch_w_lo:                float  = -10.0
    burst_ch_w_hi:                float  = 10.0
    burst_ch_wmin:                float  = -40.0
    burst_ch_wmax:                float  = 40.0
    burst_izh_a:                  float  = 0.02
    burst_izh_b:                  float  = 0.2
    burst_izh_c:                  float  = -55.0
    burst_izh_d:                  float  = 4.0

    # ── Output / logging ─────────────────────────────────────────────────────
    video_interval:            int  = 30
    # spike_panel_window_ms: how much spike history each video frame's
    # critic+DA/D1+D2/cortex rasters show (a sliding window, not just that
    # control step's own slice) -- bigger means more overlap between
    # consecutive frames, so spikes scroll off the left edge more slowly.
    # Keep this clearly below clear_rates_every*(activate_time+reward_time)
    # so most of the video shows a fully-filled window, not a
    # perpetually-truncated one right after each clear.
    spike_panel_window_ms:     float = 1800.0
    # clear_rates_every: how often (in control steps) every population
    # recorder's accumulated history is wiped, bounding how far back a
    # sliding-window spike query can look so NEST's event buffers don't
    # grow unboundedly over a long run.
    clear_rates_every:          int = 120
    spike_pdf_interval:        int  = 30
    output_folder:              str = "./experiments/runs/_default_reach_state"
    clean_output:               bool = True
    verbose:                    bool = False
