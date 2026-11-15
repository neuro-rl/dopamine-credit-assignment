import math
import nest
import numpy as np
from base_util import Base
import file_utils as futils
import recorder_tracking


class ActorD1D2Channels(Base):
    """
    Basal-ganglia inspired Go/NoGo gating with opponent ActionHead.

    Requires Dopamine to be instantiated first so that 'dopsyn' and 'adopsyn'
    synapse models are registered with NEST.
    """

    def __init__(self, cortex, dim_actions: int, verbose: bool = False,
                 ie_d1: float = 0.0,   ie_d2: float = 10.0,
                 el_d1: float = None,  el_d2: float = None,
                 v_reset_d1: float = None, v_reset_d2: float = None,
                 tau_syn_ex: float = None,
                 d1_tau_m: float = 20.0, d2_tau_m: float = 20.0,
                 grid_shape: tuple = (8, 8),
                 name_prefix: str = "",
                 disable_recurrence: bool = False,
                 disable_self_recurrence: bool = False,
                 disable_cross_inhibition: bool = False,
                 k_ctx_d1_scale: float = 1.0,
                 k_ctx_d2_scale: float = 1.0,
                 w_ctx_d1_scale: float = 1.0,
                 w_ctx_d2_scale: float = 1.0,
                 w_ctx_d1_jitter_scale: float = 1.0,
                 w_ctx_d2_jitter_scale: float = 1.0,
                 all_to_all_ctx_d1d2: bool = False,
                 topographic_ctx_d1d2: bool = False,
                 topographic_ctx_radius_frac: float = 0.15,
                 topographic_ctx_std_frac: float = 0.10,
                 topographic_ctx_p: float = 1.0,
                 cortex_extent: float = None,
                 build_plastic_ctx: bool = True,
                 d1_hidden_layer: bool = False,
                 per_dim_normalize: bool = False,
                 d1_vth_uniform: bool = False,
                 d1_vth_uniform_value: float = -53.0,
                 d1_vth_jitter_std: float = 1.0,
                 pref_decorrelate: bool = False,
                 baseline_zone_grid: tuple = (1, 1),
                 per_channel_reward: bool = False,
                 vector_readout: bool = False,
                 vector_readout_seed: int = 12345,
                 d2_stabilizer: bool = False,
                 d2_stabilizer_weight_mult: float = 3.0,
                 d2_stabilizer_radius_frac: float = 1.0,
                 unified_readout: bool = False,
                 axis_aligned_pref: bool = False,
                 axis_pref_graded: bool = False,
                 readout_baseline_correction: bool = False,
                 readout_baseline_tau: float = 50000.0,
                 d1_adapt_b: float = 0.0,
                 d1_adapt_a: float = 0.0,
                 d1_adapt_tau_w: float = 200.0):
        super().__init__()

        self.cortex       = cortex
        self.num_channels = dim_actions
        self.verbose      = verbose
        self.channels     = []
        self.ie_d1        = ie_d1
        self.ie_d2        = ie_d2
        self.el_d1        = el_d1
        self.el_d2        = el_d2
        self.v_reset_d1   = v_reset_d1
        self.v_reset_d2   = v_reset_d2
        self.tau_syn_ex   = tau_syn_ex
        self.d1_tau_m     = d1_tau_m
        self.d2_tau_m     = d2_tau_m
        # d1_adapt_b: D1-only intrinsic spike-frequency adaptation
        # (Brette-Gerstner aeif_psc_alpha's `w` current instead of
        # iaf_psc_alpha). Every D1 spike bumps that neuron's adaptation
        # current by `b` (pA), decaying with tau_w (ms), which subtracts
        # from its own drive -- so whichever D1 sub-population is firing
        # more becomes progressively harder for itself to keep firing,
        # self-limiting without touching weights, credit, or the
        # dopamine baseline. `a` defaults to 0 to keep this isolated to
        # pure spike-triggered adaptation. Default b=0.0: plain
        # iaf_psc_alpha, no behavior change unless a config opts in.
        self.d1_adapt_b     = d1_adapt_b
        self.d1_adapt_a     = d1_adapt_a
        self.d1_adapt_tau_w = d1_adapt_tau_w
        # disable_recurrence strips D1/D2 self-recurrence (Mexican-hat)
        # and cross-channel inhibition, to test whether the actor's own
        # internal dynamics are a noise source overwhelming the
        # eligibility-trace credit-assignment signal. disable_self_
        # recurrence and disable_cross_inhibition isolate the two
        # sub-mechanisms (D1/D1 and D2/D2 self-recurrence vs D1<->D2
        # cross-channel inhibition); disable_recurrence=True still
        # disables both (OR'd below) so existing test scripts keep
        # working unchanged.
        self.disable_recurrence = disable_recurrence
        self.disable_self_recurrence = disable_self_recurrence or disable_recurrence
        self.disable_cross_inhibition = disable_cross_inhibition or disable_recurrence

        self.name_prefix = name_prefix
        self.build_plastic_ctx = build_plastic_ctx
        # d1_hidden_layer: inserts a hidden population
        # cortex->D1_L1->D1_L2 instead of the plain cortex->D1
        # connection -- D1_L2 is the existing `d1` population, only its
        # input source changes. D1_L1 is a plain relay population with
        # its own independently-plastic cortex->D1_L1 connection, driven
        # by the same shared reward-deviation signal since there's no
        # backprop path to give it a layer-specific error. D2 stays
        # single-layer regardless, to isolate whether any effect comes
        # from D1's depth specifically. Default False: unchanged behavior.
        self.d1_hidden_layer = d1_hidden_layer
        # per_channel_reward: routes each channel's cortex->D1/D2
        # connections through its own dopsyn_ch{i}/adopsyn_ch{i} synapse
        # model instead of the single shared "dopsyn"/"adopsyn" -- see
        # dopamine.py's n_reward_channels comment. Default False:
        # unchanged behavior.
        self.per_channel_reward = per_channel_reward
        # vector_readout: population-vector-style decode -- each D1/D2
        # neuron gets a fixed random preferred scalar p_j in [-1,+1],
        # assigned once from a dedicated seed and never touched by STDP.
        # compute_heads() (env_adapter.py) uses this to replace the plain
        # population-rate readout with a per-neuron-weighted sum, so a
        # uniform population-wide activity shift doesn't move the
        # readout the way a plain-sum readout would. Default False:
        # unchanged behavior.
        self.vector_readout = vector_readout
        self._vector_readout_rng = np.random.default_rng(vector_readout_seed)
        # d2_stabilizer: D2 stops being a second "vote" that subtracts
        # from the readout, and becomes a targeted regulatory circuit on
        # D1 instead -- still cortex-driven and plastic, but its
        # influence on the action is entirely indirect, via suppressing
        # specific D1 neurons before D1's own weighted sum is computed.
        # The existing symmetric D1<->D2 cross-inhibition barely touches
        # D1 even widened a lot; an asymmetric version (D2->D1 alone
        # strengthened) gives a real, targeted effect on D1's active
        # fraction without touching D2's own activity.
        # d2_stabilizer_weight_mult defaults to 3.0, a moderate effect.
        self.d2_stabilizer = d2_stabilizer
        self.d2_stabilizer_weight_mult = d2_stabilizer_weight_mult
        self.d2_stabilizer_radius_frac = d2_stabilizer_radius_frac
        # unified_readout: the "single merged population, D1 outputs
        # the full action vector directly" design -- builds exactly one
        # D1/D2 pair (self._n_built_channels=1) instead
        # of one pair per action dimension, with each D1 neuron's preferred
        # value now a dim_actions-DIMENSIONAL vector (not a per-channel
        # scalar) -- one population decodes the WHOLE action at once via a
        # single weighted vector sum (env_adapter.py's compute_heads()),
        # instead of dim_actions independent per-joint scalar reads. D2
        # (when also paired with d2_stabilizer, the expected/recommended
        # combination) keeps its existing role unchanged -- a single
        # regulatory population, no preferred vector of its own, purely
        # shaping which D1 neurons fire. self.num_channels stays dim_actions
        # (the semantic action dimensionality, used for the readout vector
        # size); self._n_built_channels is the STRUCTURAL count of D1/D2
        # pairs actually constructed. Default False: every existing config,
        # including per-channel builds, is unaffected (_n_built_channels ==
        # num_channels then, identical to today).
        if unified_readout and not vector_readout:
            raise ValueError(
                "unified_readout requires vector_readout=True -- D1's "
                "per-neuron preferred VECTOR is only allocated inside the "
                "vector_readout pref-building path; without it compute_heads() "
                "would KeyError on a missing 'd1_pref' at the first training step.")
        self.unified_readout = unified_readout
        # axis_aligned_pref: each D1 neuron's pref vector is normally a
        # random continuous 2D direction, so almost every cortex->D1
        # synapse carries credit for both q1 and q2 in one weight. When
        # q1's signal dominates, its updates on shared synapses can
        # overwrite q2's weaker accumulated credit. axis_aligned_pref
        # splits D1 into pure-q1/pure-q2 sub-populations (pref in
        # {(+1,0),(-1,0),(0,+1),(0,-1)} only) so a synapse onto a q1
        # neuron is never touched by q2's credit or vice versa.
        self.axis_aligned_pref = axis_aligned_pref
        # axis_pref_graded: graded per-neuron preference magnitude
        # within each axis group instead of a plain +-1 split.
        # No-op unless axis_aligned_pref is also True.
        self.axis_pref_graded = axis_pref_graded
        # readout_baseline_correction: weighted_rate_in_window (the
        # population-vector readout that decodes D1's spikes into the
        # actual action) sums each spiking neuron's pref vector -- a
        # uniform tonic rate increase across the population cancels in
        # that sum, but a non-uniform, per-neuron excitability
        # difference does not, and biases the vote toward whichever
        # pref direction the more-excitable neurons happen to have.
        # This subtracts each neuron's own slow EMA baseline rate from
        # its count before it enters the vote, centering the readout
        # itself -- distinct from and independent of
        # actor_batched_d1_presyn_baseline_correction, which only
        # centers the learning signal, never the decoded action.
        # main.py must save/restore this state across checkpoint_every
        # rebuilds or a short rebuild cadence would wipe a baseline
        # meant to average over ~10 episodes. Default False: unchanged.
        self.readout_baseline_correction = readout_baseline_correction
        self.readout_baseline_tau = readout_baseline_tau
        self._n_built_channels = 1 if unified_readout else self.num_channels
        self.grid_shape = grid_shape
        self.N          = self.grid_shape[0] * self.grid_shape[1]
        self.scale      = math.sqrt(36 / self.N)
        # cortex_extent: without it, D1/D2's extent defaults to
        # grid_shape[0], which can mismatch cortex's own extent (e.g.
        # actor_grid_shape=(10,10) vs cortex_exc_shape=(12,12)).
        # topographic_ctx_d1d2's cortex->D1/D2 spatial mask connects two
        # periodic layers, so a mismatched extent is a real coordinate-
        # system bug. Pass cortex_extent to make D1/D2 share cortex's
        # coordinate system; omitting it preserves the old
        # grid_shape[0]-derived value for non-topographic callers. D1/D2's
        # own internal spatial connectivity (Mexican-hat self-recurrence
        # etc, all extent-relative fractions) is unaffected in relative
        # terms -- only the absolute spacing changes.
        self._extent    = float(cortex_extent) if cortex_extent is not None else float(self.grid_shape[0])

        self.w_d1_to_d2  = -3.0
        self.w_d2_to_d1  = -3.0
        self.w_self_inh  = -0.6

        # Scale indegree with population size
        # k_ctx_d1_scale: D1 is intentionally quieter than D2 (ie_d1/
        # ie_d2 + V_th asymmetry is deliberate -- do not equalize it),
        # but at low indegree the cortex groups feeding D1 can be too
        # few/inactive for STDP to ever fire on, independent of D1's
        # intrinsic excitability. Scaling indegree instead of intrinsic
        # excitability gives D1 a richer, more robust sample of cortex
        # without touching the intentional ie_d1/ie_d2 asymmetry. d2's
        # own indegree scale (k_ctx_d2_scale) mirrors k_ctx_d1_scale so
        # both D1 and D2's cortex-driven drive can be raised together.
        # Default 1.0 for both leaves existing configs unchanged.
        self.k_ctx_d1 = int(round(15 / (self.N / 25) * k_ctx_d1_scale))  # 15 for 5x5, 29 for 7x7
        # int(), NOT round() -- matches the original unscaled formula's
        # truncation exactly, so k_ctx_d2_scale=1.0 (default) reproduces
        # every existing config's k_ctx_d2 value bit-for-bit (round() here
        # would have silently changed it, e.g. 3->4 at N=100, even at the
        # "no-op" default -- caught before this shipped).
        self.k_ctx_d2 = int(15 / (self.N / 25) * k_ctx_d2_scale)
        # all_to_all_ctx_d1d2: every D1/D2 neuron connects to every
        # cortex neuron instead of a random indegree sample --
        # implemented via NEST fixed_indegree with indegree==population
        # size, functionally identical to all_to_all. Weight is
        # compensated by the original indegree/len(cortex) ratio to
        # keep total expected cortex->D1/D2 drive unchanged. Default
        # False: unchanged behavior.
        self.all_to_all_ctx_d1d2 = all_to_all_ctx_d1d2
        self._ctx_d1_weight_compensation = 1.0
        self._ctx_d2_weight_compensation = 1.0
        if self.all_to_all_ctx_d1d2:
            self._ctx_d1_weight_compensation = self.k_ctx_d1 / len(cortex)
            self._ctx_d2_weight_compensation = self.k_ctx_d2 / len(cortex)
            self.k_ctx_d1 = len(cortex)
            self.k_ctx_d2 = len(cortex)

        # topographic_ctx_d1d2: a spatially-restricted cortex->D1/D2
        # connection, radius-controlled to keep synapse count small.
        # radius_frac/std_frac are extent-relative, matching cortex's
        # own r_ee/r_ei convention. Default False leaves plain random
        # fixed_indegree unchanged.
        self.topographic_ctx_d1d2 = topographic_ctx_d1d2
        self.topographic_ctx_radius_frac = topographic_ctx_radius_frac
        self.topographic_ctx_std_frac = topographic_ctx_std_frac
        # topographic_ctx_p: the achievable neighbor count on a
        # spacing=1.0 lattice is quantized, not continuously
        # radius-tunable. p<1.0 (Bernoulli thinning within the mask)
        # reaches any expected indegree between those lattice shells.
        # Default 1.0 leaves other topographic uses unaffected.
        self.topographic_ctx_p = topographic_ctx_p

        # baseline_zone_grid: partner to dopamine.py's
        # Dopamine(n_baseline_zones=...) -- a single global `b` can't
        # express "better than expected for this context". This actor
        # owns the zone assignment since it has cortex's spatial
        # positions; dopamine.py registers one dopsyn_z{i}/adopsyn_z{i}
        # synapse model per zone, and this actor decides per cortex
        # neuron which zone's model its outgoing cortex->D1/D2
        # connections use, over a (nx, ny) grid spanning both cortex
        # axes. baseline_zone_grid=(1, 1) (default) skips all of this
        # and falls through to the original single dopsyn/adopsyn
        # connection, unchanged.
        self.baseline_zone_grid = (max(1, int(baseline_zone_grid[0])),
                                    max(1, int(baseline_zone_grid[1])))
        self.n_baseline_zones = self.baseline_zone_grid[0] * self.baseline_zone_grid[1]
        self._cortex_zones = (
            self._compute_cortex_zones(cortex) if self.n_baseline_zones > 1 else None
        )

        # Equalized cortex->D1/D2 init weight (was 26.4/19.2, a static
        # init-time Go bias applied before STDP could differentiate the
        # two). Combined with the efference-copy fix above, this
        # resolved q1/q2 saturating at their joint limits every episode
        # regardless of that episode's random reset/target.
        # w_ctx_d1_scale/w_ctx_d2_scale: k_ctx_d1/d2 scale indegree
        # inversely with actor population N so total cortex->D1/D2
        # synapse count stays roughly constant across actor sizes, but
        # per-synapse weight stays fixed, so a smaller actor's higher
        # indegree increases each neuron's total drive above what
        # larger-actor configs were calibrated for. These scales let a
        # caller compensate back down to a target total drive when
        # shrinking the actor. Default 1.0 leaves existing configs
        # unchanged.
        # w_ctx_d1/d2_jitter_scale: independent of w_ctx_d1/d2_scale
        # (which scales mean and jitter together) -- this scales only
        # the initial per-synapse weight spread, keeping the mean
        # fixed. A wider initial spread gives a weak/noisy
        # reward-correlated selection pressure more differentiated
        # starting material to select among. Default 1.0 = unchanged.
        _d1_mean, _d1_jit = 22.8 * w_ctx_d1_scale, 2. * w_ctx_d1_scale * w_ctx_d1_jitter_scale   # cortex->D1 (Go) drive
        _d2_mean, _d2_jit = 22.8 * w_ctx_d2_scale, 2. * w_ctx_d2_scale * w_ctx_d2_jitter_scale   # cortex->D2 (NoGo) drive
        # all_to_all_ctx_d1d2 weight compensation (see k_ctx_d1/d2's own
        # comment above) -- applied here, before Base.w_uni/the init-bounds
        # assertion, so the ACTUAL init distribution (not just a later
        # rescale) reflects the compensated per-synapse weight.
        _d1_mean, _d1_jit = _d1_mean * self._ctx_d1_weight_compensation, _d1_jit * self._ctx_d1_weight_compensation
        _d2_mean, _d2_jit = _d2_mean * self._ctx_d2_weight_compensation, _d2_jit * self._ctx_d2_weight_compensation
        _d1_bounds = nest.GetDefaults('dopsyn')
        _d2_bounds = nest.GetDefaults('adopsyn')
        Base.assert_synapse_init_in_bounds(
            _d1_mean, _d1_jit, _d1_bounds['Wmin'], _d1_bounds['Wmax'], label="cortex->D1 (dopsyn)")
        Base.assert_synapse_init_in_bounds(
            _d2_mean, _d2_jit, _d2_bounds['Wmin'], _d2_bounds['Wmax'], label="cortex->D2 (adopsyn)")
        self.w_ctx_d1 = Base.w_uni(_d1_mean, _d1_jit)
        self.w_ctx_d2 = Base.w_uni(_d2_mean, _d2_jit)
        # Plain-float means, kept separately for the topographic path below --
        # connect_gaussian multiplies weight by a spatial Gaussian falloff
        # elementwise, which needs a plain float peak weight, not a NEST
        # random-distribution Parameter object like self.w_ctx_d1/d2 above.
        self.w_ctx_d1_mean = _d1_mean
        self.w_ctx_d2_mean = _d2_mean
        # per_dim_normalize: compute_heads() (env_adapter.py) divides
        # both action dimensions by the same scalar `scale` before tanh,
        # but with a small population a random zero-mean pref draw's
        # per-dimension spread isn't actually equal across dimensions --
        # the dimension with more spread runs closer to tanh's
        # saturating region purely from initialization luck. This
        # corrects a per-config, per-seed effective gain mismatch
        # between q1/q2 that the shared scalar can't see. Default False:
        # unchanged behavior.
        self.per_dim_normalize = per_dim_normalize
        # d1_vth_uniform: forces D1's V_th to one constant instead of
        # nest.random.normal(-53.0, 1.0) -- tests whether V_th
        # excitability variance correlating with pref by small-N chance
        # is driving a readout bias. Not a proposed permanent fix:
        # removes D1's within-population V_th jitter entirely, which
        # risks reintroducing locked/synchronized spiking within D1
        # (jitter exists specifically to avoid that). Default False:
        # unchanged.
        self.d1_vth_uniform = d1_vth_uniform
        self.d1_vth_uniform_value = d1_vth_uniform_value
        self.d1_vth_jitter_std = d1_vth_jitter_std
        # pref_decorrelate: a readout-bias probe found the readout's raw
        # vote correlates with several independent small-N chance
        # correlations between the pref draw and other per-neuron
        # quantities (V_th, grid position/self-recurrence, connectivity
        # weight), each drawn independently but not perfectly
        # uncorrelated with only ~36 neurons. This resamples candidate
        # pref vectors (same RNG stream, deterministic) until the worst
        # correlation against V_th and grid position is below
        # pref_decorrelate_tolerance, instead of trusting one
        # independent draw to land there by luck. V_th itself isn't
        # cached like pref is, so this decorrelation guarantee only
        # holds reliably when cfg.seed is fixed. Default False:
        # unchanged behavior.
        self.pref_decorrelate = pref_decorrelate

        for ch in range(self._n_built_channels):
            d1, d2, d1_l1 = self._create_channel(ch)
            entry = {
                "d1": d1,
                "d2": d2,
                "d1_l1": d1_l1,
                "recorders": self._attach_recorders(d1, d2, d1_l1),
                # Lazily initialized to zeros on first weighted_rate_in_window
                # call (matching BatchedD1Learning's presyn_baseline pattern)
                # unless restored from a checkpoint first -- see
                # set_readout_baseline()/get_readout_baseline() below.
                "d1_readout_baseline": None,
                "d2_readout_baseline": None,
            }
            if self.vector_readout:
                # unified_readout: D1's preference is a dim_actions-D vector
                # per neuron (shape (N, dim_actions)), decoding the WHOLE
                # action from this one population. D2 gets no preference at
                # all here -- expected to be paired with d2_stabilizer,
                # whose role (shaping D1's firing, not casting its own vote)
                # never needed a preference vector regardless.
                d1_shape = (len(d1), self.num_channels) if self.unified_readout else (len(d1),)
                _decorr = None
                if self.pref_decorrelate:
                    _pos = np.array(nest.GetPosition(d1))
                    _decorr = {
                        "V_th": np.array(nest.GetStatus(d1, "V_th")),
                        "pos_x": _pos[:, 0],
                        "pos_y": _pos[:, 1],
                    }
                entry["d1_pref"] = self._load_or_create_pref_vector(
                    d1_shape, f"{self.name_prefix}d1_pref_ch{ch}", decorrelate_vars=_decorr)
                entry["d1_pref_dim_scale"] = self._pref_dim_scale(entry["d1_pref"])
                entry["d1_id_to_idx"] = {nid: i for i, nid in enumerate(d1.tolist())}
                if not self.unified_readout:
                    entry["d2_pref"] = self._load_or_create_pref_vector((len(d2),), f"{self.name_prefix}d2_pref_ch{ch}")
                    entry["d2_id_to_idx"] = {nid: i for i, nid in enumerate(d2.tolist())}
                elif not self.d2_stabilizer:
                    # unified_readout without d2_stabilizer: D2 still needs
                    # SOME readout path -- fall back to a matching D2
                    # preference vector so compute_heads() has something
                    # sane to subtract, even though this combination isn't
                    # the recommended/tested one.
                    entry["d2_pref"] = self._load_or_create_pref_vector(
                        (len(d2), self.num_channels), f"{self.name_prefix}d2_pref_ch{ch}")
                    entry["d2_pref_dim_scale"] = self._pref_dim_scale(entry["d2_pref"])
                    entry["d2_id_to_idx"] = {nid: i for i, nid in enumerate(d2.tolist())}
            self.channels.append(entry)

    @staticmethod
    def _pref_dim_scale(p):
        """per_dim_normalize's correction factor: per-dimension std,
        normalized so the AVERAGE across dimensions is 1.0 -- multiplying
        compute_heads()'s shared `scale` by this leaves the average gain
        across dimensions unchanged (existing scale_factor calibrations
        stay valid) while correcting the RELATIVE q1-vs-q2 mismatch from
        this particular random pref draw. None for scalar (non-vector)
        pref, where there's only one dimension to normalize."""
        if p.ndim != 2:
            return None
        std = p.std(axis=0)
        return std / std.mean()

    def d1_pref_dim_scale(self, i):
        return self.channels[i].get("d1_pref_dim_scale")

    def d2_pref_dim_scale(self, i):
        return self.channels[i].get("d2_pref_dim_scale")

    def _load_or_create_pref_vector(self, shape, cache_key, decorrelate_vars=None,
                                     decorrelate_tolerance: float = 0.08,
                                     decorrelate_max_tries: int = 20000):
        """Persists vector_readout's per-neuron preferred values to disk
        (model/vector_prefs/{cache_key}.npz) instead of relying on
        regenerating identical numbers from a fixed seed every build.
        `shape`: (n,) for a scalar preference per neuron, or (n, dim) for a
        dim_actions-dimensional vector preference (unified_readout).

        Why this matters: seed-based regeneration is only correct if
        population creation order is EXACTLY reproducible across every
        rebuild (same neurons -> same NodeCollection index every time) --
        true today, but an implicit assumption a future change could
        silently break, silently reassigning a preference to the WRONG
        neuron with no error. Persisting the actual generated numbers
        (same pattern as ensure_connectivity's connectivity cache, and this
        session's random_synapses explicit-pair tracking, which replaced an
        earlier seed/order-reliant approach after a real bug from exactly
        this class of assumption) makes correctness independent of that
        assumption -- load beats regenerate whenever a cache exists.

        decorrelate_vars (pref_decorrelate=True only): dict of {name:
        per-neuron array} to decorrelate the pref draw against -- e.g.
        V_th, grid position (see __init__'s pref_decorrelate comment for
        the motivation). Draws repeated candidates from the SAME rng
        stream (deterministic given a seed) until the WORST |corr|
        between any decorrelate_var and any pref dimension is below
        `decorrelate_tolerance`, or falls back to the best candidate seen
        after `decorrelate_max_tries`. No-op (single draw, as before)
        when decorrelate_vars is None.
        """
        path = futils.get_script_relative_folder("vector_prefs") / f"{cache_key}.npz"
        if path.exists():
            data = np.load(path)
            p = data["pref"]
            if p.shape == shape:
                return p
            print(f"[WARN] Stale vector_readout pref cache (shape {p.shape} != {shape}) -- regenerating: {path}")

        def _draw():
            if self.axis_aligned_pref and len(shape) == 2:
                return self._draw_axis_aligned_pref(shape)
            p = self._vector_readout_rng.uniform(-1.0, 1.0, size=shape)
            return p - p.mean(axis=0, keepdims=True)

        if not decorrelate_vars:
            p = _draw()
            np.savez(path, pref=p)
            return p

        cols = range(shape[1]) if len(shape) == 2 else [None]
        best_p, best_score, tries = None, np.inf, 0
        for tries in range(1, decorrelate_max_tries + 1):
            p = _draw()
            worst = 0.0
            for v in decorrelate_vars.values():
                for k in cols:
                    col = p[:, k] if k is not None else p
                    worst = max(worst, abs(np.corrcoef(v, col)[0, 1]))
            if worst < best_score:
                best_score, best_p = worst, p
            if worst <= decorrelate_tolerance:
                break
        print(f"[pref_decorrelate] {cache_key}: worst |corr| = {best_score:.4f} "
              f"after {tries} tries (tolerance {decorrelate_tolerance})")
        np.savez(path, pref=best_p)
        return best_p

    def _draw_axis_aligned_pref(self, shape):
        """axis_aligned_pref=True variant of _load_or_create_pref_vector's
        draw: balanced exact split of the n neurons into `dim` groups (one
        per action dimension), each neuron's pref a unit vector on ITS
        OWN axis only (+1 or -1, half/half within its group) -- exact
        (not RNG-luck) balance so no residual cross-axis component or
        mean offset exists to entangle credit across dimensions.

        axis_pref_graded: every neuron in a group getting the same +-1
        magnitude is the n=2-distinct-values case of a more general
        per-neuron-graded preference. Graded magnitude spacing via
        cos(linspace(0,pi,m)), denser near the poles and sparser near
        zero, beats a plain +-1 split by a wide margin once the readout
        is unnormalized. cos(linspace(0,pi,m)) sums to exactly zero for
        any m, same exact-balance guarantee as the binary version."""
        n, dim = shape
        group_sizes = np.full(dim, n // dim)
        group_sizes[: n % dim] += 1
        assign = np.repeat(np.arange(dim), group_sizes)
        self._vector_readout_rng.shuffle(assign)
        p = np.zeros(shape)
        for k in range(dim):
            idx = np.where(assign == k)[0]
            if self.axis_pref_graded:
                mags = np.cos(np.linspace(0, np.pi, len(idx)))
                self._vector_readout_rng.shuffle(mags)
                p[idx, k] = mags
            else:
                signs = np.resize([1.0, -1.0], len(idx))
                self._vector_readout_rng.shuffle(signs)
                p[idx, k] = signs
        return p

    def _compute_cortex_zones(self, cortex):
        """Partition `cortex` into a self.baseline_zone_grid = (nx, ny) grid
        of equal-width spatial bins over BOTH x and y, using each neuron's
        actual grid position (nest.GetPosition) -- returns a list of
        NodeCollections, one per zone (index z = ix*ny + iy), covering every
        cortex neuron exactly once. See baseline_zone_grid's __init__
        comment for why this needs no explicit context label."""
        nx, ny = self.baseline_zone_grid
        positions = np.array(nest.GetPosition(cortex))

        def _bin_axis(axis_idx, n_bins):
            vals = positions[:, axis_idx]
            lo, hi = float(vals.min()), float(vals.max())
            if hi <= lo or n_bins <= 1:
                return np.zeros(len(vals), dtype=int)
            edges = np.linspace(lo, hi, n_bins + 1)
            return np.clip(np.digitize(vals, edges[1:-1]), 0, n_bins - 1)

        ix = _bin_axis(0, nx)
        iy = _bin_axis(1, ny)
        zone_idx = ix * ny + iy
        return [cortex[zone_idx == z] for z in range(self.n_baseline_zones)]

    def _connect_ctx_to_striatum(self, cortex, target, base_synapse_model, weight, indegree, cache_key,
                                  topo_weight_mean=None):
        """cortex->target (D1 or D2) STDP dopamine connection, split across
        n_baseline_zones spatial zones when zoned (n_baseline_zones>1), or a
        single plain connection otherwise (n_baseline_zones==1 -- identical
        to the pre-zoning behavior). Zoned: each zone connects ONLY through
        its own dopsyn_z{i}/adopsyn_z{i} synapse model (registered by
        dopamine.py), and the total indegree budget is SPLIT across zones
        (not multiplied) so total cortex->target fan-in stays ~= the unzoned
        `indegree`, just sourced from each zone in proportion to its share
        of zones rather than from the whole population uniformly."""
        if self.topographic_ctx_d1d2:
            # Spatial circular-mask + Gaussian weight falloff (see this
            # class's own __init__ comment for the history/rationale).
            # `weight` (a NEST random-distribution Parameter, Base.w_uni)
            # isn't usable here -- connect_gaussian needs a plain float peak
            # weight to multiply elementwise by the spatial falloff, so
            # callers pass the pre-jitter mean via topo_weight_mean instead.
            def _build(cortex=cortex, target=target):
                Base.connect_gaussian(
                    cortex, target,
                    weight=topo_weight_mean,
                    radius=self.topographic_ctx_radius_frac * self._extent,
                    std=self.topographic_ctx_std_frac * self._extent,
                    p=self.topographic_ctx_p,
                    synapse_model=base_synapse_model,
                )
            futils.ensure_connectivity(cortex, target, cache_key, _build)
            return

        if self.n_baseline_zones <= 1:
            def _build(cortex=cortex, target=target):
                nest.Connect(
                    cortex, target,
                    conn_spec={"rule": "fixed_indegree", "indegree": indegree},
                    syn_spec={"synapse_model": base_synapse_model, "weight": weight, "delay": nest.resolution},
                )
            futils.ensure_connectivity(cortex, target, cache_key, _build)
            return

        indeg_per_zone = max(1, round(indegree / self.n_baseline_zones))
        for z, zone_cortex in enumerate(self._cortex_zones):
            def _build(zone_cortex=zone_cortex, target=target, z=z):
                nest.Connect(
                    zone_cortex, target,
                    conn_spec={"rule": "fixed_indegree", "indegree": indeg_per_zone},
                    syn_spec={"synapse_model": f"{base_synapse_model}_z{z}", "weight": weight, "delay": nest.resolution},
                )
            futils.ensure_connectivity(zone_cortex, target, f"{cache_key}_z{z}", _build)

    def _create_channel(self, ch):
        if self.verbose:
            print(f"Creating D1/D2 channel {ch}")

        d1_params = {
            "tau_m": self.d1_tau_m,
            "V_th":  self.d1_vth_uniform_value if self.d1_vth_uniform
                     else nest.random.normal(-53.0, self.d1_vth_jitter_std),
            "I_e":   self.ie_d1,
        }
        if self.el_d1      is not None: d1_params["E_L"]       = self.el_d1
        if self.v_reset_d1 is not None: d1_params["V_reset"]   = self.v_reset_d1
        if self.tau_syn_ex is not None: d1_params["tau_syn_ex"] = self.tau_syn_ex

        d1_model = "iaf_psc_alpha"
        if self.d1_adapt_b > 0.0:
            # aeif_psc_alpha, translated from the iaf_psc_alpha params
            # above (C_m/g_L chosen so C_m/g_L == tau_m, preserving the
            # same subthreshold leak) plus adaptation on top. Delta_T kept
            # small so the exponential spike-generating term stays close
            # to iaf_psc_alpha's sharp threshold -- this is meant to ADD
            # spike-frequency adaptation, not change subthreshold/spiking
            # dynamics otherwise. V_peak is a fixed spike-detection cutoff
            # (V_th is randomized per-neuron above it) -- iaf_psc_alpha has
            # no equivalent since it resets on hard threshold-crossing.
            d1_model = "aeif_psc_alpha"
            _c_m = 250.0
            d1_params["C_m"]     = _c_m
            d1_params["g_L"]     = _c_m / d1_params.pop("tau_m")
            d1_params["Delta_T"] = 1.0
            d1_params["V_peak"]  = -30.0
            d1_params["a"]       = self.d1_adapt_a
            d1_params["b"]       = self.d1_adapt_b
            d1_params["tau_w"]   = self.d1_adapt_tau_w

        ext = self._extent
        d1 = nest.Create(
            d1_model,
            params=d1_params,
            positions=nest.spatial.grid(
                shape=list(self.grid_shape),
                extent=[ext, ext],
                center=[ext / 2, ext / 2],
                edge_wrap=True,
            ),
        )
        d1_l1 = None
        if self.d1_hidden_layer:
            # Same model/params as D1_L2 (d1 above) -- a plain relay
            # population, same size/grid as D1 for simplicity (no strong
            # reason yet to make it bigger/smaller).
            d1_l1 = nest.Create(
                d1_model,
                params=d1_params,
                positions=nest.spatial.grid(
                    shape=list(self.grid_shape),
                    extent=[ext, ext],
                    center=[ext / 2, ext / 2],
                    edge_wrap=True,
                ),
            )

        d2_params = {
            "tau_m": self.d2_tau_m,
            "V_th":  nest.random.normal(-49.0, 1.5),
            "I_e":   self.ie_d2,
        }
        if self.el_d2      is not None: d2_params["E_L"]       = self.el_d2
        if self.v_reset_d2 is not None: d2_params["V_reset"]   = self.v_reset_d2
        if self.tau_syn_ex is not None: d2_params["tau_syn_ex"] = self.tau_syn_ex

        d2 = nest.Create(
            "iaf_psc_alpha",
            params=d2_params,
            positions=nest.spatial.grid(
                shape=list(self.grid_shape),
                extent=[ext, ext],
                center=[ext / 2, ext / 2],
                edge_wrap=True,
            ),
        )

        # TESTED AND REVERTED: equalized at 21.5/21.5 (average of the
        # original 18.0/25.0) to check whether D2's stronger self-recurrent
        # Mexican-hat field was a source of the persistent D1/D2 firing-rate
        # gap seen late in episode 0 of reach_state_topo2 runs. Measured no
        # meaningful change (gap stayed ~1-2Hz, same order as before) --
        # not the driver. Back to the original values.
        if not self.disable_self_recurrence:
            self.connect_mexican_hat(d1, d1, weight=18.0, std=0.12 * ext, sink=0.7, mask_radius=0.5 * ext)
            self.connect_mexican_hat(d2, d2, weight=25.0, std=0.12 * ext, sink=0.7, mask_radius=0.5 * ext)

        # Tried and reverted: a topographic cortex->D1/D2 projection
        # (spatial circular-mask + Gaussian weight falloff). The idea was
        # sound, context-preferring cortex neurons really are spatially
        # clustered, but the behavioral test got worse, not better: one
        # head locked to one polarity almost totally, because the much
        # denser connectivity this projection produced amplified a
        # pre-existing runaway-bias failure mode more than it helped
        # context-tracking win against it. Reverted to random
        # fixed_indegree below. The encoder-side fix this motivated
        # (extent-relative `radius_frac` on
        # TopographicPlaceFieldEncoder.connect_to) was independent and
        # is kept -- it didn't cause this regression.
        # build_plastic_ctx=False (NestCheckpointer rebuilds, see
        # nest_checkpoint.py): skip -- ensure_connectivity's disk cache
        # holds INITIAL random weights, not live-trained ones; leave this
        # projection for NestCheckpointer.restore_from_checkpoint() to
        # recreate from the actual saved (live) weights instead.
        if self.build_plastic_ctx:
            dopsyn_model  = f"dopsyn_ch{ch}"  if self.per_channel_reward else "dopsyn"
            adopsyn_model = f"adopsyn_ch{ch}" if self.per_channel_reward else "adopsyn"
            if self.d1_hidden_layer:
                # cortex->D1_L1->D1_L2(=d1) instead of cortex->d1 directly.
                # Both hops use the SAME dopsyn-family synapse model/Wmin-
                # Wmax/init scale -- there's no principled reason yet to
                # calibrate them differently, and keeping them identical
                # isolates "does depth help" from "did we also change the
                # per-hop calibration."
                self._connect_ctx_to_striatum(
                    self.cortex, d1_l1, dopsyn_model, self.w_ctx_d1, self.k_ctx_d1,
                    f"{self.name_prefix}cortex_d1l1_ch{ch}", topo_weight_mean=self.w_ctx_d1_mean)
                self._connect_ctx_to_striatum(
                    d1_l1, d1, dopsyn_model, self.w_ctx_d1, self.k_ctx_d1,
                    f"{self.name_prefix}d1l1_d1_ch{ch}", topo_weight_mean=self.w_ctx_d1_mean)
            else:
                self._connect_ctx_to_striatum(
                    self.cortex, d1, dopsyn_model, self.w_ctx_d1, self.k_ctx_d1,
                    f"{self.name_prefix}cortex_d1_ch{ch}", topo_weight_mean=self.w_ctx_d1_mean)
            self._connect_ctx_to_striatum(
                self.cortex, d2, adopsyn_model, self.w_ctx_d2, self.k_ctx_d2,
                f"{self.name_prefix}cortex_d2_ch{ch}", topo_weight_mean=self.w_ctx_d2_mean)

        # Gaussian/local D1↔D2 inhibition disabled for broader opponent suppression.
        # self.connect_gaussian(d1, d2, weight=-4.0, radius=0.5, std=0.5)
        # self.connect_gaussian(d2, d1, weight=-4.0, radius=0.5, std=0.5)
        if not self.disable_cross_inhibition:
            self.connect_gaussian(d1, d2, weight=-4.0, radius=0.28 * ext, std=0.10 * ext)
            # d2_stabilizer replaces the original weak/narrow D2->D1 with a
            # much wider, stronger, ASYMMETRIC one below -- D1->D2 above is
            # always left at the original baseline either way.
            if not self.d2_stabilizer:
                self.connect_gaussian(d2, d1, weight=-4.0, radius=0.28 * ext, std=0.10 * ext)
        if self.d2_stabilizer:
            _radius = self.d2_stabilizer_radius_frac * ext
            self.connect_gaussian(
                d2, d1, weight=-4.0 * self.d2_stabilizer_weight_mult,
                radius=_radius, std=(0.10 / 0.28) * _radius,
                allow_oversized_mask=True,
            )

        # Don go with weights above -3, it gets worse
        # nest.Connect(
        #     d1, d2,
        #     conn_spec={'rule': 'all_to_all'},
        #     syn_spec={'synapse_model': 'static_synapse',
        #               'weight': -2.0, 'delay': nest.resolution},
        # )
        # nest.Connect(
        #     d2, d1,
        #     conn_spec={'rule': 'all_to_all'},
        #     syn_spec={'synapse_model': 'static_synapse',
        #               'weight': -2.0, 'delay': nest.resolution},
        # )

        return d1, d2, d1_l1

    def _attach_recorders(self, d1, d2, d1_l1=None):
        rec = {
            "d1": recorder_tracking.create(),
            "d2": recorder_tracking.create(),
        }
        nest.Connect(d1, rec["d1"])
        nest.Connect(d2, rec["d2"])
        if d1_l1 is not None:
            rec["d1_l1"] = recorder_tracking.create()
            nest.Connect(d1_l1, rec["d1_l1"])
        return rec

    def d1(self, i):
        return self.channels[i]["d1"]

    def d2(self, i):
        return self.channels[i]["d2"]

    def d1_l1(self, i):
        return self.channels[i]["d1_l1"]

    def get_recorder(self, i, name):
        return self.channels[i]["recorders"][name]

    def get_readout_baseline(self, i, name):
        """name: 'd1' or 'd2'. None if never set (fresh network, or
        readout_baseline_correction=False)."""
        return self.channels[i][f"{name}_readout_baseline"]

    def set_readout_baseline(self, i, name, baseline):
        """Restores a saved baseline (plain list/array, positional order
        matching self.channels[i][name].tolist()) -- main.py calls this
        after a checkpoint rebuild, mirroring V_th's own restore."""
        self.channels[i][f"{name}_readout_baseline"] = np.asarray(baseline, dtype=float)

    def weighted_rate_in_window(self, i, name, t0, t1):
        """Population-vector-style readout (vector_readout=True only): per-
        neuron-weighted analog of Base.rate_in_window -- instead of a plain
        spike count (identical weight per neuron), sums each spiking
        neuron's FIXED preferred value (channels[i][f"{name}_pref"]), then
        normalizes by (N*duration_s) exactly like rate_in_window so the
        result is comparably scaled and a drop-in replacement in
        compute_heads(). Requires vector_readout=True (pref arrays/id maps
        only exist then). When pref is 2D (unified_readout's D1: shape
        (N, dim_actions)), returns a dim_actions-length vector (sum over
        axis 0) instead of a scalar -- one weighted vote per action
        dimension from the single merged population.

        readout_baseline_correction (see __init__'s own comment): when
        True, subtracts each neuron's own slow EMA baseline RATE before
        it enters the vote -- algebraically identical to the uncentered
        path when baseline=0 (result = sum_i pref_i*rate_i/N either way),
        just expressed per-neuron-rate instead of per-spike-count so a
        baseline can be subtracted first."""
        ch = self.channels[i]
        recorder = ch["recorders"][name]
        pref = ch[f"{name}_pref"]
        id_to_idx = ch[f"{name}_id_to_idx"]
        is_vector = pref.ndim == 2
        zero = np.zeros(pref.shape[1]) if is_vector else 0.0
        events = nest.GetStatus(recorder, keys="events")[0]
        times = np.asarray(events["times"])
        senders = np.asarray(events["senders"])
        duration_s = (t1 - t0) / 1000.0
        mask = (times >= t0) & (times < t1)
        win_senders = senders[mask]

        if not self.readout_baseline_correction:
            if len(win_senders) == 0:
                return zero
            idxs = np.fromiter((id_to_idx[s] for s in win_senders), dtype=int, count=len(win_senders))
            weighted_sum = pref[idxs].sum(axis=0)
            if not is_vector:
                weighted_sum = float(weighted_sum)
            return weighted_sum / (self.N * duration_s)

        n_neurons = pref.shape[0]
        counts = np.zeros(n_neurons)
        if len(win_senders):
            idxs = np.fromiter((id_to_idx[s] for s in win_senders), dtype=int, count=len(win_senders))
            np.add.at(counts, idxs, 1)
        rate = counts / max(duration_s, 1e-9)
        baseline_key = f"{name}_readout_baseline"
        baseline = ch[baseline_key]
        if baseline is None:
            baseline = rate.copy()   # first call: no history yet, centers to exactly 0 this step
        centered_rate = rate - baseline
        decay = np.exp(-max(duration_s * 1000.0, 1e-9) / self.readout_baseline_tau)
        ch[baseline_key] = baseline * decay + rate * (1. - decay)
        weighted_sum = (pref * centered_rate[:, None]).sum(axis=0) if is_vector \
            else float((pref * centered_rate).sum())
        return weighted_sum / self.N

    def clear_rates(self):
        for ch in self.channels:
            self.clear_spikes(ch["recorders"]["d1"])
            self.clear_spikes(ch["recorders"]["d2"])
            if "d1_l1" in ch["recorders"]:
                self.clear_spikes(ch["recorders"]["d1_l1"])

    def collect_stats(self, option_num, stats):
        for ch_i, ch in enumerate(self.channels):
            # d1_hidden_layer: cortex no longer connects to d1 directly --
            # d1_l1->d1 is the pathway that actually feeds it, so THAT's
            # what "d1_..." stats should reflect (same conceptual role the
            # direct cortex->d1 connection had). cortex->d1_l1 isn't
            # separately tracked here yet.
            d1_source = ch["d1_l1"] if ch.get("d1_l1") is not None else self.cortex
            conns_d1 = nest.GetConnections(d1_source, ch["d1"])
            conns_d2 = nest.GetConnections(self.cortex, ch["d2"])
            w_d1 = np.array(nest.GetStatus(conns_d1, "weight"))
            w_d2 = np.array(nest.GetStatus(conns_d2, "weight"))
            self.append_stat(stats, f"d1_opt_{option_num}_{ch_i}_mean", np.mean(w_d1))
            self.append_stat(stats, f"d1_opt_{option_num}_{ch_i}_var",  np.var(w_d1))
            self.append_stat(stats, f"d2_opt_{option_num}_{ch_i}_mean", np.mean(w_d2))
            self.append_stat(stats, f"d2_opt_{option_num}_{ch_i}_var",  np.var(w_d2))


class ActorD1D2ChannelsWTA(ActorD1D2Channels):
    """
    Extends ActorD1D2Channels with D1 self-recurrent excitation (efference
    copy): keeps the winning channel active into the reward phase so the
    STDP eligibility trace is refreshed when DA arrives 50 ms later.

    BipedalWalker is unaffected — it still uses ActorD1D2Channels directly.

    The class name is historical: it also carried a cross-channel WTA
    inhibitory pool, ported from a discrete-choice (bipedal-walker) actor,
    but it was never actually wired in for this project's tasks and has
    been removed -- CartPole and the arm each build exactly one D1/D2
    channel (`_n_built_channels=1` under `unified_readout`), so there was
    never a second channel for it to suppress. The real opponent
    mechanism between D1 and D2 is the base class's cross-inhibition
    (`disable_cross_inhibition`, default on), not this.
    """

    # Symmetric efference copy (both 1.0) -- was D1=1.0/D2=0.0. That old
    # split was ported from a discrete-choice actor (argmax picks one
    # action, so only reinforcing the winning Go pathway makes sense
    # there). This actor is continuous: heads = tanh(d1_rate - d2_rate),
    # and D1/D2 of the same joint fire together every step, so there's
    # no "winning channel" to selectively reinforce -- the asymmetry
    # just gave D1 an unopposed positive feedback loop into cortex that
    # D2 never got, which caused q1/q2 to saturate at their joint limits.
    W_D1_EFFERENCE = 1.0  # D1 → cortex exc (reinforces cortex activity for STDP trace refresh)
    W_D2_EFFERENCE = 1.0  # D2 → cortex exc (same, was silently a no-op at 0.0)

    def __init__(self, cortex, dim_actions: int, efference_target=None,
                 disable_efference: bool = False,
                 w_d1_efference_scale: float = 1.0,
                 w_d2_efference_scale: float = 1.0,
                 **kwargs):
        super().__init__(cortex, dim_actions, **kwargs)
        self._efference_target = efference_target if efference_target is not None else cortex
        # w_d1/d2_efference_scale: multiplies the class-default
        # W_D1/D2_EFFERENCE (both 1.0), to test whether the
        # efference-copy loop's strength matters. Default 1.0: unchanged.
        self.w_d1_efference_scale = w_d1_efference_scale
        self.w_d2_efference_scale = w_d2_efference_scale
        # disable_efference strips the D1/D2->cortex efference-copy
        # loop, separate from disable_recurrence (which only gates
        # D1/D2's own Mexican-hat/cross-inhibition). W_D1_EFFERENCE ==
        # W_D2_EFFERENCE (symmetric by design), but D2 fires
        # substantially more than D1 (the intentional ie_d1/ie_d2 +
        # V_th asymmetry), so D2's efference copy delivers
        # proportionally more re-excitation back into cortex than D1's
        # every step, which could differentially shape cortex
        # eligibility traces in a way that's self-referential rather
        # than context-driven.
        self.disable_efference = disable_efference
        if not self.disable_efference:
            self._build_efference_copy()

    def _build_efference_copy(self):
        """D1 → cortex efference copy — action feeds back to the driving layer,
        sustaining cortex activity during the reward phase so cortex→D1 STDP
        eligibility traces are refreshed when DA arrives."""
        for ch in range(self._n_built_channels):
            _d1, _d2, _tgt = self.d1(ch), self.d2(ch), self._efference_target

            def _build_d1_eff(_d1=_d1, _tgt=_tgt):
                nest.Connect(
                    _d1, _tgt,
                    conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1},
                    syn_spec={'synapse_model': 'static_synapse',
                              'weight': self.W_D1_EFFERENCE * self.w_d1_efference_scale, 'delay': nest.resolution},
                )
            futils.ensure_connectivity(_d1, _tgt, f"{self.name_prefix}d1_efference_ch{ch}", _build_d1_eff)

            def _build_d2_eff(_d2=_d2, _tgt=_tgt):
                nest.Connect(
                    _d2, _tgt,
                    conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1},
                    syn_spec={'synapse_model': 'static_synapse',
                              'weight': self.W_D2_EFFERENCE * self.w_d2_efference_scale, 'delay': nest.resolution},
                )
            futils.ensure_connectivity(_d2, _tgt, f"{self.name_prefix}d2_efference_ch{ch}", _build_d2_eff)



