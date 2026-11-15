"""batched_actor_learning.py — Python-side reimplementation of the
dopsyn (stdp_dopamine_synapse) three-factor rule for cortex->D1, applied
as ONE accumulated update per BATCH of several episodes instead of
continuously online.

Why not just reuse NEST's own c/n/b: dopsyn's eligibility trace `c`
does NOT accumulate while A_plus/A_minus=0 -- freezing the weight-update
mechanism also freezes credit tracking, since NEST's model scales both
by the same A_plus/A_minus. There is no way to "hold weights fixed but
let credit quietly build up" using the built-in synapse. So cortex->D1
must be built as a plain `static_synapse` (real weight, no live NEST
plasticity) for this to work at all, and this module reimplements the
SAME conceptual mechanism -- eligibility trace x dopamine deviation --
entirely in Python from spike data, fully decoupled from weight
application.

Two deliberate simplifications vs. dopsyn's exact behavior (see
stdp_dopamine_synapse.h), not bugs:
  1. NEST's weight update is a continuous-time closed-form integral of
     dw/dt = c(t)*(n(t)-b) between dopamine spike events. Replicating
     that exactly is a real undertaking with real bug risk for what is
     fundamentally a mechanism-level test -- this uses a discrete-time
     approximation instead (same tau_c decay, evaluated once per control
     step), not a byte-exact port.
  2. Eligibility is computed from PER-STEP SPIKE COUNTS (a rate-based
     Hebbian coincidence proxy: cortex_count[i] * d1_count[j]) rather
     than exact spike-pair timing the way dopsyn's separate Kplus
     (facilitate, post-triggered)/Kminus (depress, pre-triggered) traces
     do. A single symmetric product term is used (not separate ordered
     facilitate/depress terms) because this project's dopsyn config has
     A_plus == A_minus in magnitude (0.00275/0.00275) -- collapsing the
     pre/post ordering distinction doesn't introduce a magnitude bias
     here the way it would if they differed. Appropriate given this
     module already averages over many control steps per batch --
     millisecond-level pairing precision isn't the point.

Dopamine deviation (n-b) is NOT reimplemented -- it's read directly off
the REAL, live `critic_dopsyn` connections (cortex->critic, entirely
unaffected by this module), which already track the shared DA
population's n trace via NEST's own unmodified machinery -- composes
with cfg.teacher ("spiking" or "python") transparently, since n reflects
whatever actually drove DA regardless of which teacher produced it. Only
the actor's own eligibility/credit side is reimplemented; the
reward/critic pathway stays exactly as validated elsewhere this session.
"""
import numpy as np
import nest


class BatchedD1Learning:
    def __init__(self, cortex, d1, tau_c: float, a_plus: float,
                 wmin: float, wmax: float, batch_episodes: int, lr_scale: float = 1.0,
                 debias_pref: bool = False, center_nb_per_batch: bool = False,
                 debias_include_intercept: bool = True,
                 postsyn_baseline_correction: bool = False,
                 postsyn_baseline_tau: float = 5000.0,
                 use_action_perturbation: bool = False,
                 action_perturbation_scale: float = 1.0,
                 coincidence_bin_ms: float = None,
                 source_novelty_gate: bool = False,
                 source_novelty_tau: float = 5000.0,
                 source_novelty_gain: float = 1.0,
                 presyn_baseline_correction: bool = False,
                 presyn_baseline_tau: float = 50000.0,
                 reset_elig_per_episode: bool = False,
                 n_goals: int = None):
        # n_goals: None (default) = one shared nb_sum/nb_count/credit/
        # elig_sum, same as always. An int enables per-goal accumulators
        # and baselines, since different goals have different typical
        # returns and one pooled mean mis-centers credit across goals.
        # `elig` itself stays shared (it decays to ~0 within one step at
        # this project's tau_c, so a per-goal copy would be redundant) --
        # only `credit`/`elig_sum`/`nb_sum`/`nb_count` are goal-indexed.
        self._n_goals = n_goals
        if n_goals:
            self.nb_sum = np.zeros(n_goals)
            self.nb_count = np.zeros(n_goals, dtype=int)
        else:
            self.nb_sum = 0.0
            self.nb_count = 0
        self.tau_c = tau_c
        self.a_plus = a_plus
        self.wmin = wmin
        self.wmax = wmax
        self.batch_episodes = batch_episodes
        self.lr_scale = lr_scale
        # reset_elig_per_episode: the coincidence trace (`elig`) normally
        # persists and decays across episode boundaries within a batch,
        # only zeroed at commit(). When True, it's also zeroed at each
        # episode boundary so a new episode isn't influenced by stale
        # coincidence from the previous one (`credit`/`elig_sum` keep
        # accumulating regardless, since commit()'s math needs that).
        # Default False: unchanged behavior.
        self.reset_elig_per_episode = reset_elig_per_episode
        # postsyn_baseline_correction: coincidence normally uses D1's raw
        # spike count as the output factor, so a tonically co-active D1
        # neuron accumulates eligibility regardless of whether it actually
        # changed. This flag uses D1's deviation from its own running
        # baseline rate instead (baseline: a slow EMA, always tracked so
        # it's warm if toggled on mid-run). Default False: unchanged.
        self.postsyn_baseline_correction = postsyn_baseline_correction
        self.postsyn_baseline_tau = postsyn_baseline_tau
        self.d1_baseline = None
        # use_action_perturbation: replaces the D1-spike-count post factor
        # with the actual, noise-free exploration perturbation applied that
        # step (env_reach.py's select_action() already computes it),
        # projected onto each D1 neuron's preferred direction: coincidence
        # = cortex_count * (pref[tgt] . action_deviation). Requires
        # set_pref() and step()'s action_deviation argument every call.
        # Default False: unchanged.
        self.use_action_perturbation = use_action_perturbation
        # action_perturbation_scale: raw exploration noise (~O(exploration_
        # rate), typically well under 1) and raw D1 spike counts (small
        # positive integers) aren't necessarily on the same natural scale
        # as what a_plus was calibrated against -- this compensates,
        # analogous to actor_batched_d1_delta_python_scale's role for the
        # delta_python signal source. Default 1.0 = no compensation;
        # tune empirically if elig ends up too small/large in this mode.
        self.action_perturbation_scale = action_perturbation_scale
        # coincidence_bin_ms: coincidence normally sums spike counts over
        # the whole control-step window as one lump, with no timing
        # resolution. This subdivides the window into bins of this width
        # (ms) and applies the elig decay-then-accumulate update once per
        # bin, so pairs separated by several bins decay via tau_c instead
        # of all counting as equally coincident. None/<=0 disables it
        # (one bin spanning the whole window, same as before).
        self.coincidence_bin_ms = coincidence_bin_ms
        # source_novelty_gate: scales each cortex neuron's contribution to
        # coincidence by a novelty factor that's low when that neuron has
        # been firing tonically and high when it's been quiet recently
        # (novelty[src] = 1/(1 + source_novelty_gain * cortex_baseline[src])),
        # so constantly-active neurons don't dominate credit just by being
        # active. Composes with coincidence_bin_ms. Default False: unchanged.
        self.source_novelty_gate = source_novelty_gate
        self.source_novelty_tau = source_novelty_tau
        self.source_novelty_gain = source_novelty_gain
        self.cortex_baseline = None
        # presyn_baseline_correction: subtracts a slow EMA of each cortex
        # neuron's own spike count before forming the coincidence product
        # (coincidence = (c_counts[src] - presyn_baseline[src]) *
        # post_factor[tgt]), instead of using raw presynaptic counts, which
        # are dominated by each neuron's tonic activity and ill-condition
        # the credit signal. Uses its own, deliberately slower EMA than
        # source_novelty_gate's cortex_baseline. Default False: unchanged.
        self.presyn_baseline_correction = presyn_baseline_correction
        self.presyn_baseline_tau = presyn_baseline_tau
        self.presyn_baseline = None
        self.episodes_in_batch = 0
        self.n_commits = 0
        # center_nb_per_batch: subtracts this batch's own mean nb_value
        # from every step's contribution before it's folded into credit
        # (credit = sum_t elig(t) * (nb(t) - batch_mean_nb)), removing the
        # "was this batch's sampled targets lucky or unlucky" component and
        # leaving only the within-batch relative signal -- a policy-
        # gradient-style variance-reduction baseline applied per batch.
        # Computed exactly at commit() from elig_sum, no lookahead needed.
        # Default False: unchanged.
        self.center_nb_per_batch = center_nb_per_batch
        # debias_pref: removes the component of each commit's per-D1-neuron
        # credit that's linearly predictable from that neuron's preferred
        # direction vector (pref_q1, pref_q2), via a no-intercept
        # least-squares regression, correcting a directional bias that a
        # single population-wide dopamine scalar can't fix on its own.
        # Requires set_pref() before the first commit(); no-ops with a
        # one-time warning if pref was never set. Default False: unchanged.
        self.debias_pref = debias_pref
        # debias_include_intercept: whether debias_pref's regression fits a
        # constant term alongside [pref_q1, pref_q2]. True also zeroes the
        # batch's own average credit as a side effect of removing the
        # pref-correlated slope; False fits direction only and preserves
        # the batch's overall mean credit level. Default True.
        self.debias_include_intercept = debias_include_intercept
        self.pref = None
        self._warned_no_pref = False
        self.attach(cortex, d1)

    def set_pref(self, pref_ordered):
        """pref_ordered: (N_D1, 2) array, index i = this neuron's
        preferred (q1,q2) direction, in the same order as self.d1.tolist().
        Call once after construction and again after every attach() (ids
        can renumber across a checkpoint rebuild)."""
        self.pref = np.asarray(pref_ordered, dtype=float)

    def attach(self, cortex, d1):
        """(Re)acquire connection handles -- call again after any
        nest.ResetKernel() rebuild. Uses canonical (source,target) order
        (np.lexsort), not raw GetConnections order, which is not
        rebuild-stable even though the underlying connections/weights are
        unchanged. elig/credit accumulators are plain Python/numpy arrays
        owned by this object, not NEST state, so they survive a kernel
        reset; only the connection handles need re-fetching.

        A rebuild mid-batch remaps the existing accumulator arrays onto
        the new native order via each synapse's (cortex_idx, d1_idx)
        pair, which is stable across rebuilds, instead of leaving them
        indexed by the old (unstable) native order."""
        _had_prev = hasattr(self, "native_src_idx") and hasattr(self, "elig")
        if _had_prev:
            _old_src_idx = self.native_src_idx
            _old_tgt_idx = self.native_tgt_idx
            _old_elig, _old_credit, _old_elig_sum = self.elig, self.credit, self.elig_sum
        self.cortex = cortex
        self.d1 = d1
        conns = nest.GetConnections(cortex, d1)
        src = np.array(conns.get("source"))
        tgt = np.array(conns.get("target"))
        self.order = np.lexsort((tgt, src))
        self.conns = conns
        cortex_id_to_idx = {nid: i for i, nid in enumerate(cortex.tolist())}
        d1_id_to_idx = {nid: i for i, nid in enumerate(d1.tolist())}
        # Per-synapse (cortex_idx, d1_idx), in NATIVE conns order -- used to
        # scatter each step's coincidence product onto the right synapse
        # without a per-synapse Python loop.
        self.native_src_idx = np.array([cortex_id_to_idx[s] for s in src])
        self.native_tgt_idx = np.array([d1_id_to_idx[t] for t in tgt])
        self._cortex_id_to_idx = cortex_id_to_idx
        self._d1_id_to_idx = d1_id_to_idx
        n = len(conns)
        # credit/elig_sum are (n,) in the default (n_goals=None) case, or
        # (n_goals, n) in context-gated mode -- elig itself is ALWAYS
        # (n,), see __init__'s n_goals comment for why it's not
        # duplicated per goal. `arr[..., perm]` below applies the
        # permutation along the LAST axis regardless of 1D/2D, so the
        # exact same remap logic (and the same rebuild-safety guarantee)
        # covers both shapes with no branching.
        _credit_shape = (n,) if not self._n_goals else (self._n_goals, n)
        if not _had_prev or len(_old_elig) != n:
            self.elig = np.zeros(n)               # native order
            self.credit = np.zeros(_credit_shape)    # native order
            self.elig_sum = np.zeros(_credit_shape)  # native order -- see center_nb_per_batch
        else:
            # Remap old-native-order accumulators onto the new native
            # order through the (cortex_idx, d1_idx) key -- a no-op
            # (identity permutation) whenever the rebuild happened to
            # land on a batch boundary (accumulators already zero), a
            # real reorder otherwise.
            _key_to_old = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(_old_src_idx, _old_tgt_idx))}
            _perm = np.array([_key_to_old[(int(s), int(t))]
                               for s, t in zip(self.native_src_idx, self.native_tgt_idx)])
            self.elig = _old_elig[_perm]
            self.credit = _old_credit[..., _perm]
            self.elig_sum = _old_elig_sum[..., _perm]
        n_d1 = len(d1)
        if self.d1_baseline is not None and len(self.d1_baseline) != n_d1:
            self.d1_baseline = None      # d1 population resized -- re-warm from scratch
        n_cortex = len(cortex)
        if self.cortex_baseline is not None and len(self.cortex_baseline) != n_cortex:
            self.cortex_baseline = None  # cortex population resized -- re-warm from scratch
        if self.presyn_baseline is not None and len(self.presyn_baseline) != n_cortex:
            self.presyn_baseline = None
    def step(self, cortex_rec, d1_rec, t0: float, t1: float, nb_value, action_deviation=None,
              goal_idx: int = None):
        """Call once per control step, AFTER the step's spikes have
        occurred (t0,t1 = that step's activation window). nb_value: this
        step's dopamine-deviation sample -- caller reads this from the
        real critic_dopsyn pathway (e.g. net.dopamine._nb_samples[-1]
        right after net.dopamine.sample_nb()), NOT reimplemented here.

        nb_value may be a scalar (every prior behavior, shared identically
        by every D1 neuron regardless of its own preferred direction) OR a
        (dim_actions,) vector (DIRECTIONAL mode -- e.g. per-
        joint deltas from python_critics_per_joint, see main.py). In
        vector mode, each SYNAPSE'S OWN target-neuron preferred-direction
        vector (self.pref, set via set_pref() -- MUST be called before
        step() in this mode) projects the shared vector down to a per-
        synapse scalar via a dot product (self.pref[native_tgt_idx[i]] .
        nb_value) -- different neurons get genuinely different credit
        depending on which direction they represent, not a uniform
        broadcast. Falls back to plain scalar broadcast when nb_value is
        a plain float/int (np.ndim==0) -- zero behavior change for every
        existing (non-directional) config.

        action_deviation: (dim_actions,) array, this step's REALIZED minus
        MEAN action (e.g. env_reach.py's `action - stat_action`, i.e. the
        exact injected exploration noise before clipping) -- required, and
        only used, when self.use_action_perturbation=True (see __init__)."""
        dt_ms = t1 - t0
        nbins = 1
        if self.coincidence_bin_ms and self.coincidence_bin_ms > 0:
            nbins = max(1, int(round(dt_ms / self.coincidence_bin_ms)))
        if nbins > 1:
            c_counts_bins = self._counts_binned(
                cortex_rec, self._cortex_id_to_idx, len(self.cortex), t0, t1, nbins)
            d1_counts_bins = self._counts_binned(
                d1_rec, self._d1_id_to_idx, len(self.d1), t0, t1, nbins)
            d1_counts = d1_counts_bins.sum(axis=0)
            c_counts_total = c_counts_bins.sum(axis=0)
        else:
            c_counts_bins = None
            d1_counts = self._counts(d1_rec, self._d1_id_to_idx, len(self.d1), t0, t1)
            c_counts_total = self._counts(cortex_rec, self._cortex_id_to_idx, len(self.cortex), t0, t1)
        if self.d1_baseline is None:
            self.d1_baseline = d1_counts.copy()
        d1_deviation = d1_counts - self.d1_baseline
        # Baseline updates AFTER computing this step's deviation (compares
        # against the PRIOR estimate, not one already nudged by this step)
        # -- always tracked, regardless of postsyn_baseline_correction, so
        # the baseline is warm if the flag is toggled on later. Updated
        # ONCE per step (not per bin) -- it's a slow, step-scale quantity.
        baseline_decay = np.exp(-dt_ms / self.postsyn_baseline_tau)
        self.d1_baseline = self.d1_baseline * baseline_decay + d1_counts * (1. - baseline_decay)

        if self.cortex_baseline is None:
            self.cortex_baseline = c_counts_total.copy()
        if self.source_novelty_gate:
            # Novelty computed from the PRIOR baseline (before this step's
            # own spikes fold in) -- a source isn't penalized for the very
            # spike that's earning it credit right now, only for its
            # activity BEFORE this step.
            novelty = 1.0 / (1.0 + self.source_novelty_gain * self.cortex_baseline)
        else:
            novelty = None
        cortex_novelty_decay = np.exp(-dt_ms / self.source_novelty_tau)
        self.cortex_baseline = self.cortex_baseline * cortex_novelty_decay + c_counts_total * (1. - cortex_novelty_decay)

        if self.presyn_baseline is None:
            self.presyn_baseline = c_counts_total.copy()
        # Deviation computed from the PRIOR baseline (before this step's own
        # spikes fold in), same convention as d1_deviation above.
        c_counts_centered = c_counts_total - self.presyn_baseline
        presyn_decay = np.exp(-dt_ms / self.presyn_baseline_tau)
        self.presyn_baseline = self.presyn_baseline * presyn_decay + c_counts_total * (1. - presyn_decay)

        if self.use_action_perturbation:
            # Per-synapse projection of the KNOWN (noise-free) exploration
            # perturbation through each synapse's own target D1 neuron's
            # pref -- same idiom as nb_value's directional-mode projection
            # below, just applied to the "post" factor instead of "reward".
            # Constant across bins (action_deviation is drawn once per
            # control step, not resolved at sub-step timing).
            post_signal = self.action_perturbation_scale * (
                self.pref[self.native_tgt_idx] @ np.asarray(action_deviation))
        elif self.postsyn_baseline_correction:
            post_signal = d1_deviation[self.native_tgt_idx]
        else:
            post_signal = d1_counts[self.native_tgt_idx]

        novelty_per_synapse = novelty[self.native_src_idx] if novelty is not None else None
        c_counts_for_coincidence = c_counts_centered if self.presyn_baseline_correction else c_counts_total
        if nbins == 1:
            coincidence = c_counts_for_coincidence[self.native_src_idx] * post_signal
            if novelty_per_synapse is not None:
                coincidence = coincidence * novelty_per_synapse
            self.elig *= np.exp(-dt_ms / self.tau_c)
            self.elig += self.a_plus * coincidence
        else:
            # Same decay-then-accumulate update as above, applied once per
            # fine bin instead of once for the whole window -- a cortex/D1
            # pair coincident in the SAME bin contributes near-full
            # strength; pairs several bins apart get progressively decayed
            # via the same tau_c mechanism, just at finer resolution. Only
            # c_counts varies bin-to-bin here (post_signal, above, is
            # either constant across bins already -- action_perturbation,
            # postsyn_baseline_correction -- or intentionally re-derived
            # per bin for the raw-count mode via d1_counts_bins below).
            # novelty_per_synapse is a single per-step value (from the
            # step-scale cortex_baseline), applied identically to every
            # bin -- same rationale as action_deviation being constant
            # across bins.
            bin_ms = dt_ms / nbins
            bin_decay = np.exp(-bin_ms / self.tau_c)
            use_binned_post = (not self.use_action_perturbation and not self.postsyn_baseline_correction)
            presyn_baseline_share = (self.presyn_baseline / nbins) if self.presyn_baseline_correction else None
            for b in range(nbins):
                c_bin = c_counts_bins[b][self.native_src_idx]
                if presyn_baseline_share is not None:
                    c_bin = c_bin - presyn_baseline_share[self.native_src_idx]
                if novelty_per_synapse is not None:
                    c_bin = c_bin * novelty_per_synapse
                if use_binned_post:
                    post_bin = d1_counts_bins[b][self.native_tgt_idx]
                else:
                    post_bin = post_signal
                self.elig *= bin_decay
                self.elig += self.a_plus * c_bin * post_bin
        if np.ndim(nb_value) > 0:
            effective_nb = self.pref[self.native_tgt_idx] @ np.asarray(nb_value)
        else:
            effective_nb = nb_value
        if self._n_goals:
            # context-gated mode: route this step's contribution into
            # its OWN goal's bucket (see __init__'s n_goals comment) --
            # goal_idx is the CURRENT episode's commanded goal, constant
            # for every step within one episode, supplied by the caller
            # (nb_value is assumed scalar here -- vector/directional mode
            # isn't combined with context-gating by any config yet).
            if goal_idx is None:
                raise ValueError("BatchedD1Learning: goal_idx is required every "
                                  "step() call in context-gated mode (n_goals set).")
            self.elig_sum[goal_idx] += self.elig
            self.credit[goal_idx] += self.elig * effective_nb
            self.nb_sum[goal_idx] += nb_value
            self.nb_count[goal_idx] += 1
        else:
            self.elig_sum += self.elig
            self.credit += self.elig * effective_nb
            self.nb_sum = self.nb_sum + nb_value
            self.nb_count += 1

    @staticmethod
    def _counts(rec, id_to_idx, n, t0, t1):
        events = nest.GetStatus(rec, keys="events")[0]
        times = np.asarray(events["times"])
        senders = np.asarray(events["senders"])
        counts = np.zeros(n)
        if len(times) == 0:
            return counts
        mask = (times >= t0) & (times < t1)
        senders = senders[mask]
        if len(senders) == 0:
            return counts
        idxs = np.fromiter((id_to_idx[s] for s in senders), dtype=int, count=len(senders))
        np.add.at(counts, idxs, 1)
        return counts

    @staticmethod
    def _counts_binned(rec, id_to_idx, n, t0, t1, nbins):
        """Same event fetch as _counts, but returns (nbins, n): spikes
        bucketed by which fine time bin within [t0,t1) they fall in
        (coincidence_bin_ms). One GetStatus call, same as _counts -- no
        extra NEST round-trips for the extra resolution."""
        events = nest.GetStatus(rec, keys="events")[0]
        times = np.asarray(events["times"])
        senders = np.asarray(events["senders"])
        counts = np.zeros((nbins, n))
        if len(times) == 0:
            return counts
        mask = (times >= t0) & (times < t1)
        times = times[mask]
        senders = senders[mask]
        if len(senders) == 0:
            return counts
        idxs = np.fromiter((id_to_idx[s] for s in senders), dtype=int, count=len(senders))
        bin_idx = np.clip(((times - t0) / (t1 - t0) * nbins).astype(int), 0, nbins - 1)
        np.add.at(counts, (bin_idx, idxs), 1)
        return counts

    def end_episode(self) -> bool:
        """Call once per episode boundary. Commits (applies the
        accumulated batch update, resets accumulators) and returns True
        exactly every `batch_episodes` calls; otherwise returns False.

        reset_elig_per_episode (see __init__'s own comment): zeroes ONLY
        `elig` (the coincidence trace), every episode boundary, whether
        or not this call also happens to commit -- `credit`/`elig_sum`
        are untouched here (commit() alone resets those, at batch end,
        exactly as before) so the batch-level accumulation this class is
        built around still works unchanged."""
        if self.reset_elig_per_episode and hasattr(self, "elig"):
            self.elig[:] = 0.0
        self.episodes_in_batch += 1
        if self.episodes_in_batch >= self.batch_episodes:
            self.commit()
            return True
        return False

    def commit(self):
        # last_raw_credit/last_residual_credit: stashed here since `credit`
        # is zeroed at the end of this method and an external
        # episode_callback only gets control back after commit() has run.
        # Always computed, regardless of debias_pref, so a diagnostic can
        # compare both forms on any run.
        self.last_raw_credit = self.credit.copy()
        # _debias_pref_correlation expects a flat (n_synapses,) array --
        # self.credit is (n_goals, n_synapses) in context-gated mode, so
        # this EARLY diagnostic stash is skipped there; the meaningful
        # one (on the already goal-combined, always-flat `credit` local
        # below) still runs unconditionally a few lines down.
        self.last_residual_credit = (
            self._debias_pref_correlation(self.credit)
            if self.pref is not None and not self._n_goals else None)
        # last_elig: the eligibility trace alone, decoupled from nb_value,
        # stashed for the same reason as last_raw_credit. Tests whether
        # which synapses accumulate eligibility is itself a consistent,
        # structured signal batch-to-batch, independent of the reward sign.
        self.last_elig = self.elig.copy()

        credit = self.credit
        # center_nb_per_batch: credit_centered = credit_raw - batch_mean_nb
        # * elig_sum (exact, see __init__'s comment -- linear in nb(t), so
        # the batch-mean can be factored out and subtracted post-hoc from
        # the already-accumulated credit/elig_sum, no lookahead needed).
        # DIRECTIONAL mode (nb_sum/batch_mean_nb is a (dim_actions,)
        # vector, not a scalar): the same linearity argument holds
        # component-wise -- credit_raw = sum_t elig(t) * (pref[tgt] .
        # nb(t)), so credit_centered = credit_raw - (pref[tgt] .
        # batch_mean_nb) * elig_sum, i.e. project the BATCH's mean vector
        # through each synapse's own target-neuron pref (exactly the same
        # per-step projection step() already does), then subtract that
        # scalar times elig_sum, per synapse.
        self.last_centered_credit = None
        if self._n_goals:
            # Per-goal generalization of the batch-pooled centering
            # above: instead of ONE scalar baseline shared by every
            # goal's episodes in this batch, each goal g gets its OWN
            # baseline (nb_sum[g]/nb_count[g], computed from only that
            # goal's own episodes this batch) subtracted from only its
            # own accumulator (credit[g]/elig_sum[g]) -- same exact
            # linear-factoring identity as center_nb_per_batch, applied
            # per goal, then summed (weights are shared across goals, so
            # the committed update is the sum of each goal's own
            # centered contribution -- matches the spec's "Δw += lr *
            # elig[g]*(credit[g]-baseline[g])" summed over g). A goal
            # that got zero episodes this batch contributes nothing
            # (its credit[g]/elig_sum[g] are already exactly zero).
            credit = np.zeros(self.credit.shape[-1])
            for _g in range(self._n_goals):
                if self.nb_count[_g] > 0:
                    _baseline_g = self.nb_sum[_g] / self.nb_count[_g]
                    credit += self.credit[_g] - _baseline_g * self.elig_sum[_g]
                else:
                    credit += self.credit[_g]
            self.last_centered_credit = credit.copy()
        elif self.center_nb_per_batch and self.nb_count > 0:
            batch_mean_nb = self.nb_sum / self.nb_count
            if np.ndim(batch_mean_nb) > 0:
                batch_mean_effective = self.pref[self.native_tgt_idx] @ np.asarray(batch_mean_nb)
            else:
                batch_mean_effective = batch_mean_nb
            credit = credit - batch_mean_effective * self.elig_sum
            self.last_centered_credit = credit.copy()
        self.last_residual_credit = (
            self._debias_pref_correlation(credit) if self.pref is not None else None)

        if self.debias_pref:
            if self.pref is not None:
                credit = self._debias_pref_correlation(credit)
            elif not self._warned_no_pref:
                print("[BatchedD1Learning] debias_pref=True but set_pref() was "
                      "never called -- skipping debiasing this commit.")
                self._warned_no_pref = True
        weights = np.array(nest.GetStatus(self.conns, "weight"))
        weights = np.clip(weights + self.lr_scale * credit, self.wmin, self.wmax)
        nest.SetStatus(self.conns, "weight", weights.tolist())
        self.credit[:] = 0.0
        self.elig[:] = 0.0
        self.elig_sum[:] = 0.0
        if self._n_goals:
            self.nb_sum[:] = 0.0
            self.nb_count[:] = 0
        else:
            self.nb_sum = 0.0
            self.nb_count = 0
        self.episodes_in_batch = 0
        self.n_commits += 1

    def _debias_pref_correlation(self, credit):
        """Regress this commit's per-D1-neuron mean credit against
        [pref_q1, pref_q2] (plus an intercept column iff
        debias_include_intercept) and return a corrected per-synapse
        credit array with that linear component removed (see debias_pref's
        __init__ comment for the full motivation, and
        debias_include_intercept's comment for why the intercept is
        excluded by default)."""
        n_d1 = self.pref.shape[0]
        indegree = np.bincount(self.native_tgt_idx, minlength=n_d1)
        credit_sum = np.bincount(self.native_tgt_idx, weights=credit, minlength=n_d1)
        safe_indegree = np.where(indegree > 0, indegree, 1)
        neuron_credit_mean = credit_sum / safe_indegree
        cols = [self.pref[:, 0], self.pref[:, 1]]
        if self.debias_include_intercept:
            cols.append(np.ones(n_d1))
        X = np.column_stack(cols)
        coeffs, *_ = np.linalg.lstsq(X, neuron_credit_mean, rcond=None)
        predicted = X @ coeffs
        # Additive per-neuron shift that removes exactly the predicted
        # (pref-correlated, plus mean-level ONLY if debias_include_
        # intercept) component, applied uniformly to every synapse feeding
        # that neuron -- preserves the WITHIN-neuron relative structure
        # across its own cortex afferents.
        correction_per_neuron = -predicted
        return credit + correction_per_neuron[self.native_tgt_idx]
