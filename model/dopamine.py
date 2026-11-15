import nest
import numpy as np
from base_util import Base
import recorder_tracking


class Dopamine(Base):
    """
    Dopamine / reward system.

    Creates DA neuron pool and STDP dopamine synapse models:
      - critic_dopsyn  (cortex → critic L1)
      - dopsyn         (cortex → D1 striatum, Go pathway)
      - adopsyn        (cortex → D2 striatum, NoGo pathway, inverted STDP)

    IMPORTANT: Instantiate Dopamine BEFORE Critic and ActorD1D2Channels,
    since nest.CopyModel calls here register the synapse types used downstream.

    Reward shaping is intentionally NOT part of this class.
    Call start_reward(reshaped_reward, pos_scaling, neg_scaling) from the
    training loop after computing the shaped reward externally.
    """

    def __init__(
            self,
            *,
            n_da_neurons: int   = 20,
            da_convert_hz_to_nest: float = 7e-4,
            tau_c: float        = 500.0,
            tau_c_adopsyn: float = 120.0,
            tau_n: float        = 4.0,
            tau_plus: float     = 20.0,
            tau_syn_ex_da: float = None,
            da_baseline_init: float = 10.0,
            reward_da_weight: float = 10.0,
            tonic_rate_hz: float = 1600.0,
            verbose: bool       = False,
            dopsyn_a_plus: float = 0.002,
            dopsyn_a_minus: float = 0.0035,
            adopsyn_a_plus: float = -0.0035,
            adopsyn_a_minus: float = -0.002,
            dopsyn_wmin: float = 10.,
            dopsyn_wmax: float = 40.,
            adopsyn_wmin: float = 8.,
            adopsyn_wmax: float = 30.,
            eligibility_weighted_nb: bool = True,
            b_meas_correction_factor: float = 0.95,
            n_baseline_zones: int = 1,
            condition_b_on_c: bool = False,
            context_gated_da: bool = False,
            da_zone_v_th: float = -52.0,
            da_zone_gate_weight: float = 20.0,
            da_zone_tonic_weight: float = 5.0,
            da_tau_m: float = 20.0,
            da_tonic_weight: float = 5.0,
            connect_da_to_actor_vt: bool = True,
            n_reward_channels: int = 0,
            da_baseline_ema_alpha: float = 0.041,
            da_baseline_fixed: bool = False,
    ):
        self.da_baseline_ema_alpha = da_baseline_ema_alpha
        # da_baseline_fixed: replaces the continuously-tracked EMA
        # baseline with Izhikevich (2007)'s simpler low-fixed-floor
        # design -- `b` is set once from da_baseline_init/_b_init and
        # never re-estimated. The EMA baseline's own `n-b` mean drifts
        # sign over training, and a static correction on top of the EMA
        # can't fix a sign-flipping drift. Freezing `b` removes the
        # tracking mechanism that produces the drift, at the cost of `b`
        # no longer adapting if the network's baseline firing genuinely
        # shifts over training. Default False: unchanged behavior.
        self.da_baseline_fixed = da_baseline_fixed
        self.eligibility_weighted_nb = eligibility_weighted_nb
        self.b_meas_correction_factor = b_meas_correction_factor
        # condition_b_on_c: sample_nb() already eligibility-weights n
        # within a step, which is mathematically the value of `b` that
        # zeros out this step's eligibility-weighted (n-b). But
        # update_da_baseline() then averages those already-conditioned
        # per-step values plainly across the sample window, treating a
        # near-zero-eligibility step the same as a strongly-eligible
        # one. This weights the temporal average itself by each
        # sampled step's own total eligibility magnitude instead.
        # Default off: unchanged behavior (plain temporal mean).
        self.condition_b_on_c = condition_b_on_c
        # n_baseline_zones: `b` is a NEST CommonSynapseProperty, one
        # shared value per synapse model. A single global baseline can
        # only express "better/worse than average across all contexts
        # combined," never "better/worse than expected for this
        # context" -- the actual root cause of this project's
        # context-tracking failure. n_baseline_zones>1 approximates a
        # state-conditioned baseline by giving spatially-distinct zones
        # of cortex->D1/D2 synapses their own independent `b` (and
        # hence their own `n`, `c`, via dedicated dopsyn_z{i}/
        # adopsyn_z{i} synapse models). Cortex is topographically
        # organized, so nearby target positions activate nearby cortex
        # zones -- this needs no explicit "which context is active"
        # signal, just a fixed spatial partition decided once at
        # construction (see actor.py, which owns the zone assignment).
        # Default 1: unchanged behavior (original unzoned models).
        self.n_baseline_zones = max(1, int(n_baseline_zones))
        # context_gated_da: n_baseline_zones alone only changes where
        # `b` is compared -- `n` itself stays identical across every
        # zone, because every zone's CopyModel pointed at the same
        # shared volume_transmitter. Zoning `b` alone doesn't move the
        # oracle result, because `(n-b)` still measures "surprising
        # relative to a global trace," never "surprising for this
        # zone."
        #
        # This flag makes `n` itself context-specific instead: each
        # zone gets its own DA neuron population and its own volume
        # transmitter. Each zone population gets the same tonic+reward
        # drive as the original DA_neurons, but at a raised V_th
        # (da_zone_v_th), so reaching a full phasic response needs an
        # extra excitatory kick from that zone's own cortex
        # (da_zone_gate_weight) -- wired in network.py. Net effect: a
        # reward pulse landing while zone Z's cortex wasn't active no
        # longer reliably elevates n_Z.
        #
        # Requires n_baseline_zones>1 (silently no-op otherwise).
        # Uncalibrated: da_zone_v_th/da_zone_gate_weight are starting
        # guesses, not validated against any measurement.
        self.context_gated_da = context_gated_da and self.n_baseline_zones > 1
        # n_reward_channels (per-channel/per-joint reward decomposition):
        # see the full block below for the actual DA population/
        # volume-transmitter/synapse-model setup -- set here, early,
        # since baseline_model_names() needs it already. Default 1:
        # behaves exactly as before, one shared "dopsyn"/"adopsyn" pair.
        # Uses max(0, ...), not max(1, ...), so the "disabled" sentinel
        # (network.py passes 0) stays distinguishable from "enabled
        # with exactly 1 channel" (a real case for n_actions=1 tasks).
        self.n_reward_channels = max(0, int(n_reward_channels))
        super().__init__()
        self.da_convert_hz_to_nest = da_convert_hz_to_nest
        self.reward_da_weight = reward_da_weight
        self.verbose = verbose

        self.da_baseline = da_baseline_init
        self.da_phasic   = 0.0
        self.da_rate_hz  = 0.0
        self.episode_da_effective = []
        # da_phasic used to be logged as whatever measure_da_activity's
        # last call that episode happened to compute -- a single
        # noisy Poisson sample, not an episode-representative value.
        # Accumulate-then-average per episode instead, same pattern as
        # episode_da_effective.
        self.episode_da_phasic = []
        self.learning_active = False
        # da_baseline_init should be set close to the empirically-
        # observed DA population rate (~20-25Hz) rather than a high
        # guess, since a too-high baseline keeps conditioned (n-b)
        # pushed below zero for outcomes that should register
        # net-positive. This formula seeds both the DA population's own
        # baseline (self.da_baseline above) and the STDP synapses' `b`
        # coherently from that one calibrated number.
        _b_init = n_da_neurons * da_convert_hz_to_nest * da_baseline_init
        # baseline_model_names() (below) returns ('critic_dopsyn', 'dopsyn',
        # 'adopsyn') when n_baseline_zones==1 (unchanged default) or
        # 'critic_dopsyn' plus one dopsyn_z{i}/adopsyn_z{i} pair per zone
        # otherwise -- these two dicts, and everywhere else in this file
        # that used to hardcode the 3-tuple, now key off that single source
        # of truth so zoning doesn't need separate bookkeeping.
        self._da_b_smoothed = {name: _b_init for name in self.baseline_model_names()}
        self.last_time_dop_base_estimated = 0
        self._nb_samples = []   # whole-episode diagnostic (nb_mean / nb_pos_frac) -- dopsyn-specific
        # Previously one shared _n_samples list, populated from
        # dopsyn's own 'n' only, was used to calibrate all three
        # models' baseline 'b' -- critic_dopsyn and adopsyn never got
        # their own baseline. Each model now gets its own connection
        # cache and its own recent-window sample list.
        self._n_samples_by_model = {name: [] for name in self.baseline_model_names()}
        # Parallel per-step eligibility-magnitude samples
        # (condition_b_on_c above), same indexing as
        # _n_samples_by_model. Populated regardless of condition_b_on_c
        # (cheap) so the flag can be flipped without a rerun to
        # backfill history.
        self._c_samples_by_model = {name: [] for name in self.baseline_model_names()}
        self._conns_by_model = {}
        self._nb_mean = 0.0
        self._nb_pos_frac = 0.0

        self.pos_reward_pg     = nest.Create('poisson_generator', 20, {'rate': 0.0})
        self.neg_reward_pg     = nest.Create('poisson_generator', 20, {'rate': 0.0})
        self.pos_reward_parrot = nest.Create('parrot_neuron', 20)
        self.neg_reward_parrot = nest.Create('parrot_neuron', 20)
        nest.Connect(self.pos_reward_pg, self.pos_reward_parrot, "one_to_one")
        nest.Connect(self.neg_reward_pg, self.neg_reward_parrot, "one_to_one")

        self.recorders = {
            'da':         recorder_tracking.create(),
            'pos_reward': recorder_tracking.create(),
            'neg_reward': recorder_tracking.create(),
        }
        nest.Connect(self.pos_reward_parrot, self.recorders["pos_reward"])
        nest.Connect(self.neg_reward_parrot, self.recorders["neg_reward"])

        # da_tau_m/da_tonic_weight exposed together so a shorter tau_m
        # can be tested without breaking the calibrated tonic/reward
        # balance. A neuron's steady-state response to a constant-rate
        # input scales with tau_m (R_m=tau_m/C_m), so shrinking tau_m
        # shrinks every synaptic input's steady-state contribution
        # proportionally, not just reward's -- callers should scale
        # da_tonic_weight and reward_da_weight by the same factor
        # together, not just reward_da_weight alone. Defaults (20.0,
        # 5.0) match the previously-hardcoded values.
        da_params = {
            'tau_m':  da_tau_m,
            'V_reset': -65.0,
            'V_th':   -57.0,
            't_ref':  3.0,
            'C_m':    200.0,
            'I_e':    220.0,
        }
        if tau_syn_ex_da is not None:
            da_params['tau_syn_ex'] = tau_syn_ex_da
            da_params['tau_syn_in'] = tau_syn_ex_da
        self.DA_neurons = nest.Create('iaf_psc_alpha', n_da_neurons, da_params)
        # Since a Poisson rate can't go negative, a too-low baseline
        # leaves little room for a negative reward to push DA below
        # baseline before it floor-saturates at 0Hz, crushing small and
        # large mistakes to the same near-zero rate -- a near-binary
        # "punish" signal instead of a graded one on the channel
        # (D2/adopsyn) that's supposed to encode "how much worse than
        # expected". Doubling the tonic drive gives negative rewards
        # real headroom. da_baseline_init in the configs must be
        # re-seeded to match the resulting empirical rate.
        tonic = nest.Create("poisson_generator", n_da_neurons, {"rate": tonic_rate_hz})
        nest.Connect(tonic, self.DA_neurons, "one_to_one", {"weight": da_tonic_weight})        # Per-neuron independent noise to break synchrony
        # noise = nest.Create("poisson_generator", n_da_neurons, {"rate": 500.0})
        # nest.Connect(noise, self.DA_neurons, "one_to_one", {"weight": 0.1})
        nest.Connect(self.DA_neurons, self.recorders["da"])

        factor = 1.0
        self.vol_trans_critic = nest.Create('volume_transmitter', 1)
        self.vol_trans_actor  = nest.Create('volume_transmitter', 1)

        # vol_trans_actor_zones: which volume transmitter each zone's
        # dopsyn_z{i}/adopsyn_z{i} points at. Default (context_gated_da=
        # False, every existing config): all zones point at the SAME
        # self.vol_trans_actor -- byte-identical to the original
        # n_baseline_zones-only behavior (only `b` differs per zone, `n`
        # doesn't). context_gated_da=True replaces this with N distinct
        # populations+VTs below, so `n` differs per zone too.
        self.vol_trans_actor_zones = [self.vol_trans_actor] * self.n_baseline_zones
        self.da_zone_populations = None
        self.da_zone_gate_weight = da_zone_gate_weight
        if self.context_gated_da:
            zone_da_params = dict(da_params)
            zone_da_params['V_th'] = da_zone_v_th
            self.da_zone_populations = []
            self.vol_trans_actor_zones = []
            for _z in range(self.n_baseline_zones):
                da_pop_z = nest.Create('iaf_psc_alpha', n_da_neurons, zone_da_params)
                # Tonic drive at da_zone_tonic_weight (default matches
                # self.DA_neurons' hardcoded 5.0, but exposed here --
                # CALIBRATION FINDING: raising V_th alone barely moved the
                # matching/non-matching discrimination ratio (~1.1x even at
                # V_th=-38, 19mV above default) because the 1600Hz tonic
                # Poisson drive dominates the neuron's operating point
                # almost independent of a few-mV threshold shift -- shot-
                # noise-driven high-rate synaptic bombardment, not a clean
                # DC offset a hard threshold can gate against. Reducing
                # THIS weight is what actually lowers the ungated baseline
                # enough for the gate's extra kick to matter.
                nest.Connect(tonic, da_pop_z, "one_to_one", {"weight": da_zone_tonic_weight})
                if reward_da_weight != 0.0:
                    nest.Connect(self.pos_reward_parrot, da_pop_z, 'all_to_all',
                                 {'weight':  reward_da_weight})
                    nest.Connect(self.neg_reward_parrot, da_pop_z, 'all_to_all',
                                 {'weight': -reward_da_weight})
                vt_z = nest.Create('volume_transmitter', 1)
                nest.Connect(da_pop_z, vt_z, syn_spec={"weight": 1.0})
                self.da_zone_populations.append(da_pop_z)
                self.vol_trans_actor_zones.append(vt_z)

                # Per-zone recorder: the 'da' recorder above only ever
                # covers self.DA_neurons (the global, critic-facing
                # pool) -- da_zone_populations, the actor-facing pools
                # that actually drive dopsyn_z{i}/adopsyn_z{i} learning
                # under context_gated_da, had no recorder at all. Keyed
                # 'da_z{i}' in the same self.recorders dict so
                # get_recorder()/clear_rates pick them up with no
                # separate bookkeeping.
                self.recorders[f'da_z{_z}'] = recorder_tracking.create()
                nest.Connect(da_pop_z, self.recorders[f'da_z{_z}'])
            # The cortex-zone -> da_zone_populations[_z] gating connection
            # itself is NOT built here -- Dopamine is instantiated before
            # Actor (module docstring), so cortex's zone partition
            # (actor._cortex_zones) doesn't exist yet. network.py wires it
            # once both are built, using self.da_zone_populations and
            # self.da_zone_gate_weight exposed above.

        # Wmin/Wmax recomputed against THIS project's inits, not pasted from
        # the acrobot source. critic_dopsyn was the one bug actually present
        # here: Wmin=5 with critic.py's init landing at exactly w_mean=5.0
        # (bottom_clamped at episode 0, could only potentiate). Wmin=1/Wmax=12
        # puts init ~35-50% of range. dopsyn/adopsyn were already clear of
        # their init range in this copy (D1 init 22±2 vs [10,40]; D2 init
        # 16±2 vs [8,30]) — left unchanged, each still gets the init-bounds
        # assert at its connection site regardless.
        nest.CopyModel('stdp_dopamine_synapse', "critic_dopsyn", {
            'volume_transmitter': self.vol_trans_critic,
            # Rebalancing this A_plus/A_minus ratio the same way
            # dopsyn's was made value-tracking worse, not better --
            # critic_dopsyn's problem isn't the same STDP-ratio issue
            # that fixed dopsyn. Back to the original values; see
            # critic.py's build_input_to_critic_l1 for the actual next
            # lever.
            'A_plus':  0.002 * factor,
            'A_minus': 0.0008 * factor,
            'tau_plus': tau_plus,
            'Wmin': 5., 'Wmax': 15.,
            'tau_n': tau_n, 'tau_c': tau_c, 'b': _b_init,
        })
        nest.CopyModel('stdp_dopamine_synapse', "dopsyn", {
            'volume_transmitter': self.vol_trans_actor,
            # The old 1:1 A_plus:A_minus ratio assumed a reward
            # distribution that isn't this task's -- env_reach.py's
            # reward is negative-skewed, so (n-b) is negative on
            # average, and D1's mostly-causal eligibility trace under
            # the old ratio produced net depression every episode,
            # collapsing D1 to near-zero firing. dopsyn_a_plus/
            # dopsyn_a_minus are exposed so this ratio and
            # update_da_baseline's b_meas correction (which shifts the
            # (n-b) reference point instead) can be tuned independently.
            'A_plus':  dopsyn_a_plus * factor,
            'A_minus': dopsyn_a_minus * factor,
            'tau_plus': tau_plus,
            'Wmin': dopsyn_wmin, 'Wmax': dopsyn_wmax,
            'tau_n': tau_n, 'tau_c': tau_c, 'b': _b_init,
        })
        # Rebalanced 1:1.2 (A_plus=-0.0026/A_minus=-0.003, "matching mountain
        # car") -> 1.75:1 (A_plus=-0.0035/A_minus=-0.002). The old ratio was
        # never recalibrated for THIS task's reward distribution the way
        # dopsyn's was (see dopsyn's own A_plus/A_minus comment above) --
        # even from a symmetric cortex->D1/D2 init weight, it let
        # cortex->D2 weight drift down monotonically while D1 climbed
        # the mirror amount -- an ongoing STDP-driven divergence, not
        # just an initial-condition bias, recreating the same D1/D2
        # imbalance responsible for q1/q2 saturating at their joint
        # limits. adopsyn's inverted-STDP sign convention (both A_plus
        # and A_minus negative) flips which ratio direction helps
        # relative to dopsyn; the current ratio held the D1/D2 gap
        # stable across a real run where the symmetric default did not.
        nest.CopyModel('stdp_dopamine_synapse', "adopsyn", {
            'volume_transmitter': self.vol_trans_actor,
            'A_plus':  adopsyn_a_plus * factor,
            'A_minus': adopsyn_a_minus * factor,
            'tau_plus': tau_plus,
            'Wmin': adopsyn_wmin, 'Wmax': adopsyn_wmax,
            'tau_n': tau_n, 'tau_c': tau_c_adopsyn, 'b': _b_init,
        })
        # Zone-qualified copies (n_baseline_zones>1 only -- see __init__'s
        # top comment). nest.CopyModel(existing_model, new_name, params)
        # duplicates ALL of "dopsyn"/"adopsyn"'s just-set defaults
        # (A_plus/A_minus/Wmin/Wmax/tau_n/tau_c/b etc.) verbatim -- each
        # zone starts identical to the unzoned model and only diverges as
        # its OWN 'b' (updated independently by update_da_baseline()) and,
        # when context_gated_da, its OWN 'n' (driven by its own zone's DA
        # population/VT instead of the shared one) evolve. The
        # `volume_transmitter` override is always passed explicitly (not
        # left to plain inheritance) via vol_trans_actor_zones -- which is
        # every zone pointing at the SAME self.vol_trans_actor when
        # context_gated_da is off (byte-identical to the original
        # behavior), or each zone's own distinct VT when it's on. The base
        # "dopsyn"/"adopsyn" models above are still created either way (so
        # actor.py's Wmin/Wmax bounds check and any other code keying off
        # those exact names keeps working) -- they just end up with zero
        # connections once actor.py routes zone traffic through the
        # zone-qualified names instead (that wiring is a separate step).
        if self.n_baseline_zones > 1:
            for _z in range(self.n_baseline_zones):
                nest.CopyModel("dopsyn",  f"dopsyn_z{_z}",
                                {"volume_transmitter": self.vol_trans_actor_zones[_z]})
                nest.CopyModel("adopsyn", f"adopsyn_z{_z}",
                                {"volume_transmitter": self.vol_trans_actor_zones[_z]})
        # Representative dopsyn/adopsyn model name for logging only (the
        # base "dopsyn"/"adopsyn" names still exist and share identical
        # A_plus/A_minus with every zone right after CopyModel above, so
        # either works as a log sample -- zone 0 is used once zoning is on
        # since it's the one actually carrying connections in that case).
        self._dopsyn_log_name = "dopsyn" if self.n_baseline_zones <= 1 else "dopsyn_z0"

        # n_reward_channels>0: each channel gets its own DA
        # population, volume transmitter, dopsyn_ch{i}/adopsyn_ch{i} synapse
        # models (CopyModel'd from the just-configured "dopsyn"/"adopsyn" --
        # inherits their A_plus/A_minus/Wmin/Wmax/tau_n/tau_c/b verbatim,
        # only volume_transmitter differs, exact same pattern as the zone
        # copies above), and its OWN reward Poisson-generator/parrot pool
        # (separate from pos_reward_pg/neg_reward_pg -- those keep driving
        # only the shared DA_neurons/critic pathway). Motivated by: today
        # every D1/D2 channel shares one DA population/volume_transmitter,
        # so a reward pulse reflecting only joint 0's motion potentiates/
        # depresses joint 1's synapses identically. Deliberately independent
        # of n_baseline_zones (spatial cortex context) -- not combined with
        # zones in this first build; if n_reward_channels>0, callers should
        # leave n_baseline_zones at 1. Default 1: this whole block is
        # skipped, every existing config's behavior is unchanged.
        self.channel_da_populations = None
        self.vol_trans_actor_channels = None
        self.channel_pos_reward_pg = None
        self.channel_neg_reward_pg = None
        if self.n_reward_channels > 0:
            self.channel_da_populations = []
            self.vol_trans_actor_channels = []
            self.channel_pos_reward_pg = []
            self.channel_neg_reward_pg = []
            for _ch in range(self.n_reward_channels):
                da_pop_ch = nest.Create('iaf_psc_alpha', n_da_neurons, da_params)
                tonic_ch = nest.Create("poisson_generator", n_da_neurons, {"rate": tonic_rate_hz})
                nest.Connect(tonic_ch, da_pop_ch, "one_to_one", {"weight": da_tonic_weight})

                vt_ch = nest.Create('volume_transmitter', 1)
                nest.Connect(da_pop_ch, vt_ch, syn_spec={"weight": 1.0})

                pos_pg_ch = nest.Create('poisson_generator', 20, {'rate': 0.0})
                neg_pg_ch = nest.Create('poisson_generator', 20, {'rate': 0.0})
                pos_parrot_ch = nest.Create('parrot_neuron', 20)
                neg_parrot_ch = nest.Create('parrot_neuron', 20)
                nest.Connect(pos_pg_ch, pos_parrot_ch, "one_to_one")
                nest.Connect(neg_pg_ch, neg_parrot_ch, "one_to_one")
                if reward_da_weight != 0.0:
                    nest.Connect(pos_parrot_ch, da_pop_ch, 'all_to_all', {'weight':  reward_da_weight})
                    nest.Connect(neg_parrot_ch, da_pop_ch, 'all_to_all', {'weight': -reward_da_weight})

                nest.CopyModel("dopsyn",  f"dopsyn_ch{_ch}",  {"volume_transmitter": vt_ch})
                nest.CopyModel("adopsyn", f"adopsyn_ch{_ch}", {"volume_transmitter": vt_ch})

                self.channel_da_populations.append(da_pop_ch)
                self.vol_trans_actor_channels.append(vt_ch)
                self.channel_pos_reward_pg.append(pos_pg_ch)
                self.channel_neg_reward_pg.append(neg_pg_ch)
            print(f"[DA init] {self.n_reward_channels} per-channel DA populations/dopsyn_ch{{i}}/"
                  f"adopsyn_ch{{i}} models registered.")

        # VT weight does NOT gate n_: VT.handle() uses e.get_multiplicity() not e.get_weight().
        # Learning is disabled by zeroing A_plus/A_minus (CommonProperties) instead.
        # Save initial learning rates, then zero them during warmup.
        self._a_init = {}
        for _m in self.baseline_model_names():
            _d = nest.GetDefaults(_m)
            self._a_init[_m] = {'A_plus': _d['A_plus'], 'A_minus': _d['A_minus']}
            nest.SetDefaults(_m, {'A_plus': 0.0, 'A_minus': 0.0})
        print(f"[DA init] STDP disabled (warmup). "
              f"A_plus_dopsyn={self._a_init[self._dopsyn_log_name]['A_plus']:.6f} stored.")

        nest.Connect(self.DA_neurons, self.vol_trans_critic, syn_spec={"weight": 1.0})
        # connect_da_to_actor_vt: default True leaves existing configs
        # unchanged. False skips wiring the spiking DA_neurons
        # population into the actor's volume transmitter, so an
        # external diagnostic can drive it directly from its own
        # poisson_generator without DA_neurons' tonic background
        # spiking leaking into the clean injected signal.
        # vol_trans_critic is unaffected either way.
        if connect_da_to_actor_vt:
            nest.Connect(self.DA_neurons, self.vol_trans_actor, syn_spec={"weight": 1.0})

        if reward_da_weight != 0.0:
            nest.Connect(self.pos_reward_parrot, self.DA_neurons, 'all_to_all',
                         {'weight':  reward_da_weight})
            nest.Connect(self.neg_reward_parrot, self.DA_neurons, 'all_to_all',
                         {'weight': -reward_da_weight})
        else:
            print("*************************************************************")
            print("* Reward is NOT connected to dopamine (reward_da_weight=0)  *")
            print("*************************************************************")

    # ── Volume transmitter control ────────────────────────────────────────────

    def reset_eligibility_traces(self):
        """Zero `c` (eligibility) and `n` (dopamine trace) on every STDP-
        dopamine synapse model this class tracks (baseline_model_names()).
        Factored out of control_volume_transmitter()'s learning-disabled->
        enabled transition so callers that keep learning permanently off
        for an entire run can still call this periodically -- helps keep
        per-generation wall-clock time from growing across a run."""
        for _model in self.baseline_model_names():
            _conns = nest.GetConnections(synapse_model=_model)
            if len(_conns) > 0:
                nest.SetStatus(_conns, {'c': 0.0, 'n': 0.0})

    def control_volume_transmitter(self, learning_enabled: bool, warmup_until_ms: float):
        # `current_time` comes from NEST's kernel clock, which
        # nest.ResetKernel() (checkpoint_every's periodic rebuild)
        # resets to 0 -- re-deriving should_be_active from only the time
        # check would mean every checkpoint's post-reset segment
        # re-enters "warmup" from scratch, and if checkpoint_every is
        # short enough, warmup could mathematically never complete.
        # Once self.learning_active has been set True -- whether
        # freshly triggered here, or restored from a checkpoint's
        # extra_state -- a later time regression must not re-arm
        # warmup; only an explicit learning_enabled=False can turn it
        # back off.
        current_time = nest.GetKernelStatus("biological_time")
        should_be_active = learning_enabled and (self.learning_active or current_time >= warmup_until_ms)

        if should_be_active and not self.learning_active:
            # Reset accumulated eligibility traces before enabling
            self.reset_eligibility_traces()

            # Restore A_plus/A_minus — these are CommonProperties, SetDefaults affects all synapses
            for _model, _vals in self._a_init.items():
                nest.SetDefaults(_model, {'A_plus': _vals['A_plus'], 'A_minus': _vals['A_minus']})

            self.learning_active = True
            print(f"*****************************************************************")
            print(f"[DA] Learning ENABLED at t={current_time:.1f} ms  "
                  f"A_dopsyn={self._a_init[self._dopsyn_log_name]['A_plus']:.6f}")
            print(f"*****************************************************************")

        elif not should_be_active and self.learning_active:
            # Save current (possibly decayed) rates, then zero them
            for _model in self.baseline_model_names():
                _d = nest.GetDefaults(_model)
                self._a_init[_model] = {'A_plus': _d['A_plus'], 'A_minus': _d['A_minus']}
                nest.SetDefaults(_model, {'A_plus': 0.0, 'A_minus': 0.0})
            self.learning_active = False
            print(f"[DA] Learning DISABLED at t={current_time:.1f} ms")

    # ── Reward injection ─────────────────────────────────────────────────────

    def start_reward(self, reshaped_reward: float, pos_scaling: float, neg_scaling: float,
                     time=None) -> float:
        """
        Inject shaped reward into DA neurons via Poisson generators.

        Args:
            reshaped_reward: already-shaped scalar reward (computed externally)
            pos_scaling:     multiplier applied when reward is positive
            neg_scaling:     multiplier applied when reward is negative
            time:            current simulation time (for logging only)
        """
        if reshaped_reward <= 0.0:
            scaled_reward = abs(reshaped_reward) * neg_scaling
            self.neg_reward_pg.rate = np.clip(scaled_reward, 0.0, 400.0)
            self.pos_reward_pg.rate = 0.0
            # print("reward: ", scaled_reward, ", rate:", self.neg_reward_pg.rate)
        else:
            scaled_reward = abs(reshaped_reward) * pos_scaling
            self.pos_reward_pg.rate = np.clip(scaled_reward, 0.0, 400.0)
            self.neg_reward_pg.rate = 0.0
            # print("reward: ", scaled_reward, ", rate:", self.pos_reward_pg.rate)

        if self.verbose:
            print(f"Reward reshaped={reshaped_reward:.4f} scaled={scaled_reward:.4f} t={time}")
        return reshaped_reward

    def start_reward_per_channel(self, rewards, pos_scaling: float, neg_scaling: float,
                                 time=None):
        """Per-channel counterpart to start_reward() -- injects a DISTINCT
        reward value per channel into that channel's own reward Poisson-
        generator pool (channel_pos_reward_pg[i]/channel_neg_reward_pg[i]),
        which only drives that channel's own DA population/volume
        transmitter/dopsyn_ch{i}/adopsyn_ch{i} synapses. No-op if
        n_reward_channels<=0 (channel_pos_reward_pg is None then). Call
        ALONGSIDE start_reward() (still needed for the critic's pooled
        pathway), not instead of it -- they drive disjoint generator pools."""
        if self.channel_pos_reward_pg is None:
            return
        for ch, r in enumerate(rewards):
            if r <= 0.0:
                scaled = abs(r) * neg_scaling
                self.channel_neg_reward_pg[ch].rate = np.clip(scaled, 0.0, 400.0)
                self.channel_pos_reward_pg[ch].rate = 0.0
            else:
                scaled = abs(r) * pos_scaling
                self.channel_pos_reward_pg[ch].rate = np.clip(scaled, 0.0, 400.0)
                self.channel_neg_reward_pg[ch].rate = 0.0
            if self.verbose:
                print(f"Channel {ch} reward={r:.4f} scaled={scaled:.4f} t={time}")

    def stop_reward(self):
        self.pos_reward_pg.rate = 0.0
        self.neg_reward_pg.rate = 0.0
        if self.channel_pos_reward_pg is not None:
            for ch in range(self.n_reward_channels):
                self.channel_pos_reward_pg[ch].rate = 0.0
                self.channel_neg_reward_pg[ch].rate = 0.0

    # ── DA activity measurement ───────────────────────────────────────────────

    def measure_da_activity(self, t1, t2):
        bin_width_ms = t2 - t1
        if bin_width_ms <= 0:
            return
        times = nest.GetStatus(self.recorders['da'], keys='events')[0]['times']
        n_spikes = np.sum((times >= t1) & (times < t2))
        N_DA = len(self.da())
        bin_width_sec = bin_width_ms / 1000.0
        da_rate_hz = n_spikes / (N_DA * bin_width_sec)
        tau_b = 60.0
        alpha = 1.0 - np.exp(-bin_width_sec / tau_b)
        self.da_baseline = (1 - alpha) * self.da_baseline + alpha * da_rate_hz
        self.da_phasic   = da_rate_hz - self.da_baseline
        self.episode_da_phasic.append(self.da_phasic)
        self.da_rate_hz  = da_rate_hz
        self.last_time_dop_base_estimated = t2
        da_effective = self.da_convert_hz_to_nest * (self.da_rate_hz - self.da_baseline)
        self.episode_da_effective.append(da_effective)

    def measure_da_window_only(self, t1, t2):
        times = nest.GetStatus(self.recorders['da'], keys='events')[0]['times']
        n_spikes = np.sum((times >= t1) & (times < t2))
        bin_width_sec = (t2 - t1) / 1000.0
        return n_spikes / (len(self.DA_neurons) * bin_width_sec)

    def diagnose_da_response(self, t_before_reward, t_after_reward):
        rate_before = self.measure_da_window_only(t_before_reward - 30, t_before_reward)
        rate_after  = self.measure_da_window_only(t_after_reward  - 30, t_after_reward)
        return rate_before, rate_after

    def update_da_baseline(self, stdp_synapse_model):
        if self.da_baseline_fixed:
            # `b` stays pinned at its CopyModel-time value (_b_init) --
            # skip both the sample bookkeeping's consumption and the
            # nest.SetDefaults call entirely. sample_nb() still appends to
            # _n_samples_by_model (needed for the nb diagnostic/log), just
            # nothing reads it back into `b` here.
            return
        samples = self._n_samples_by_model.get(stdp_synapse_model)
        if not samples:
            return
        # alpha=0.02 gives an effective time constant of ~10 episodes,
        # which risks lagging behind a shifting target for a meaningful
        # fraction of a run since the true baseline level tracks the
        # evolving policy, not a fixed constant. Too fast a baseline
        # risks b chasing n itself and collapsing (n-b) toward zero
        # regardless of outcome.
        alpha = self.da_baseline_ema_alpha
        # Weighting population-averaged n by population-averaged |c|
        # per step didn't move the baseline at all -- population-
        # averaging n and c separately throws away exactly the
        # per-connection covariation that matters. sample_nb() now does
        # the eligibility weighting at the connection level every step,
        # so each stored sample is already a correctly-weighted scalar;
        # this is now a plain temporal average of those.
        window = samples[-50:]
        # condition_b_on_c weights the window average of the
        # already-per-connection-weighted n_val samples by each
        # sampled step's own total |c|, so steps where this model's
        # sampled connections carried near-zero eligibility count for
        # less than steps where its synapses were actually eligible.
        if self.condition_b_on_c:
            c_window = self._c_samples_by_model.get(stdp_synapse_model, [])[-50:]
            c_window = np.array(c_window[-len(window):])
            n_window = np.array(window)
            c_sum_total = float(c_window.sum())
            if c_sum_total > 1e-9:
                b_meas = float(np.sum(n_window * c_window) / c_sum_total)
            else:
                b_meas = float(np.mean(window))   # no eligibility anywhere in the window -- fall back
        else:
            b_meas = float(np.mean(window))
        b_meas = b_meas * self.b_meas_correction_factor
        prev_b = self._da_b_smoothed.get(stdp_synapse_model, b_meas)
        b = (1.0 - alpha) * prev_b + alpha * b_meas
        self._da_b_smoothed[stdp_synapse_model] = b
        nest.SetDefaults(stdp_synapse_model, {"b": float(b)})

    # ── Stats & accessors ────────────────────────────────────────────────────

    def baseline_model_names(self):
        """All STDP dopamine synapse MODEL names this class tracks/updates a
        `b` baseline for. n_baseline_zones==1 (default, every existing
        config): unchanged ('critic_dopsyn', 'dopsyn', 'adopsyn'). >1: swaps
        'dopsyn'/'adopsyn' for one dopsyn_z{i}/adopsyn_z{i} pair per zone
        (each a distinct NEST synapse model, its own CommonSynapseProperty
        `b` -- see __init__'s top comment); critic_dopsyn stays global
        either way, only cortex->D1/D2 is zoned. n_reward_channels>0
        additionally appends one dopsyn_ch{i}/adopsyn_ch{i} pair per
        channel, on top of whichever of the above applies -- the base
        'dopsyn'/'adopsyn' (or zone) names stay listed too even though
        they'll carry zero connections once actor.py routes through the
        channel-qualified names instead, same forgiving convention as the
        zone case."""
        if self.n_baseline_zones <= 1:
            names = ['critic_dopsyn', 'dopsyn', 'adopsyn']
        else:
            names = ['critic_dopsyn']
            for i in range(self.n_baseline_zones):
                names.append(f'dopsyn_z{i}')
                names.append(f'adopsyn_z{i}')
        if self.n_reward_channels > 0:
            for i in range(self.n_reward_channels):
                names.append(f'dopsyn_ch{i}')
                names.append(f'adopsyn_ch{i}')
        return tuple(names)

    def sample_nb(self):
        # Each model's own 'n' trace feeds its OWN baseline window (the fix
        # -- see __init__'s comment). dopsyn's samples additionally feed the
        # whole-episode nb_mean/nb_pos_frac diagnostic, unchanged.
        #
        # MEASURED, two failed attempts before this one -- kept here because
        # the reasoning matters for anyone tempted to retry a variant:
        #   1. Weight-by-population-mean-|c|, sampling only 50-of-2000
        #      connections: no change (conditioned high-elig (n-b) stayed
        #      -0.045/-0.642 good/bad, same as unweighted).
        #   2. Same weighting, sampling 100% of connections: also no change
        #      (-0.080/-0.608). Widening the sample wasn't the problem.
        # Root cause: both attempts computed mean(n) and mean(|c|)
        # SEPARATELY across connections each step, then weighted the
        # resulting scalars across TIME. That only captures whether the
        # population's AVERAGE eligibility is elevated at a given moment --
        # and population-mean eligibility apparently doesn't covary with
        # population-mean n across steps, so weighting by it changes
        # nothing. The real effect (from the original diagnostic probe) is
        # at PER-CONNECTION granularity: specific synapses' own n correlates
        # with their own c. Averaging n and c separately before weighting
        # throws that away. Fixed by computing the eligibility-weighted
        # mean ACROSS CONNECTIONS within each step (sum(n_i*|c_i|) /
        # sum(|c_i|)) -- this is the correctly-weighted scalar for that
        # step; update_da_baseline just averages these over time.
        # Subsampled back down to ~50 connections/model (was the full
        # population -- 2900 for critic_dopsyn, 600 each for dopsyn/adopsyn,
        # all read via GetStatus for BOTH n and c every single step of every
        # episode). The two failed attempts above already established that
        # sample width (50-of-2000 vs 100%) doesn't change the measured
        # (n-b) result -- what mattered was the per-connection eligibility
        # weighting, not population coverage -- so there's no accuracy cost
        # to going back to a small sample, only a real per-step cost saved.
        # Evenly-strided slice (not just the first 50) so the sample isn't
        # biased toward whichever neuron/channel GetConnections happens to
        # list first.
        for model in self.baseline_model_names():
            if model not in self._conns_by_model:
                _all = nest.GetConnections(synapse_model=model)
                stride = max(1, len(_all) // 50)
                self._conns_by_model[model] = _all[::stride]
            conns = self._conns_by_model[model]
            if len(conns) == 0:
                continue
            n_arr = np.array(nest.GetStatus(conns, 'n'))
            # c_arr/c_sum computed unconditionally (cheap, reuses the same
            # GetStatus call the eligibility-weighted branch already needs)
            # so condition_b_on_c's per-step eligibility-magnitude sample is
            # always available, regardless of eligibility_weighted_nb.
            c_arr = np.abs(np.array(nest.GetStatus(conns, 'c')))
            c_sum = float(c_arr.sum())
            if self.eligibility_weighted_nb:
                if c_sum > 1e-9:
                    n_val = float(np.sum(n_arr * c_arr) / c_sum)   # per-connection eligibility-weighted, this step
                else:
                    n_val = float(n_arr.mean())   # degenerate step (no eligibility anywhere) -- fall back
            else:
                # CONTROLLED TEST: plain unweighted mean(n), no |c| weighting
                # at all -- isolates whether b_meas*0.95's benefit is
                # compensating for a bias the eligibility-weighting itself
                # introduces (e.g. high-eligibility moments correlating with
                # elevated n for reasons unrelated to actual outcome
                # quality, such as the D1 efference-copy loop keeping both
                # cortex activity AND recent DA elevated together) rather
                # than compensating for env_reach's reward skew.
                n_val = float(n_arr.mean())
            self._n_samples_by_model[model].append(n_val)
            self._c_samples_by_model[model].append(c_sum)
            # Episode-level (n-b) diagnostic -- originally dopsyn-only;
            # when zoned, every dopsyn_z{i} feeds the same pooled diagnostic
            # (there's no single "the" dopsyn model anymore to pick one from).
            if model == self._dopsyn_log_name or (self.n_baseline_zones > 1 and model.startswith('dopsyn_z')):
                b_val = self._da_b_smoothed[model]
                self._nb_samples.append(n_val - b_val)

    def log_nb_episode(self):
        if self._nb_samples:
            arr = self._nb_samples
            pos = sum(1 for x in arr if x > 0)
            self._nb_mean = float(np.mean(arr))
            self._nb_pos_frac = pos / len(arr)
            print(f"[n-b] mean={self._nb_mean:+.4f}  "
                  f"positive={pos}/{len(arr)}")
        else:
            self._nb_mean = 0.0
            self._nb_pos_frac = 0.0

        # Per-zone baselines: _da_b_smoothed already holds one
        # independently-updated `b` per dopsyn_z{i}/adopsyn_z{i} model (see
        # baseline_model_names()) but nothing printed them -- the console
        # only ever showed the single rate-domain da_baseline (main.py's
        # "base=" field), a DIFFERENT quantity (smoothed da_rate_hz, used
        # for da_phasic/da_effective) unrelated to these NEST-unit STDP
        # baselines. Skipped entirely when unzoned (n_baseline_zones<=1,
        # every existing config) to avoid changing existing console output.
        if self.n_baseline_zones > 1:
            d1 = "  ".join(f"z{i}={self._da_b_smoothed[f'dopsyn_z{i}']:6.2f}"
                            for i in range(self.n_baseline_zones))
            d2 = "  ".join(f"z{i}={self._da_b_smoothed[f'adopsyn_z{i}']:6.2f}"
                            for i in range(self.n_baseline_zones))
            print(f"[b] D1(dopsyn)  {d1}  crit={self._da_b_smoothed['critic_dopsyn']:.2f}")
            print(f"[b] D2(adopsyn) {d2}")

    def reset_episode_samples(self):
        self._nb_samples = []
        for model in self._n_samples_by_model:
            self._n_samples_by_model[model] = []
            self._c_samples_by_model[model] = []

    def collect_stats(self, stats):
        self.append_stat(stats, "da_base",     self.da_baseline)
        self.append_stat(stats, "da_phasic",   np.mean(self.episode_da_phasic))
        self.append_stat(stats, "da_eff_mean", np.mean(self.episode_da_effective))
        self.append_stat(stats, "da_eff_min",  np.min(self.episode_da_effective))
        self.append_stat(stats, "da_eff_max",  np.max(self.episode_da_effective))
        self.episode_da_phasic = []
        self.episode_da_effective = []

    def da(self):
        return self.DA_neurons

    def da_channel(self, i):
        return self.channel_da_populations[i]

    def get_recorder(self, rec):
        return self.recorders[rec]

    def clear_rates(self):
        self.clear_spikes(self.recorders["da"])
        self.clear_spikes(self.recorders["pos_reward"])
        self.clear_spikes(self.recorders["neg_reward"])
        if self.da_zone_populations is not None:
            for _z in range(self.n_baseline_zones):
                self.clear_spikes(self.recorders[f"da_z{_z}"])
