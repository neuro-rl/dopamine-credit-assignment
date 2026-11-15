import nest
import numpy as np
import file_utils as futils
import recorder_tracking
from base_util import Base


class Critic(Base):
    """
    Single-layer spiking critic: L1 (grid-shaped) is the value readout
    itself (value() returns it directly) -- cortex->L1 is a plain
    static_synapse, trained by an external Python-side learner
    (model/spiking_critic_learning.py) with a centered, delta-gated
    update, the same structure as python_critic.py's linear TD(0) but
    over real spiking cortex activity. The input synapse model is exposed
    as self.input_synapse_model so that learner can find its connections
    via nest.GetConnections(synapse_model=...).

    Requires Dopamine to be instantiated first so that 'critic_dopsyn' is
    registered (used only if signed_pools/native-STDP paths need it).

    Two things deliberately NOT copied from the acrobot source (measured
    10 Hz recurrent-inhibition oscillator that contaminated the dopamine
    signal):
      - L2->L2 all-to-all inhibition
      - L1 Mexican-hat
    If the critic later saturates without any inhibition, add it back
    DESYNCHRONISED (p=0.3, delay=nest.random.uniform(1.0, 5.0)) — a uniform
    delay is what let the population lock into a rhythm.
    """

    def __init__(self, cortex, name_prefix: str = "",
                 l1_shape=(10, 10),
                 tau_syn_ex_l1: float = None,
                 critic_da_factor: float = 100.0,
                 d: float = 30.0, tau_r: float = 1000.0,
                 l1_ie: float = 120.0,
                 target_total_l1: float = 300.0,
                 p_input_l1: float = 0.3,
                 build_plastic_input: bool = True,
                 signed_pools: bool = False):
        super().__init__()
        self.critic_da_factor = critic_da_factor
        # signed_pools: a single population's firing RATE can only
        # represent V(s)>=0, a real ceiling since this task's reward is
        # negative-skewed. Splits L1 into two half-size populations
        # (layer1_pos, layer1_neg), mirroring the actor's D1/D2
        # opponent-pool convention -- V(s) = rate(pos) - rate(neg), able
        # to go negative. Default False: single population.
        self.signed_pools = signed_pools

        self.d      = d
        self.tau_r  = tau_r

        # ── Layer 1 ───────────────────────────────────────────────────────────
        # l1_ie default (140.0) TESTED AND KEPT, mixed result. MEASURED
        # (direct sim, single neuron, zero synaptic input, C_m=250pF/
        # tau_m=20ms/E_L=-65/V_th=-54): the old inline comment's "V_ss=-54.6,
        # gap 0.6mV" doesn't match I_e=140 (that's actually V_ss=-53.8, ABOVE
        # threshold -- 11.6Hz of PURE AUTONOMOUS firing with zero synaptic
        # input, no cortex needed at all). I_e=130 actually gives gap=0.6mV/
        # V_ss=-54.6 (0.0Hz autonomous, confirmed in the same sim), and a
        # reach_state_topo2 A/B (critic_l1_ie=130 vs default 140, 100
        # episodes each) confirmed the DIAGNOSIS: population CV across L1
        # (std/mean of per-neuron rate) went 0.03-0.06 at I_e=140 (nearly
        # every neuron firing the same regardless of which cortex neurons it
        # samples, despite cortex's OWN cv=~1.0) -> 0.37 at I_e=130 (healthy,
        # genuinely input-driven, L1 mean rate stayed a healthy ~20Hz). But
        # that mechanism fix did NOT improve the actual outcome metric --
        # corr(value_rate, d_start) went -0.15 -> -0.016, if anything
        # slightly worse. Pushed further (I_e=110, gap=2.2mV) L1 collapsed
        # toward silence (mean 1.6Hz) instead of becoming a better
        # differentiator -- ie130's healthy-but-uncorrelated result isn't a
        # "didn't push hard enough" issue. So L1's own excitability wasn't
        # gating value-tracking after all; left at the original 140 since
        # neither direction helped and 140 is the empirically-run-tested
        # value. critic_l1_ie is still exposed below for future probing.
        l1_params = {
            'I_e':    l1_ie,
            'tau_m':  10.0,
            'V_th':  -55.0,
            'V_reset': -65.0,
            'E_L':   -65.0,
            't_ref':   4.0,
        }
        if tau_syn_ex_l1 is not None:
            l1_params['tau_syn_ex'] = tau_syn_ex_l1
            l1_params['tau_syn_in'] = tau_syn_ex_l1
        n_l1_total = l1_shape[0] * l1_shape[1]
        if self.signed_pools:
            # Flat populations (no spatial grid) -- L1's grid position was
            # never used for any actual connectivity rule even in the
            # non-signed case (cortex->L1 and L1->L2 are both plain
            # fixed_indegree, confirmed by the "topographic projection
            # TRIED AND REVERTED TWICE" comment below), so dropping it
            # here loses nothing functional. Split the ORIGINAL total L1
            # neuron count in half, one pool each -- per-neuron excitability/
            # drive calibration (l1_params, w_mean/w_jit below) is UNCHANGED
            # per neuron, only the per-pool population size halves (a real,
            # flagged tradeoff: smaller N per pool means somewhat noisier
            # per-pool rate estimates, same 1/sqrt(N) scaling theme as
            # everywhere else in this project).
            n_pos = n_l1_total // 2
            n_neg = n_l1_total - n_pos
            self.layer1_pos = nest.Create('iaf_psc_alpha', n_pos, params=l1_params)
            self.layer1_neg = nest.Create('iaf_psc_alpha', n_neg, params=l1_params)
            self.layer1 = None
        else:
            l1_ext = float(l1_shape[0])  # spacing = extent/n = 1.0 always
            self.layer1 = nest.Create(
                'iaf_psc_alpha',
                params=l1_params,
                positions=nest.spatial.grid(
                    shape=list(l1_shape),
                    extent=[l1_ext, l1_ext],
                    center=[l1_ext / 2, l1_ext / 2],
                    edge_wrap=True,
                ),
            )

        # w_mean/w_jit recomputed against THIS cortex size, not pasted --
        # the acrobot source's w_mean landed EXACTLY on Wmin=5 (half the
        # synapses clamped from episode 0). critic_dopsyn's
        # Wmin/Wmax were fixed in dopamine.py; this assert makes it
        # impossible for this connection's own init to repeat that bug
        # regardless of cortex size changes later.
        # p_input_l1: was a hardcoded 0.3,
        # calibrated for cortex_exc_shape=(12,12)=144 (giving K=43,
        # w_mean=300/43=6.98, safely inside critic_dopsyn's [5,15]
        # bounds). At a much smaller cortex (e.g. 49 for reach_state_
        # topo2_small.py) the SAME 0.3 drops K to 15, which would need
        # w_mean=300/15=20 to hit the SAME target_total_l1 -- past Wmax
        # entirely. Raising p_input_l1 (more of the smaller cortex
        # sampled per L1 neuron) restores K enough to hit the ORIGINAL
        # total-drive target without exceeding Wmax, instead of lowering
        # the target itself (which was tried first and left L1/L2
        # badly underdriven relative to the original config: MEASURED
        # 6.22Hz/0.24Hz vs the original's 21.64Hz/10.84Hz). Default 0.3
        # leaves every existing config's behavior unchanged.
        p_input_L1    = p_input_l1
        K_input_l1    = int(round(p_input_L1 * len(cortex)))
        target_total_L1 = target_total_l1
        w_mean = target_total_L1 / K_input_l1
        w_jit  = 0.05 * w_mean
        _bounds = nest.GetDefaults('critic_dopsyn')
        Base.assert_synapse_init_in_bounds(
            w_mean, w_jit, _bounds['Wmin'], _bounds['Wmax'], label="critic input->L1")
        print(f"[Critic] Input->L1: K={K_input_l1}, w_mean={w_mean:.2f}, w_jit={w_jit:.2f}")

        # TRIED AND REVERTED TWICE: a topographic spatial-mask projection
        # here, mirroring the joints->cortex fix (L1 already has its own
        # spatial grid layout). First attempt (pre-topo_both, when cortex's
        # output was still a MIX of topographically-organized joints signal
        # and randomly-projected object signal): corr(value_rate, d_start)
        # went -0.165 -> -0.039, worse. Re-tested under topo_both (BOTH
        # streams spatially organized, removing that exact precondition,
        # plus L1 given a matching extent so the mask lands in the same
        # coordinate system as cortex -- topographic_input param, since
        # reverted): corr went -0.15 -> +0.06, worse AND wrong-signed, not
        # just weaker. So the original "mixed signal" explanation for the
        # first failure wasn't the real reason -- topographic pooling
        # specifically hurts the CRITIC (L2 needs to integrate a scalar
        # value across the whole state space; local spatial pooling likely
        # just samples a narrower, less representative slice of cortex than
        # random sampling does), unlike the ACTOR (D1/D2, which each
        # represent one specific joint and genuinely benefit from a
        # topographically consistent input). Back to random fixed_indegree.
        # cortex->L1 is a plain static_synapse (named, so
        # model/spiking_critic_learning.py can find its connections via
        # nest.GetConnections(synapse_model=self.input_synapse_model)) --
        # NEST applies zero automatic plasticity here, an external Python
        # learner handles cortex->L1 entirely.
        if not self.signed_pools:
            self.input_synapse_model = f"{name_prefix}critic_input_static"
            nest.CopyModel('static_synapse', self.input_synapse_model)
        else:
            self.input_synapse_model_pos = f"{name_prefix}critic_input_static_pos"
            self.input_synapse_model_neg = f"{name_prefix}critic_input_static_neg"
            nest.CopyModel('static_synapse', self.input_synapse_model_pos)
            nest.CopyModel('static_synapse', self.input_synapse_model_neg)
            self.input_synapse_model = None   # no single model in this mode -- see the two above

        if self.signed_pools:
            def build_input_to_critic_l1_pos():
                nest.Connect(
                    cortex, self.layer1_pos,
                    conn_spec={'rule': 'fixed_indegree', 'indegree': K_input_l1},
                    syn_spec={'synapse_model': self.input_synapse_model_pos,
                              'weight': Base.w_uni(w_mean, w_jit),
                              'delay': nest.resolution},
                )

            def build_input_to_critic_l1_neg():
                nest.Connect(
                    cortex, self.layer1_neg,
                    conn_spec={'rule': 'fixed_indegree', 'indegree': K_input_l1},
                    syn_spec={'synapse_model': self.input_synapse_model_neg,
                              'weight': Base.w_uni(w_mean, w_jit),
                              'delay': nest.resolution},
                )

            if build_plastic_input:
                futils.ensure_connectivity(cortex, self.layer1_pos,
                                           f"{name_prefix}input_critic_l1_pos", build_input_to_critic_l1_pos)
                futils.ensure_connectivity(cortex, self.layer1_neg,
                                           f"{name_prefix}input_critic_l1_neg", build_input_to_critic_l1_neg)
        else:
            def build_input_to_critic_l1():
                nest.Connect(
                    cortex, self.layer1,
                    conn_spec={'rule': 'fixed_indegree', 'indegree': K_input_l1},
                    syn_spec={'synapse_model': self.input_synapse_model,
                              'weight': Base.w_uni(w_mean, w_jit),
                              'delay': nest.resolution},
                )

            # build_plastic_input=False (NestCheckpointer rebuilds, see
            # nest_checkpoint.py): skip BOTH branches of ensure_connectivity --
            # the disk cache holds this connection's INITIAL random weights,
            # not live-trained ones, so loading it here would silently discard
            # trained state instead of leaving the projection for
            # NestCheckpointer.restore_from_checkpoint() to recreate from the
            # actual saved (live) weights.
            if build_plastic_input:
                futils.ensure_connectivity(cortex, self.layer1,
                                           f"{name_prefix}input_critic_l1", build_input_to_critic_l1)

        # ── Recorders ────────────────────────────────────────────────────────
        if self.signed_pools:
            # Two SEPARATE rate recorders -- diagnose_value_signal() needs
            # each pool's rate individually to compute the signed
            # difference; a single combined recorder would throw away the
            # sign information entirely. "value" (combined, both pools)
            # kept too, for debug_util.py's existing viz hooks -- display
            # only, not used by diagnose_value_signal().
            self.recorders = {
                "features":   recorder_tracking.create(),
                "value":      recorder_tracking.create(),
                "value_pos":  recorder_tracking.create(),
                "value_neg":  recorder_tracking.create(),
            }
            nest.Connect(self.layer1_pos + self.layer1_neg, self.recorders["features"], 'all_to_all')
            nest.Connect(self.layer1_pos + self.layer1_neg, self.recorders["value"], 'all_to_all')
            nest.Connect(self.layer1_pos, self.recorders["value_pos"], 'all_to_all')
            nest.Connect(self.layer1_neg, self.recorders["value_neg"], 'all_to_all')
        else:
            self.recorders = {
                "features": recorder_tracking.create(),
                "value":    recorder_tracking.create(),
            }
            nest.Connect(self.layer1, self.recorders["features"], 'all_to_all')
            nest.Connect(self.value(), self.recorders["value"], 'all_to_all')

    # ── Accessors ─────────────────────────────────────────────────────────────

    def features(self):
        return (self.layer1_pos + self.layer1_neg) if self.signed_pools else self.layer1

    def value(self):
        """Note: in signed_pools mode this returns the COMBINED population
        (layer1_pos + layer1_neg) -- useful for code that just needs "the
        value population" as a NodeCollection (e.g. connect_dopamine's
        per-channel broadcast loop still wires through this), but its RAW
        rate is NOT the signed V(s) -- see diagnose_value_signal() for
        that. connect_dopamine() itself does NOT use value() when
        signed_pools -- it wires layer1_pos/layer1_neg separately with
        opposite sign (see below)."""
        if self.signed_pools:
            return self.layer1_pos + self.layer1_neg
        return self.layer1

    def get_recorder(self, name):
        return self.recorders[name]

    def gamma(self):
        return 1 - (self.d / self.tau_r)

    # ── Dopamine connection ───────────────────────────────────────────────────

    def connect_dopamine(self, dopamine):
        """
        Doya (2000) continuous-time TD error via dual synaptic delay:
            L2 → DA  weight=+w_fast  delay=resolution   (V(t))
            L2 → DA  weight=-w_slow  delay=d ms         (V(t-d))
        Net DA drive ≈ -factor*(1/tau_r)*V(t) + factor*dV/dt  (MEASURED: an
        earlier version of this docstring had the V(t) term's sign backwards
        -- +factor/tau_r instead of -factor/tau_r. Verified two ways: (1)
        expanding w_fast*V(t) + w_slow*V(t-d) algebraically gives
        factor*(1/d)*(V(t)-V(t-d)) - factor*V(t)/tau_r, i.e. MINUS; (2) a
        controlled test (stand-in L2 with a deterministic rate step, wired
        through this exact w_fast/w_slow formula, no reward/reset) measured
        steady-state DA rate strictly LOWER at high V than at low V
        (-5.49Hz), matching the code's actual -V/tau_r term, not the old
        docstring's claimed +V/tau_r. This IS the correct Doya (2000)
        continuous-TD formula (delta = -V/tau + dV/dt, the continuous-time
        analogue of delta = r + gamma*V' - V) -- the CODE was always right,
        only the comment was wrong.
        At steady state the two arms nearly cancel (tau_r=1000 is large, so
        the residual -V/tau_r term is small relative to phasic dV/dt
        bursts, though not exactly zero -- the same control test measured
        +16.4Hz on a step up and -5.6Hz on a step down, confirming the
        phasic term is correctly signed and dominant).
        Requires DA tau_syn_ex == tau_syn_in so both arms integrate identically.
        """
        factor = self.critic_da_factor
        if factor == 0.0:
            print("*************************************************************")
            print("* Critic is NOT connected to dopamine (critic_da_factor=0)  *")
            print("*************************************************************")
            return
        w_fast =  (1.0 / self.d - 1.0 / self.tau_r) * factor
        w_slow = -(1.0 / self.d) * factor

        def _wire(src, w_f, w_s):
            nest.Connect(src, dopamine.da(),
                         conn_spec={'rule': 'all_to_all'},
                         syn_spec={'weight': w_f, 'delay': nest.resolution})
            nest.Connect(src, dopamine.da(),
                         conn_spec={'rule': 'all_to_all'},
                         syn_spec={'weight': w_s, 'delay': float(self.d)})
            if dopamine.channel_da_populations is not None:
                for da_pop_ch in dopamine.channel_da_populations:
                    nest.Connect(src, da_pop_ch,
                                 conn_spec={'rule': 'all_to_all'},
                                 syn_spec={'weight': w_f, 'delay': nest.resolution})
                    nest.Connect(src, da_pop_ch,
                                 conn_spec={'rule': 'all_to_all'},
                                 syn_spec={'weight': w_s, 'delay': float(self.d)})

        if self.signed_pools:
            # layer1_pos: SAME sign as the single-population case (pushes DA
            # up when it fires more). layer1_neg: NEGATED (pushes DA down
            # when IT fires more) -- the real-time, network-computed TD
            # signal is now signed, mirroring the Python-side diagnostic's
            # rate(pos)-rate(neg) decode (see diagnose_value_signal below)
            # and the two-learner training convention (main.py).
            _wire(self.layer1_pos, w_fast, w_slow)
            _wire(self.layer1_neg, -w_fast, -w_slow)
        else:
            _wire(self.value(), w_fast, w_slow)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnose_value_signal(self, t0, t1):
        if self.signed_pools:
            v_pos, v_neg = self.diagnose_pool_rates(t0, t1)
            return v_pos - v_neg
        return Base.rate_in_window(self.recorders["value"], len(self.value()), t0, t1)

    def diagnose_pool_rates(self, t0, t1):
        """(rate_pos, rate_neg) separately -- only meaningful when
        signed_pools (raises otherwise, not silently returns a pair of
        Nones, since any caller of this is specifically investigating the
        opponent-pool mechanism)."""
        if not self.signed_pools:
            raise ValueError("diagnose_pool_rates() only meaningful when signed_pools=True")
        v_pos = Base.rate_in_window(self.recorders["value_pos"], len(self.layer1_pos), t0, t1)
        v_neg = Base.rate_in_window(self.recorders["value_neg"], len(self.layer1_neg), t0, t1)
        return v_pos, v_neg

    def diagnose_weight_changes(self, cortex):
        if self.signed_pools:
            conns_pos = nest.GetConnections(cortex, self.layer1_pos)
            conns_neg = nest.GetConnections(cortex, self.layer1_neg)
            weights = np.concatenate([
                np.array(nest.GetStatus(conns_pos, "weight")),
                np.array(nest.GetStatus(conns_neg, "weight")),
            ])
            return float(np.mean(weights)), float(np.std(weights)), float(np.min(weights)), float(np.max(weights))
        conns   = nest.GetConnections(cortex, self.layer1)
        weights = np.array(nest.GetStatus(conns, "weight"))
        return np.mean(weights), np.std(weights), np.min(weights), np.max(weights)

    def clear_rates(self):
        self.clear_spikes(self.recorders["features"])
        self.clear_spikes(self.recorders["value"])
        if self.signed_pools:
            self.clear_spikes(self.recorders["value_pos"])
            self.clear_spikes(self.recorders["value_neg"])
