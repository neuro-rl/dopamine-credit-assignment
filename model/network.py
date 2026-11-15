"""network.py — assembles Dopamine, Cortex, the SensoryInput encoders, Actor
and Critic from a Config. main.py stays task-agnostic; this is where the
pieces get wired together.

Cortex does NOT wire its own input (unlike the acrobot source) -- each
encoder owns its projection rule (encoders.py's connect_to()) and this module
wires each stream onto cortex.output() separately, so joints and object can
be tuned independently (neither stream should saturate the cortex) and each
config's encoders (place fields, or a retina+pool) connect however suits
their own code.
"""
import nest

import file_utils as futils
import recorder_tracking
from base_util import Base
from cortex import Cortex
from dopamine import Dopamine
from actor import ActorD1D2ChannelsWTA
from critic import Critic


class Network:

    def __init__(self, enc, cfg, name_prefix: str = "", load_plastic: bool = True):
        """load_plastic=False skips building the plastic (dopamine-modulated)
        cortex->D1/D2 and cortex->critic-L1 projections -- used by
        NestCheckpointer (see nest_checkpoint.py) when rebuilding after a
        periodic nest.ResetKernel(): those get reconnected from saved live
        weights instead of fresh random init / the on-disk topology cache
        (which only ever holds the INITIAL random weights)."""
        self.enc = enc

        # baseline_zone_grid=(nx, ny): actor.py partitions
        # cortex into a 2D grid over both x and y; Dopamine only
        # needs the total zone count to allocate that many independent DA
        # populations/dopsyn_z{i} models, geometry-agnostic.
        _zone_grid = getattr(cfg, "actor_baseline_zone_grid", (1, 1))
        _n_baseline_zones = _zone_grid[0] * _zone_grid[1]

        # Dopamine MUST be built first -- it registers critic_dopsyn/dopsyn/
        # adopsyn via nest.CopyModel, which Critic and Actor depend on.
        self.dopamine = Dopamine(
            n_da_neurons          = cfg.da_neurons,
            da_convert_hz_to_nest = cfg.da_convert_hz_to_nest,
            tau_c                 = cfg.da_tau_c,
            tau_c_adopsyn         = cfg.da_tau_c_adopsyn,
            tau_n                 = cfg.da_tau_n,
            tau_plus              = cfg.da_tau_plus,
            tau_syn_ex_da         = cfg.tau_syn_ex_da,
            da_baseline_init      = cfg.da_baseline_init,
            reward_da_weight      = cfg.reward_da_weight,
            tonic_rate_hz         = cfg.da_tonic_rate_hz,
            verbose               = cfg.verbose,
            dopsyn_a_plus         = getattr(cfg, "dopsyn_a_plus", 0.002),
            dopsyn_a_minus        = getattr(cfg, "dopsyn_a_minus", 0.0035),
            adopsyn_a_plus        = getattr(cfg, "adopsyn_a_plus", -0.0035),
            adopsyn_a_minus       = getattr(cfg, "adopsyn_a_minus", -0.002),
            dopsyn_wmin           = getattr(cfg, "dopsyn_wmin", 10.),
            dopsyn_wmax           = getattr(cfg, "dopsyn_wmax", 40.),
            adopsyn_wmin          = getattr(cfg, "adopsyn_wmin", 8.),
            adopsyn_wmax          = getattr(cfg, "adopsyn_wmax", 30.),
            eligibility_weighted_nb = getattr(cfg, "eligibility_weighted_nb", True),
            b_meas_correction_factor = getattr(cfg, "b_meas_correction_factor", 0.95),
            n_baseline_zones      = _n_baseline_zones,
            condition_b_on_c      = getattr(cfg, "condition_b_on_c", False),
            context_gated_da      = getattr(cfg, "actor_context_gated_da", False),
            da_zone_v_th          = getattr(cfg, "actor_da_zone_v_th", -52.0),
            da_zone_gate_weight   = getattr(cfg, "actor_da_zone_gate_weight", 20.0),
            da_zone_tonic_weight  = getattr(cfg, "actor_da_zone_tonic_weight", 5.0),
            da_tau_m              = getattr(cfg, "da_tau_m", 20.0),
            da_tonic_weight       = getattr(cfg, "da_tonic_weight", 5.0),
            n_reward_channels     = cfg.n_actions if getattr(cfg, "actor_per_channel_reward", False) else 0,
            da_baseline_ema_alpha = getattr(cfg, "da_baseline_ema_alpha", 0.041),
            da_baseline_fixed     = getattr(cfg, "da_baseline_fixed", False),
        )

        self.cortex = Cortex(
            exc_shape=cfg.cortex_exc_shape,
            inh_shape=cfg.cortex_inh_shape,
            extent=getattr(cfg, "cortex_extent", None),
            bg_n_sources=cfg.cortex_bg_n_sources,
            bg_rate_hz=cfg.cortex_bg_rate_hz,
            bg_p=cfg.cortex_bg_p,
            bg_w_exc=cfg.cortex_bg_w_exc,
            bg_w_inh=cfg.cortex_bg_w_inh,
            bg_shared=cfg.cortex_bg_shared,
            bg_one_to_one=cfg.cortex_bg_one_to_one,
            # r_ee/w_ee were previously hardcoded class defaults, invisible
            # to every config -- exposed here for the E->E recurrence-vs-
            # decodability control test (grid-local E->E recurrence
            # propagates activity by spatial topology, but the feedforward
            # projection into cortex is a random, non-spatial fixed_indegree
            # -- suspected to wash out state-specific structure regardless
            # of healthy firing rate). getattr default matches the old
            # hardcoded value exactly, so any config that doesn't set these
            # behaves identically to before.
            r_ee=getattr(cfg, "cortex_r_ee", 0.15),
            w_ee=getattr(cfg, "cortex_w_ee", 4),
            # MEASURED: with w_ei=20 (class default), inhibitory neurons'
            # V_m only reached -63 (V_th=-56, E_L=-65) despite 300 E->I
            # connections and robust excitatory firing (782 spikes/500ms)
            # -- each inh neuron's ~12 E->I synapses fire too sparsely
            # (~25ms apart on average) relative to tau_syn_ex=2.0ms for
            # any temporal summation to happen, so PSCs never accumulate
            # toward threshold regardless of the per-synapse weight.
            # Exposed the same way as r_ee/w_ee for testing a much larger
            # weight (getattr default matches the old hardcoded value).
            r_ei=getattr(cfg, "cortex_r_ei", 0.20),
            w_ei=getattr(cfg, "cortex_w_ei", 20.0),
            # r_ie/w_ie: the I->E feedback weight that actually suppresses
            # cortex_exc firing. Raising cortex_w_ei alone barely moves
            # cortex_active_fraction -- the inhibitory population's own
            # drive isn't the bottleneck, how hard each inh spike
            # suppresses exc is. Exposed the same getattr way as
            # r_ee/w_ee/r_ei/w_ei; default matches the old hardcoded value.
            r_ie=getattr(cfg, "cortex_r_ie", 0.15),
            w_ie=getattr(cfg, "cortex_w_ie", -11.0),
            name_prefix=name_prefix,
        )

        # Each stream wires itself onto cortex with its own projection rule
        # and its own weight/indegree, cached the same way as any other
        # connectivity (ensure_connectivity).
        #
        # actor_bypass_cortex_joints: under bypass, nothing reads cortex
        # any more -- D1/D2 read the encoder population directly (below).
        # Wiring joints/object/joints_vel onto cortex anyway would just
        # drive it to saturation for no functional purpose and slow
        # episodes down, so this skips that wiring entirely with no
        # behavior change.
        if not getattr(cfg, "actor_bypass_cortex_joints", False):
            def _build_joints():
                enc["joints"].connect_to(self.cortex.output(), **cfg.joints_connect_kw)
            futils.ensure_connectivity(
                enc["joints"].output(), self.cortex.output(),
                f"{name_prefix}cortex_input_joints", _build_joints)

            # actor_reach_vec_input: replaces the raw-target "object"
            # stream with the reach_vec = fk(joints)-target population
            # instead -- the same population already validated for the
            # critic (Grid2DPlaceFieldEncoder). Its own connect_to() takes
            # weight/delay/indegree only (no spatial mask), so this is a
            # dedicated kwarg dict, not object_connect_kw reused.
            if getattr(cfg, "actor_reach_vec_input", False):
                def _build_object():
                    enc["reach_vec"].connect_to(self.cortex.output(), **cfg.reach_vec_cortex_connect_kw)
                futils.ensure_connectivity(
                    enc["reach_vec"].output(), self.cortex.output(),
                    f"{name_prefix}cortex_input_reach_vec", _build_object)
            else:
                def _build_object():
                    enc["object"].connect_to(self.cortex.output(), **cfg.object_connect_kw)
                futils.ensure_connectivity(
                    enc["object"].output(), self.cortex.output(),
                    f"{name_prefix}cortex_input_object", _build_object)

            # velocity_input: same wiring pattern as joints/
            # object, only present when main.py's _build_encoders actually
            # built the "joints_vel" stream.
            if getattr(cfg, "velocity_input", False):
                def _build_joints_vel():
                    enc["joints_vel"].connect_to(self.cortex.output(), **cfg.joints_vel_connect_kw)
                futils.ensure_connectivity(
                    enc["joints_vel"].output(), self.cortex.output(),
                    f"{name_prefix}cortex_input_joints_vel", _build_joints_vel)

        # actor_burst_hidden: inserts an Izhikevich intrinsically-bursting
        # population H between cortex and D1/D2 -- H.output() is passed as
        # the Actor's "cortex" argument below instead of the real cortex,
        # so every existing cortex->D1/D2 wiring mechanism builds H->D1/D2
        # unmodified. Built here because H's NEST-side infrastructure is
        # rebuilt fresh every checkpoint cycle like cortex/actor/critic;
        # only burst_hidden.py's Python-side state (fb_weight, x_bar)
        # needs to survive a rebuild, and that lives in main.py's
        # separately-constructed net.burst_teacher, re-attached to the
        # fresh pop_H/conns_ch built here after every rebuild.
        self.burst_h_pop = None
        self.burst_h_rec_bd1 = None
        self.burst_h_rec_burst = None
        self.burst_h_ngs = None
        self.actor_bypass_pop = None
        self.actor_bypass_rec = None
        self.actor_fixed_hidden_pop = None
        self.actor_gainfield_hidden_pop = None
        self.actor_cortex_hidden_pop = None
        _actor_cortex_input = self.cortex.output()
        # _efference_target_override: None for every existing config
        # (efference feeds back into whatever _actor_cortex_input ends up
        # being, the pre-existing default). CANN mode is the one case
        # that must override this explicitly -- see below for why.
        _efference_target_override = None
        # actor_bypass_cortex_joints: the recurrent cortex was measured to
        # collapse the combined-encoder reach_vec signal, so this routes
        # around it entirely -- D1/D2 reads the joints+object encoder
        # output directly (enc["joints"] and enc["object"] are both plain
        # nest.Create parrot populations, so '+' concatenation is valid).
        # Cortex itself is still built and still receives joints/object
        # normally; this only redirects what the actor reads, the same
        # single intervention point actor_burst_hidden uses below.
        if getattr(cfg, "actor_bypass_cortex_joints", False):
            _bypass_encoder_pop = enc["joints"].output() + enc["object"].output()
            # actor_bypass_fixed_hidden: inserts a frozen random
            # feedforward hidden layer between the bypass encoder and
            # D1/D2. Decode R^2 on this system doesn't predict control
            # quality, so this is trained and measured directly rather
            # than gated on an offline decode number. Encoder->H weights
            # are frozen for the run's lifetime via futils.ensure_
            # connectivity so a checkpoint rebuild replays the same
            # random basis instead of re-randomizing it. Only H->D1/D2
            # (sourced from H) is trained by BatchedD1Learning.
            if getattr(cfg, "actor_gainfield_hidden", False) and getattr(cfg, "actor_bypass_fixed_hidden", False):
                # Stacks a frozen random fixed-hidden layer after the
                # gain-field relay, between gain-field and D1/D2 --
                # tests whether an extra fixed nonlinearity helps, hurts,
                # or does nothing on top of the gain-field representation.
                # Builds the gain-field relay exactly as the gainfield-only
                # branch below does, then feeds it into a fixed_indegree
                # frozen random projection to a second iaf_psc_alpha
                # population.
                _ff_weight = getattr(cfg, "actor_gainfield_ff_weight", 300.0)
                _gf_n = len(enc["gainfield"].output())
                self.actor_gainfield_hidden_pop = nest.Create("iaf_psc_alpha", _gf_n)
                nest.Connect(
                    enc["gainfield"].output(), self.actor_gainfield_hidden_pop,
                    "one_to_one",
                    syn_spec={"synapse_model": "static_synapse",
                              "weight": _ff_weight, "delay": 0.5})

                _h_n = getattr(cfg, "actor_fixed_hidden_size", 750)
                _h_indegree = getattr(cfg, "actor_fixed_hidden_indegree", 30)
                _h_weight = getattr(cfg, "actor_fixed_hidden_weight", 300.0)
                self.actor_fixed_hidden_pop = nest.Create("iaf_psc_alpha", _h_n)

                def _build_gainfield_fixed_hidden():
                    nest.Connect(
                        self.actor_gainfield_hidden_pop, self.actor_fixed_hidden_pop,
                        conn_spec={"rule": "fixed_indegree", "indegree": _h_indegree},
                        syn_spec={"synapse_model": "static_synapse",
                                  "weight": nest.random.uniform(-_h_weight, _h_weight),
                                  "delay": 0.5})
                futils.ensure_connectivity(
                    self.actor_gainfield_hidden_pop, self.actor_fixed_hidden_pop,
                    f"{name_prefix}gainfield_fixed_hidden", _build_gainfield_fixed_hidden)

                # Efference targets the final real iaf relay (fixed_hidden),
                # same safety rationale as every other bypass branch here --
                # no bare-parrot runaway risk.
                self.actor_bypass_pop = self.actor_fixed_hidden_pop
                _efference_target_override = self.actor_fixed_hidden_pop
            elif getattr(cfg, "actor_gainfield_hidden", False):
                # actor_gainfield_hidden: D1/D2 reads a real iaf_psc_alpha
                # relay, one_to_one from enc["gainfield"]'s parrot units
                # (plain/gain-x/gain-y -- see encoders.py's GainFieldEncoder).
                # Target info is already folded into the gain-x/gain-y
                # units themselves, so actor_bypass_pop is the relay
                # population directly. Deterministic one_to_one weight, no
                # RNG draw, so no ensure_connectivity caching needed.
                _ff_weight = getattr(cfg, "actor_gainfield_ff_weight", 300.0)
                _gf_n = len(enc["gainfield"].output())
                self.actor_gainfield_hidden_pop = nest.Create("iaf_psc_alpha", _gf_n)
                nest.Connect(
                    enc["gainfield"].output(), self.actor_gainfield_hidden_pop,
                    "one_to_one",
                    syn_spec={"synapse_model": "static_synapse",
                              "weight": _ff_weight, "delay": 0.5})
                self.actor_bypass_pop = self.actor_gainfield_hidden_pop
                # Efference targets the relay (real iaf units) -- same
                # safety rationale as CANN/fixedhidden, no bare-parrot
                # runaway risk since there's no raw-target concatenation
                # to worry about here anyway, but kept explicit/consistent.
                _efference_target_override = self.actor_gainfield_hidden_pop
            else:
                self.actor_bypass_pop = _bypass_encoder_pop
            self.actor_bypass_rec = recorder_tracking.create()
            nest.Connect(self.actor_bypass_pop, self.actor_bypass_rec)
            _actor_cortex_input = self.actor_bypass_pop
        elif getattr(cfg, "actor_burst_hidden", False):
            _h_size = getattr(cfg, "actor_burst_hidden_size", None) or len(self.cortex.output())
            self.burst_h_pop = nest.Create(
                "izhikevich", _h_size,
                params={"a": cfg.burst_izh_a, "b": cfg.burst_izh_b,
                        "c": cfg.burst_izh_c, "d": cfg.burst_izh_d,
                        "I_e": 0.0, "V_m": cfg.burst_izh_c})
            self.burst_h_rec_bd1 = recorder_tracking.create()
            self.burst_h_rec_burst = recorder_tracking.create()
            nest.Connect(self.burst_h_pop, self.burst_h_rec_bd1)
            nest.Connect(self.burst_h_pop, self.burst_h_rec_burst)
            nest.CopyModel("static_synapse", "static_ch_burst")
            # build_plastic_ctx (load_plastic)-gated, same rationale as
            # Actor's own cortex->D1/D2 build (actor.py's "if self.
            # build_plastic_ctx:"): on a NestCheckpointer rebuild
            # (load_plastic=False), this connection is left for
            # restore_from_checkpoint()'s tracked-projection replay to
            # recreate from the actual saved LIVE weights instead --
            # building it here too would double-connect.
            if load_plastic:
                nest.Connect(
                    self.cortex.output(), self.burst_h_pop, "all_to_all",
                    syn_spec={"synapse_model": "static_ch_burst",
                              "weight": nest.random.uniform(cfg.burst_ch_w_lo, cfg.burst_ch_w_hi),
                              "delay": nest.resolution})
            # Per-H-neuron top-down noise_generator -- one device per H
            # neuron, so each gets its own independently-modulated
            # amplitude during the teaching window (NEST noise_generator
            # mean/std are per-device parameters).
            self.burst_h_ngs = nest.Create(
                "noise_generator", _h_size,
                params={"mean": 0.0, "std": cfg.burst_base_std, "dt": nest.resolution})
            nest.Connect(self.burst_h_ngs, self.burst_h_pop, "one_to_one")
            _actor_cortex_input = self.burst_h_pop
        elif getattr(cfg, "actor_cortex_fixed_hidden", False):
            # actor_cortex_fixed_hidden: inserts a frozen random
            # fixed-hidden layer between the real recurrent cortex and
            # D1/D2 -- cortex is not bypassed here (contrast with
            # actor_bypass_fixed_hidden, which sources from the bypass
            # encoder when cortex is skipped entirely). Same
            # construction/caching pattern as every other fixed-hidden
            # branch, just sourced from self.cortex.output() instead.
            _h_n = getattr(cfg, "actor_fixed_hidden_size", 750)
            _h_indegree = getattr(cfg, "actor_fixed_hidden_indegree", 30)
            _h_weight = getattr(cfg, "actor_fixed_hidden_weight", 300.0)
            self.actor_cortex_hidden_pop = nest.Create("iaf_psc_alpha", _h_n)

            def _build_cortex_fixed_hidden():
                nest.Connect(
                    self.cortex.output(), self.actor_cortex_hidden_pop,
                    conn_spec={"rule": "fixed_indegree", "indegree": _h_indegree},
                    syn_spec={"synapse_model": "static_synapse",
                              "weight": nest.random.uniform(-_h_weight, _h_weight),
                              "delay": 0.5})
            futils.ensure_connectivity(
                self.cortex.output(), self.actor_cortex_hidden_pop,
                f"{name_prefix}cortex_fixed_hidden", _build_cortex_fixed_hidden)

            _actor_cortex_input = self.actor_cortex_hidden_pop

        self.actor = ActorD1D2ChannelsWTA(
            _actor_cortex_input, cfg.n_actions,
            efference_target=_efference_target_override,
            verbose=cfg.verbose,
            ie_d1=cfg.ie_d1,           ie_d2=cfg.ie_d2,
            el_d1=cfg.el_d1,           el_d2=cfg.el_d2,
            v_reset_d1=cfg.v_reset_d1, v_reset_d2=cfg.v_reset_d2,
            tau_syn_ex=cfg.tau_syn_ex_actor,
            d1_tau_m=getattr(cfg, "actor_d1_tau_m", 20.0),
            d2_tau_m=getattr(cfg, "actor_d2_tau_m", 20.0),
            grid_shape=cfg.actor_grid_shape,
            name_prefix=name_prefix,
            disable_recurrence=getattr(cfg, "actor_disable_recurrence", False),
            disable_self_recurrence=getattr(cfg, "actor_disable_self_recurrence", False),
            disable_cross_inhibition=getattr(cfg, "actor_disable_cross_inhibition", False),
            disable_efference=getattr(cfg, "actor_disable_efference", False),
            all_to_all_ctx_d1d2=getattr(cfg, "actor_all_to_all_ctx_d1d2", False),
            w_d1_efference_scale=getattr(cfg, "actor_w_d1_efference_scale", 1.0),
            w_d2_efference_scale=getattr(cfg, "actor_w_d2_efference_scale", 1.0),
            k_ctx_d1_scale=getattr(cfg, "actor_k_ctx_d1_scale", 1.0),
            k_ctx_d2_scale=getattr(cfg, "actor_k_ctx_d2_scale", 1.0),
            w_ctx_d1_scale=getattr(cfg, "actor_w_ctx_d1_scale", 1.0),
            w_ctx_d2_scale=getattr(cfg, "actor_w_ctx_d2_scale", 1.0),
            w_ctx_d1_jitter_scale=getattr(cfg, "actor_w_ctx_d1_jitter_scale", 1.0),
            w_ctx_d2_jitter_scale=getattr(cfg, "actor_w_ctx_d2_jitter_scale", 1.0),
            d1_hidden_layer=getattr(cfg, "actor_d1_hidden_layer", False),
            per_dim_normalize=getattr(cfg, "actor_per_dim_normalize", False),
            d1_vth_uniform=getattr(cfg, "actor_d1_vth_uniform", False),
            pref_decorrelate=getattr(cfg, "actor_pref_decorrelate", False),
            d1_adapt_b=getattr(cfg, "actor_d1_adapt_b", 0.0),
            d1_adapt_a=getattr(cfg, "actor_d1_adapt_a", 0.0),
            d1_adapt_tau_w=getattr(cfg, "actor_d1_adapt_tau_w", 200.0),
            topographic_ctx_d1d2=getattr(cfg, "actor_topographic_ctx_d1d2", False),
            topographic_ctx_radius_frac=getattr(cfg, "actor_topographic_ctx_radius_frac", 0.15),
            topographic_ctx_std_frac=getattr(cfg, "actor_topographic_ctx_std_frac", 0.10),
            cortex_extent=self.cortex.extent,
            topographic_ctx_p=getattr(cfg, "actor_topographic_ctx_p", 1.0),
            baseline_zone_grid=_zone_grid,
            build_plastic_ctx=load_plastic,
            per_channel_reward=getattr(cfg, "actor_per_channel_reward", False),
            vector_readout=getattr(cfg, "actor_vector_readout", False),
            vector_readout_seed=getattr(cfg, "actor_vector_readout_seed", 12345),
            d2_stabilizer=getattr(cfg, "actor_d2_stabilizer", False),
            d2_stabilizer_weight_mult=getattr(cfg, "actor_d2_stabilizer_weight_mult", 5.0),
            d2_stabilizer_radius_frac=getattr(cfg, "actor_d2_stabilizer_radius_frac", 1.0),
            unified_readout=getattr(cfg, "actor_unified_readout", False),
            axis_aligned_pref=getattr(cfg, "actor_axis_aligned_pref", False),
            axis_pref_graded=getattr(cfg, "actor_axis_pref_graded", False),
            readout_baseline_correction=getattr(cfg, "actor_readout_baseline_correction", False),
            readout_baseline_tau=getattr(cfg, "actor_readout_baseline_tau", 50000.0),
            d1_vth_uniform_value=getattr(cfg, "actor_d1_vth_uniform_value", -53.0),
            d1_vth_jitter_std=getattr(cfg, "actor_d1_vth_jitter_std", 1.0),
        )

        # context_gated_da's missing piece: Dopamine (built first, per the
        # ordering requirement in its own docstring) exposes N per-zone DA
        # populations but couldn't wire their gating input itself -- it
        # doesn't have cortex's zone partition yet. Actor computes/owns that
        # partition (actor._cortex_zones, same one it uses to route its own
        # cortex->D1/D2 connections). Now that both exist, connect each
        # zone's cortex directly to that zone's own DA population as the
        # extra excitatory "this zone is currently active" kick (see
        # dopamine.py's context_gated_da comment for why this needs to be a
        # kick, not the sole drive -- reward+tonic alone must stay weaker
        # than full response, calibrated via da_zone_v_th).
        if getattr(self.dopamine, "da_zone_populations", None) is not None:
            gate_weight = self.dopamine.da_zone_gate_weight
            gate_indegree = getattr(cfg, "actor_da_zone_gate_indegree", 30)
            for _z, zone_cortex in enumerate(self.actor._cortex_zones):
                nest.Connect(
                    zone_cortex, self.dopamine.da_zone_populations[_z],
                    conn_spec={"rule": "fixed_indegree",
                               "indegree": min(gate_indegree, len(zone_cortex))},
                    syn_spec={"weight": gate_weight, "delay": nest.resolution},
                )

        # critic_reach_vec_input: the critic's L1 input source -- a
        # dedicated reach_vec place-field population (enc["reach_vec"],
        # built by main.py's _build_encoders) instead of cortex, since
        # this population alone decodes V* far better than cortex does.
        # Critic.__init__ doesn't care what
        # population it's handed -- it just connects it to layer1 -- so no
        # change to critic.py itself. Stored on self so main.py's
        # SpikingCriticLearning construction/attach (which otherwise
        # defaults to net.cortex.output()) can reuse the same source
        # without duplicating this conditional.
        self.critic_input_pop = (
            enc["reach_vec"].output() if getattr(cfg, "critic_reach_vec_input", False)
            else self.cortex.output())
        self.critic = Critic(
            self.critic_input_pop,
            name_prefix=name_prefix,
            l1_shape=cfg.critic_l1_shape,
            tau_syn_ex_l1=cfg.tau_syn_ex_critic_l1,
            critic_da_factor=cfg.critic_da_factor,
            d=cfg.critic_d,
            tau_r=cfg.critic_tau_r,
            l1_ie=getattr(cfg, "critic_l1_ie", 140.0),
            target_total_l1=getattr(cfg, "critic_target_total_l1", 300.0),
            p_input_l1=getattr(cfg, "critic_p_input_l1", 0.3),
            build_plastic_input=load_plastic,
            signed_pools=getattr(cfg, "critic_signed_pools", False),
        )
        self.critic.connect_dopamine(self.dopamine)
        # critic_learner (SpikingCriticLearning) is deliberately NOT
        # constructed here -- this
        # whole Network is rebuilt FRESH on every checkpoint_every rebuild
        # (main.py's nonlocal `net = Network(...)`), which would silently
        # discard a critic_learner's self.W/self.x_bar every few episodes.
        # It's built ONCE in main.py instead (same scope/pattern as
        # batched_d1_list's elig/credit arrays) and re-attached to each
        # rebuilt net via .attach(), not recreated. See main.py.

        # Always-on cortex recorder for the active-fraction diagnostic (log
        # every episode; a saturated cortex (>~50% active) means nothing
        # downstream can discriminate), same as the acrobot source's
        # _cortex_diag_rec.
        self._cortex_diag_rec = recorder_tracking.create()
        nest.Connect(self.cortex.output(), self._cortex_diag_rec)

        # actor_cortex_fixed_hidden: dedicated recorder on the frozen
        # hidden layer sitting between cortex and D1/D2, so
        # BatchedD1Learning's presynaptic
        # source can be redirected there (main.py's _bd1_populations) --
        # same non-destructive time-filtered read convention as every
        # other recorder BatchedD1Learning consumes, not a private
        # self-clearing one like burst_h_rec_burst.
        self._actor_cortex_hidden_diag_rec = None
        if self.actor_cortex_hidden_pop is not None:
            self._actor_cortex_hidden_diag_rec = recorder_tracking.create()
            nest.Connect(self.actor_cortex_hidden_pop, self._actor_cortex_hidden_diag_rec)

        # critic_input_rec: SpikingCriticLearning.step() needs a recorder on
        # whatever population critic_input_pop actually is. When that's
        # cortex (the default), reuse _cortex_diag_rec -- no second
        # recorder on the same population. When it's the dedicated
        # reach_vec population (critic_reach_vec_input=True), it needs its
        # own recorder -- _cortex_diag_rec only ever sees cortex spikes, so
        # main.py's learner call would silently look up reach_vec sender
        # ids in an empty/wrong event stream (KeyError on the first step --
        # confirmed via smoke test) without this.
        if getattr(cfg, "critic_reach_vec_input", False):
            self.critic_input_rec = recorder_tracking.create()
            nest.Connect(self.critic_input_pop, self.critic_input_rec)
        else:
            self.critic_input_rec = self._cortex_diag_rec

        # Inhibitory population's own diagnostic recorder -- previously
        # unrecorded entirely (only cortex.exc had one), so cortex_raster/
        # cortex_spike_bins PDFs never showed I population activity at all.
        self._cortex_inh_diag_rec = recorder_tracking.create()
        nest.Connect(self.cortex.inh, self._cortex_inh_diag_rec)

        # Always-on input recorders (joints/object encoder outputs) for
        # debug_util.py's spike PDFs -- no single flat "sd_state_inputs"
        # population here (unlike the acrobot source) since inputs are two
        # separately-named encoders, not one parrot pool.
        self._joints_diag_rec = recorder_tracking.create()
        nest.Connect(enc["joints"].output(), self._joints_diag_rec)
        self._object_diag_rec = None
        if hasattr(enc["object"], "output"):
            try:
                self._object_diag_rec = recorder_tracking.create()
                nest.Connect(enc["object"].output(), self._object_diag_rec)
            except Exception:
                # A future encoder with a spatial (non-plain) pool might
                # not connect to a plain recorder -- don't let diagnostics
                # break network construction.
                self._object_diag_rec = None

    def cortex_active_fraction(self, t0, t1) -> float:
        events = nest.GetStatus(self._cortex_diag_rec, "events")[0]
        times = events["times"]
        senders = events["senders"]
        mask = (times >= t0) & (times < t1)
        active = len(set(senders[mask].tolist()))
        return active / len(self.cortex.output())

    def clear_rates(self):
        self.critic.clear_rates()
        self.actor.clear_rates()
        self.dopamine.clear_rates()
        self.cortex.clear_rates()
        Base.clear_spikes(self._cortex_diag_rec)
        if self.critic_input_rec is not self._cortex_diag_rec:
            Base.clear_spikes(self.critic_input_rec)
        if self._actor_cortex_hidden_diag_rec is not None:
            Base.clear_spikes(self._actor_cortex_hidden_diag_rec)
        Base.clear_spikes(self._cortex_inh_diag_rec)
        Base.clear_spikes(self._joints_diag_rec)
        if self._object_diag_rec is not None:
            Base.clear_spikes(self._object_diag_rec)
        if self.burst_h_rec_bd1 is not None:
            # rec_H_burst is NOT cleared here -- it's self-clearing every
            # step (burst_hidden.py's apply_burst_update), private to
            # that one consumer. rec_H_bd1 is read non-destructively
            # (time-filtered) by BatchedD1Learning every step, like every
            # other recorder in this periodic sweep.
            Base.clear_spikes(self.burst_h_rec_bd1)
