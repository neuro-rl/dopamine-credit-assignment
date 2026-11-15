import numpy as np

# nest: imported LAZILY inside
# each method that actually needs it, not at module level -- this class is
# otherwise pure-Python glue (append_stat/assert_synapse_init_in_bounds have
# no NEST dependency at all), and env_reach.py/env_reach_camera.py's
# ReachAdapter/CameraMarkerRewardAdapter subclass EnvAdapter (which
# subclasses nothing NEST-specific) purely to reuse THOSE two methods --
# their own reset()/step() never touch a NEST-dependent Base method. An
# unconditional top-level `import nest` made this whole file (and therefore
# the adapters) unimportable without a real NEST install, which blocked
# reusing the adapters' pure-numpy kinematics/reward logic from a separate,
# NEST-free training pipeline for no actual behavioral reason. Every
# existing NEST-based caller is unaffected -- the import still happens on
# first use, just deferred instead of at module load.


class Base:

    @staticmethod
    def clear_spikes(recorder):
        import nest
        nest.SetStatus(recorder, {'n_events': 0})

    @staticmethod
    def rate_in_window(recorder, N, t0, t1):
        """Population rate (Hz) over the half-open window [t0, t1)."""
        import nest
        events = nest.GetStatus(recorder, keys="events")[0]
        times = events["times"]

        if len(times) == 0:
            return 0.0

        mask = (times >= t0) & (times < t1)
        n_spikes = np.sum(mask)
        duration_s = (t1 - t0) / 1000.0
        return n_spikes / (N * duration_s)

    @staticmethod
    def w_uni(base, spread):
        import nest
        return nest.random.uniform(min=base - spread, max=base + spread)

    @staticmethod
    def connect_mexican_hat(source, target, weight=40., std=0.3, sink=0.7, mask_radius=0.5):
        """
        Connect populations with Mexican hat connectivity (center excitation, surround inhibition).
        """
        import nest
        nest.Connect(
            source, target,
            conn_spec={
                'rule': 'pairwise_bernoulli',
                'p': 1.0,
                'mask': {'circular': {'radius': mask_radius}}
            },
            syn_spec={
                'weight': weight * (
                        nest.spatial_distributions.gaussian(
                            nest.spatial.distance, std=std
                        ) - sink
                ),
                'delay': nest.resolution
            }
        )

    @staticmethod
    def connect_gaussian(source, target, *, weight, radius, std, p=1.0, delay=None,
                          synapse_model=None, allow_oversized_mask=False):
        """Spatial Gaussian connectivity; sign of weight sets excitation/inhibition.
        synapse_model: defaults to None, i.e. NEST's own default
        ('static_synapse') -- every existing caller (D1<->D2 cross-inhibition
        etc.) is unaffected. Pass e.g. 'dopsyn'/'adopsyn' to make a spatial
        Gaussian connection use a plastic STDP-dopamine synapse model instead,
        for a topographic cortex->D1/D2 projection (see actor.py's
        topographic_ctx_d1d2).
        allow_oversized_mask: NEST rejects a circular mask
        radius larger than half the periodic layer's extent by default
        ("Mask size must not exceed layer size") -- needed for
        actor.py's d2_stabilizer, whose wide-radius D2->D1 suppression
        needs radius roughly equal to the layer extent to actually
        move D1's activity, which exceeds
        that. Default False: every existing caller unchanged."""
        import nest
        syn_spec = {
            "weight": float(weight) *
                      nest.spatial_distributions.gaussian(
                          nest.spatial.distance,
                          std=float(std)
                      ),
            "delay": nest.resolution if delay is None else float(delay)
        }
        if synapse_model is not None:
            syn_spec["synapse_model"] = synapse_model
        conn_spec = {
            "rule": "pairwise_bernoulli",
            "p": float(p),
            "mask": {"circular": {"radius": float(radius)}}
        }
        if allow_oversized_mask:
            conn_spec["allow_oversized_mask"] = True
        nest.Connect(
            source,
            target,
            conn_spec=conn_spec,
            syn_spec=syn_spec
        )

    @staticmethod
    def append_stat(stats, key, value):
        if key not in stats:
            stats[key] = []
        stats[key].append(value)

    @staticmethod
    def count_spikes(recorder, t0, t1):
        import nest
        events = nest.GetStatus(recorder, keys="events")[0]
        times = events["times"]
        if len(times) == 0:
            return 0
        return int(np.sum((times >= t0) & (times < t1)))

    @staticmethod
    def assert_synapse_init_in_bounds(w_mean, w_jit, Wmin, Wmax, label=""):
        """A plastic population initialised on its own floor/ceiling can only
        move one direction (the critic's Wmin=5, init=5.0 bug — it could
        only potentiate, ratcheting 5->16.6 while "learning" was really
        just leaving the floor). Make this class of bug impossible at
        construction time, not discoverable-in-a-CSV."""
        assert Wmin < w_mean - w_jit and w_mean + w_jit < Wmax, (
            f"{label} init {w_mean}±{w_jit} clamps against [{Wmin},{Wmax}]")
