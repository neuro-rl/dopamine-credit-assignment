"""
encoders.py — observations in, spiking populations out.

Same role for the *input* side that EnvAdapter plays for the environment side:
isolate what varies (how an observation becomes spikes) from what doesn't
(cortex, D1/D2, critic, dopamine).

    enc = SensoryInput({
        "joints": TopographicPlaceFieldEncoder(n_features=2, bins=15,
                                                lo=[-pi, -pi], hi=[pi, pi]),
        "object": LinearRateEncoder(n_features=2, ...),
    })

    cortex = Cortex(...)
    enc.connect_to(cortex, weight=45.0)          # each encoder projects its own way

    # per control step
    enc.reset()
    enc.observe("joints", q)
    enc.commit()
    nest.Run(50.0)

WHY connect_to() lives on the encoder
-------------------------------------
Different codes need different projections, and getting it wrong is silent:
place fields are already a code, so random projection (fixed_indegree) loses
nothing -- scrambling which cortex neuron reads which field is harmless. A
spatial/topographic code needs a circular mask instead, or the layout (which
IS the information) gets destroyed. So the encoder owns its own projection
rule; the network just asks for it.
"""

from abc import ABC, abstractmethod
import numpy as np
import nest
import file_utils as futils


class Encoder(ABC):
    """Observation -> spiking population."""

    @abstractmethod
    def output(self):
        """NodeCollection the network connects FROM."""

    @abstractmethod
    def connect_to(self, target, **kw):
        """Project onto target using whatever rule suits this code."""

    def reset(self):
        """Start a new control step."""

    @abstractmethod
    def observe(self, value, t_us=None):
        """Take an observation. May be called once, or per rendered frame."""

    def commit(self, t_origin_ms=None):
        """Load into NEST. Called once, before Run(). Returns spikes loaded."""
        return 0

    @property
    @abstractmethod
    def size(self):
        """Number of output neurons."""


# ──────────────────────────────────────────────────────────────────────────
class TopographicPlaceFieldEncoder(Encoder):
    """Gaussian place fields, same rate code as a plain random-projection
    place-field encoder would use -- but
    the parrots live on a spatial grid (bin k next to bin k+1, each feature
    its own row) and connect_to() uses a circular mask instead of random
    fixed_indegree.

    WHY: measured (this project's own decodability probes) that cortex's
    recurrent E->E connectivity is spatially local (grid-neighbour, radius
    ~1.5 on a 10-unit extent) while a random-projection place-field code
    has zero relationship to that layout -- decodability stayed near zero
    through cortex (mean~0.14) even though the SAME rate code measured
    mean~0.44 at the encoder's own output, one hop earlier. A free control
    (E->E weight=0) proved recurrence alone wasn't the story; an
    activity-matched indegree sweep (7/15/30, same nominal drive) showed
    decodability tracking DOWN with indegree, not up, meaning the loss
    wasn't random-sampling variance either -- it's the coordinate-system
    mismatch itself. Laying bins out on cortex's own grid and projecting
    with a spatial mask means recurrence reinforces the input's structure
    (nearby bins -> nearby cortex neurons -> mutual E->E reinforcement)
    instead of spreading activity along a topology unrelated to it.

    extent MUST match the target population's extent (the cortex this
    connects to), or the mask silently lands in the wrong coordinate
    system -- this trap has already bitten three times in this project.
    Pass extent=cfg.cortex_extent-equivalent explicitly.
    """

    def __init__(self, n_features, bins=15, lo=None, hi=None,
                 max_rate=800.0, sigma_bins=1.0, extent=10.0,
                 x_lo=None, x_hi=None, y_lo=None, y_hi=None):
        self.n_features = n_features
        self.bins = bins
        self.lo = np.full(n_features, -1.0) if lo is None else np.asarray(lo, float)
        self.hi = np.full(n_features, +1.0) if hi is None else np.asarray(hi, float)
        self.max_rate = max_rate
        self.centres = np.linspace(0.0, 1.0, bins)
        self.sigma = sigma_bins / max(bins - 1, 1)
        self.extent = float(extent)

        self.gen = nest.Create("poisson_generator", n_features * bins, {"rate": 0.0})

        # x_lo/x_hi/y_lo/y_hi confine this encoder's bins to a SUB-REGION of
        # the shared extent (default: the full extent, backward compatible)
        # -- lets two streams occupy disjoint zones of the same cortex
        # (Step 2 of the topographic fix: joints and object each get their
        # own region, rather than both spanning the full grid and landing
        # on IDENTICAL positions if instantiated independently with
        # defaults, which would collide instead of separating).
        x_lo = 0.0 if x_lo is None else float(x_lo)
        x_hi = self.extent if x_hi is None else float(x_hi)
        y_lo = 0.0 if y_lo is None else float(y_lo)
        y_hi = self.extent if y_hi is None else float(y_hi)

        # One row per feature, spread evenly across [y_lo,y_hi] so different
        # features don't overlap in space (row f centred at y_lo +
        # (f+0.5)/n_features * (y_hi-y_lo)); bins spread across [x_lo,x_hi]
        # in order, so bin k sits next to bin k+1. All positions built into
        # ONE nest.spatial.free() call -- NodeCollections from separate
        # spatial Create() calls carry different metadata and can't be
        # joined with '+' (NESTErrors.BadProperty), so per-feature rows
        # must be laid out as one combined position list instead of one
        # grid Create per row.
        positions = []
        for f in range(n_features):
            y = y_lo + (f + 0.5) / n_features * (y_hi - y_lo)
            for b in range(bins):
                x = x_lo + (b + 0.5) / bins * (x_hi - x_lo)
                positions.append([x, y])
        # extent/edge_wrap MUST be declared here to match the target cortex
        # grid's own [extent, extent]/edge_wrap=True (see cortex.py) --
        # without them, nest.spatial.free() defaults to a non-periodic,
        # extent-less layer, and pairwise_bernoulli's circular mask against
        # a periodic target silently ignores the radius entirely (verified:
        # radius swept 0.3->100.0 all gave the identical connection count).
        # With this fix, radius correctly scales connection count and NEST
        # correctly rejects radius > extent/2 (oversized for a periodic
        # layer) instead of silently no-op'ing.
        self.par = nest.Create("parrot_neuron", positions=nest.spatial.free(
            positions, extent=[self.extent, self.extent], edge_wrap=True))

        nest.Connect(self.gen, self.par, "one_to_one")

    @property
    def size(self):
        return self.n_features * self.bins

    def output(self):
        return self.par

    def observe(self, value, t_us=None):
        """value: array of n_features. Idempotent -- last call wins."""
        v = np.asarray(value, float)
        x = np.clip((v - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)
        r = self.max_rate * np.exp(-((x[:, None] - self.centres[None, :]) ** 2)
                                   / (2 * self.sigma ** 2))
        self.gen.rate = r.ravel().tolist()

    def connect_to(self, target, weight=10.0, delay=0.5, radius=3.0,
                   radius_frac=None, p=1.0, indegree=None, spatial_indegree=True,
                   synapse_model="static_synapse", **kw):
        """Spatial mask -- pairwise_bernoulli by default, OR fixed_indegree
        (spatially-restricted OR fully random) if `indegree` is given.
        target must live on a grid with the SAME extent as this encoder
        (see class docstring) whenever a mask is in play.

        radius_frac: extent-relative alternative to `radius` (matches the
        convention used everywhere else in this codebase -- cortex's own
        r_ee/r_ei, actor's connect_gaussian calls -- unlike the old fixed
        `radius`, this auto-scales if cortex_extent ever changes instead of
        needing manual re-tuning). If given, overrides `radius`.

        indegree: if given, uses 'fixed_indegree' instead of
        'pairwise_bernoulli' -- every target neuron gets EXACTLY `indegree`
        connections. Two modes:
          - spatial_indegree=True (default): sampled from source neurons
            within `radius` only (mask still restricts the candidate pool;
            `p` is ignored). Decouples "how many inputs per neuron" from
            "how wide the receptive field is" -- pairwise_bernoulli+p=1.0
            connects to EVERY candidate within radius, so indegree varies
            with local encoder-bin density near each target.
          - spatial_indegree=False: NO mask at all -- every target neuron's
            `indegree` sources are drawn uniformly at random from the
            ENTIRE encoder population, position/topography ignored
            entirely (a plain random projection, just via this class's
            rate code). `radius` is unused in this mode.
        """
        if radius_frac is not None:
            radius = radius_frac * self.extent
        if indegree is not None and spatial_indegree:
            conn_spec = {"rule": "fixed_indegree", "indegree": indegree,
                         "mask": {"circular": {"radius": radius}}}
        elif indegree is not None:
            conn_spec = {"rule": "fixed_indegree", "indegree": indegree}
        else:
            conn_spec = {"rule": "pairwise_bernoulli", "p": p,
                         "mask": {"circular": {"radius": radius}}}
        nest.Connect(self.par, target,
                     conn_spec=conn_spec,
                     syn_spec={"synapse_model": synapse_model,
                               "weight": weight, "delay": delay, **kw})


# ──────────────────────────────────────────────────────────────────────────
class Grid2DPlaceFieldEncoder(Encoder):
    """TRUE joint 2-D Gaussian receptive fields over a 2-D feature (e.g.
    reach_vec = ee_xy - target) -- bins x bins neurons, each tuned to a
    (cx, cy) point in the full 2-D plane, unlike a SEPARABLE per-axis
    place-field code's n_features*bins tuning (each of its neurons reads
    only one axis, so a linear readout downstream can never recover a
    joint nonlinearity like distance/norm from it -- confirmed via the
    reach-vector decodability probe: this class's own offline
    measurement -- before being wired into any live network -- reached
    R2=0.55-0.59 (ridge) where cortex's own representation gave 0.23-0.28).

    lo/hi: per-axis range, calibrated by the caller (e.g. sampling real
    (fk(q)-target) pairs) -- this class does no calibration of its own,
    just takes lo/hi explicitly.
    """

    def __init__(self, bins=12, lo=None, hi=None, max_rate=800.0, sigma_bins=1.0):
        self.bins = bins
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.max_rate = max_rate
        cx, cy = np.meshgrid(np.linspace(self.lo[0], self.hi[0], bins),
                              np.linspace(self.lo[1], self.hi[1], bins))
        self.centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
        self.sigma = (self.hi - self.lo) / max(bins - 1, 1) * sigma_bins

        self.gen = nest.Create("poisson_generator", bins * bins, {"rate": 0.0})
        self.par = nest.Create("parrot_neuron", bins * bins)
        nest.Connect(self.gen, self.par, "one_to_one")

    @property
    def size(self):
        return self.bins * self.bins

    def output(self):
        return self.par

    def observe(self, value, t_us=None):
        """value: array of shape (2,). Idempotent -- last call wins."""
        v = np.asarray(value, float)
        d2 = (((v[0] - self.centers[:, 0]) / self.sigma[0]) ** 2 +
              ((v[1] - self.centers[:, 1]) / self.sigma[1]) ** 2)
        self.gen.rate = (self.max_rate * np.exp(-d2 / 2.0)).tolist()

    def connect_to(self, target, weight=25.0, delay=0.5, indegree=9,
                   synapse_model="static_synapse", **kw):
        """Random projection -- a place field is already a code."""
        nest.Connect(self.par, target,
                     conn_spec={"rule": "fixed_indegree", "indegree": indegree},
                     syn_spec={"synapse_model": synapse_model,
                               "weight": weight, "delay": delay, **kw})


# ──────────────────────────────────────────────────────────────────────────
class GainFieldEncoder(Encoder):
    """Fixed conjunctive gain-field basis for resolved-rate IK: the action the actor needs, dq* =
    J(q)^T . (target - fk(q)), is a q-dependent linear map applied to the
    target plus a q-dependent bias -- exactly the function class a
    gain-field basis spans (Salinas & Abbott 1995; Pouget & Sejnowski 1997;
    Deneve et al. 2001), unlike a random fixed hidden layer which only
    spans it by luck.

    Three K-unit groups over a bins x bins place-field grid on (q1,q2)
    (K = bins*bins, same meshgrid/ravel row-major convention as
    Grid2DPlaceFieldEncoder):
      - plain:  place_k(q)                 -- bias term
      - gain-x: place_k(q) * norm(tx)       -- q-dependent gain on target x
      - gain-y: place_k(q) * norm(ty)       -- q-dependent gain on target y
    norm(t) = (t - lo_t) / (hi_t - lo_t), clipped to [0,1] -- this task's
    target workspace is already non-negative in both axes, so no signed
    +/- channel split is needed (picked the simplest encoding that clears
    the offline decode gate, not the literal two-spike-train-converging
    multiplier the spec's own writeup floats as the general case).

    The product is computed in the ANALOG RATE domain (same as every
    other encoder in this file sets its Poisson rate from an analytic
    function of the observed state) and Poisson-sampled from there --
    this is squarely on the "fixed representation" side of the spec's own
    cheating boundary (same status as place fields making value linearly
    decodable), NOT feeding the action or supervising the readout: the
    readout weights (D1/D2) still have to learn the right w_k combination
    from reward via the existing rule.
    """

    def __init__(self, bins=12, lo_q=None, hi_q=None, lo_t=None, hi_t=None,
                 max_rate=400.0, sigma_bins=1.0):
        self.bins = bins
        self.lo_q = np.asarray(lo_q, float)
        self.hi_q = np.asarray(hi_q, float)
        self.lo_t = np.asarray(lo_t, float)
        self.hi_t = np.asarray(hi_t, float)
        self.max_rate = max_rate
        cx, cy = np.meshgrid(np.linspace(self.lo_q[0], self.hi_q[0], bins),
                              np.linspace(self.lo_q[1], self.hi_q[1], bins))
        self.centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
        self.sigma = (self.hi_q - self.lo_q) / max(bins - 1, 1) * sigma_bins
        self.K = bins * bins

        self.gen_plain = nest.Create("poisson_generator", self.K, {"rate": 0.0})
        self.par_plain = nest.Create("parrot_neuron", self.K)
        nest.Connect(self.gen_plain, self.par_plain, "one_to_one")
        self.gen_gx = nest.Create("poisson_generator", self.K, {"rate": 0.0})
        self.par_gx = nest.Create("parrot_neuron", self.K)
        nest.Connect(self.gen_gx, self.par_gx, "one_to_one")
        self.gen_gy = nest.Create("poisson_generator", self.K, {"rate": 0.0})
        self.par_gy = nest.Create("parrot_neuron", self.K)
        nest.Connect(self.gen_gy, self.par_gy, "one_to_one")
        self.par = self.par_plain + self.par_gx + self.par_gy

    @property
    def size(self):
        return 3 * self.K

    def output(self):
        return self.par

    def observe(self, value, t_us=None):
        """value: (q, target), each array-like of shape (2,)."""
        q, target = value
        q = np.asarray(q, float)
        target = np.asarray(target, float)
        d2 = (((q[0] - self.centers[:, 0]) / self.sigma[0]) ** 2 +
              ((q[1] - self.centers[:, 1]) / self.sigma[1]) ** 2)
        a = np.exp(-d2 / 2.0)
        nt = (target - self.lo_t) / (self.hi_t - self.lo_t + 1e-12)
        nt = np.clip(nt, 0.0, 1.0)
        self.gen_plain.rate = (self.max_rate * a).tolist()
        self.gen_gx.rate = (self.max_rate * a * nt[0]).tolist()
        self.gen_gy.rate = (self.max_rate * a * nt[1]).tolist()

    def connect_to(self, target, weight=25.0, delay=0.5, indegree=9,
                   synapse_model="static_synapse", **kw):
        """Random projection -- same rationale as Grid2DPlaceFieldEncoder.
        Not used for the actor pathway (network.py wires a dedicated real
        iaf_psc_alpha relay one_to_one instead, for a safe efference
        target) -- kept for interface completeness / cortex wiring if
        ever needed."""
        nest.Connect(self.par, target,
                     conn_spec={"rule": "fixed_indegree", "indegree": indegree},
                     syn_spec={"synapse_model": synapse_model,
                               "weight": weight, "delay": delay, **kw})


# ──────────────────────────────────────────────────────────────────────────
class LinearRateEncoder(Encoder):
    """One neuron PER FEATURE (not per bin) -- rate scales linearly with the
    feature's normalised value, bigger value = higher frequency. The
    simplest possible rate code, as opposed to a 15-bin Gaussian place-field
    population code.

    Same Encoder interface as every other encoder here, so it's a drop-in
    swap anywhere one is constructed.
    """

    def __init__(self, n_features, lo=None, hi=None, min_rate=0.0, max_rate=400.0):
        self.n_features = n_features
        self.lo = np.full(n_features, -1.0) if lo is None else np.asarray(lo, float)
        self.hi = np.full(n_features, +1.0) if hi is None else np.asarray(hi, float)
        self.min_rate = min_rate
        self.max_rate = max_rate

        self.gen = nest.Create("poisson_generator", n_features, {"rate": 0.0})
        self.par = nest.Create("parrot_neuron", n_features)
        nest.Connect(self.gen, self.par, "one_to_one")

    @property
    def size(self):
        return self.n_features

    def output(self):
        return self.par

    def observe(self, value, t_us=None):
        """value: array of n_features. Idempotent -- last call wins."""
        v = np.asarray(value, float)
        x = np.clip((v - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)
        r = self.min_rate + x * (self.max_rate - self.min_rate)
        self.gen.rate = r.tolist()

    def connect_to(self, target, weight=25.0, delay=0.5, indegree=9,
                   synapse_model="static_synapse", **kw):
        """Random projection -- with only
        2 source neurons total this indegree is really "how many of the 2
        each target neuron connects to" (indegree=1 or 2 in practice, NEST
        clips to n_features if indegree exceeds it)."""
        nest.Connect(self.par, target,
                     conn_spec={"rule": "fixed_indegree", "indegree": min(indegree, self.n_features)},
                     syn_spec={"synapse_model": synapse_model,
                               "weight": weight, "delay": delay, **kw})


# ──────────────────────────────────────────────────────────────────────────
class SensoryInput:
    """Named encoders, driven together."""

    def __init__(self, encoders: dict):
        self.enc = dict(encoders)

    def __getitem__(self, name):
        return self.enc[name]

    @property
    def size(self):
        return sum(e.size for e in self.enc.values())

    def reset(self):
        for e in self.enc.values():
            e.reset()

    def observe(self, name, value, t_us=None):
        return self.enc[name].observe(value, t_us)

    def commit(self, t_origin_ms=None):
        return sum(e.commit(t_origin_ms) or 0 for e in self.enc.values())

    def connect_to(self, target, **per_encoder):
        """connect_to(cortex, joints=dict(indegree=9), object=dict(radius=10.))

        Each encoder uses its own rule; kwargs are routed by name.
        Cannot join NodeCollections here — spatial ones refuse — so each
        encoder connects itself. Which is what you want anyway.
        """
        for name, e in self.enc.items():
            e.connect_to(target, **per_encoder.get(name, {}))

    def report(self):
        return {n: {"type": type(e).__name__, "size": e.size}
                for n, e in self.enc.items()}
