import nest
from base_util import Base
import file_utils as futils
import recorder_tracking


class Cortex(Base):
    """
    Balanced random E/I cortex that:
      - has E->E, E->I, I->E, I->I recurrence
      - includes decorrelated background drive for BOTH E and I
      - exposes:
          output() -> excitatory population (used downstream)
          clear_rates() -> resets recorders

    Unlike the acrobot original, this Cortex does NOT wire its own input.
    The reach task has two heterogeneous streams (joints, object) that need
    separately-tuned weight/indegree and different projection rules (place
    fields use random projection, a retinotopic layer needs a spatial
    mask) -- so encoders own connect_to() and network.py wires each stream
    onto cortex.output() after both exist.
    """

    def __init__(
            self,
            *,
            exc_shape=(20, 20),
            inh_shape=(10, 10),
            r_ee=0.15, w_ee=4,
            r_ei=0.20, w_ei=20.0,
            r_ie=0.15, w_ie=-11.0,
            p_ii=0.05, w_ii=-4.0,
            bg_n_sources=50,
            bg_rate_hz=640.0,
            bg_p=0.10,
            bg_w_exc=1.0,
            bg_w_inh=1.0,
            bg_shared=True,
            bg_one_to_one=False,
            exc_params=None,
            inh_params=None,
            extent=None,
            name_prefix: str = "",
    ):
        super().__init__()

        if exc_params is None:
            exc_params = {
                "tau_m": 20.0,
                "V_reset": -65.0,
                "E_L": -65.0,
                "V_th": -54.0,
                "t_ref": 3.0,
                "I_e": 0.0,
            }
        if inh_params is None:
            inh_params = {
                "tau_m": 10.0,
                "V_reset": -65.0,
                "E_L": -65.0,
                "V_th": -56.0,
                "t_ref": 3.0,
                "I_e": 10.0,
            }

        # Shared extent; spacing = ext/n. All the internal E/I radii below are
        # expressed as ext-relative fractions (e.g. r_ee=0.15*ext), so the
        # actual fan-in ratios are extent-invariant -- ext only matters when
        # an EXTERNAL population needs to connect via a spatial mask and must
        # share this exact coordinate system (e.g. mode B's EventEncoder pool
        # lives at a much smaller extent than the default 10.0; network.py
        # passes extent=cfg.cortex_extent to match it — see reach_events.py).
        ext = float(max(exc_shape[0], inh_shape[0])) if extent is None else float(extent)
        self._extent = ext

        self.exc = nest.Create(
            "iaf_psc_alpha",
            positions=nest.spatial.grid(
                shape=list(exc_shape),
                extent=[ext, ext],
                center=[ext / 2, ext / 2],
                edge_wrap=True
            ),
            params=exc_params
        )

        self.inh = nest.Create(
            "iaf_psc_alpha",
            positions=nest.spatial.grid(
                shape=list(inh_shape),
                extent=[ext, ext],
                center=[ext / 2, ext / 2],
                edge_wrap=True
            ),
            params=inh_params
        )

        self.out = self.exc

        self.recorder_exc = recorder_tracking.create()
        self.recorder_inh = recorder_tracking.create()
        nest.Connect(self.exc, self.recorder_exc)
        nest.Connect(self.inh, self.recorder_inh)

        if bg_shared:
            bg = nest.Create("poisson_generator", int(bg_n_sources), {"rate": float(bg_rate_hz)})

            def _build_bg_exc(bg=bg, exc=self.exc):
                nest.Connect(
                    bg, exc,
                    {"rule": "pairwise_bernoulli", "p": float(bg_p)},
                    {"weight": float(bg_w_exc), "delay": nest.resolution},
                )
            futils.ensure_connectivity(bg, self.exc, f"{name_prefix}cortex_bg_exc", _build_bg_exc)

            def _build_bg_inh(bg=bg, inh=self.inh):
                nest.Connect(
                    bg, inh,
                    {"rule": "pairwise_bernoulli", "p": float(bg_p)},
                    {"weight": float(bg_w_inh), "delay": nest.resolution},
                )
            futils.ensure_connectivity(bg, self.inh, f"{name_prefix}cortex_bg_inh", _build_bg_inh)

        if bg_one_to_one:
            # One private generator per neuron — fully decorrelated across exc and inh.
            # Rate chosen so mean drive matches shared-pool equivalent:
            #   bg_n_sources * bg_p * bg_rate_hz
            bg_rate_one_to_one = float(bg_n_sources) * float(bg_p) * float(bg_rate_hz)

            bg_exc = nest.Create("poisson_generator", len(self.exc), {"rate": bg_rate_one_to_one})
            nest.Connect(bg_exc, self.exc, "one_to_one",
                         {"weight": float(bg_w_exc), "delay": nest.resolution})

            bg_inh = nest.Create("poisson_generator", len(self.inh), {"rate": bg_rate_one_to_one})
            nest.Connect(bg_inh, self.inh, "one_to_one",
                         {"weight": float(bg_w_inh), "delay": nest.resolution})

        nest.Connect(
            self.exc, self.exc,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 1.0,
                "mask": {"circular": {"radius": r_ee * ext}}
            },
            syn_spec={"weight": w_ee, "delay": nest.resolution}
        )

        nest.Connect(
            self.exc, self.inh,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 1.0,
                "mask": {"circular": {"radius": r_ei * ext}}
            },
            syn_spec={"weight": w_ei, "delay": nest.resolution}
        )

        nest.Connect(
            self.inh, self.exc,
            conn_spec={
                "rule": "pairwise_bernoulli",
                "p": 1.0,
                "mask": {"circular": {"radius": r_ie * ext}}
            },
            syn_spec={"weight": w_ie, "delay": nest.resolution}
        )

        def _build_inh_inh(inh=self.inh):
            nest.Connect(
                inh, inh,
                {"rule": "pairwise_bernoulli", "p": float(p_ii), "allow_autapses": False},
                {"weight": float(w_ii), "delay": nest.resolution},
            )
        futils.ensure_connectivity(self.inh, self.inh, f"{name_prefix}cortex_inh_inh", _build_inh_inh)

    def output(self):
        return self.out

    @property
    def extent(self):
        return self._extent

    def clear_rates(self):
        Base.clear_spikes(self.recorder_exc)
        Base.clear_spikes(self.recorder_inh)
