"""
burst-dependent plasticity for training a genuine hidden layer H (cortex
-> H -> D1/D2, replacing the flat cortex -> D1/D2 projection) in the arm
actor. Ported from cart_pole/nest_cartpole_centered_burst_test.py's H
block -- see that module's own docstring for the full calibration
history (Stage-0 feasibility, BASE_STD/burst_frac relationship). H's
NEST-side infrastructure (population, recorders, noise_generators,
cortex->H connection) is built in network.py (rebuilt every checkpoint
cycle, same as cortex/actor/critic). This module is the PERSISTENT
Python-side piece -- constructed once, like BatchedD1Learning
(batched_actor_learning.py) -- holding state with no NEST equivalent
(fb_weight, x_bar) that must survive nest.ResetKernel() untouched, and
re-attached to fresh connection/recorder handles after every rebuild.

Two-recorder design (a deliberate deviation from the CartPole
reference, required by this project's different per-step call order --
see the plan's own writeup): the reference clears H's ONE recorder at
the end of both the decision and teaching windows, in strict order, so
there's never stale data. Here, BatchedD1Learning.step() (which needs
H's DECISION-window spikes) runs AFTER the teaching window has already
executed (main.py's step loop). A single shared recorder would mean
whichever of the two consumers reads second loses the other's data.
Fixed by giving H two recorders: rec_H_bd1 (read non-destructively,
time-filtered, by BatchedD1Learning/_bd1_populations -- owned by
network.py, cleared only by the existing periodic clear_rates_every
sweep) and rec_H_burst (private to THIS module, cleared immediately
before the teaching window and read-and-cleared immediately after)."""
import numpy as np
import nest


def burst_fraction(spike_times, thresh):
    """Fraction of spikes that are part of a burst (ISI < thresh on
    either side). Ported verbatim from the CartPole reference."""
    if len(spike_times) < 2:
        return 0.0
    isi = np.diff(spike_times)
    is_burst = np.zeros(len(spike_times), dtype=bool)
    for j, d in enumerate(isi):
        if d < thresh:
            is_burst[j] = True
            is_burst[j + 1] = True
    return float(is_burst.sum() / len(spike_times))


def _read_times_and_clear(rec, ids):
    events = rec.get("events")
    times = np.asarray(events["times"])
    senders = np.asarray(events["senders"])
    out = []
    for nid in ids:
        t = times[senders == nid]
        out.append(np.sort(t))
    rec.set({"n_events": 0})
    return out


def _counts_in_window(rec, ids, t0, t1):
    events = rec.get("events")
    times = np.asarray(events["times"])
    senders = np.asarray(events["senders"])
    counts = np.zeros(len(ids))
    if len(times) == 0:
        return counts
    mask = (times >= t0) & (times < t1)
    senders = senders[mask]
    if len(senders) == 0:
        return counts
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    idxs = np.fromiter((id_to_idx[s] for s in senders), dtype=int, count=len(senders))
    np.add.at(counts, idxs, 1)
    return counts


class BurstHiddenTeacher:
    def __init__(self, h_size: int, seed: int,
                 isi_ms: float = 8.0, base_std: float = 35.0,
                 std_min: float = 5.0, std_max: float = 80.0,
                 gain_td: float = 40.0, p_baseline: float = 0.40,
                 lr: float = 0.05, fb_weight_lo: float = -1.0,
                 fb_weight_hi: float = 1.0, wmin_ch: float = -40.0,
                 wmax_ch: float = 40.0, xbar_alpha: float = 0.1):
        self.h_size = h_size
        self.isi_ms = isi_ms
        self.base_std = base_std
        self.std_min = std_min
        self.std_max = std_max
        self.gain_td = gain_td
        self.p_baseline = p_baseline
        self.lr = lr
        self.wmin_ch = wmin_ch
        self.wmax_ch = wmax_ch
        self.xbar_alpha = xbar_alpha
        # Fixed random feedback weights (feedback-alignment style) --
        # drawn ONCE via a dedicated RNG stream (not NEST's own
        # rng_seed), then held fixed for the whole run, same as the
        # CartPole reference.
        fb_rng = np.random.default_rng(seed + 100000)
        self.fb_weight = fb_rng.uniform(fb_weight_lo, fb_weight_hi, size=h_size)
        self.x_bar = np.zeros(0)   # sized on first attach() (cortex size)
        self.h_ids = None
        self.conns_ch = None
        self.cortex_ids = None

    def attach(self, cortex_pop, h_pop, conns_ch):
        """Re-fetch handles after a nest.ResetKernel() rebuild. conns_ch:
        the live cortex->H connections (static_ch_burst), already
        restored by NestCheckpointer to their LIVE (not re-randomized)
        weights before this is called."""
        self.cortex_ids = cortex_pop.tolist()
        self.h_ids = h_pop.tolist()
        self.conns_ch = conns_ch
        src = np.array(conns_ch.get("source"))
        tgt = np.array(conns_ch.get("target"))
        cortex_id_to_idx = {nid: i for i, nid in enumerate(self.cortex_ids)}
        h_id_to_idx = {nid: i for i, nid in enumerate(self.h_ids)}
        self._src_idx = np.array([cortex_id_to_idx[s] for s in src])
        self._tgt_idx = np.array([h_id_to_idx[t] for t in tgt])
        if len(self.x_bar) != len(self.cortex_ids):
            self.x_bar = np.zeros(len(self.cortex_ids))

    def set_teaching_std(self, dev: float, ngs):
        std_vec = np.clip(self.base_std + self.gain_td * dev * self.fb_weight,
                           self.std_min, self.std_max)
        ngs.std = std_vec.tolist()

    def reset_decision_std(self, ngs):
        ngs.std = self.base_std

    def apply_burst_update(self, cortex_rec, t_decision0, t_decision1,
                            rec_h_burst, t_teach0, t_teach1):
        """Reads H's teaching-window spike times (rec_h_burst, already
        cleared by the caller right before the teaching window started)
        and cortex's decision-window spike counts (cortex_rec, read non-
        destructively -- that recorder is shared with other diagnostics
        and cleared only by the periodic clear_rates_every sweep), then
        applies the burst-dependent cortex->H weight update. Returns a
        stats dict for logging."""
        h_spiketimes = _read_times_and_clear(rec_h_burst, self.h_ids)
        burst_frac = np.array([burst_fraction(t, self.isi_ms) for t in h_spiketimes])
        h_rate_hz = np.array([len(t) for t in h_spiketimes], dtype=float) / \
            ((t_teach1 - t_teach0) / 1000.0)

        x_j = _counts_in_window(cortex_rec, self.cortex_ids, t_decision0, t_decision1)
        centered_x = x_j - self.x_bar
        self.x_bar = (1.0 - self.xbar_alpha) * self.x_bar + self.xbar_alpha * x_j

        weights = np.array(self.conns_ch.get("weight"))
        delta_ch = self.lr * (centered_x[self._src_idx] * (burst_frac - self.p_baseline)[self._tgt_idx])
        weights = np.clip(weights + delta_ch, self.wmin_ch, self.wmax_ch)
        self.conns_ch.set(weight=weights.tolist())

        return {
            "burst_frac_mean": float(burst_frac.mean()),
            "h_rate_hz_mean": float(h_rate_hz.mean()),
            "w_ch_mean": float(weights.mean()),
            "w_ch_std": float(weights.std()),
        }

    def extract_state(self):
        return {"fb_weight": self.fb_weight.copy(), "x_bar": self.x_bar.copy()}

    def restore_state(self, state):
        self.fb_weight = state["fb_weight"].copy()
        self.x_bar = state["x_bar"].copy()
