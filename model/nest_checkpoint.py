"""
Reusable NEST network checkpoint/restore helper.

Works around a NEST memory-growth characteristic tied to
stdp_dopamine_synapse + volume_transmitter (and plausibly other
continuously-active plasticity mechanisms) accumulating internal state
across many Simulate() calls. Confirmed present identically in NEST
3.8.0 and 3.10.0, reproduced in a minimal network with none of a
calling project's own Python code involved -- so it isn't fixable at
the application level. Periodically calling nest.ResetKernel() and
rebuilding the network from scratch, then replaying the saved
connection weights, cuts the leak by roughly 45x in isolated testing
(~11.2 MB/999-step episode unmitigated, down to ~0.25 MB/episode).

What is NOT preserved across a restore, and why that's usually fine:
  - NEST's internal RNG state. There is no API to serialize/restore it,
    so every restore effectively reseeds stochastic dynamics
    (exploration noise, Poisson generators, etc.). Treat this the same
    as ordinary run-to-run seed variance.
  - Per-connection STDP eligibility-trace state (e.g. 'c'/'n' on
    stdp_dopamine_synapse). These decay on a timescale usually much
    shorter than a checkpoint interval, and most spiking networks
    already have some natural inter-trial gap where this state mostly
    decays anyway.

Usage:
    ckpt = NestCheckpointer(build_fn=build_network)
    ckpt.track_projection("cortex_d1", lambda: cortex_pop, lambda: d1_pop)
    ckpt.track_projection("cortex_d2", lambda: cortex_pop, lambda: d2_pop)

    # First build, e.g. loading a topology from disk however this
    # project already does that -- NOT through this class.
    build_network()

    # ... train for a while ...
    ckpt.save_checkpoint(extra_state={"da_baseline": dopamine.da_baseline})

    # ... nest.ResetKernel() would invalidate every NodeCollection the
    # caller is holding, so do this at a point where the caller is
    # about to rebuild its own references anyway ...
    extra_state = ckpt.restore_from_checkpoint()
    dopamine.da_baseline = extra_state["da_baseline"]

Each tracked projection's getters must return a *fresh* NodeCollection
reflecting whatever currently exists -- typically a lambda closing over
a mutable holder (e.g. a dict or object attribute) that the caller
updates when build_fn() runs, since the NodeCollection objects from
before a nest.ResetKernel() call are no longer valid afterward.
"""
import pickle

import nest
import numpy as np


class NestCheckpointer:
    def __init__(self, build_fn):
        """
        build_fn: callable, () -> None. Rebuilds every NODE the tracked
            projections' getters will resolve to (neurons, generators,
            recorders, and any *non*-tracked/fixed connections). Must
            be safe to call again after nest.ResetKernel(). Should NOT
            recreate the tracked plastic projections themselves --
            restore_from_checkpoint() does that from the saved
            checkpoint instead, to avoid double-connecting.
        """
        self.build_fn = build_fn
        self._projections = {}   # name -> (src_getter, tgt_getter)
        self._saved_conns = {}   # name -> dict of arrays, or None
        self._saved_extra = {}
        self._has_checkpoint = False

    def track_projection(self, name, src_getter, tgt_getter, synapse_model=None):
        """Register a projection to snapshot/restore.

        src_getter/tgt_getter: zero-arg callables returning the current
        NodeCollection for that side of the projection. Called both at
        save time and at restore time (after build_fn() has run), so
        they must resolve to whatever's current, not a stale reference
        captured once at registration time.

        synapse_model (optional): restrict save/restore to connections
        using exactly this synapse model. Needed when src/tgt cover a
        heterogeneous pool that ALSO carries other, separately-tracked
        or fixed connections between its own members (e.g. a random-
        connectivity diagnostic scattered across many populations that
        already have their own structured projections between each
        other) -- without this, GetConnections(src, tgt) would also
        pick up those unrelated connections. None (default) preserves
        the original behavior (every connection between src and tgt,
        any model) for every existing caller.
        """
        self._projections[name] = (src_getter, tgt_getter, synapse_model)

    def save_checkpoint(self, extra_state=None):
        """Snapshot every tracked projection's (src_idx, tgt_idx,
        weight, delay, synapse_model) into memory, plus an arbitrary
        extra_state dict for whatever scalar/Python-side state the
        caller wants carried across the next restore."""
        self._saved_conns = {}
        for name, (src_getter, tgt_getter, synapse_model) in self._projections.items():
            src = src_getter()
            tgt = tgt_getter()
            src_index = {gid: i for i, gid in enumerate(src.tolist())}
            tgt_index = {gid: j for j, gid in enumerate(tgt.tolist())}

            if synapse_model is not None:
                conns = nest.GetConnections(src, tgt, synapse_model=synapse_model)
            else:
                conns = nest.GetConnections(src, tgt)
            if len(conns) == 0:
                self._saved_conns[name] = None
                continue
            conn_info = nest.GetStatus(conns, ["source", "target", "weight", "delay", "synapse_model"])
            n = len(conn_info)
            src_idx = np.zeros(n, dtype=np.int32)
            tgt_idx = np.zeros(n, dtype=np.int32)
            weights = np.zeros(n, dtype=np.float64)
            delays = np.zeros(n, dtype=np.float64)
            syn_models = np.empty(n, dtype=object)
            for i, (s, t, w, d, syn) in enumerate(conn_info):
                src_idx[i] = src_index[s]
                tgt_idx[i] = tgt_index[t]
                weights[i] = w
                delays[i] = d
                syn_models[i] = syn
            self._saved_conns[name] = {
                "src_idx": src_idx, "tgt_idx": tgt_idx,
                "weights": weights, "delays": delays, "syn_models": syn_models,
            }
        self._saved_extra = dict(extra_state) if extra_state else {}
        self._has_checkpoint = True

    def restore_from_checkpoint(self):
        """nest.ResetKernel(), rebuild via build_fn(), replay every
        tracked projection's saved connections/weights, and return the
        extra_state dict passed to the matching save_checkpoint()."""
        if not self._has_checkpoint:
            raise RuntimeError("No checkpoint saved yet -- call save_checkpoint() first.")

        nest.ResetKernel()
        self.build_fn()

        for name, (src_getter, tgt_getter, synapse_model) in self._projections.items():
            saved = self._saved_conns.get(name)
            if not saved:
                continue
            src = src_getter()
            tgt = tgt_getter()
            for s, t, w, d, syn in zip(
                saved["src_idx"], saved["tgt_idx"], saved["weights"], saved["delays"], saved["syn_models"]
            ):
                nest.Connect(
                    src[int(s)], tgt[int(t)],
                    syn_spec={"weight": float(w), "delay": float(d), "synapse_model": str(syn)},
                )

        return self._saved_extra

    # ── Disk persistence ───────────────────────────────────────
    # save_checkpoint()/restore_from_checkpoint() above only ever held state
    # in memory within one process (across a nest.ResetKernel(), not across
    # a process boundary) -- there was no way to train once, exit, and later
    # load the trained network in a SEPARATE process (e.g. to generate an
    # eval video without re-running training). These two methods are thin
    # wrappers: save_checkpoint_to_disk persists exactly what save_
    # checkpoint() already collects; load_checkpoint_from_disk loads it back
    # into self._saved_conns/_saved_extra so the EXISTING restore_from_
    # checkpoint() (ResetKernel + build_fn + replay) works completely
    # unchanged, whether the checkpoint came from this process's own memory
    # or from a file written by a now-dead one. Pickle is fine here -- this
    # is a controlled, same-codebase round trip (we write it, we read it),
    # not deserialization of untrusted external input.
    def save_checkpoint_to_disk(self, path, extra_state=None):
        """save_checkpoint(extra_state), then persist the snapshot to
        `path`. The caller must have already registered the SAME
        track_projection() names it wants restorable -- this does not
        change what's tracked, only where the snapshot ends up."""
        self.save_checkpoint(extra_state=extra_state)
        with open(path, "wb") as f:
            pickle.dump({"saved_conns": self._saved_conns, "saved_extra": self._saved_extra}, f)

    def load_checkpoint_from_disk(self, path):
        """Load a snapshot written by save_checkpoint_to_disk() into this
        instance. Call restore_from_checkpoint() right after to actually
        apply it -- this method only loads the data; it does not touch
        NEST. The caller must register the SAME track_projection() names
        the writing run used, or restore_from_checkpoint() silently skips
        whatever doesn't match (same contract as the in-memory path)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._saved_conns = data["saved_conns"]
        self._saved_extra = data["saved_extra"]
        self._has_checkpoint = True
