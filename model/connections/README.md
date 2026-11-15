# model/connections

Cached connectivity for NEST's "fixed" (non-trained) connections, written
by `model/file_utils.py`'s `ensure_connectivity()`. Filenames are
`<run_label>_<connection_name>.npz`.

These `.npz` files are not source data — they regenerate automatically
and identically the next time the same config+seed runs, since the
connectivity is built deterministically from that seed. They're
gitignored (see `.gitignore` here) and safe to delete.

**Exception:** `cartpole_toy_v2_cortex_inh_inh.npz` is tracked in git,
not ignored. Its cache key isn't seed/config-specific (every CartPole
run shares it, see `cart_pole/state_encoder.py`), so deleting it would
let the next CartPole run regenerate a different shared connectivity
that every other run would then reuse — breaking reproducibility of
already-locked CartPole results. Keep it.
