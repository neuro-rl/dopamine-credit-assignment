import nest
import numpy as np
import os
import uuid
from typing import Mapping, Any
from pathlib import Path


def save_experiment_params(
        experiment_id: uuid.UUID,
        output_folder: str,
        experiment_data: Mapping[str, Any],) -> Path:

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    marker_file = output_path / "experiment.txt"

    with marker_file.open("w", encoding="utf-8") as f:
        f.write(f"experiment_id={experiment_id}\n")
        for key, value in experiment_data.items():
            f.write(f"{key}={value}\n")

    return marker_file


def write_stats_csv(stats, output_folder, filename="stats.csv"):
    """Dump all arrays in `stats` into one CSV; rows aligned by index, missing values -> NaN."""

    if not stats:
        return

    os.makedirs(output_folder, exist_ok=True)

    keys = sorted(stats.keys())
    max_len = max(len(stats[k]) for k in keys)

    data = np.full((max_len, len(keys)), np.nan)

    for col, key in enumerate(keys):
        values = np.asarray(stats[key])
        data[:len(values), col] = values

    header = ",".join(keys)
    path = os.path.join(output_folder, filename)

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.4f"
    )


def save_spikes(spike_detector, base_name, output_folder, bins=50):
    """Save spikes to CSV and a combined raster + PSTH PNG."""

    import matplotlib.pyplot as plt

    events  = nest.GetStatus(spike_detector, "events")[0]
    times   = events["times"]
    senders = events["senders"]

    csv_path = f"{output_folder}/{base_name}.csv"

    if len(times) == 0:
        with open(csv_path, "w") as f:
            f.write("gids,ts\n")
        print(f"[INFO] No spikes in {base_name}; saved empty CSV.")
    else:
        arr = np.vstack((senders, times)).T
        np.savetxt(csv_path, arr, comments='',
                   fmt='%d,%.3f', header='gids,ts', delimiter=',')
        print(f"[OK] Saved spikes to {csv_path}")

    png_path = f"{output_folder}/{base_name}.png"

    fig, (ax_raster, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=False,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    if len(times) == 0:
        ax_raster.text(0.5, 0.5, "NO SPIKES", ha='center', va='center', fontsize=14)
        ax_raster.set_title(f"{base_name}: No Spikes Recorded")
        ax_raster.set_xlabel("Time (ms)")
        ax_raster.set_ylabel("Neuron ID")
    else:
        ax_raster.plot(times, senders, '.', markersize=3)
        ax_raster.set_title(f"{base_name}: Raster Plot")
        ax_raster.set_ylabel("Neuron ID")

    if len(times) == 0:
        ax_hist.text(0.5, 0.5, "NO SPIKES", ha='center', va='center', fontsize=12)
        ax_hist.set_xlabel("Time (ms)")
        ax_hist.set_ylabel("Spike Count")
    else:
        ax_hist.hist(times, bins=bins, color="gray", edgecolor="black")
        time_range = times.max() - times.min()
        bin_width = time_range / bins if bins > 0 else 0

        ax_hist.set_title(f"Spike-Time Histogram (bin width ≈ {bin_width:.2f} ms)")
        ax_hist.set_xlabel("Time (ms)")
        ax_hist.set_ylabel("Spike Count")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[OK] Saved raster+time-histogram plot to {png_path}")


class MinMaxScaler:
    def __init__(self, feature_range=(0., 1.), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def fit(self, X):
        X = np.asarray(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1  # avoid division by zero
        return self

    def transform(self, X):
        X = np.asarray(X, float if self.copy else None)
        X_scaled = (X - self.data_min_) / self.data_range_

        min_, max_ = self.feature_range
        X_scaled = X_scaled * (max_ - min_) + min_

        if self.clip:
            X_scaled = np.clip(X_scaled, min_, max_)

        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def ensure_connectivity(
        src_population,
        tgt_population,
        filename,
        build_function,   # function that builds connectivity
):
    """
    If a connectivity file exists and matches current population sizes, load it.
    Otherwise build it and save it. Stale files (wrong population size) are
    deleted and rebuilt automatically.
    """

    output_folder = get_script_relative_folder("connections")
    path = Path(output_folder) / f"{filename}.npz"

    if path.exists():
        data = np.load(path, allow_pickle=True)
        src_idx = data["src_idx"]
        tgt_idx = data["tgt_idx"]
        saved_src_size = int(data["src_size"]) if "src_size" in data else None
        saved_tgt_size = int(data["tgt_size"]) if "tgt_size" in data else None
        stale = (
            (len(src_idx) > 0 and int(src_idx.max()) >= len(src_population)) or
            (len(tgt_idx) > 0 and int(tgt_idx.max()) >= len(tgt_population)) or
            (saved_src_size is not None and saved_src_size != len(src_population)) or
            (saved_tgt_size is not None and saved_tgt_size != len(tgt_population))
        )
        if stale:
            print(f"[WARN] Stale connectivity cache (population size changed) — rebuilding: {path}")
            path.unlink()
        else:
            print(f"[INFO] Loading connectivity from {path}")
            load_connectivity_recoverable(src_population, tgt_population, path, data)
            return

    print("[INFO] Connectivity file not found. Building manually...")
    build_function()
    save_connectivity_recoverable(
        src_population,
        tgt_population,
        path
    )


def load_connectivity_recoverable(src_population, tgt_population, filepath, data=None):
    if data is None:
        data = np.load(filepath, allow_pickle=True)

    src_idx = data["src_idx"]
    tgt_idx = data["tgt_idx"]
    weights = data["weights"]
    delays = data["delays"]
    syn_models = data["syn_models"]

    for s, t, w, d, syn in zip(src_idx, tgt_idx, weights, delays, syn_models):
        nest.Connect(
            src_population[s],
            tgt_population[t],
            syn_spec={
                "weight": float(w),
                "delay": float(max(d,nest.resolution)),
                "synapse_model": str(syn)
            }
        )

    print(f"[OK] Connectivity restored from {filepath}")


def save_connectivity_recoverable(src_population, tgt_population, path):
    src_ids = src_population.tolist()
    tgt_ids = tgt_population.tolist()

    src_index = {gid: i for i, gid in enumerate(src_ids)}
    tgt_index = {gid: j for j, gid in enumerate(tgt_ids)}

    conns = nest.GetConnections(src_population, tgt_population)
    keys = ["source", "target", "weight", "delay", "synapse_model"]
    conn_info = nest.GetStatus(conns, keys)

    n = len(conn_info)

    if n == 0:
        print("[save_connectivity_recoverable] No connections found.")
        return

    src_idx = np.zeros(n, dtype=np.int32)
    tgt_idx = np.zeros(n, dtype=np.int32)
    weights = np.zeros(n, dtype=np.float32)
    delays = np.zeros(n, dtype=np.float32)
    syn_models = np.empty(n, dtype=object)

    for i, (s, t, w, d, syn) in enumerate(conn_info):
        src_idx[i] = src_index[s]
        tgt_idx[i] = tgt_index[t]
        weights[i] = w
        delays[i] = d
        syn_models[i] = syn

    np.savez_compressed(
        path,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        weights=weights,
        delays=delays,
        syn_models=syn_models,
        src_size=np.int32(len(src_population)),
        tgt_size=np.int32(len(tgt_population)),
    )

    print(f"[OK] Recoverable connectivity saved to {path}")


def get_script_relative_folder(subfolder="connections"):
    base = Path(__file__).resolve().parent
    out = base / subfolder
    out.mkdir(exist_ok=True)
    return out