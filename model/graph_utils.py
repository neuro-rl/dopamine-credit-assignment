import os

import nest
import numpy as np
import matplotlib.pyplot as plt


def save_spike_data(
        spike_recorders,
        base_name,
        output_folder,
        n_bins=50,
):
    """Save all spike recorder data into a single NPZ file."""
    os.makedirs(output_folder, exist_ok=True)

    data = {}
    labels = []

    for label, sr in spike_recorders:
        events = nest.GetStatus(sr, "events")[0]
        times = np.asarray(events["times"], dtype=np.float32)
        senders = np.asarray(events["senders"], dtype=np.int64)

        data[f"{label}_times"] = times
        data[f"{label}_senders"] = senders
        labels.append(label)

    out_path = os.path.join(output_folder, f"{base_name}_spikes.npz")
    np.savez_compressed(
        out_path,
        labels=np.array(labels),
        n_bins=n_bins,
        **data
    )

    print(f"[OK] Spike data saved to: {out_path}")
    return out_path


def load_spike_data(path):
    data = np.load(path, allow_pickle=True)
    labels = list(data["labels"])
    n_bins = int(data["n_bins"])

    events = {}
    for label in labels:
        events[label] = (
            data[f"{label}_times"],
            data[f"{label}_senders"]
        )

    return events, n_bins


def plot_spike_raster_pdf(
        spike_data_file,
        output_pdf,
        populations=None,              # technical names
        label_map=None,                # technical -> display label
        time_window=None,
        subtitle=None,
        base_width=10.0,
        row_height=1.2,
):
    """Publication-ready raster plot with technical->display label mapping."""

    data = np.load(spike_data_file, allow_pickle=True)

    all_events = {}
    for key in data.files:
        if key.endswith("_times"):
            label = key.replace("_times", "")
            times = data[f"{label}_times"]
            senders = data[f"{label}_senders"]
            all_events[label] = (times, senders)

    if populations is not None:
        all_events = {k: v for k, v in all_events.items() if k in populations}

    n_rows = len(all_events)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(base_width, row_height * n_rows),
        sharex=True,
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = [axes]

    for ax, (tech_name, (times, senders)) in zip(axes, all_events.items()):
        if time_window is not None:
            t0, t1 = time_window
            mask = (times >= t0) & (times <= t1)
            times = times[mask]
            senders = senders[mask]

        if len(times) > 0:
            ax.plot(times, senders, ".", markersize=2)
        else:
            ax.text(
                0.5, 0.5,
                "No spikes",
                transform=ax.transAxes,
                ha="center",
                va="center"
            )

        display_label = label_map[tech_name] if label_map and tech_name in label_map else tech_name
        ax.set_ylabel(display_label, rotation=0, labelpad=40)
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel("Time (ms)")

    if subtitle:
        fig.suptitle(subtitle, y=1.02)

    plt.savefig(output_pdf)
    plt.close(fig)

    print(f"[OK] Raster PDF saved to: {output_pdf}")


def plot_spike_bins_only_pdf(
        spike_data_file,
        output_pdf,
        populations=None,          # dict: {"internal_name": "Display Label"}
        subtitle=None,
        base_width=12.0,
):
    """
    populations:
        dict mapping population key in spike_data to label shown in plot.
        If None -> plot all populations using their original names.
    """

    all_events, n_bins = load_spike_data(spike_data_file)

    if populations is None:
        selected = list(all_events.items())
    else:
        selected = [
            (key, all_events[key])
            for key in populations.keys()
            if key in all_events
        ]

    n_recorders = len(selected)

    if n_recorders == 0:
        print("[WARN] No matching populations found.")
        return

    fig_height = 1.1 * n_recorders + 1.5
    fig, axes = plt.subplots(
        n_recorders, 1,
        figsize=(base_width, fig_height),
        sharex=True
    )

    if n_recorders == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    valid_times = [t for (_, (t, _)) in selected if len(t) > 0]

    if len(valid_times) == 0:
        min_t, max_t = 0.0, 1.0
    else:
        all_times = np.hstack(valid_times)
        min_t = float(all_times.min())
        max_t = float(all_times.max())
        if max_t == min_t:
            max_t = min_t + 1.0

    # MEASURED bug: when only one population has spikes in the window and
    # they're all within ~1ms of each other (e.g. a single isolated spike --
    # increasingly common once a policy collapses and most D1/D2 go quiet),
    # the old code still split that ~1ms into n_bins=50 razor-thin bins.
    # rate_hz = count/(n_neurons*bin_width_s) then blows up to tens of
    # thousands of Hz from a single spike in a ~0.02ms bin -- physically
    # impossible (these neurons' own t_ref caps them far below that) and
    # produces an axis with such an extreme dynamic range it crashed
    # matplotlib's freetype text renderer (FT_Render_Glyph raster overflow)
    # outright, corrupting/truncating the whole PDF. Floor the bin width
    # instead of forcing a fixed bin COUNT -- adapt n_bins down for narrow
    # windows rather than let bins shrink to near-zero width.
    MIN_BIN_WIDTH_MS = 5.0
    span_ms = max(max_t - min_t, 1e-9)
    effective_n_bins = max(1, min(n_bins, int(span_ms / MIN_BIN_WIDTH_MS)))
    bins = np.linspace(min_t, max_t, effective_n_bins + 1)
    bin_width_ms = bins[1] - bins[0]
    bin_width_s = bin_width_ms / 1000.0

    for i, ax in enumerate(axes):
        key, (times, senders) = selected[i]
        color = color_cycle[i % len(color_cycle)]
        label = key if populations is None else populations.get(key, key)

        if len(times) == 0:
            ax.text(
                0.5, 0.5, f"{label}: no spikes",
                ha="center", va="center"
            )
            ax.set_yticks([])
            continue

        counts, _ = np.histogram(times, bins=bins)
        n_neurons = len(np.unique(senders))
        rates_hz = counts / (n_neurons * bin_width_s)

        ax.bar(
            bins[:-1],
            rates_hz,
            width=bin_width_ms,
            align="edge",
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel(f"{label}\n(Hz)", rotation=0, labelpad=35, ha="center")

    axes[-1].set_xlabel("Time (ms)")

    if subtitle:
        fig.suptitle(subtitle)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close(fig)

    print(f"[OK] Spike-bin PDF (Hz per neuron) saved to: {output_pdf}")