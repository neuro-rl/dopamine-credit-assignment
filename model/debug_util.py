"""debug_util.py — spike PDFs (raw spikes + binned rate histograms) per
episode. A finer-grained diagnostic than viz.py's episode video and
summary plots: what did each population's spikes actually look like
during a specific episode.
"""
import graph_utils as gu


def _safe_plot(fn, **kwargs):
    # These PDFs are diagnostic-only and never read by training/eval/
    # checkpoint logic, so a rendering failure (matplotlib/freetype can
    # occasionally raise on glyph rasterization) shouldn't take down a
    # long training run. Swallow and warn instead.
    try:
        fn(**kwargs)
    except Exception as e:
        print(f"[print_epoch_spikes] WARNING: {kwargs.get('output_pdf')} failed to render ({e!r}), skipping")


def print_epoch_spikes(episode, net, output_folder):
    """Save spike data to a compressed NPZ and generate PDFs for every
    population in the network (inputs, cortex, dopamine, critic, D1/D2).

    Captures whatever's accumulated in the always-on diagnostic recorders
    since the last clear_rates() call (main.py clears every 10 steps) --
    a recent-activity snapshot, not a full-episode dump, matching the
    reference project's own behaviour.

    Produces:
      cortex_spike_bins_<ep>.pdf   : inputs + cortex, binned rate (Hz)
      cortex_raster_<ep>.pdf       : inputs + cortex, raster
      da_critic_spike_bins_<ep>.pdf: dopamine + critic L1/L2 + reward gens
      da_critic_raster_<ep>.pdf    : same, raster
      actor_d1d2_<ep>.pdf          : D1/D2 per action channel, binned rate
      inputs_raster_<ep>.pdf       : joints/object encoder outputs, raster
    """
    spike_list = [
        ('joints_input', net._joints_diag_rec),
    ]
    if net._object_diag_rec is not None:
        spike_list.append(('object_input', net._object_diag_rec))
    if getattr(net, "_object_right_diag_rec", None) is not None:
        spike_list.append(('object_right_input', net._object_right_diag_rec))
    spike_list += [
        ('cortex_exc',      net._cortex_diag_rec),
        ('cortex_inh',      net._cortex_inh_diag_rec),
        ('da',              net.dopamine.get_recorder("da")),
        ('critic_features', net.critic.get_recorder("features")),
        ('critic_value',    net.critic.get_recorder("value")),
        ('reward_pos',      net.dopamine.get_recorder("pos_reward")),
        ('reward_neg',      net.dopamine.get_recorder("neg_reward")),
    ]
    # With context_gated_da=True, also plot the per-zone DA pools that
    # drive dopsyn_z{i}/adopsyn_z{i} -- 'da' above only covers the global,
    # critic-facing pool. Skipped when zoning is off (da_zone_populations
    # is None).
    if net.dopamine.da_zone_populations is not None:
        for _z in range(net.dopamine.n_baseline_zones):
            spike_list.append((f'da_z{_z}', net.dopamine.get_recorder(f"da_z{_z}")))
    for i, ch in enumerate(net.actor.channels):
        spike_list.append((f'd1_ch{i}', net.actor.get_recorder(i, "d1")))
        spike_list.append((f'd2_ch{i}', net.actor.get_recorder(i, "d2")))

    data_file = gu.save_spike_data(
        spike_list,
        base_name=f"all_{episode}",
        output_folder=output_folder,
    )

    # ── Inputs + cortex ──────────────────────────────────────────────────
    input_pops = {"joints_input": "Joints (Parrots)"}
    if net._object_diag_rec is not None:
        input_pops["object_input"] = "Object (Parrots)"
    if getattr(net, "_object_right_diag_rec", None) is not None:
        # "(Pool)" not "(Parrots)": EventEncoder.output() is the
        # retinotopic pool of real spiking neurons, not a parrot-relay
        # layer, unlike this project's other encoder types.
        input_pops["object_right_input"] = "Object Right (Pool)"
    cortex_pops = dict(input_pops)
    cortex_pops["cortex_exc"] = "Cortex (Exc)"
    cortex_pops["cortex_inh"] = "Cortex (Inh)"

    _safe_plot(gu.plot_spike_bins_only_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/cortex_spike_bins_{episode}.pdf",
        populations=cortex_pops,
        subtitle=f"Episode {episode} — Cortex & Input Activity",
    )
    _safe_plot(gu.plot_spike_raster_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/cortex_raster_{episode}.pdf",
        populations=list(cortex_pops.keys()),
        label_map=cortex_pops,
        subtitle=f"Episode {episode} — Cortex & Input Raster",
    )
    _safe_plot(gu.plot_spike_raster_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/inputs_raster_{episode}.pdf",
        populations=list(input_pops.keys()),
        label_map=input_pops,
        subtitle=f"Episode {episode} — Input Spike Raster",
    )

    # ── Dopamine + critic ─────────────────────────────────────────────────
    da_critic_pops = {
        "da":              "Dopamine",
        "critic_features": "Critic (L1)",
        "critic_value":    "Critic (L2)",
        "reward_pos":      "Reward (+)",
        "reward_neg":      "Reward (−)",
    }
    if net.dopamine.da_zone_populations is not None:
        for _z in range(net.dopamine.n_baseline_zones):
            da_critic_pops[f"da_z{_z}"] = f"Dopamine (zone {_z})"
    _safe_plot(gu.plot_spike_bins_only_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/da_critic_spike_bins_{episode}.pdf",
        populations=da_critic_pops,
        subtitle=f"Episode {episode} — Dopamine & Critic Activity",
    )
    _safe_plot(gu.plot_spike_raster_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/da_critic_raster_{episode}.pdf",
        populations=list(da_critic_pops.keys()),
        label_map=da_critic_pops,
        subtitle=f"Episode {episode} — Dopamine & Critic Raster",
    )

    # ── Actor D1/D2 ───────────────────────────────────────────────────────
    actor_dict = {}
    for i in range(len(net.actor.channels)):
        actor_dict[f"d1_ch{i}"] = f"D1 ch{i} (Go)"
        actor_dict[f"d2_ch{i}"] = f"D2 ch{i} (NoGo)"
    _safe_plot(gu.plot_spike_bins_only_pdf,
        spike_data_file=data_file,
        output_pdf=f"{output_folder}/actor_d1d2_{episode}.pdf",
        populations=actor_dict,
        subtitle=f"Episode {episode} — Striatum D1/D2 Activity",
    )

    print(f"[Spikes] Episode {episode}: NPZ + PDFs saved to {output_folder}/")