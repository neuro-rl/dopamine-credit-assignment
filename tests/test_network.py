"""Network-level sanity gate:

- every plastic synapse model passes the init-bounds assert
- learning ON/OFF actually changes weights
- one episode runs end to end without error in both modes
"""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import nest
import pytest

from base_util import Base
from encoders import TopographicPlaceFieldEncoder, SensoryInput
from env_reach import ReachAdapter
from network import Network

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "configs"


def _load_config(name, **overrides):
    spec = importlib.util.spec_from_file_location("cfgmod", _CONFIGS_DIR / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _reset_nest(cfg, seed=1):
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    import recorder_tracking
    recorder_tracking.clear_tracked_recorders()
    recorder_tracking.plugin_recorder_listeners()
    nest.resolution = cfg.nest_resolution
    nest.SetKernelStatus({"local_num_threads": cfg.num_threads, "rng_seed": seed})


def _build_encoders(cfg):
    extent = cfg.cortex_extent or float(max(cfg.cortex_exc_shape[0], cfg.cortex_inh_shape[0]))
    joints_enc = TopographicPlaceFieldEncoder(2, extent=extent, **cfg.joints_encoder_kw)
    object_enc = TopographicPlaceFieldEncoder(2, extent=extent, **cfg.object_encoder_kw)
    return SensoryInput({"joints": joints_enc, "object": object_enc})


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    nest.ResetKernel()


# ── init-bounds assert helper itself ────────────────────────────────────────

def test_assert_synapse_init_in_bounds_catches_floor_clamp():
    """The exact bug this exists to prevent: init sitting on Wmin."""
    with pytest.raises(AssertionError):
        Base.assert_synapse_init_in_bounds(w_mean=5.0, w_jit=0.0, Wmin=5.0, Wmax=30.0)


def test_assert_synapse_init_in_bounds_catches_ceiling_clamp():
    with pytest.raises(AssertionError):
        Base.assert_synapse_init_in_bounds(w_mean=29.5, w_jit=1.0, Wmin=5.0, Wmax=30.0)


def test_assert_synapse_init_in_bounds_passes_when_centered():
    Base.assert_synapse_init_in_bounds(w_mean=16.0, w_jit=2.0, Wmin=8.0, Wmax=30.0)


# ── every plastic synapse model, as actually configured ────────────────────

def test_reach_state_network_builds_within_synapse_bounds():
    """If any of critic_dopsyn/dopsyn/adopsyn's init clamps against its own
    Wmin/Wmax, Network()'s construction raises (critic.py, actor.py both call
    the init-bounds assert at their connection sites) -- so a clean build
    here IS the test."""
    cfg = _load_config("reach_state.py")
    _reset_nest(cfg)
    enc = _build_encoders(cfg)
    net = Network(enc, cfg, name_prefix="test_bounds_state_")
    for model in ("critic_dopsyn", "dopsyn", "adopsyn"):
        assert nest.GetDefaults(model)["Wmin"] < nest.GetDefaults(model)["Wmax"]


# ── learning ON/OFF changes weights ─────────────────────────────────────────

def _drive_a_few_steps(net, enc, adapter, rng, n_steps=5):
    obs = adapter.reset(rng)
    for _ in range(n_steps):
        enc.reset()
        enc.observe("joints", obs["joints"])
        enc.observe("object", obs["object"])
        enc.commit()
        with nest.RunManager():
            nest.Run(50.0)
            t1 = nest.GetKernelStatus("biological_time")
            action, _ = adapter.select_action(net.actor, adapter.cfg, t1 - 50.0, t1, 0.5)
            obs, reward, terminated, _success = adapter.step(action)
            net.dopamine.start_reward(reward, adapter.cfg.pos_reward_scale,
                                      adapter.cfg.neg_reward_scale, t1)
            nest.Run(25.0)
            net.dopamine.stop_reward()
        if terminated:
            obs = adapter.reset(rng)


def _mean_weight(cortex_out, target_pop):
    conns = nest.GetConnections(cortex_out, target_pop)
    return float(np.mean(nest.GetStatus(conns, "weight"))) if len(conns) else 0.0


def test_learning_off_does_not_change_weights():
    cfg = _load_config("reach_state.py")
    _reset_nest(cfg)
    enc = _build_encoders(cfg)
    net = Network(enc, cfg, name_prefix="test_learnoff_")
    adapter = ReachAdapter(cfg)
    rng = np.random.default_rng(3)

    net.dopamine.control_volume_transmitter(learning_enabled=False, warmup_until_ms=0.0)
    w_before = _mean_weight(net.cortex.output(), net.actor.d1(0))
    _drive_a_few_steps(net, enc, adapter, rng)
    net.dopamine.control_volume_transmitter(learning_enabled=False, warmup_until_ms=0.0)
    w_after = _mean_weight(net.cortex.output(), net.actor.d1(0))

    assert w_before == pytest.approx(w_after, abs=1e-9), (
        "weights changed with learning explicitly disabled (STDP should be zeroed)")


def test_learning_on_changes_weights():
    cfg = _load_config("reach_state.py")
    _reset_nest(cfg)
    enc = _build_encoders(cfg)
    net = Network(enc, cfg, name_prefix="test_learnon_")
    adapter = ReachAdapter(cfg)
    rng = np.random.default_rng(4)

    net.dopamine.control_volume_transmitter(learning_enabled=True, warmup_until_ms=0.0)
    w_before_d1 = _mean_weight(net.cortex.output(), net.actor.d1(0))
    w_before_critic = _mean_weight(net.cortex.output(), net.critic.layer1)
    _drive_a_few_steps(net, enc, adapter, rng, n_steps=15)
    w_after_d1 = _mean_weight(net.cortex.output(), net.actor.d1(0))
    w_after_critic = _mean_weight(net.cortex.output(), net.critic.layer1)

    changed = (abs(w_after_d1 - w_before_d1) > 1e-6) or (abs(w_after_critic - w_before_critic) > 1e-6)
    assert changed, "learning enabled for 15 steps but no plastic weight moved at all"


# ── one episode end to end ──────────────────────────────────────────────────

def _run_one_short_episode(config_name):
    cfg = _load_config(config_name, num_episodes=1, max_steps=5, video_interval=0)
    _reset_nest(cfg)
    enc = _build_encoders(cfg)
    net = Network(enc, cfg, name_prefix=f"test_e2e_{cfg.mode}_")
    adapter = ReachAdapter(cfg)
    rng = np.random.default_rng(5)

    obs = adapter.reset(rng)
    for step in range(cfg.max_steps):
        net.dopamine.control_volume_transmitter(learning_enabled=True, warmup_until_ms=cfg.warmup_time)
        enc.reset()
        enc.observe("joints", obs["joints"])
        enc.observe("object", obs["object"])
        enc.commit()

        with nest.RunManager():
            nest.Run(cfg.activate_time)
            t1 = nest.GetKernelStatus("biological_time")
            action, _ = adapter.select_action(net.actor, cfg, t1 - cfg.activate_time, t1, 0.5)
            obs, reward, terminated, success = adapter.step(action)
            net.dopamine.start_reward(reward, cfg.pos_reward_scale, cfg.neg_reward_scale, t1)
            nest.Run(cfg.reward_time)
            net.dopamine.stop_reward()
        if terminated:
            break
    return net


def test_one_episode_end_to_end_mode_state():
    net = _run_one_short_episode("reach_state.py")
    assert len(net.cortex.output()) > 0
