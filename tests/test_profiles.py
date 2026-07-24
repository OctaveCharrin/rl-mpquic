"""Tests for role/use-case profiles and the profile selector (Phase 3)."""

import os

import yaml

from src.train.config import _deep_merge, load_config

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT = os.path.join(_HERE, "configs", "default.yaml")


def test_deep_merge_override_wins_and_recurses():
    base = {"reward": {"b_latency": 0.5, "a_quality": 1.0}, "run": {"seed": 1}}
    override = {"reward": {"b_latency": 0.2}, "deadline_ms": 400}
    merged = _deep_merge(base, override)
    assert merged["reward"]["b_latency"] == 0.2      # override wins
    assert merged["reward"]["a_quality"] == 1.0      # untouched key preserved
    assert merged["run"]["seed"] == 1                # sibling block preserved
    assert merged["deadline_ms"] == 400


def test_no_profile_is_legacy_byte_identical():
    cfg = load_config(_DEFAULT)
    # Default (no profile) keeps the provisional weights + code-default deadline.
    assert (cfg.weights.b_latency, cfg.weights.c_jitter, cfg.weights.d_loss) == (0.5, 0.5, 1.0)
    assert cfg.deadline_ms == 180.0


def test_profile_flag_applies_reward_and_deadline():
    interactive = load_config(_DEFAULT, profile="interactive")
    presenter = load_config(_DEFAULT, profile="presenter")
    passive = load_config(_DEFAULT, profile="passive")

    assert interactive.deadline_ms == 180.0
    assert presenter.deadline_ms == 400.0
    assert passive.deadline_ms == 800.0

    # Presenter downweights delay vs interactive; passive relaxes loss.
    assert presenter.weights.b_latency < interactive.weights.b_latency
    assert presenter.weights.latency_norm_ms == 400.0
    assert passive.weights.d_loss < 1.0

    # a_quality stays the numeraire in every profile.
    for c in (interactive, presenter, passive):
        assert c.weights.a_quality == 1.0
    # Topology comes from the base config (profile is reward+deadline only).
    assert len(presenter.paths) == len(load_config(_DEFAULT).paths)


def test_in_file_profile_key(tmp_path):
    base = {
        "topology": {"paths": [{"rate": "8Mbps", "delay": "10ms"}]},
        "profile": "passive",
        "reward": {"b_latency": 0.5, "d_loss": 1.0},  # should be overridden by profile
    }
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(base))
    # Resolve the profile from the repo configs dir by symlinking profiles/.
    os.symlink(os.path.join(_HERE, "configs", "profiles"), tmp_path / "profiles")
    cfg = load_config(str(p))
    assert cfg.deadline_ms == 800.0
    assert cfg.weights.d_loss == 0.7  # profile won over the in-file reward block


def test_explicit_profile_arg_overrides_in_file_key(tmp_path):
    base = {"topology": {"paths": [{"rate": "8Mbps", "delay": "10ms"}]}, "profile": "passive"}
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(base))
    os.symlink(os.path.join(_HERE, "configs", "profiles"), tmp_path / "profiles")
    cfg = load_config(str(p), profile="presenter")  # CLI wins
    assert cfg.deadline_ms == 400.0
