"""Tests for src/utils/config.py."""

import os
import textwrap
from pathlib import Path

import pytest

from src.utils.config import Config, get_config, reset_config


@pytest.fixture(autouse=True)
def _reset():
    """Reset the singleton between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent("""\
            data:
              raw_path: data/raw/creditcard.csv
              sample_path: data/sample/
            model:
              registry_path: models/
              default_model: xgboost
            api:
              host: 0.0.0.0
              port: 8000
            database:
              path: data/predictions.db
            streaming:
              rate: 10
              fraud_rate: 0.02
        """)
    )
    return p


# ------------------------------------------------------------------
# Basic loading
# ------------------------------------------------------------------


def test_load_yaml(config_file: Path):
    cfg = Config(config_file)
    assert cfg.get("api", "port") == 8000
    assert cfg.get("data", "raw_path") == "data/raw/creditcard.csv"


def test_get_section(config_file: Path):
    cfg = Config(config_file)
    api = cfg.get("api")
    assert isinstance(api, dict)
    assert api["host"] == "0.0.0.0"


def test_get_missing_key_returns_default(config_file: Path):
    cfg = Config(config_file)
    assert cfg.get("api", "missing_key", "fallback") == "fallback"


def test_get_missing_section_returns_default(config_file: Path):
    cfg = Config(config_file)
    assert cfg.get("nonexistent", default="nope") == "nope"


def test_getitem(config_file: Path):
    cfg = Config(config_file)
    assert cfg["model"]["default_model"] == "xgboost"


def test_getitem_missing_section_raises(config_file: Path):
    cfg = Config(config_file)
    with pytest.raises(KeyError):
        _ = cfg["does_not_exist"]


def test_data_property(config_file: Path):
    cfg = Config(config_file)
    assert isinstance(cfg.data, dict)
    assert "streaming" in cfg.data


# ------------------------------------------------------------------
# Environment variable overrides
# ------------------------------------------------------------------


def test_env_override_int(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAUD_API_PORT", "9000")
    cfg = Config(config_file)
    assert cfg.get("api", "port") == 9000


def test_env_override_float(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAUD_STREAMING_RATE", "50")
    cfg = Config(config_file)
    assert cfg.get("streaming", "rate") == 50


def test_env_override_string(config_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRAUD_MODEL_DEFAULT_MODEL", "random_forest")
    cfg = Config(config_file)

    # NOTE: env key split on first "_" after prefix → section="model", name="default_model"
    # The current split("_", 1) produces section="model", name="default_model" — correct.
    assert cfg.get("model", "default_model") == "random_forest"


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------


def test_singleton(config_file: Path):
    c1 = get_config(config_file)
    c2 = get_config()
    assert c1 is c2


def test_reset_singleton(config_file: Path):
    c1 = get_config(config_file)
    reset_config()
    c2 = get_config(config_file)
    assert c1 is not c2


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_missing_config_file(tmp_path: Path):
    cfg = Config(tmp_path / "does_not_exist.yaml")
    assert cfg.data == {}


def test_empty_config_file(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    cfg = Config(p)
    assert cfg.data == {}
