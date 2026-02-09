"""Configuration management with YAML loading and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

_instance: "Config | None" = None


class Config:
    """Application configuration backed by a YAML file.

    Supports environment variable overrides using the naming convention
    ``FRAUD_<SECTION>_<KEY>`` (upper-case, underscore-separated).  For example,
    ``FRAUD_API_PORT=9000`` overrides ``config["api"]["port"]``.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                self._data: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            self._data = {}
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    # Environment variable overrides
    # ------------------------------------------------------------------

    def _apply_env_overrides(self) -> None:
        """Override config values with ``FRAUD_<SECTION>_<KEY>`` env vars."""
        prefix = "FRAUD_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, name = parts
            if section in self._data:
                self._data[section][name] = self._cast(value, section, name)

    def _cast(self, value: str, section: str, name: str) -> Any:
        """Attempt to cast *value* to the same type as the current setting."""
        current = self._data.get(section, {}).get(name)
        if current is None:
            return value
        target_type = type(current)
        try:
            if target_type is bool:
                return value.lower() in ("1", "true", "yes")
            return target_type(value)
        except (ValueError, TypeError):
            return value

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Return a config value.

        ``get("api")`` returns the whole *api* section dict.
        ``get("api", "port")`` returns a single value.
        """
        sect = self._data.get(section, default if key is None else {})
        if key is None:
            return sect
        return sect.get(key, default) if isinstance(sect, dict) else default

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self._data[section]

    @property
    def data(self) -> dict[str, Any]:
        return self._data


# ------------------------------------------------------------------
# Singleton access
# ------------------------------------------------------------------


def get_config(config_path: str | Path | None = None) -> Config:
    """Return the global :class:`Config` singleton, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = Config(config_path)
    return _instance


def reset_config() -> None:
    """Reset the singleton (useful for testing)."""
    global _instance
    _instance = None
