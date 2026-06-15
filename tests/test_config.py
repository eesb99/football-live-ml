from __future__ import annotations

import pytest

from src.config import (
    MissingApiKeyError,
    MissingSportMonksTokenError,
    SPORTMONKS_BASE_URL,
    load_settings,
)


def test_load_settings_allows_missing_api_keys_when_not_required(monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)

    settings = load_settings(require_api_key=False, env_file=None)

    assert settings.api_key == ""
    assert settings.sportmonks_api_token == ""
    assert settings.sportmonks_base_url == SPORTMONKS_BASE_URL


def test_load_settings_requires_api_football_key_when_requested(monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    with pytest.raises(MissingApiKeyError):
        load_settings(require_api_key=True, env_file=None)


def test_load_settings_loads_optional_sportmonks_token(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "api-football-key")
    monkeypatch.setenv("SPORTMONKS_API_TOKEN", "sportmonks-token")
    monkeypatch.setenv("SPORTMONKS_BASE_URL", "https://example.test/v3/football")

    settings = load_settings(env_file=None)

    assert settings.api_key == "api-football-key"
    assert settings.sportmonks_api_token == "sportmonks-token"
    assert settings.sportmonks_base_url == "https://example.test/v3/football"


def test_load_settings_can_require_sportmonks_token(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "api-football-key")
    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)

    with pytest.raises(MissingSportMonksTokenError):
        load_settings(require_sportmonks_token=True, env_file=None)
