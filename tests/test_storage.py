from __future__ import annotations

from src.storage import (
    ensure_sportmonks_dirs,
    load_api_prediction_cache,
    load_latest_sportmonks_fixtures_cache,
    load_latest_sportmonks_odds_cache,
    load_latest_sportmonks_audit,
    save_sportmonks_fixtures_cache,
    save_sportmonks_odds_cache,
    save_api_prediction_cache,
    save_sportmonks_audit,
    sanitize_api_prediction_error,
)


def test_api_prediction_cache_round_trips_normalized_fields(tmp_path):
    prediction = {
        "available": True,
        "status": "available",
        "endpoint": "/predictions?fixture=9001",
        "raw_response_count": 1,
        "home_probability": 0.45,
        "draw_probability": 0.35,
        "away_probability": 0.20,
        "advice": "Double chance : Alpha or draw",
        "winner_name": "Alpha",
        "unexpected_raw_payload": {"secret": "not persisted"},
    }

    path = save_api_prediction_cache(prediction, 9001, cache_dir=tmp_path)
    cached = load_api_prediction_cache(9001, cache_dir=tmp_path)

    assert path == tmp_path / "9001.json"
    assert cached is not None
    assert cached["fixture_id"] == 9001
    assert cached["available"] is True
    assert cached["home_probability"] == 0.45
    assert cached["winner_name"] == "Alpha"
    assert "captured_at" in cached
    assert "unexpected_raw_payload" not in cached


def test_api_prediction_cache_sanitizes_error_text(tmp_path):
    prediction = {
        "available": False,
        "status": "error",
        "endpoint": "/predictions?fixture=9001",
        "last_error": "API request failed x-apisports-key: abc123",
    }

    save_api_prediction_cache(prediction, 9001, cache_dir=tmp_path)
    cached = load_api_prediction_cache(9001, cache_dir=tmp_path)

    assert cached is not None
    assert "abc123" not in cached["last_error"]
    assert "[redacted]" in cached["last_error"]


def test_sanitize_api_prediction_error_truncates_long_values():
    sanitized = sanitize_api_prediction_error("x" * 700)

    assert len(sanitized) == 500


def test_sportmonks_audit_round_trips_from_generated_directory(tmp_path):
    audit_dir = tmp_path / "sportmonks" / "audits"
    audit = {
        "provider": "sportmonks",
        "summary": {"accessible_categories": ["leagues"]},
    }

    path = save_sportmonks_audit(audit, audit_dir=audit_dir)
    loaded = load_latest_sportmonks_audit(audit_dir=audit_dir)

    assert path.parent == audit_dir
    assert path.name.endswith("_sportmonks_access_audit.json")
    assert loaded is not None
    assert loaded["summary"]["accessible_categories"] == ["leagues"]
    assert loaded["audit_file"] == str(path)


def test_ensure_sportmonks_dirs_returns_expected_directories(monkeypatch, tmp_path):
    import src.storage as storage

    custom_dirs = [
        tmp_path / "audits",
        tmp_path / "fixtures",
        tmp_path / "odds",
    ]
    monkeypatch.setattr(storage, "SPORTMONKS_GENERATED_DIRS", custom_dirs)

    created = ensure_sportmonks_dirs()

    assert created == custom_dirs
    assert all(path.exists() for path in custom_dirs)


def test_sportmonks_fixture_cache_round_trips_without_token(tmp_path):
    payload = {
        "data": [{"id": 19606945, "name": "Alpha vs Beta"}],
        "api_token": "sportmonks-secret",
        "message": "called with api_token=sportmonks-secret",
    }

    path = save_sportmonks_fixtures_cache(
        payload,
        26618,
        cache_dir=tmp_path,
        token="sportmonks-secret",
    )
    loaded = load_latest_sportmonks_fixtures_cache(26618, cache_dir=tmp_path)
    saved_text = path.read_text()

    assert loaded is not None
    assert loaded["cache_key"] == "26618"
    assert loaded["payload"]["data"][0]["id"] == 19606945
    assert "sportmonks-secret" not in saved_text
    assert "api_token=sportmonks-secret" not in saved_text


def test_sportmonks_odds_cache_round_trips_without_token(tmp_path):
    payload = {
        "data": [{"fixture_id": 19609127, "market_id": 1, "label": "Home"}],
        "message": "called with api_token=sportmonks-secret",
    }

    path = save_sportmonks_odds_cache(
        payload,
        19609127,
        cache_dir=tmp_path,
        token="sportmonks-secret",
    )
    loaded = load_latest_sportmonks_odds_cache(19609127, cache_dir=tmp_path)
    saved_text = path.read_text()

    assert loaded is not None
    assert loaded["cache_key"] == "19609127"
    assert loaded["payload"]["data"][0]["fixture_id"] == 19609127
    assert "sportmonks-secret" not in saved_text
