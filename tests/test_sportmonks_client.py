from __future__ import annotations

import pytest

from src.config import Settings
from src.sportmonks_client import (
    SportMonksAuthError,
    SportMonksClient,
    redact_sportmonks_secret,
    sanitize_sportmonks_payload,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": []}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def settings(token="sportmonks-secret") -> Settings:
    return Settings(
        sportmonks_api_token=token,
        sportmonks_base_url="https://api.sportmonks.test/v3/football",
    )


def test_client_sends_token_as_query_param_and_not_header():
    session = FakeSession(FakeResponse(payload={"data": [{"id": 1, "name": "World Cup"}]}))
    client = SportMonksClient(settings(), session=session)

    payload = client.get_leagues({"per_page": 10})

    assert payload["data"][0]["name"] == "World Cup"
    assert session.calls[0]["url"] == "https://api.sportmonks.test/v3/football/leagues"
    assert session.calls[0]["params"]["api_token"] == "sportmonks-secret"
    assert session.calls[0]["params"]["per_page"] == 10


def test_client_filters_fixtures_by_season():
    session = FakeSession(FakeResponse(payload={"data": []}))
    client = SportMonksClient(settings(), session=session)

    client.get_fixtures(26618)

    assert (
        session.calls[0]["url"]
        == "https://api.sportmonks.test/v3/football/fixtures"
    )
    assert session.calls[0]["params"]["filters"] == "fixtureSeasons:26618"


def test_client_uses_documented_prediction_fixture_endpoint():
    session = FakeSession(FakeResponse(payload={"data": []}))
    client = SportMonksClient(settings(), session=session)

    client.get_prediction_probabilities(19606945)

    assert (
        session.calls[0]["url"]
        == "https://api.sportmonks.test/v3/football/predictions/probabilities/fixtures/19606945"
    )


def test_client_uses_documented_pre_match_news_endpoint():
    session = FakeSession(FakeResponse(payload={"data": []}))
    client = SportMonksClient(settings(), session=session)

    client.get_news(19606945)

    assert (
        session.calls[0]["url"]
        == "https://api.sportmonks.test/v3/football/news/pre-match/upcoming"
    )


def test_client_auth_error_redacts_token_from_message():
    session = FakeSession(
        FakeResponse(
            status_code=401,
            text="Invalid api_token=sportmonks-secret token: sportmonks-secret",
        )
    )
    client = SportMonksClient(settings(), session=session)

    with pytest.raises(SportMonksAuthError) as exc_info:
        client.get_leagues()

    message = str(exc_info.value)
    assert "sportmonks-secret" not in message
    assert "[redacted]" in message


def test_redaction_sanitizes_nested_payloads():
    sanitized = sanitize_sportmonks_payload(
        {
            "api_token": "sportmonks-secret",
            "message": "failed with api_token=sportmonks-secret",
            "nested": [{"authorization": "Bearer sportmonks-secret"}],
        },
        token="sportmonks-secret",
    )

    dumped = str(sanitized)
    assert "sportmonks-secret" not in dumped
    assert sanitized["api_token"] == "[redacted]"


def test_redact_sportmonks_secret_handles_query_and_key_value_forms():
    text = "url?api_token=sportmonks-secret token: sportmonks-secret"

    redacted = redact_sportmonks_secret(text, token="sportmonks-secret")

    assert "sportmonks-secret" not in redacted
    assert "[redacted]" in redacted
