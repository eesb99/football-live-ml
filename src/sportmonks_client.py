from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests

from src.config import MissingSportMonksTokenError, Settings


SENSITIVE_KEYWORDS = {
    "api_token",
    "authorization",
    "bearer",
    "sportmonks_api_token",
    "token",
}


class SportMonksError(RuntimeError):
    """Base exception for SportMonks failures."""


class SportMonksAuthError(SportMonksError):
    """Raised when SportMonks rejects the configured token."""


class SportMonksRateLimitError(SportMonksError):
    """Raised when SportMonks reports rate-limit exhaustion."""


def redact_sportmonks_secret(value: Any, token: str = "") -> str:
    text = str(value or "")
    if token:
        text = text.replace(token, "[redacted]")
    text = re.sub(r"(?i)(api_token=)[^&\s'\"]+", r"\1[redacted]", text)
    text = re.sub(
        r"(?i)(sportmonks_api_token|api_token|authorization|token)"
        r"(['\"]?\s*[:=]\s*)['\"]?[^,'\"\s}&]+",
        r"\1\2[redacted]",
        text,
    )
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._\-]+", r"\1[redacted]", text)
    return text


def sanitize_sportmonks_payload(value: Any, token: str = "") -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in SENSITIVE_KEYWORDS:
                sanitized[key_text] = "[redacted]"
            else:
                sanitized[key_text] = sanitize_sportmonks_payload(item, token)
        return sanitized
    if isinstance(value, list):
        return [sanitize_sportmonks_payload(item, token) for item in value]
    if isinstance(value, tuple):
        return [sanitize_sportmonks_payload(item, token) for item in value]
    if isinstance(value, str):
        return redact_sportmonks_secret(value, token)
    return value


def sportmonks_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [record for record in data if isinstance(record, dict)]
        if isinstance(data, dict):
            return [data]
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    return []


class SportMonksClient:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
    ) -> None:
        if not settings.sportmonks_api_token:
            raise MissingSportMonksTokenError(
                "Missing SPORTMONKS_API_TOKEN. Add it to your environment or local .env file."
            )
        self.settings = settings
        self.api_token = settings.sportmonks_api_token
        self.session = session or requests.Session()

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.settings.sportmonks_base_url.rstrip('/')}{normalized_path}"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_params = dict(params or {})
        request_params["api_token"] = self.api_token
        try:
            response = self.session.get(
                self._url(path),
                params=request_params,
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            message = redact_sportmonks_secret(exc, self.api_token)
            raise SportMonksError(f"SportMonks request failed: {message}") from exc

        if response.status_code in {401, 403}:
            raise SportMonksAuthError(self._error_message(response))
        if response.status_code == 429:
            raise SportMonksRateLimitError(self._error_message(response))
        if response.status_code >= 400:
            raise SportMonksError(self._error_message(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise SportMonksError("SportMonks returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise SportMonksError("SportMonks returned an unexpected JSON shape.")
        return sanitize_sportmonks_payload(payload, self.api_token)

    def _error_message(self, response: requests.Response) -> str:
        body = redact_sportmonks_secret(response.text[:700], self.api_token)
        return f"SportMonks HTTP {response.status_code}: {body}"

    def get_leagues(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._get("/leagues", params=params)

    def search_leagues(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._get(f"/leagues/search/{quote(query, safe='')}", params=params)

    def get_seasons(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._get("/seasons", params=params)

    def get_fixtures(
        self,
        season_id: int,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_params["filters"] = f"fixtureSeasons:{int(season_id)}"
        return self._get("/fixtures", params=request_params)

    def get_fixture_detail(
        self,
        fixture_id: int,
        includes: str = "participants;league;season;scores;state;statistics;metadata",
    ) -> dict[str, Any]:
        return self._get(f"/fixtures/{int(fixture_id)}", params={"include": includes})

    def get_prediction_probabilities(self, fixture_id: int) -> dict[str, Any]:
        return self._get(f"/predictions/probabilities/fixtures/{int(fixture_id)}")

    def get_pre_match_odds(self, fixture_id: int) -> dict[str, Any]:
        return self._get(f"/odds/pre-match/fixtures/{int(fixture_id)}")

    def get_expected_goals(self, fixture_id: int) -> dict[str, Any]:
        del fixture_id
        return self._get("/expected/fixtures", params={"include": "fixture", "per_page": 25})

    def get_expected_goals_page(
        self,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = {"include": "fixture", "per_page": 25}
        request_params.update(params or {})
        return self._get("/expected/fixtures", params=request_params)

    def get_news(self, fixture_id: int) -> dict[str, Any]:
        del fixture_id
        return self._get("/news/pre-match/upcoming", params={"per_page": 25})

    def get_news_page(
        self,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_params = {"per_page": 25}
        request_params.update(params or {})
        return self._get("/news/pre-match/upcoming", params=request_params)

    def get_match_facts(self, fixture_id: int) -> dict[str, Any]:
        return self._get(f"/match-facts/{int(fixture_id)}")
