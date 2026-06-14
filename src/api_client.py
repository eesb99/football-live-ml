from __future__ import annotations

from typing import Any

import requests

from src.config import Settings


class ApiFootballError(RuntimeError):
    """Base exception for API-Football failures."""


class ApiFootballRateLimitError(ApiFootballError):
    """Raised when API-Football reports quota or rate-limit exhaustion."""


class ApiFootballClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": settings.api_key})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ApiFootballError(f"API request failed: {exc}") from exc

        if response.status_code in {429, 499}:
            raise ApiFootballRateLimitError(
                "API-Football quota or rate limit reached. Try again later."
            )

        if response.status_code >= 400:
            raise ApiFootballError(
                f"API request failed with HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiFootballError("API returned invalid JSON.") from exc

        self._raise_for_payload_errors(payload)
        return payload

    @staticmethod
    def _raise_for_payload_errors(payload: dict[str, Any]) -> None:
        errors = payload.get("errors")
        if not errors:
            return

        error_text = str(errors).lower()
        if any(token in error_text for token in ("rate", "quota", "limit", "requests")):
            raise ApiFootballRateLimitError(
                f"API-Football quota or rate-limit response: {errors}"
            )
        raise ApiFootballError(f"API-Football error response: {errors}")

    def get_live_fixtures(self) -> list[dict[str, Any]]:
        payload = self._get("/fixtures", params={"live": "all"})
        return payload.get("response") or []

    def get_fixtures(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self._get("/fixtures", params=params)
        return payload.get("response") or []

    def get_fixture_events(self, fixture_id: int) -> list[dict[str, Any]]:
        payload = self._get("/fixtures/events", params={"fixture": fixture_id})
        return payload.get("response") or []

    def get_fixture_statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        payload = self._get("/fixtures/statistics", params={"fixture": fixture_id})
        return payload.get("response") or []
