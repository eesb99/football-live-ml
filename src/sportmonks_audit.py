from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.config import Settings, load_settings
from src.sportmonks_client import (
    SportMonksClient,
    SportMonksError,
    sanitize_sportmonks_payload,
    sportmonks_records,
)
from src.storage import save_sportmonks_audit


WORLD_CUP_YEAR = 2026
WORLD_CUP_SEARCH_QUERY = "World Cup"


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for item in value.values():
            found.extend(_iter_dicts(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_iter_dicts(item))
    return found


def response_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("subscription", "rate_limit", "timezone", "pagination"):
        if key in payload:
            metadata[key] = payload[key]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        for key in ("subscription", "rate_limit", "pagination"):
            if key in meta:
                metadata[key] = meta[key]
    return metadata


def compact_payload(payload: dict[str, Any], max_records: int = 5) -> dict[str, Any]:
    records = sportmonks_records(payload)
    return {
        "record_count": len(records),
        "sample": records[:max_records],
        "metadata": response_metadata(payload),
    }


def not_checked(reason: str) -> dict[str, Any]:
    return {
        "status": "not_checked",
        "available": False,
        "record_count": 0,
        "reason": reason,
    }


def run_check(
    name: str,
    callback: Callable[[], dict[str, Any]],
    token: str,
    require_records: bool = True,
) -> dict[str, Any]:
    try:
        payload = callback()
    except SportMonksError as exc:
        return {
            "status": "error",
            "available": False,
            "record_count": 0,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive audit boundary
        return {
            "status": "error",
            "available": False,
            "record_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    compact = compact_payload(payload)
    record_count = int(compact["record_count"])
    available = record_count > 0 or not require_records
    return {
        "status": "available" if available else "empty",
        "available": available,
        "record_count": record_count,
        "endpoint": name,
        **sanitize_sportmonks_payload(compact, token),
    }


def league_candidates(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in payloads:
        for record in _iter_dicts(payload):
            name = str(record.get("name") or "")
            league_id = record.get("id")
            if "world cup" not in name.casefold() or league_id is None:
                continue
            try:
                numeric_id = int(league_id)
            except (TypeError, ValueError):
                continue
            if numeric_id in seen:
                continue
            seen.add(numeric_id)
            candidates.append(
                {
                    "id": numeric_id,
                    "name": name,
                    "type": record.get("type"),
                    "sub_type": record.get("sub_type"),
                    "country_id": record.get("country_id"),
                }
            )
    return candidates


def season_candidates(
    payloads: list[dict[str, Any]],
    year: int = WORLD_CUP_YEAR,
) -> list[dict[str, Any]]:
    year_text = str(year)
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in payloads:
        for record in _iter_dicts(payload):
            record_id = record.get("id")
            if record_id is None:
                continue
            direct_text = " ".join(
                str(record.get(key) or "")
                for key in ("name", "year", "starting_at", "ending_at")
            ).casefold()
            looks_like_season = (
                "league_id" in record
                or "finished" in record
                or "starting_at" in record
                or "ending_at" in record
                or str(record.get("name") or record.get("year") or "") == year_text
            )
            if year_text not in direct_text or not looks_like_season:
                continue
            try:
                numeric_id = int(record_id)
            except (TypeError, ValueError):
                continue
            if numeric_id in seen:
                continue
            seen.add(numeric_id)
            candidates.append(
                {
                    "id": numeric_id,
                    "name": str(record.get("name") or record.get("year") or year_text),
                    "league_id": record.get("league_id"),
                    "starting_at": record.get("starting_at"),
                    "ending_at": record.get("ending_at"),
                }
            )
    return candidates


def prioritize_world_cup_seasons(
    seasons: list[dict[str, Any]],
    leagues: list[dict[str, Any]],
    year: int = WORLD_CUP_YEAR,
) -> list[dict[str, Any]]:
    final_world_cup_league_ids = {
        int(league["id"])
        for league in leagues
        if "world cup" in str(league.get("name") or "").casefold()
        and "qualifier" not in str(league.get("name") or "").casefold()
        and "qualifying" not in str(league.get("name") or "").casefold()
    }

    def sort_key(season: dict[str, Any]) -> tuple[int, int, str]:
        try:
            league_id = int(season.get("league_id"))
        except (TypeError, ValueError):
            league_id = -1
        final_competition_rank = 0 if league_id in final_world_cup_league_ids else 1
        exact_year_rank = 0 if str(season.get("name") or "") == str(year) else 1
        return (final_competition_rank, exact_year_rank, str(season.get("starting_at") or ""))

    return sorted(seasons, key=sort_key)


def first_fixture_id(check: dict[str, Any]) -> int | None:
    for record in check.get("sample") or []:
        fixture_id = record.get("id")
        try:
            return int(fixture_id)
        except (TypeError, ValueError):
            continue
    return None


def _search_world_cup(client: SportMonksClient) -> dict[str, Any]:
    try:
        return client.search_leagues(
            WORLD_CUP_SEARCH_QUERY,
            {"include": "seasons", "per_page": 50},
        )
    except SportMonksError:
        return client.search_leagues(WORLD_CUP_SEARCH_QUERY, {"per_page": 50})


def _first_available_fixture_check(
    client: SportMonksClient,
    season_ids: list[int],
    token: str,
) -> tuple[dict[str, Any], int | None]:
    if not season_ids:
        return not_checked("No SportMonks World Cup 2026 season ID was discovered."), None

    last_check: dict[str, Any] = not_checked("No season checked.")
    for season_id in season_ids[:5]:
        check = run_check(
            f"fixtures?filters=fixtureSeasons:{season_id}",
            lambda season_id=season_id: client.get_fixtures(
                season_id,
                {
                    "include": "participants;league;season;scores;state;venue",
                    "per_page": 100,
                },
            ),
            token,
        )
        check["season_id"] = season_id
        last_check = check
        fixture_id = first_fixture_id(check)
        if check["available"] and fixture_id is not None:
            return check, fixture_id
    return last_check, first_fixture_id(last_check)


def audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    checks = audit.get("checks") or {}
    accessible_categories = [
        name for name, check in checks.items() if (check or {}).get("available")
    ]
    error_categories = [
        name for name, check in checks.items() if (check or {}).get("status") == "error"
    ]
    metadata: dict[str, Any] = {}
    for check in checks.values():
        check_metadata = (check or {}).get("metadata") or {}
        for key in ("subscription", "rate_limit", "pagination"):
            if key in check_metadata and key not in metadata:
                metadata[key] = check_metadata[key]
    return {
        "accessible_categories": accessible_categories,
        "error_categories": error_categories,
        "world_cup_league_ids": [
            candidate["id"] for candidate in audit.get("world_cup_league_candidates", [])
        ],
        "world_cup_2026_season_ids": [
            candidate["id"] for candidate in audit.get("world_cup_2026_season_candidates", [])
        ],
        "selected_fixture_id": audit.get("selected_fixture_id"),
        "metadata": metadata,
    }


def run_sportmonks_access_audit(
    settings: Settings | None = None,
    client: SportMonksClient | None = None,
    audit_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    settings = settings or load_settings(
        require_api_key=False,
        require_sportmonks_token=True,
    )
    token = settings.sportmonks_api_token
    client = client or SportMonksClient(settings)
    captured_at = datetime.now().isoformat(timespec="seconds")

    checks: dict[str, dict[str, Any]] = {}
    checks["leagues"] = run_check(
        "leagues",
        lambda: client.get_leagues({"per_page": 100}),
        token,
    )
    checks["world_cup_search"] = run_check(
        "leagues/search/World Cup",
        lambda: _search_world_cup(client),
        token,
    )
    checks["seasons"] = run_check(
        "seasons",
        lambda: client.get_seasons({"include": "league", "per_page": 100}),
        token,
    )

    search_payloads = [
        {"data": checks["world_cup_search"].get("sample") or []},
        {"data": checks["leagues"].get("sample") or []},
    ]
    season_payloads = [
        {"data": checks["world_cup_search"].get("sample") or []},
        {"data": checks["seasons"].get("sample") or []},
    ]
    leagues = league_candidates(search_payloads)
    seasons = prioritize_world_cup_seasons(
        season_candidates(season_payloads),
        leagues,
    )
    season_ids = [int(candidate["id"]) for candidate in seasons]

    fixtures_check, fixture_id = _first_available_fixture_check(client, season_ids, token)
    checks["world_cup_2026_fixtures"] = fixtures_check

    if fixture_id is None:
        checks["fixture_detail"] = not_checked("No fixture ID was available.")
        checks["sportmonks_predictions"] = not_checked("No fixture ID was available.")
        checks["pre_match_odds"] = not_checked("No fixture ID was available.")
        checks["expected_goals"] = not_checked("No fixture ID was available.")
        checks["news"] = not_checked("No fixture ID was available.")
        checks["match_facts"] = not_checked("No fixture ID was available.")
    else:
        checks["fixture_detail"] = run_check(
            f"fixtures/{fixture_id}",
            lambda: client.get_fixture_detail(fixture_id),
            token,
            require_records=False,
        )
        checks["sportmonks_predictions"] = run_check(
            f"predictions/probabilities/fixture/{fixture_id}",
            lambda: client.get_prediction_probabilities(fixture_id),
            token,
        )
        checks["pre_match_odds"] = run_check(
            f"odds/pre-match/fixtures/{fixture_id}",
            lambda: client.get_pre_match_odds(fixture_id),
            token,
        )
        checks["expected_goals"] = run_check(
            f"expected/fixtures/{fixture_id}",
            lambda: client.get_expected_goals(fixture_id),
            token,
        )
        checks["news"] = run_check(
            f"news/pre-match/fixtures/{fixture_id}",
            lambda: client.get_news(fixture_id),
            token,
        )
        checks["match_facts"] = run_check(
            f"match-facts/fixture/{fixture_id}",
            lambda: client.get_match_facts(fixture_id),
            token,
        )

    audit: dict[str, Any] = {
        "provider": "sportmonks",
        "captured_at": captured_at,
        "base_url": settings.sportmonks_base_url,
        "token_present": bool(token),
        "token_value": "[redacted]" if token else "",
        "world_cup_league_candidates": leagues,
        "world_cup_2026_season_candidates": seasons,
        "selected_fixture_id": fixture_id,
        "checks": checks,
    }
    audit["summary"] = audit_summary(audit)
    sanitized = sanitize_sportmonks_payload(audit, token)
    if audit_dir is None:
        path = save_sportmonks_audit(sanitized)
    else:
        path = save_sportmonks_audit(sanitized, audit_dir=audit_dir)
    return sanitized, path


def main() -> int:
    settings = load_settings(require_api_key=False, require_sportmonks_token=True)
    audit, path = run_sportmonks_access_audit(settings=settings)
    print(f"SportMonks audit saved: {path}")
    print(
        json.dumps(
            {
                "accessible_categories": audit["summary"]["accessible_categories"],
                "error_categories": audit["summary"]["error_categories"],
                "world_cup_league_ids": audit["summary"]["world_cup_league_ids"],
                "world_cup_2026_season_ids": audit["summary"][
                    "world_cup_2026_season_ids"
                ],
                "selected_fixture_id": audit["summary"]["selected_fixture_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
