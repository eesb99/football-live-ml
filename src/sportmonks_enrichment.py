from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import Settings, load_settings
from src.sportmonks_client import SportMonksClient, sportmonks_records
from src.sportmonks_mapping import match_sportmonks_fixture, sportmonks_fixture_key
from src.storage import (
    SPORTMONKS_FIXTURE_DIR,
    list_sportmonks_cache_files,
    load_latest_sportmonks_fixtures_cache,
    load_latest_sportmonks_news_cache,
    load_latest_sportmonks_xg_cache,
    save_sportmonks_fixture_detail_cache,
    save_sportmonks_fixtures_cache,
    save_sportmonks_news_cache,
    save_sportmonks_xg_cache,
)


WORLD_CUP_2026_SPORTMONKS_SEASON_ID = 26618
WORLD_CUP_FIXTURE_INCLUDES = "participants;league;season;scores;state;venue"
WORLD_CUP_DETAIL_INCLUDES = (
    "participants;league;season;scores;state;statistics;metadata;venue"
)


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


def _pagination(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = response_metadata(payload)
    pagination = metadata.get("pagination") or {}
    return pagination if isinstance(pagination, dict) else {}


def combine_paginated_payloads(
    payloads: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"source": source, "pages": len(payloads)}
    for payload in payloads:
        records.extend(sportmonks_records(payload))
        for key, value in response_metadata(payload).items():
            metadata.setdefault(key, value)
    metadata["record_count"] = len(records)
    return {"data": records, "metadata": metadata}


def fetch_paginated_fixtures(
    client: SportMonksClient,
    season_id: int = WORLD_CUP_2026_SPORTMONKS_SEASON_ID,
    *,
    max_pages: int = 5,
    per_page: int = 100,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get_fixtures(
            season_id,
            {
                "include": WORLD_CUP_FIXTURE_INCLUDES,
                "per_page": per_page,
                "page": page,
            },
        )
        payloads.append(payload)
        pagination = _pagination(payload)
        if pagination and not pagination.get("has_more"):
            break
        if not sportmonks_records(payload):
            break
    combined = combine_paginated_payloads(payloads, source="fixtures")
    combined["season_id"] = int(season_id)
    return combined


def fetch_paginated_expected_goals(
    client: SportMonksClient,
    *,
    max_pages: int = 5,
    per_page: int = 100,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get_expected_goals_page(
            {"include": "fixture", "per_page": per_page, "page": page}
        )
        payloads.append(payload)
        pagination = _pagination(payload)
        if pagination and not pagination.get("has_more"):
            break
        if not sportmonks_records(payload):
            break
    return combine_paginated_payloads(payloads, source="expected_goals")


def fetch_paginated_news(
    client: SportMonksClient,
    *,
    max_pages: int = 5,
    per_page: int = 100,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get_news_page({"per_page": per_page, "page": page})
        payloads.append(payload)
        pagination = _pagination(payload)
        if pagination and not pagination.get("has_more"):
            break
        if not sportmonks_records(payload):
            break
    return combine_paginated_payloads(payloads, source="news")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _cache_payload(cache_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cache_record, dict):
        return {}
    payload = cache_record.get("payload")
    return payload if isinstance(payload, dict) else {}


def cache_records(cache_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    return sportmonks_records(_cache_payload(cache_record))


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fixture_detail_priority(record: dict[str, Any]) -> tuple[int, str, int]:
    placeholder_rank = 1 if record.get("placeholder") else 0
    starting_at = str(record.get("starting_at") or "9999-99-99 99:99:99")
    fixture_id = _as_int(record.get("id")) or 0
    return placeholder_rank, starting_at, fixture_id


def _metric_value(record: dict[str, Any]) -> float | None:
    data = record.get("data") or {}
    value = data.get("value") if isinstance(data, dict) else None
    if value is None:
        value = record.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def xg_by_fixture_from_cache(
    xg_cache: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    captured_at = str((xg_cache or {}).get("captured_at") or "")
    by_fixture: dict[int, dict[str, Any]] = {}
    for record in cache_records(xg_cache):
        fixture_id = _as_int(record.get("fixture_id"))
        if fixture_id is None:
            fixture = record.get("fixture") or {}
            if isinstance(fixture, dict):
                fixture_id = _as_int(fixture.get("id"))
        if fixture_id is None:
            continue
        location = str(record.get("location") or "").casefold()
        metric_value = _metric_value(record)
        if metric_value is None:
            continue
        entry = by_fixture.setdefault(
            fixture_id,
            {
                "sportmonks_fixture_id": fixture_id,
                "captured_at": captured_at,
                "availability": "post_match_only",
                "available_before_kickoff": False,
                "source": "sportmonks_expected_fixtures",
            },
        )
        if location == "home":
            entry["home_sportmonks_xg"] = metric_value
        elif location == "away":
            entry["away_sportmonks_xg"] = metric_value
    for entry in by_fixture.values():
        entry["xg_pair_available"] = (
            entry.get("home_sportmonks_xg") is not None
            and entry.get("away_sportmonks_xg") is not None
        )
    return by_fixture


def news_by_fixture_from_cache(
    news_cache: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    captured_at = str((news_cache or {}).get("captured_at") or "")
    by_fixture: dict[int, dict[str, Any]] = {}
    for record in cache_records(news_cache):
        fixture_id = _as_int(record.get("fixture_id"))
        if fixture_id is None:
            continue
        entry = by_fixture.setdefault(
            fixture_id,
            {
                "sportmonks_fixture_id": fixture_id,
                "captured_at": captured_at,
                "availability": "pre_match_news",
                "available_before_kickoff": None,
                "news_count": 0,
                "news_titles": [],
            },
        )
        entry["news_count"] += 1
        title = str(record.get("title") or "").strip()
        if title and len(entry["news_titles"]) < 3:
            entry["news_titles"].append(title)
    return by_fixture


def latest_fixture_detail_caches() -> dict[int, dict[str, Any]]:
    latest_by_fixture: dict[int, dict[str, Any]] = {}
    for path in list_sportmonks_cache_files(prefix="fixture_detail"):
        record = _load_json(path)
        if not record:
            continue
        fixture_id = _as_int(record.get("cache_key"))
        if fixture_id is None:
            continue
        latest_by_fixture.setdefault(fixture_id, {**record, "cache_file": str(path)})
    return latest_by_fixture


def load_latest_world_cup_enrichment(
    season_id: int = WORLD_CUP_2026_SPORTMONKS_SEASON_ID,
) -> dict[str, Any]:
    fixtures_cache = load_latest_sportmonks_fixtures_cache(season_id)
    xg_cache = load_latest_sportmonks_xg_cache()
    news_cache = load_latest_sportmonks_news_cache()
    detail_caches = latest_fixture_detail_caches()
    return {
        "season_id": int(season_id),
        "fixtures_cache": fixtures_cache,
        "xg_cache": xg_cache,
        "news_cache": news_cache,
        "detail_caches": detail_caches,
        "sportmonks_fixtures": cache_records(fixtures_cache),
        "xg_by_fixture": xg_by_fixture_from_cache(xg_cache),
        "news_by_fixture": news_by_fixture_from_cache(news_cache),
    }


def sportmonks_cache_status_rows(bundle: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bundle = bundle or load_latest_world_cup_enrichment()
    fixtures_cache = bundle.get("fixtures_cache") or {}
    xg_cache = bundle.get("xg_cache") or {}
    news_cache = bundle.get("news_cache") or {}
    detail_caches = bundle.get("detail_caches") or {}
    return [
        {
            "cache": "World Cup fixtures",
            "status": "available" if fixtures_cache else "missing",
            "records": len(bundle.get("sportmonks_fixtures") or []),
            "detail": fixtures_cache.get("cache_file", "Run python3 -m src.sportmonks_enrichment"),
        },
        {
            "cache": "Fixture detail",
            "status": "available" if detail_caches else "missing",
            "records": len(detail_caches),
            "detail": SPORTMONKS_FIXTURE_DIR.as_posix(),
        },
        {
            "cache": "Pre-kickoff odds",
            "status": "available" if list_sportmonks_cache_files(prefix="odds") else "missing",
            "records": len(list_sportmonks_cache_files(prefix="odds")),
            "detail": "Run python3 -m src.market_intelligence capture",
        },
        {
            "cache": "Expected goals",
            "status": "available" if xg_cache else "missing",
            "records": len(cache_records(xg_cache)),
            "detail": xg_cache.get("cache_file", "Run python3 -m src.sportmonks_enrichment"),
        },
        {
            "cache": "Pre-match news",
            "status": "available" if news_cache else "missing",
            "records": len(cache_records(news_cache)),
            "detail": news_cache.get("cache_file", "Run python3 -m src.sportmonks_enrichment"),
        },
    ]


def sportmonks_mapping_coverage_rows(
    api_fixtures: list[dict[str, Any]],
    bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    bundle = bundle or load_latest_world_cup_enrichment()
    sportmonks_fixtures = bundle.get("sportmonks_fixtures") or []
    detail_caches = bundle.get("detail_caches") or {}
    xg_by_fixture = bundle.get("xg_by_fixture") or {}
    news_by_fixture = bundle.get("news_by_fixture") or {}
    rows: list[dict[str, Any]] = []
    for fixture in api_fixtures:
        match = match_sportmonks_fixture(fixture, sportmonks_fixtures)
        sportmonks_fixture_id = _as_int(match.get("sportmonks_fixture_id"))
        xg = xg_by_fixture.get(sportmonks_fixture_id or -1, {})
        news = news_by_fixture.get(sportmonks_fixture_id or -1, {})
        detail_available = sportmonks_fixture_id in detail_caches
        api_fixture_id = _as_int((fixture.get("fixture") or {}).get("id"))
        sportmonks_key = {}
        if sportmonks_fixture_id is not None:
            sportmonks_record = next(
                (
                    record
                    for record in sportmonks_fixtures
                    if _as_int(record.get("id")) == sportmonks_fixture_id
                ),
                {},
            )
            sportmonks_key = sportmonks_fixture_key(sportmonks_record)
        rows.append(
            {
                "api_fixture_id": api_fixture_id,
                "api_match": match.get("local_match"),
                "sportmonks_fixture_id": sportmonks_fixture_id,
                "sportmonks_match": (
                    f"{sportmonks_key.get('home') or ''} vs "
                    f"{sportmonks_key.get('away') or ''}"
                ).strip(),
                "mapping_confidence": match.get("confidence"),
                "mapping_score": match.get("score"),
                "fixture_detail_available": detail_available,
                "xg_pair_available": bool(xg.get("xg_pair_available")),
                "home_sportmonks_xg": xg.get("home_sportmonks_xg"),
                "away_sportmonks_xg": xg.get("away_sportmonks_xg"),
                "xg_availability": xg.get("availability"),
                "xg_captured_at": xg.get("captured_at"),
                "xg_available_before_kickoff": xg.get("available_before_kickoff"),
                "news_count": int(news.get("news_count") or 0),
                "news_captured_at": news.get("captured_at"),
                "news_titles": "; ".join(news.get("news_titles") or []),
            }
        )
    return rows


def sportmonks_mapping_coverage_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    mapped = [
        row
        for row in rows
        if row.get("mapping_confidence") in {"exact", "likely", "ambiguous"}
    ]
    exact = [row for row in rows if row.get("mapping_confidence") == "exact"]
    likely = [row for row in rows if row.get("mapping_confidence") == "likely"]
    ambiguous = [row for row in rows if row.get("mapping_confidence") == "ambiguous"]
    detail = [row for row in rows if row.get("fixture_detail_available")]
    xg = [row for row in rows if row.get("xg_pair_available")]
    news = [row for row in rows if int(row.get("news_count") or 0) > 0]
    return {
        "api_fixtures": total,
        "mapped": len(mapped),
        "mapping_rate": len(mapped) / total if total else None,
        "exact": len(exact),
        "likely": len(likely),
        "ambiguous": len(ambiguous),
        "no_match": total - len(mapped),
        "detail_available": len(detail),
        "xg_pair_available": len(xg),
        "news_available": len(news),
    }


def sportmonks_candidate_enrichment_by_api_fixture(
    coverage_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    enrichment: dict[int, dict[str, Any]] = {}
    for row in coverage_rows:
        fixture_id = _as_int(row.get("api_fixture_id"))
        if fixture_id is None:
            continue
        enrichment[fixture_id] = {
            "sportmonks_fixture_id": row.get("sportmonks_fixture_id"),
            "mapping_confidence": row.get("mapping_confidence"),
            "mapping_score": row.get("mapping_score"),
            "home_sportmonks_xg": row.get("home_sportmonks_xg"),
            "away_sportmonks_xg": row.get("away_sportmonks_xg"),
            "xg_pair_available": row.get("xg_pair_available"),
            "captured_at": row.get("xg_captured_at") or row.get("news_captured_at"),
            "availability": row.get("xg_availability") or "missing",
            "available_before_kickoff": row.get("xg_available_before_kickoff"),
            "news_count": row.get("news_count"),
        }
    return enrichment


def fetch_and_cache_world_cup_enrichment(
    settings: Settings | None = None,
    client: SportMonksClient | None = None,
    *,
    season_id: int = WORLD_CUP_2026_SPORTMONKS_SEASON_ID,
    max_pages: int = 2,
    per_page: int = 100,
    max_detail_fixtures: int = 25,
) -> dict[str, Any]:
    settings = settings or load_settings(
        require_api_key=False,
        require_sportmonks_token=True,
    )
    client = client or SportMonksClient(settings)
    token = settings.sportmonks_api_token

    fixtures_payload = fetch_paginated_fixtures(
        client,
        season_id,
        max_pages=max_pages,
        per_page=per_page,
    )
    fixtures_path = save_sportmonks_fixtures_cache(
        fixtures_payload,
        season_id,
        token=token,
        metadata={"max_pages": max_pages, "per_page": per_page},
    )

    detail_paths: list[Path] = []
    fixture_ids = [
        int(record["id"])
        for record in sorted(
            sportmonks_records(fixtures_payload),
            key=_fixture_detail_priority,
        )
        if _as_int(record.get("id")) is not None
    ]
    for fixture_id in fixture_ids[:max_detail_fixtures]:
        detail_payload = client.get_fixture_detail(
            fixture_id,
            includes=WORLD_CUP_DETAIL_INCLUDES,
        )
        detail_paths.append(
            save_sportmonks_fixture_detail_cache(
                detail_payload,
                fixture_id,
                token=token,
            )
        )

    xg_payload = fetch_paginated_expected_goals(
        client,
        max_pages=max_pages,
        per_page=per_page,
    )
    xg_path = save_sportmonks_xg_cache(
        xg_payload,
        token=token,
        metadata={"max_pages": max_pages, "per_page": per_page},
    )

    news_payload = fetch_paginated_news(
        client,
        max_pages=max_pages,
        per_page=per_page,
    )
    news_path = save_sportmonks_news_cache(
        news_payload,
        token=token,
        metadata={"max_pages": max_pages, "per_page": per_page},
    )

    return {
        "season_id": int(season_id),
        "fixtures_cached": len(sportmonks_records(fixtures_payload)),
        "fixture_details_cached": len(detail_paths),
        "expected_goals_cached": len(sportmonks_records(xg_payload)),
        "news_cached": len(sportmonks_records(news_payload)),
        "paths": {
            "fixtures": str(fixtures_path),
            "fixture_details": [str(path) for path in detail_paths],
            "xg": str(xg_path),
            "news": str(news_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and cache sanitized SportMonks World Cup enrichment data."
    )
    parser.add_argument("--season-id", type=int, default=WORLD_CUP_2026_SPORTMONKS_SEASON_ID)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-detail-fixtures", type=int, default=25)
    args = parser.parse_args(argv)

    summary = fetch_and_cache_world_cup_enrichment(
        season_id=args.season_id,
        max_pages=args.max_pages,
        per_page=args.per_page,
        max_detail_fixtures=args.max_detail_fixtures,
    )
    public_summary = {
        key: value
        for key, value in summary.items()
        if key != "paths"
    }
    public_summary["paths"] = {
        key: value if key != "fixture_details" else f"{len(value)} files"
        for key, value in summary["paths"].items()
    }
    print(json.dumps(public_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
