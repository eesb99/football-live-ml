from __future__ import annotations

from src.sportmonks_enrichment import (
    cache_records,
    news_by_fixture_from_cache,
    sportmonks_candidate_enrichment_by_api_fixture,
    sportmonks_mapping_coverage_rows,
    sportmonks_mapping_coverage_summary,
    xg_by_fixture_from_cache,
)


def api_fixture() -> dict:
    return {
        "fixture": {
            "id": 9001,
            "date": "2026-06-14T04:00:00+00:00",
            "status": {"short": "NS"},
        },
        "teams": {
            "home": {"id": 10, "name": "Australia"},
            "away": {"id": 20, "name": "Turkiye"},
        },
        "goals": {"home": None, "away": None},
    }


def sportmonks_fixture() -> dict:
    return {
        "id": 19609156,
        "name": "Australia vs Turkiye",
        "starting_at": "2026-06-14 04:00:00",
        "participants": [
            {"name": "Australia", "meta": {"location": "home"}},
            {"name": "Turkiye", "meta": {"location": "away"}},
        ],
    }


def test_cache_records_unwraps_sanitized_cache_payload():
    cache = {"payload": {"data": [sportmonks_fixture()]}}

    assert cache_records(cache)[0]["id"] == 19609156


def test_xg_and_news_indexes_by_fixture():
    xg_cache = {
        "captured_at": "2026-06-15T00:00:00+00:00",
        "payload": {
            "data": [
                {
                    "fixture_id": 19609156,
                    "location": "home",
                    "data": {"value": 1.4},
                },
                {
                    "fixture_id": 19609156,
                    "location": "away",
                    "data": {"value": 0.8},
                },
            ]
        },
    }
    news_cache = {
        "captured_at": "2026-06-13T00:00:00+00:00",
        "payload": {
            "data": [
                {"fixture_id": 19609156, "title": "Preview one"},
                {"fixture_id": 19609156, "title": "Preview two"},
            ]
        },
    }

    xg = xg_by_fixture_from_cache(xg_cache)
    news = news_by_fixture_from_cache(news_cache)

    assert xg[19609156]["home_sportmonks_xg"] == 1.4
    assert xg[19609156]["away_sportmonks_xg"] == 0.8
    assert xg[19609156]["xg_pair_available"] is True
    assert xg[19609156]["available_before_kickoff"] is False
    assert news[19609156]["news_count"] == 2


def test_mapping_coverage_rows_join_fixture_detail_xg_and_news():
    bundle = {
        "sportmonks_fixtures": [sportmonks_fixture()],
        "detail_caches": {19609156: {"cache_file": "/tmp/detail.json"}},
        "xg_by_fixture": {
            19609156: {
                "home_sportmonks_xg": 1.4,
                "away_sportmonks_xg": 0.8,
                "xg_pair_available": True,
                "captured_at": "2026-06-15T00:00:00+00:00",
                "availability": "post_match_only",
                "available_before_kickoff": False,
            }
        },
        "news_by_fixture": {
            19609156: {
                "news_count": 1,
                "captured_at": "2026-06-13T00:00:00+00:00",
                "news_titles": ["Preview"],
            }
        },
    }

    rows = sportmonks_mapping_coverage_rows([api_fixture()], bundle)
    summary = sportmonks_mapping_coverage_summary(rows)
    candidate = sportmonks_candidate_enrichment_by_api_fixture(rows)

    assert rows[0]["mapping_confidence"] == "exact"
    assert rows[0]["fixture_detail_available"] is True
    assert rows[0]["xg_pair_available"] is True
    assert rows[0]["news_count"] == 1
    assert summary["mapped"] == 1
    assert summary["detail_available"] == 1
    assert candidate[9001]["sportmonks_fixture_id"] == 19609156
