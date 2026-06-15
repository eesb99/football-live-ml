from __future__ import annotations

import json

from src.config import Settings
from src.sportmonks_audit import run_sportmonks_access_audit
from src.sportmonks_client import SportMonksError


class FakeSportMonksClient:
    def __init__(self, token="sportmonks-secret"):
        self.token = token

    def get_leagues(self, params=None):
        return {
            "data": [{"id": 999, "name": "FIFA World Cup"}],
            "rate_limit": {"remaining": 2999},
            "subscription": {"plan": "World Cup All-in"},
        }

    def search_leagues(self, query, params=None):
        return {
            "data": [
                {
                    "id": 999,
                    "name": "FIFA World Cup",
                    "seasons": [{"id": 202699, "name": "2026", "league_id": 999}],
                }
            ]
        }

    def get_seasons(self, params=None):
        return {
            "data": [
                {
                    "id": 202699,
                    "name": "2026",
                    "league_id": 999,
                    "starting_at": "2026-06-11",
                }
            ]
        }

    def get_fixtures(self, season_id, params=None):
        return {
            "data": [
                {
                    "id": 7001,
                    "name": "Mexico vs South Africa",
                    "starting_at": "2026-06-12 03:00:00",
                }
            ]
        }

    def get_fixture_detail(self, fixture_id):
        return {"data": {"id": fixture_id, "name": "Mexico vs South Africa"}}

    def get_prediction_probabilities(self, fixture_id):
        return {"data": [{"fixture_id": fixture_id, "home": 0.45, "draw": 0.30, "away": 0.25}]}

    def get_pre_match_odds(self, fixture_id):
        return {"data": [{"fixture_id": fixture_id, "market": "Fulltime Result"}]}

    def get_expected_goals(self, fixture_id):
        return {"data": [{"fixture_id": fixture_id, "home_xg": 1.2, "away_xg": 0.9}]}

    def get_news(self, fixture_id):
        return {"data": [{"fixture_id": fixture_id, "title": "Team news"}]}

    def get_match_facts(self, fixture_id):
        raise SportMonksError(f"api_token={self.token} not allowed")


def test_sportmonks_access_audit_normalizes_mocked_access(tmp_path):
    settings = Settings(sportmonks_api_token="sportmonks-secret")

    audit, path = run_sportmonks_access_audit(
        settings=settings,
        client=FakeSportMonksClient(),
        audit_dir=tmp_path,
    )

    assert path.exists()
    assert audit["summary"]["world_cup_league_ids"] == [999]
    assert audit["summary"]["world_cup_2026_season_ids"] == [202699]
    assert audit["summary"]["selected_fixture_id"] == 7001
    assert "pre_match_odds" in audit["summary"]["accessible_categories"]
    assert audit["checks"]["match_facts"]["status"] == "error"


def test_sportmonks_audit_file_never_persists_token(tmp_path):
    settings = Settings(sportmonks_api_token="sportmonks-secret")

    _, path = run_sportmonks_access_audit(
        settings=settings,
        client=FakeSportMonksClient(),
        audit_dir=tmp_path,
    )

    saved = path.read_text()
    parsed = json.loads(saved)
    assert "sportmonks-secret" not in saved
    assert "api_token=sportmonks-secret" not in saved
    assert parsed["token_value"] == "[redacted]"
