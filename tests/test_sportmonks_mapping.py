from __future__ import annotations

from src.sportmonks_mapping import match_sportmonks_fixture, normalize_team_name


def local_fixture(home="Mexico", away="South Africa", date="2026-06-12T03:00:00+00:00"):
    return {
        "fixture": {"id": 9001, "date": date},
        "teams": {
            "home": {"name": home},
            "away": {"name": away},
        },
    }


def sportmonks_fixture(
    fixture_id=7001,
    home="Mexico",
    away="South Africa",
    starting_at="2026-06-12 03:00:00",
):
    return {
        "id": fixture_id,
        "starting_at": starting_at,
        "participants": [
            {"name": home, "meta": {"location": "home"}},
            {"name": away, "meta": {"location": "away"}},
        ],
    }


def test_normalize_team_name_applies_basic_aliases():
    assert normalize_team_name("USA") == "united states"
    assert normalize_team_name("Korea Republic") == "south korea"


def test_match_sportmonks_fixture_exact_match():
    result = match_sportmonks_fixture(local_fixture(), [sportmonks_fixture()])

    assert result["sportmonks_fixture_id"] == 7001
    assert result["confidence"] == "exact"


def test_match_sportmonks_fixture_likely_match_for_longer_provider_names():
    result = match_sportmonks_fixture(
        local_fixture(),
        [sportmonks_fixture(home="Mexico U23", away="South Africa U23")],
    )

    assert result["sportmonks_fixture_id"] == 7001
    assert result["confidence"] == "likely"


def test_match_sportmonks_fixture_marks_ambiguous_when_candidates_are_tied():
    result = match_sportmonks_fixture(
        local_fixture(),
        [
            sportmonks_fixture(fixture_id=7001),
            sportmonks_fixture(fixture_id=7002),
        ],
    )

    assert result["confidence"] == "ambiguous"
    assert result["sportmonks_fixture_id"] == 7001


def test_match_sportmonks_fixture_no_match_when_time_and_teams_do_not_align():
    result = match_sportmonks_fixture(
        local_fixture(),
        [sportmonks_fixture(home="France", away="Germany", starting_at="2026-06-13 03:00:00")],
    )

    assert result["confidence"] == "no_match"
    assert result["sportmonks_fixture_id"] is None
