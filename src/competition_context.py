from __future__ import annotations

from typing import Any


WORLD_CUP_HOST_TEAMS = {
    "canada": "canada",
    "mexico": "mexico",
    "united states": "usa",
    "usa": "usa",
    "us": "usa",
}

WORLD_CUP_HOST_CITY_COUNTRY = {
    "atlanta": "usa",
    "arlington": "usa",
    "boston": "usa",
    "dallas": "usa",
    "east rutherford": "usa",
    "guadalajara": "mexico",
    "houston": "usa",
    "inglewood": "usa",
    "kansas city": "usa",
    "los angeles": "usa",
    "miami": "usa",
    "mexico city": "mexico",
    "monterrey": "mexico",
    "new york": "usa",
    "philadelphia": "usa",
    "santa clara": "usa",
    "seattle": "usa",
    "toronto": "canada",
    "vancouver": "canada",
}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def is_world_cup_fixture(fixture: dict[str, Any]) -> bool:
    league = fixture.get("league") or {}
    league_name = _normalized_text(league.get("name"))
    return int(league.get("id") or 0) == 1 or "world cup" in league_name


def venue_city(fixture: dict[str, Any]) -> str:
    fixture_data = fixture.get("fixture") or {}
    venue = fixture_data.get("venue") or fixture.get("venue") or {}
    return _normalized_text(venue.get("city"))


def team_host_country(team_name: Any) -> str | None:
    return WORLD_CUP_HOST_TEAMS.get(_normalized_text(team_name))


def venue_host_country(fixture: dict[str, Any]) -> str | None:
    return WORLD_CUP_HOST_CITY_COUNTRY.get(venue_city(fixture))


def effective_home_advantage_elo(
    fixture: dict[str, Any],
    default_home_advantage_elo: float,
) -> float:
    if not is_world_cup_fixture(fixture):
        return float(default_home_advantage_elo)

    teams = fixture.get("teams") or {}
    home_name = (teams.get("home") or {}).get("name")
    home_country = team_host_country(home_name)
    venue_country = venue_host_country(fixture)
    if home_country and venue_country and home_country == venue_country:
        return float(default_home_advantage_elo)
    return 0.0


def is_neutral_world_cup_fixture(
    fixture: dict[str, Any],
    default_home_advantage_elo: float,
) -> bool:
    return (
        is_world_cup_fixture(fixture)
        and effective_home_advantage_elo(fixture, default_home_advantage_elo) == 0.0
    )
