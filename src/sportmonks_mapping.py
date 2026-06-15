from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any


TEAM_ALIASES = {
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "usa": "united states",
    "u s a": "united states",
    "us": "united states",
}

TEAM_STOPWORDS = {
    "fc",
    "football",
    "national",
    "team",
    "the",
}


def normalize_team_name(name: Any) -> str:
    text = str(name or "").casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in TEAM_STOPWORDS]
    normalized = " ".join(tokens)
    return TEAM_ALIASES.get(normalized, normalized)


def parse_fixture_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        text = text.replace(" ", "T") + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def api_football_fixture_key(fixture: dict[str, Any]) -> dict[str, Any]:
    fixture_data = fixture.get("fixture") or {}
    teams = fixture.get("teams") or {}
    return {
        "fixture_id": fixture_data.get("id"),
        "kickoff": parse_fixture_datetime(fixture_data.get("date")),
        "home": str((teams.get("home") or {}).get("name") or ""),
        "away": str((teams.get("away") or {}).get("name") or ""),
    }


def sportmonks_fixture_key(fixture: dict[str, Any]) -> dict[str, Any]:
    kickoff = parse_fixture_datetime(
        fixture.get("starting_at")
        or fixture.get("start_time")
        or fixture.get("fixture_starting_at")
    )
    home = ""
    away = ""
    participants = fixture.get("participants") or []
    if isinstance(participants, dict):
        participants = participants.get("data") or []
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            location = str((participant.get("meta") or {}).get("location") or "")
            name = str(participant.get("name") or "")
            if location == "home":
                home = name
            elif location == "away":
                away = name

    if not home or not away:
        name = str(fixture.get("name") or "")
        if " vs " in name:
            home, away = [part.strip() for part in name.split(" vs ", 1)]
        elif " - " in name:
            home, away = [part.strip() for part in name.split(" - ", 1)]

    return {
        "fixture_id": fixture.get("id"),
        "kickoff": kickoff,
        "home": home,
        "away": away,
    }


def _team_score(local_name: str, sportmonks_name: str) -> float:
    local_normalized = normalize_team_name(local_name)
    sportmonks_normalized = normalize_team_name(sportmonks_name)
    if not local_normalized or not sportmonks_normalized:
        return 0.0
    if local_normalized == sportmonks_normalized:
        return 1.0
    local_tokens = set(local_normalized.split())
    sportmonks_tokens = set(sportmonks_normalized.split())
    if local_tokens and local_tokens.issubset(sportmonks_tokens):
        return 0.9
    if sportmonks_tokens and sportmonks_tokens.issubset(local_tokens):
        return 0.9
    return SequenceMatcher(None, local_normalized, sportmonks_normalized).ratio()


def fixture_pair_score(
    local_fixture: dict[str, Any],
    sportmonks_fixture: dict[str, Any],
    kickoff_tolerance_minutes: int = 180,
) -> float:
    local_key = api_football_fixture_key(local_fixture)
    sportmonks_key = sportmonks_fixture_key(sportmonks_fixture)
    local_kickoff = local_key["kickoff"]
    sportmonks_kickoff = sportmonks_key["kickoff"]
    if local_kickoff is None or sportmonks_kickoff is None:
        return 0.0
    kickoff_delta = abs((local_kickoff - sportmonks_kickoff).total_seconds()) / 60
    if kickoff_delta > kickoff_tolerance_minutes:
        return 0.0

    home_score = _team_score(local_key["home"], sportmonks_key["home"])
    away_score = _team_score(local_key["away"], sportmonks_key["away"])
    if home_score == 0.0 or away_score == 0.0:
        return 0.0
    kickoff_score = max(0.0, 1.0 - (kickoff_delta / kickoff_tolerance_minutes))
    return (home_score * 0.4) + (away_score * 0.4) + (kickoff_score * 0.2)


def match_sportmonks_fixture(
    local_fixture: dict[str, Any],
    sportmonks_fixtures: list[dict[str, Any]],
    kickoff_tolerance_minutes: int = 180,
) -> dict[str, Any]:
    scored = [
        {
            "sportmonks_fixture_id": sportmonks_fixture.get("id"),
            "score": fixture_pair_score(
                local_fixture,
                sportmonks_fixture,
                kickoff_tolerance_minutes=kickoff_tolerance_minutes,
            ),
            "sportmonks_fixture": sportmonks_fixture,
        }
        for sportmonks_fixture in sportmonks_fixtures
    ]
    candidates = [candidate for candidate in scored if candidate["score"] >= 0.78]
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    local_key = api_football_fixture_key(local_fixture)
    base_result = {
        "local_fixture_id": local_key["fixture_id"],
        "local_match": f"{local_key['home']} vs {local_key['away']}",
        "sportmonks_fixture_id": None,
        "confidence": "no_match",
        "score": 0.0,
    }
    if not candidates:
        return base_result

    best = candidates[0]
    if len(candidates) > 1 and abs(best["score"] - candidates[1]["score"]) < 0.03:
        return {
            **base_result,
            "sportmonks_fixture_id": best["sportmonks_fixture_id"],
            "confidence": "ambiguous",
            "score": round(best["score"], 3),
        }

    confidence = "exact" if best["score"] >= 0.995 else "likely"
    return {
        **base_result,
        "sportmonks_fixture_id": best["sportmonks_fixture_id"],
        "confidence": confidence,
        "score": round(best["score"], 3),
    }
