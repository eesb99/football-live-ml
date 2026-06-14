from __future__ import annotations

from typing import Any


API_FOOTBALL_PREDICTIONS_ENDPOINT = "/predictions"


def parse_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        parsed = float(value)
    else:
        return None

    if parsed < 0:
        return None
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(parsed, 1.0))


def probability_display(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _endpoint_label(fixture_id: int) -> str:
    return f"{API_FOOTBALL_PREDICTIONS_ENDPOINT}?fixture={fixture_id}"


def unavailable_prediction(
    fixture_id: int,
    status: str,
    last_error: str = "",
) -> dict[str, Any]:
    return {
        "available": False,
        "status": status,
        "endpoint": _endpoint_label(fixture_id),
        "last_error": last_error,
        "raw_response_count": 0,
        "home_probability": None,
        "draw_probability": None,
        "away_probability": None,
        "home_display": "-",
        "draw_display": "-",
        "away_display": "-",
        "advice": "",
        "winner_id": None,
        "winner_name": "",
        "winner_comment": "",
        "win_or_draw": None,
        "under_over": "",
        "goals_home": "",
        "goals_away": "",
    }


def normalize_api_football_prediction(
    response: list[dict[str, Any]] | None,
    fixture_id: int,
) -> dict[str, Any]:
    records = response or []
    if not records:
        return unavailable_prediction(fixture_id, "missing")

    record = records[0] or {}
    prediction = record.get("predictions") or {}
    percent = prediction.get("percent") or {}
    winner = prediction.get("winner") or {}
    goals = prediction.get("goals") or {}

    home_probability = parse_probability(percent.get("home"))
    draw_probability = parse_probability(percent.get("draw"))
    away_probability = parse_probability(percent.get("away"))
    available = any(
        value is not None
        for value in (home_probability, draw_probability, away_probability)
    ) or bool(prediction.get("advice") or winner)

    if not available:
        return unavailable_prediction(fixture_id, "missing")

    return {
        "available": True,
        "status": "available",
        "endpoint": _endpoint_label(fixture_id),
        "last_error": "",
        "raw_response_count": len(records),
        "home_probability": home_probability,
        "draw_probability": draw_probability,
        "away_probability": away_probability,
        "home_display": probability_display(home_probability),
        "draw_display": probability_display(draw_probability),
        "away_display": probability_display(away_probability),
        "advice": str(prediction.get("advice") or ""),
        "winner_id": winner.get("id"),
        "winner_name": str(winner.get("name") or ""),
        "winner_comment": str(winner.get("comment") or ""),
        "win_or_draw": prediction.get("win_or_draw"),
        "under_over": str(prediction.get("under_over") or ""),
        "goals_home": str(goals.get("home") or ""),
        "goals_away": str(goals.get("away") or ""),
    }
