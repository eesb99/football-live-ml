from src.external_predictions import (
    normalize_api_football_prediction,
    parse_probability,
    unavailable_prediction,
)


def test_parse_probability_handles_api_percent_strings():
    assert parse_probability("45%") == 0.45
    assert parse_probability("12.5%") == 0.125
    assert parse_probability(0.3) == 0.3
    assert parse_probability(30) == 0.3
    assert parse_probability(None) is None
    assert parse_probability("bad") is None


def test_normalize_api_football_prediction_extracts_core_fields():
    normalized = normalize_api_football_prediction(
        [
            {
                "predictions": {
                    "winner": {
                        "id": 10,
                        "name": "Alpha",
                        "comment": "Win or draw",
                    },
                    "win_or_draw": True,
                    "under_over": "-3.5",
                    "goals": {"home": "-2.5", "away": "-1.5"},
                    "advice": "Double chance : Alpha or draw",
                    "percent": {"home": "45%", "draw": "35%", "away": "20%"},
                }
            }
        ],
        fixture_id=9001,
    )

    assert normalized["available"] is True
    assert normalized["status"] == "available"
    assert normalized["endpoint"] == "/predictions?fixture=9001"
    assert normalized["home_probability"] == 0.45
    assert normalized["draw_display"] == "35.0%"
    assert normalized["away_display"] == "20.0%"
    assert normalized["advice"] == "Double chance : Alpha or draw"
    assert normalized["winner_name"] == "Alpha"
    assert normalized["winner_comment"] == "Win or draw"
    assert normalized["win_or_draw"] is True
    assert normalized["under_over"] == "-3.5"
    assert normalized["goals_home"] == "-2.5"
    assert normalized["goals_away"] == "-1.5"


def test_normalize_api_football_prediction_handles_empty_response():
    normalized = normalize_api_football_prediction([], fixture_id=9001)

    assert normalized == unavailable_prediction(9001, "missing")


def test_unavailable_prediction_can_store_error_without_sensitive_data():
    normalized = unavailable_prediction(
        9001,
        "error",
        last_error="API-Football error response: {'plan': 'not available'}",
    )

    assert normalized["available"] is False
    assert normalized["status"] == "error"
    assert normalized["last_error"] == "API-Football error response: {'plan': 'not available'}"
    assert normalized["home_display"] == "-"
