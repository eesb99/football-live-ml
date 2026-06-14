from src.api_client import ApiFootballError
from src.ratings import TeamRating

from app.streamlit_app import (
    arrow_safe_dataframe,
    data_source_rows,
    elo_prior_rows,
    expected_goal_rows,
    extracted_feature_rows,
    fixture_events_display_rows,
    fixture_statistics_display_rows,
    probability_rows,
    should_try_free_plan_fallback,
    strength_component_rows,
)


def test_should_try_free_plan_fallback_detects_api_football_plan_error():
    error = ApiFootballError(
        "API-Football error response: {'plan': 'Free plans do not have access "
        "to this season, try from 2022 to 2024.'}"
    )

    assert should_try_free_plan_fallback(error) is True


def test_should_try_free_plan_fallback_ignores_unrelated_error():
    error = ApiFootballError("API-Football error response: {'token': 'bad key'}")

    assert should_try_free_plan_fallback(error) is False


def fixture():
    return {
        "fixture": {
            "id": 9001,
            "date": "2026-06-14T20:00:00+00:00",
            "status": {"elapsed": 65, "long": "Second Half", "short": "2H"},
        },
        "league": {
            "id": 1,
            "name": "World Cup",
            "country": "World",
            "season": 2026,
            "round": "Group Stage - 1",
        },
        "teams": {
            "home": {"id": 10, "name": "Alpha"},
            "away": {"id": 20, "name": "Beta"},
        },
        "goals": {"home": 1, "away": 0},
    }


def features():
    return {
        "minute": 65,
        "home_goals": 1,
        "away_goals": 0,
        "score_difference": 1,
        "home_red_cards": 0,
        "away_red_cards": 1,
        "red_card_difference": -1,
        "home_shots": 14,
        "away_shots": 4,
        "shot_difference": 10,
        "home_xg": 1.62,
        "away_xg": 0.51,
        "xg_difference": 1.11,
        "home_proxy_xg": 1.12,
        "away_proxy_xg": 0.44,
        "proxy_xg_difference": 0.68,
        "home_effective_xg": 1.62,
        "away_effective_xg": 0.51,
        "effective_xg_difference": 1.11,
        "xg_source": "api_football_real_xg",
        "real_xg_available": True,
        "proxy_xg_available": True,
        "home_shots_on_target": 6,
        "away_shots_on_target": 1,
        "shots_on_target_difference": 5,
        "home_possession": 61.0,
        "away_possession": 39.0,
        "possession_difference": 22.0,
        "home_corners": 7,
        "away_corners": 2,
        "corner_difference": 5,
        "home_shots_inside_box": 8,
        "away_shots_inside_box": 2,
        "home_blocked_shots": 3,
        "away_blocked_shots": 1,
        "home_pass_accuracy": 87.5,
        "away_pass_accuracy": 79.1,
        "home_pressure_score": 4.2,
        "away_pressure_score": 1.4,
        "home_pressure_share": 0.75,
        "away_pressure_share": 0.25,
        "home_shot_share": 0.78,
        "away_shot_share": 0.22,
        "home_shots_on_target_share": 0.86,
        "away_shots_on_target_share": 0.14,
        "home_recent_events": 2,
        "away_recent_events": 1,
        "home_recent_goals": 1,
        "away_recent_goals": 0,
    }


def prediction():
    return {
        "prediction_mode": "live",
        "home_win_probability": 0.68,
        "draw_probability": 0.21,
        "away_win_probability": 0.11,
        "next_goal_probability": 0.44,
        "home_scores_next_probability": 0.31,
        "away_scores_next_probability": 0.13,
        "no_next_goal_probability": 0.56,
        "home_strength_score": 4.8,
        "away_strength_score": 2.1,
        "home_proxy_xg": 1.12,
        "away_proxy_xg": 0.44,
        "home_effective_xg": 1.62,
        "away_effective_xg": 0.51,
        "xg_source": "api_football_real_xg",
        "home_expected_goals": 1.9,
        "away_expected_goals": 0.8,
        "home_expected_remaining_goals": 0.52,
        "away_expected_remaining_goals": 0.22,
        "model_confidence": 0.72,
        "model_version": "world-cup-rules-v2",
        "odds_available": False,
        "real_xg_available": True,
        "injuries_available": False,
        "news_available": False,
        "odds_source": "not configured",
        "real_xg_source": "api_football_real_xg",
        "injuries_source": "not configured",
        "news_source": "not configured",
    }


def test_probability_rows_group_outcome_and_next_goal_metrics():
    rows = probability_rows(prediction())

    assert [row["metric"] for row in rows[:3]] == ["Home win", "Draw", "Away win"]
    assert {row["group"] for row in rows} == {"match outcome", "next goal"}
    assert rows[0]["display"] == "68.0%"


def test_elo_prior_rows_include_ratings_and_prematch_probabilities():
    ratings = {
        10: TeamRating(team_id=10, team_name="Alpha", rating=1600, matches_played=4),
        20: TeamRating(team_id=20, team_name="Beta", rating=1500, matches_played=4),
    }

    rows = elo_prior_rows(fixture(), ratings)

    metrics = {row["metric"]: row for row in rows}
    assert metrics["Home Elo"]["value"] == "1600.0"
    assert metrics["Away Elo"]["value"] == "1500.0"
    assert "Pre-match home" in metrics
    assert metrics["Pre-match home"]["value"].endswith("%")


def test_extracted_feature_rows_include_core_features_and_xg_when_available():
    rows = extracted_feature_rows(features())
    metrics = {row["signal"]: row for row in rows}

    assert metrics["Minute"]["match"] == "65'"
    assert metrics["Shots"]["home"] == "14"
    assert metrics["Possession"]["home"] == "61.0%"
    assert metrics["Real expected goals"]["home"] == "1.62"
    assert metrics["Proxy expected goals"]["home"] == "1.12"
    assert metrics["Effective expected goals"]["match"] == "api_football_real_xg"


def test_strength_and_expected_goal_rows_expose_model_layers():
    strength_rows = strength_component_rows(features(), prediction())
    xg_rows = expected_goal_rows(prediction())

    assert strength_rows[0]["component"] == "Live strength score"
    assert strength_rows[0]["home"] == "4.80"
    assert strength_rows[1]["component"] == "Effective xG"
    assert xg_rows[1]["metric"] == "Effective xG input"
    assert xg_rows[2]["metric"] == "Remaining expected goals"
    assert xg_rows[3]["home"] == "31.0%"


def test_data_source_rows_show_model_and_adapter_status():
    rows = data_source_rows(features(), prediction())
    sources = {row["source"]: row for row in rows}

    assert sources["Model mode"]["status"] == "live"
    assert sources["xG"]["status"] == "real xG"
    assert sources["Odds adapter"]["status"] == "missing"
    assert sources["Real xG adapter"]["detail"] == "api_football_real_xg"


def test_fixture_statistics_display_rows_flatten_percentages_and_mixed_values():
    statistics = [
        {
            "team": {"id": 1, "name": "Home"},
            "statistics": [
                {"type": "Ball Possession", "value": "61%"},
                {"type": "Total Shots", "value": 12},
                {"type": "Expected Goals", "value": None},
                {"type": "Nested", "value": {"raw": [1, "2"]}},
            ],
        },
        {"team": {"id": 2, "name": "Away"}, "statistics": []},
        "unexpected",
    ]

    rows = fixture_statistics_display_rows(statistics)

    assert rows[0] == {
        "team_id": "1",
        "team": "Home",
        "statistic": "Ball Possession",
        "value": "61%",
    }
    assert rows[1]["value"] == "12"
    assert rows[2]["value"] == ""
    assert rows[3]["value"] == '{"raw": [1, "2"]}'
    assert rows[4] == {"team_id": "2", "team": "Away", "statistic": "", "value": ""}
    assert rows[5]["value"] == "unexpected"


def test_fixture_events_display_rows_flatten_nested_objects_and_lists():
    events = [
        {
            "time": {"elapsed": 64, "extra": None},
            "team": {"id": 1, "name": "Home"},
            "player": {"id": 10, "name": "Scorer"},
            "assist": {"id": None, "name": None},
            "type": "Goal",
            "detail": "Normal Goal",
            "comments": None,
            "cards": [{"type": "yellow", "minute": 55}],
        },
        {"time": {}, "team": {"name": "Away"}, "type": "Card"},
        "raw event",
    ]

    rows = fixture_events_display_rows(events)

    assert rows[0]["time.elapsed"] == "64"
    assert rows[0]["time.extra"] == ""
    assert rows[0]["team.name"] == "Home"
    assert rows[0]["player.name"] == "Scorer"
    assert rows[0]["cards"] == '[{"minute": 55, "type": "yellow"}]'
    assert rows[1]["team.name"] == "Away"
    assert rows[2] == {"event": "raw event"}


def test_arrow_safe_dataframe_converts_display_rows_to_string_columns():
    rows = [
        {"statistic": "Ball Possession", "value": "61%", "raw": {"a": 1}},
        {"statistic": "Total Shots", "value": 12, "raw": [1, 2]},
    ]

    frame = arrow_safe_dataframe(rows)

    assert all(str(dtype) == "string" for dtype in frame.dtypes)
    assert frame.loc[0, "value"] == "61%"
    assert frame.loc[1, "value"] == "12"

    import pyarrow as pa

    pa.Table.from_pandas(frame)
