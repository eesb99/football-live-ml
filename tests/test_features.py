import pandas as pd

from src.features import (
    build_live_match_table,
    build_match_features,
    event_summary,
    parse_percentage,
)
from src.model import predict_match_probabilities
from src.storage import save_snapshot


def sample_fixture():
    return {
        "fixture": {
            "id": 123,
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
            "home": {"id": 1, "name": "Home FC"},
            "away": {"id": 2, "name": "Away FC"},
        },
        "goals": {"home": 2, "away": 1},
    }


def sample_statistics():
    return [
        {
            "team": {"id": 1, "name": "Home FC"},
            "statistics": [
                {"type": "Total Shots", "value": 12},
                {"type": "Expected Goals", "value": "1.74"},
                {"type": "Shots on Goal", "value": 5},
                {"type": "Ball Possession", "value": "58%"},
                {"type": "Corner Kicks", "value": 6},
                {"type": "Yellow Cards", "value": 1},
                {"type": "Total passes", "value": 410},
                {"type": "Passes accurate", "value": 360},
                {"type": "Goalkeeper Saves", "value": 1},
            ],
        },
        {
            "team": {"id": 2, "name": "Away FC"},
            "statistics": [
                {"type": "Total Shots", "value": 7},
                {"type": "Expected Goals", "value": "0.82"},
                {"type": "Shots on Goal", "value": 2},
                {"type": "Ball Possession", "value": "42%"},
                {"type": "Corner Kicks", "value": 3},
                {"type": "Yellow Cards", "value": 2},
                {"type": "Total passes", "value": 300},
                {"type": "Passes accurate", "value": 240},
                {"type": "Goalkeeper Saves", "value": 3},
            ],
        },
    ]


def sample_events():
    return [
        {
            "type": "Card",
            "detail": "Red Card",
            "time": {"elapsed": 61},
            "team": {"id": 2, "name": "Away FC"},
        },
        {
            "type": "Card",
            "detail": "Yellow Card",
            "time": {"elapsed": 59},
            "team": {"id": 1, "name": "Home FC"},
        },
        {
            "type": "Goal",
            "detail": "Normal Goal",
            "time": {"elapsed": 64},
            "team": {"id": 1, "name": "Home FC"},
        }
    ]


def test_parse_percentage_handles_api_strings():
    assert parse_percentage("58%") == 58.0
    assert parse_percentage(None) == 0.0
    assert parse_percentage("bad") == 0.0


def test_build_match_features_extracts_live_state():
    features = build_match_features(
        sample_fixture(),
        statistics=sample_statistics(),
        events=sample_events(),
    )

    assert features["fixture_id"] == 123
    assert features["fixture_date"] == "2026-06-14T20:00:00+00:00"
    assert features["league_id"] == 1
    assert features["league_name"] == "World Cup"
    assert features["league_season"] == 2026
    assert features["league_round"] == "Group Stage - 1"
    assert features["is_live"] is True
    assert features["minute"] == 65
    assert features["home_team"] == "Home FC"
    assert features["away_team"] == "Away FC"
    assert features["home_goals"] == 2
    assert features["away_goals"] == 1
    assert features["score_difference"] == 1
    assert features["home_red_cards"] == 0
    assert features["away_red_cards"] == 1
    assert features["home_shots"] == 12
    assert features["home_xg"] == 1.74
    assert features["away_xg"] == 0.82
    assert round(features["xg_difference"], 2) == 0.92
    assert features["away_shots_on_target"] == 2
    assert features["home_possession"] == 58.0
    assert features["away_corners"] == 3
    assert features["home_yellow_cards"] == 1
    assert features["away_yellow_cards"] == 2
    assert features["home_goalkeeper_saves"] == 1
    assert features["away_goalkeeper_saves"] == 3
    assert round(features["home_pass_accuracy"], 1) == 87.8
    assert features["home_recent_events"] == 2
    assert features["home_recent_goals"] == 1
    assert features["away_recent_events"] == 1
    assert 0.0 <= features["home_pressure_share"] <= 1.0
    assert features["data_completeness_score"] == 1.0


def test_event_summary_counts_cards_goals_and_recent_events():
    summary = event_summary(
        sample_events(),
        home_team_id=1,
        away_team_id=2,
        current_minute=65,
    )

    assert summary["home_red_cards"] == 0
    assert summary["away_red_cards"] == 1
    assert summary["home_yellow_cards"] == 1
    assert summary["home_goal_events"] == 1
    assert summary["home_recent_goals"] == 1


def test_build_live_match_table_formats_score():
    rows = build_live_match_table([sample_fixture()])

    assert rows == [
        {
            "fixture_id": 123,
            "fixture_date": "2026-06-14T20:00:00+00:00",
            "minute": 65,
            "status": "Second Half",
            "status_short": "2H",
            "league_id": 1,
            "league_name": "World Cup",
            "league_country": "World",
            "league_season": 2026,
            "league_round": "Group Stage - 1",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "score": "2-1",
        }
    ]


def test_poisson_prediction_probabilities_are_valid():
    features = build_match_features(
        sample_fixture(),
        statistics=sample_statistics(),
        events=sample_events(),
    )
    predictions = predict_match_probabilities(features)

    outcome_sum = (
        predictions["home_win_probability"]
        + predictions["draw_probability"]
        + predictions["away_win_probability"]
    )
    next_goal_sum = (
        predictions["home_scores_next_probability"]
        + predictions["away_scores_next_probability"]
        + predictions["no_next_goal_probability"]
    )

    assert abs(outcome_sum - 1.0) < 0.001
    assert abs(next_goal_sum - 1.0) < 0.001
    assert 0.0 <= predictions["next_goal_probability"] <= 1.0
    assert predictions["home_win_probability"] > predictions["away_win_probability"]
    assert predictions["home_strength_score"] > 0
    assert predictions["away_strength_score"] > 0
    assert predictions["home_expected_remaining_goals"] > 0
    assert predictions["model_confidence"] > 0


def test_save_snapshot_can_log_empty_refresh(tmp_path):
    path = save_snapshot([], snapshot_dir=tmp_path, allow_empty=True)

    assert path is not None
    frame = pd.read_csv(path)
    assert frame.loc[0, "snapshot_row_count"] == 0


def test_save_snapshot_adds_audit_columns(tmp_path):
    path = save_snapshot(
        [{"fixture_id": 123, "home_team": "Home FC"}],
        snapshot_dir=tmp_path,
        allow_empty=True,
    )

    frame = pd.read_csv(path)
    assert list(frame.columns[:2]) == ["snapshot_captured_at", "snapshot_row_count"]
    assert frame.loc[0, "snapshot_row_count"] == 1
    assert frame.loc[0, "fixture_id"] == 123
