import pandas as pd

from src.features import build_match_features
from src.predictor import predict_fixture, prediction_snapshot_row
from src.ratings import (
    DEFAULT_RATING,
    TeamRating,
    expected_score,
    save_rating_snapshot,
    update_ratings_from_results,
)
from src.storage import save_prediction_snapshot


def fixture(status_short="NS", elapsed=None, home_goals=None, away_goals=None):
    return {
        "fixture": {
            "id": 9001,
            "date": "2026-06-14T20:00:00+00:00",
            "status": {
                "elapsed": elapsed,
                "long": "Not Started" if status_short == "NS" else "Match Finished",
                "short": status_short,
            },
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
        "goals": {"home": home_goals, "away": away_goals},
    }


def live_statistics():
    return [
        {
            "team": {"id": 10, "name": "Alpha"},
            "statistics": [
                {"type": "Total Shots", "value": 14},
                {"type": "Shots on Goal", "value": 6},
                {"type": "Ball Possession", "value": "61%"},
                {"type": "Corner Kicks", "value": 7},
            ],
        },
        {
            "team": {"id": 20, "name": "Beta"},
            "statistics": [
                {"type": "Total Shots", "value": 4},
                {"type": "Shots on Goal", "value": 1},
                {"type": "Ball Possession", "value": "39%"},
                {"type": "Corner Kicks", "value": 2},
            ],
        },
    ]


def live_events():
    return [
        {
            "type": "Card",
            "detail": "Red Card",
            "time": {"elapsed": 55},
            "team": {"id": 20, "name": "Beta"},
        }
    ]


def test_expected_score_is_symmetric():
    alpha = expected_score(1500, 1500)
    beta = expected_score(1500, 1500)

    assert alpha == 0.5
    assert alpha + beta == 1.0


def test_update_ratings_from_completed_results():
    ratings = update_ratings_from_results([fixture("FT", 90, 2, 0)])

    assert ratings[10].rating > DEFAULT_RATING
    assert ratings[20].rating < DEFAULT_RATING
    assert ratings[10].matches_played == 1


def test_save_rating_snapshot(tmp_path):
    ratings = {
        10: TeamRating(team_id=10, team_name="Alpha", rating=1512, matches_played=1)
    }
    path = save_rating_snapshot(ratings, snapshot_dir=tmp_path)

    frame = pd.read_csv(path)
    assert frame.loc[0, "team_id"] == 10
    assert frame.loc[0, "rating"] == 1512
    assert "snapshot_captured_at" in frame.columns


def test_prematch_prediction_prefers_stronger_team():
    ratings = {
        10: TeamRating(team_id=10, team_name="Alpha", rating=1700, matches_played=8),
        20: TeamRating(team_id=20, team_name="Beta", rating=1400, matches_played=8),
    }
    prediction = predict_fixture(fixture(), ratings=ratings)

    outcome_sum = (
        prediction["home_win_probability"]
        + prediction["draw_probability"]
        + prediction["away_win_probability"]
    )
    assert prediction["prediction_mode"] == "prematch"
    assert abs(outcome_sum - 1.0) < 0.001
    assert prediction["home_win_probability"] > prediction["away_win_probability"]
    assert prediction["model_drivers"]


def test_live_prediction_blends_prior_and_live_features():
    ratings = {
        10: TeamRating(team_id=10, team_name="Alpha", rating=1500, matches_played=4),
        20: TeamRating(team_id=20, team_name="Beta", rating=1500, matches_played=4),
    }
    prediction = predict_fixture(
        fixture("2H", 65, 1, 0),
        statistics=live_statistics(),
        events=live_events(),
        ratings=ratings,
    )

    assert prediction["prediction_mode"] == "live"
    assert prediction["home_win_probability"] > prediction["away_win_probability"]
    assert prediction["next_goal_probability"] > 0
    assert any("red cards" in driver for driver in prediction["model_drivers"])


def test_prediction_snapshot_row_and_storage(tmp_path):
    match = fixture("2H", 65, 1, 0)
    features = build_match_features(match, statistics=live_statistics(), events=live_events())
    prediction = predict_fixture(match, statistics=live_statistics(), events=live_events())
    row = prediction_snapshot_row(features, prediction)
    path = save_prediction_snapshot([row], snapshot_dir=tmp_path)

    frame = pd.read_csv(path)
    assert frame.loc[0, "fixture_id"] == 9001
    assert frame.loc[0, "prediction_mode"] == "live"
    assert frame.loc[0, "model_version"] == "world-cup-predictor-v1"
    assert frame.loc[0, "snapshot_row_count"] == 1
