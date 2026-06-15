import pandas as pd

from src.adapters import PaidDataSnapshot
from src.competition_context import effective_home_advantage_elo
from src.features import build_match_features
from src.predictor import (
    apply_paid_data_to_features,
    calibrate_draw_probability,
    predict_fixture,
    prediction_snapshot_row,
)
from src.ratings import (
    DEFAULT_RATING,
    TeamRating,
    expected_score,
    load_ratings,
    save_rating_snapshot,
    save_ratings,
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


def test_load_ratings_handles_blank_file(tmp_path):
    path = tmp_path / "team_ratings.csv"
    path.write_text("\n")

    assert load_ratings(path) == {}


def test_save_ratings_writes_headers_for_empty_ratings(tmp_path):
    path = tmp_path / "team_ratings.csv"

    save_ratings({}, path=path)

    frame = pd.read_csv(path)
    assert list(frame.columns) == [
        "team_id",
        "team_name",
        "rating",
        "matches_played",
        "snapshot_captured_at",
    ]
    assert frame.empty
    assert load_ratings(path) == {}


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


def test_prematch_confidence_reflects_probability_margin_and_rating_depth():
    close_prediction = predict_fixture(fixture())
    strong_prediction = predict_fixture(
        fixture(),
        ratings={
            10: TeamRating(team_id=10, team_name="Alpha", rating=1900, matches_played=10),
            20: TeamRating(team_id=20, team_name="Beta", rating=1250, matches_played=10),
        },
    )

    assert strong_prediction["model_confidence"] > close_prediction["model_confidence"]
    assert (
        "Confidence reflects top probability"
        in strong_prediction["model_driver_summary"]
    )


def test_draw_calibration_lifts_close_low_tempo_draw_and_normalizes():
    base = (0.39, 0.28, 0.33)
    calibrated = calibrate_draw_probability(
        base,
        1.15,
        1.05,
        rating_gap=35.0,
        form_gap=0.05,
    )

    assert calibrated[1] > base[1]
    assert abs(sum(calibrated) - 1.0) < 0.001


def test_draw_calibration_keeps_cold_start_neutral_draw_material_but_not_forced_top():
    base = (0.41, 0.31, 0.28)
    calibrated = calibrate_draw_probability(
        base,
        1.30,
        1.05,
        rating_gap=0.0,
        form_gap=0.0,
        neutral_site=True,
        home_matches_played=0,
        away_matches_played=0,
    )

    assert calibrated[1] > base[1]
    assert calibrated[1] < calibrated[0]
    assert abs(sum(calibrated) - 1.0) < 0.001


def test_draw_calibration_does_not_boost_clear_mismatch():
    base = (0.65, 0.20, 0.15)
    calibrated = calibrate_draw_probability(
        base,
        2.1,
        0.8,
        rating_gap=220.0,
        form_gap=0.6,
    )

    assert calibrated == base


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
    assert prediction["xg_source"] == "proxy_xg"
    assert prediction["home_proxy_xg"] > 0
    assert prediction["home_effective_xg"] == prediction["home_proxy_xg"]
    assert prediction["odds_available"] is False
    assert "proxy xG" in prediction["model_driver_summary"]
    assert any("red cards" in driver for driver in prediction["model_drivers"])


def test_neutral_world_cup_fixture_removes_home_advantage_and_lifts_draw():
    neutral_fixture = fixture()
    prediction = predict_fixture(neutral_fixture)

    assert effective_home_advantage_elo(neutral_fixture, 60.0) == 0.0
    assert prediction["draw_probability"] > prediction["away_win_probability"]
    assert prediction["home_win_probability"] > prediction["draw_probability"]
    assert (
        "Neutral World Cup venue removes the standard home-advantage Elo boost"
        in prediction["model_driver_summary"]
    )


def test_world_cup_host_fixture_keeps_host_home_advantage():
    host_fixture = fixture()
    host_fixture["teams"]["home"]["name"] = "Mexico"
    host_fixture["fixture"]["venue"] = {
        "id": 1069,
        "name": "Estadio Azteca",
        "city": "Mexico City",
    }

    prediction = predict_fixture(host_fixture)

    assert effective_home_advantage_elo(host_fixture, 60.0) == 60.0
    assert prediction["home_win_probability"] > prediction["draw_probability"]
    assert "Home or host advantage adds 60 Elo points" in prediction["model_driver_summary"]


def test_live_prediction_uses_paid_real_xg_when_available():
    paid_data = PaidDataSnapshot(
        real_xg_available=True,
        real_xg_source="paid-test-xg",
        home_real_xg=2.1,
        away_real_xg=0.4,
    )
    prediction = predict_fixture(
        fixture("2H", 65, 1, 0),
        statistics=live_statistics(),
        events=live_events(),
        ratings={
            10: TeamRating(team_id=10, team_name="Alpha", rating=1500, matches_played=4),
            20: TeamRating(team_id=20, team_name="Beta", rating=1500, matches_played=4),
        },
        paid_data=paid_data,
    )

    assert prediction["prediction_mode"] == "live"
    assert prediction["real_xg_available"] is True
    assert prediction["real_xg_source"] == "paid-test-xg"
    assert prediction["xg_source"] == "paid-test-xg"
    assert prediction["home_effective_xg"] == 2.1
    assert prediction["away_effective_xg"] == 0.4
    assert "Real xG source is available" in prediction["model_driver_summary"]


def test_apply_paid_data_to_features_overrides_effective_xg():
    match = fixture("2H", 65, 1, 0)
    features = build_match_features(match, statistics=live_statistics(), events=live_events())
    paid_data = PaidDataSnapshot(
        real_xg_available=True,
        real_xg_source="paid-test-xg",
        home_real_xg=1.8,
        away_real_xg=0.7,
    )

    enriched = apply_paid_data_to_features(features, paid_data)

    assert enriched["home_xg"] == 1.8
    assert enriched["away_xg"] == 0.7
    assert enriched["home_effective_xg"] == 1.8
    assert enriched["away_effective_xg"] == 0.7
    assert enriched["xg_source"] == "paid-test-xg"


def test_prediction_snapshot_row_and_storage(tmp_path):
    match = fixture("2H", 65, 1, 0)
    features = build_match_features(match, statistics=live_statistics(), events=live_events())
    prediction = predict_fixture(match, statistics=live_statistics(), events=live_events())
    row = prediction_snapshot_row(features, prediction)
    path = save_prediction_snapshot([row], snapshot_dir=tmp_path)

    frame = pd.read_csv(path)
    assert frame.loc[0, "fixture_id"] == 9001
    assert frame.loc[0, "prediction_mode"] == "live"
    assert frame.loc[0, "model_version"] == "world-cup-rules-v2"
    assert frame.loc[0, "xg_source"] == "proxy_xg"
    assert frame.loc[0, "odds_available"] == False
    assert frame.loc[0, "snapshot_row_count"] == 1
