from src.api_client import ApiFootballError
from src.ratings import TeamRating

from app.streamlit_app import (
    api_football_accuracy_summary,
    arrow_safe_dataframe,
    api_football_advice_rows,
    api_football_predicted_outcome,
    api_football_running_accuracy_rows,
    comparison_status_rows,
    data_source_rows,
    elo_prior_rows,
    expected_goal_rows,
    extracted_feature_rows,
    fixture_events_display_rows,
    fixture_statistics_display_rows,
    fetch_api_football_prediction,
    load_public_odds_refresh_state,
    model_comparison_rows,
    model_result_banner_data,
    public_odds_refresh_remaining_seconds,
    public_odds_refresh_summary,
    probability_difference,
    probability_rows,
    running_accuracy_rows,
    save_public_odds_refresh_state,
    secret_or_env_value,
    should_try_free_plan_fallback,
    sportmonks_audit_check_rows,
    sportmonks_mapping_metric_rows,
    sportmonks_provider_status_rows,
    sportmonks_token_configured,
    strength_component_rows,
)


def test_public_odds_refresh_summary_excludes_paths_and_fixture_ids():
    summary = public_odds_refresh_summary(
        {
            "season_id": 26618,
            "fixtures_cached": 72,
            "fixture_details_cached": 20,
            "fixtures_considered": 3,
            "odds_cached": 2,
            "empty_odds": 1,
            "errors": 0,
            "cached_paths": ["/tmp/private/path.json"],
            "empty_fixture_ids": [123],
            "error_fixture_ids": [456],
        }
    )

    assert summary == {
        "season_id": 26618,
        "fixtures_cached": 72,
        "fixture_details_cached": 20,
        "fixtures_considered": 3,
        "odds_cached": 2,
        "empty_odds": 1,
        "errors": 0,
    }


def test_sportmonks_token_configured_reads_streamlit_secrets(monkeypatch):
    import app.streamlit_app as app_module

    monkeypatch.delenv("SPORTMONKS_API_TOKEN", raising=False)
    monkeypatch.setattr(
        app_module.st,
        "secrets",
        {"SPORTMONKS_API_TOKEN": "streamlit-secret-token"},
    )

    assert secret_or_env_value("SPORTMONKS_API_TOKEN") == "streamlit-secret-token"
    assert sportmonks_token_configured() is True


def test_public_odds_refresh_state_round_trips_sanitized_summary(tmp_path):
    path = tmp_path / "refresh_state.json"
    save_public_odds_refresh_state(
        {
            "season_id": 26618,
            "fixtures_cached": 72,
            "fixture_details_cached": 20,
            "fixtures_considered": 3,
            "odds_cached": 2,
            "empty_odds": 1,
            "errors": 0,
            "cached_paths": ["/tmp/private/path.json"],
        },
        now_epoch=1000,
        path=path,
    )

    state = load_public_odds_refresh_state(path)

    assert state["last_refresh_epoch"] == 1000
    assert state["summary"] == {
        "season_id": 26618,
        "fixtures_cached": 72,
        "fixture_details_cached": 20,
        "fixtures_considered": 3,
        "odds_cached": 2,
        "empty_odds": 1,
        "errors": 0,
    }
    assert "cached_paths" not in state["summary"]


def test_public_odds_refresh_remaining_seconds_enforces_cooldown():
    state = {"last_refresh_epoch": 1000}

    assert public_odds_refresh_remaining_seconds(
        state,
        now_epoch=1100,
        cooldown_seconds=300,
    ) == 200
    assert public_odds_refresh_remaining_seconds(
        state,
        now_epoch=1400,
        cooldown_seconds=300,
    ) == 0


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


def final_fixture(home_goals=1, away_goals=0):
    match = fixture()
    match["fixture"]["status"] = {
        "elapsed": 90,
        "long": "Match Finished",
        "short": "FT",
    }
    match["goals"] = {"home": home_goals, "away": away_goals}
    return match


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


def final_features(home_goals=1, away_goals=0):
    data = features()
    data["status"] = "Match Finished"
    data["status_short"] = "FT"
    data["minute"] = 90
    data["home_goals"] = home_goals
    data["away_goals"] = away_goals
    data["score_difference"] = home_goals - away_goals
    return data


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


def final_prediction():
    data = prediction()
    data["prediction_mode"] = "final"
    data["home_win_probability"] = 1.0
    data["draw_probability"] = 0.0
    data["away_win_probability"] = 0.0
    data["model_confidence"] = 1.0
    return data


def test_probability_rows_group_outcome_and_next_goal_metrics():
    rows = probability_rows(prediction())

    assert [row["metric"] for row in rows[:3]] == ["Home win", "Draw", "Away win"]
    assert {row["group"] for row in rows} == {"match outcome", "next goal"}
    assert rows[0]["display"] == "68.0%"


def test_model_result_banner_data_marks_completed_model_win():
    banner = model_result_banner_data(
        final_fixture(1, 0),
        final_features(1, 0),
        final_prediction(),
        ratings={
            10: TeamRating(team_id=10, team_name="Alpha", rating=1700, matches_played=8),
            20: TeamRating(team_id=20, team_name="Beta", rating=1400, matches_played=8),
        },
    )

    assert banner["status"] == "win"
    assert banner["correct"] is True
    assert banner["predicted"] == "home"
    assert banner["actual"] == "home"
    assert banner["basis"] == "pre-match prior"
    assert "Our model WIN" == banner["headline"]


def test_model_result_banner_data_marks_completed_model_loss():
    banner = model_result_banner_data(
        final_fixture(1, 0),
        final_features(1, 0),
        final_prediction(),
        ratings={
            10: TeamRating(team_id=10, team_name="Alpha", rating=1300, matches_played=8),
            20: TeamRating(team_id=20, team_name="Beta", rating=1750, matches_played=8),
        },
    )

    assert banner["status"] == "loss"
    assert banner["correct"] is False
    assert banner["predicted"] == "away"
    assert banner["actual"] == "home"
    assert banner["basis"] == "pre-match prior"
    assert "Our model LOSS" == banner["headline"]


def test_model_result_banner_data_is_pending_for_unfinished_match():
    banner = model_result_banner_data(
        fixture(),
        features(),
        prediction(),
        ratings={},
    )

    assert banner["status"] == "pending"
    assert banner["correct"] is None
    assert banner["actual"] is None
    assert banner["basis"] == "live"


def test_running_accuracy_rows_sort_and_add_cumulative_accuracy():
    rows = running_accuracy_rows(
        [
            {
                "fixture_id": 3,
                "kickoff_utc": "2026-06-14T10:00:00+00:00",
                "match": "Gamma vs Delta",
                "correct": True,
            },
            {
                "fixture_id": 1,
                "kickoff_utc": "2026-06-12T10:00:00+00:00",
                "match": "Alpha vs Beta",
                "correct": True,
            },
            {
                "fixture_id": 2,
                "kickoff_utc": "2026-06-13T10:00:00+00:00",
                "match": "Alpha vs Gamma",
                "correct": False,
            },
        ]
    )

    assert [row["fixture_id"] for row in rows] == [1, 2, 3]
    assert [row["match_number"] for row in rows] == [1, 2, 3]
    assert [row["cumulative_correct"] for row in rows] == [1, 1, 2]
    assert rows[0]["running_accuracy_display"] == "100.0%"
    assert rows[1]["running_accuracy_display"] == "50.0%"
    assert rows[2]["running_accuracy_display"] == "66.7%"


def test_api_football_predicted_outcome_uses_highest_available_probability():
    assert api_football_predicted_outcome(api_prediction()) == "home"
    unavailable = api_prediction()
    unavailable["available"] = False

    assert api_football_predicted_outcome(unavailable) is None


def test_api_football_running_accuracy_rows_skip_unavailable_predictions():
    rows = running_accuracy_rows(
        [
            {
                "fixture_id": 1,
                "kickoff_utc": "2026-06-12T10:00:00+00:00",
                "match": "Alpha vs Beta",
                "actual": "home",
                "correct": True,
            },
            {
                "fixture_id": 2,
                "kickoff_utc": "2026-06-13T10:00:00+00:00",
                "match": "Gamma vs Delta",
                "actual": "away",
                "correct": False,
            },
        ]
    )
    home_prediction = api_prediction()
    away_prediction = api_prediction()
    away_prediction["home_probability"] = 0.2
    away_prediction["draw_probability"] = 0.1
    away_prediction["away_probability"] = 0.7

    enriched = api_football_running_accuracy_rows(
        rows,
        {
            1: home_prediction,
            2: away_prediction,
        },
    )

    assert enriched[0]["api_football_predicted"] == "home"
    assert enriched[0]["api_football_correct"] is True
    assert enriched[0]["api_football_running_accuracy_display"] == "100.0%"
    assert enriched[1]["api_football_predicted"] == "away"
    assert enriched[1]["api_football_correct"] is True
    assert enriched[1]["api_football_running_accuracy_display"] == "100.0%"


def test_api_football_accuracy_summary_counts_available_rows_only():
    rows = [
        {"api_football_correct": True},
        {"api_football_correct": False},
        {"api_football_correct": None},
    ]
    summary = api_football_accuracy_summary(rows)

    assert summary == {
        "evaluated": 2,
        "correct": 1,
        "accuracy": 0.5,
        "unavailable": 1,
    }


def test_fetch_api_football_prediction_uses_cache_before_live_api(monkeypatch):
    import app.streamlit_app as app_module

    cached = {
        "fixture_id": 9001,
        "available": True,
        "status": "available",
        "home_probability": 0.55,
        "draw_probability": 0.25,
        "away_probability": 0.20,
    }
    fetch_api_football_prediction.clear()
    monkeypatch.setattr(
        app_module,
        "load_api_prediction_cache",
        lambda fixture_id: cached if fixture_id == 9001 else None,
    )
    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda: (_ for _ in ()).throw(AssertionError("settings should not load")),
    )

    assert fetch_api_football_prediction(9001) == cached
    fetch_api_football_prediction.clear()


def api_prediction():
    return {
        "available": True,
        "status": "available",
        "endpoint": "/predictions?fixture=9001",
        "last_error": "",
        "home_probability": 0.55,
        "draw_probability": 0.25,
        "away_probability": 0.20,
        "home_display": "55.0%",
        "draw_display": "25.0%",
        "away_display": "20.0%",
        "advice": "Double chance : Alpha or draw",
        "winner_name": "Alpha",
        "winner_comment": "Win or draw",
        "win_or_draw": True,
        "under_over": "-3.5",
        "goals_home": "-2.5",
        "goals_away": "-1.5",
    }


def test_probability_difference_formats_percentage_points():
    assert probability_difference(0.68, 0.55) == "+13.0 pp"
    assert probability_difference(0.21, 0.25) == "-4.0 pp"
    assert probability_difference(0.21, None) == "-"


def test_model_comparison_rows_show_our_model_and_api_football():
    rows = model_comparison_rows(prediction(), api_prediction())
    metrics = {row["metric"]: row for row in rows}

    assert metrics["Home win"]["our_model"] == "68.0%"
    assert metrics["Home win"]["api_football"] == "55.0%"
    assert metrics["Home win"]["difference"] == "+13.0 pp"
    assert metrics["Home scores next"]["api_football"] == "-"


def test_comparison_status_rows_show_endpoint_and_availability():
    rows = comparison_status_rows(api_prediction())

    assert rows[0] == {
        "source": "API-Football prediction endpoint",
        "status": "available",
        "detail": "/predictions?fixture=9001",
    }
    assert rows[1]["status"] == "available"


def test_api_football_advice_rows_show_prediction_details():
    rows = api_football_advice_rows(api_prediction())
    values = {row["field"]: row["value"] for row in rows}

    assert values["Advice"] == "Double chance : Alpha or draw"
    assert values["Winner"] == "Alpha"
    assert values["Win or draw"] == "true"
    assert values["Predicted goals"] == "-2.5 - -1.5"


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


def test_sportmonks_provider_status_rows_do_not_expose_token(monkeypatch):
    import app.streamlit_app as app_module

    monkeypatch.setattr(
        app_module,
        "load_settings",
        lambda require_api_key=False: type(
            "Settings",
            (),
            {
                "sportmonks_api_token": "sportmonks-secret",
                "sportmonks_base_url": "https://api.sportmonks.test/v3/football",
            },
        )(),
    )
    monkeypatch.setattr(
        app_module,
        "load_latest_sportmonks_audit",
        lambda: {
            "audit_file": "/tmp/audit.json",
            "summary": {
                "accessible_categories": ["leagues", "world_cup_search"],
                "world_cup_2026_season_ids": [202699],
                "metadata": {
                    "subscription": {"plan": "World Cup All-in"},
                    "rate_limit": {"remaining": 2999},
                },
            },
            "checks": {},
        },
    )

    rows = sportmonks_provider_status_rows()
    dumped = str(rows)

    assert "sportmonks-secret" not in dumped
    assert rows[0]["status"] == "present"
    assert any(row["source"] == "Accessible categories" for row in rows)


def test_sportmonks_audit_check_rows_flatten_latest_audit():
    rows = sportmonks_audit_check_rows(
        {
            "checks": {
                "leagues": {
                    "status": "available",
                    "available": True,
                    "record_count": 4,
                    "endpoint": "leagues",
                },
                "odds": {
                    "status": "error",
                    "available": False,
                    "record_count": 0,
                    "error": "SportMonks HTTP 403: package missing",
                },
            }
        }
    )

    assert rows[0] == {
        "category": "leagues",
        "status": "available",
        "available": True,
        "records": 4,
        "detail": "leagues",
    }
    assert rows[1]["category"] == "odds"
    assert rows[1]["status"] == "error"


def test_sportmonks_mapping_metric_rows_summarize_provider_coverage():
    rows = sportmonks_mapping_metric_rows(
        [
            {
                "mapping_confidence": "exact",
                "fixture_detail_available": True,
                "xg_pair_available": True,
                "news_count": 1,
            },
            {
                "mapping_confidence": "no_match",
                "fixture_detail_available": False,
                "xg_pair_available": False,
                "news_count": 0,
            },
        ]
    )
    metrics = {row["metric"]: row for row in rows}

    assert metrics["API-Football fixtures"]["value"] == 2
    assert metrics["Mapped to SportMonks"]["value"] == 1
    assert metrics["xG pair cache"]["value"] == 1
    assert metrics["News cache"]["value"] == 1


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
