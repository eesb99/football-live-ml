from __future__ import annotations

from src.benchmark import (
    benchmark_diagnostic_counts,
    benchmark_diagnostic_rows,
    brier_score,
    draw_diagnostic_summary,
    draw_miss_diagnostic_rows,
    draw_risk_label,
    fair_api_comparison_rows,
    fair_benchmark_summary,
    log_loss,
    sportmonks_candidate_rows,
    sportmonks_candidate_summary,
    sportmonks_enrichment_is_non_leaky,
    walk_forward_backtest_rows,
)
from src.ratings import DEFAULT_RATING


def fixture(
    fixture_id: int,
    date: str,
    home_id: int,
    home_name: str,
    away_id: int,
    away_name: str,
    home_goals: int,
    away_goals: int,
) -> dict:
    return {
        "fixture": {
            "id": fixture_id,
            "date": date,
            "status": {
                "elapsed": 90,
                "long": "Match Finished",
                "short": "FT",
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
            "home": {"id": home_id, "name": home_name},
            "away": {"id": away_id, "name": away_name},
        },
        "goals": {"home": home_goals, "away": away_goals},
    }


def api_prediction(home=0.6, draw=0.2, away=0.2, available=True) -> dict:
    return {
        "available": available,
        "status": "available" if available else "missing",
        "home_probability": home if available else None,
        "draw_probability": draw if available else None,
        "away_probability": away if available else None,
    }


def test_walk_forward_scores_fixture_before_updating_ratings_from_that_result():
    rows = walk_forward_backtest_rows(
        [
            fixture(
                2,
                "2026-06-13T10:00:00+00:00",
                10,
                "Alpha",
                20,
                "Beta",
                2,
                0,
            ),
            fixture(
                1,
                "2026-06-12T10:00:00+00:00",
                10,
                "Alpha",
                20,
                "Beta",
                0,
                1,
            ),
        ]
    )

    assert [row["fixture_id"] for row in rows] == [1, 2]
    assert rows[0]["home_rating_before"] == DEFAULT_RATING
    assert rows[0]["away_rating_before"] == DEFAULT_RATING
    assert rows[0]["home_matches_before"] == 0
    assert rows[0]["away_matches_before"] == 0
    assert rows[1]["home_matches_before"] == 1
    assert rows[1]["away_matches_before"] == 1
    assert rows[1]["home_rating_before"] < DEFAULT_RATING
    assert rows[1]["away_rating_before"] > DEFAULT_RATING
    assert rows[0]["home_form_matches_before"] == 0
    assert rows[0]["away_form_matches_before"] == 0
    assert rows[1]["home_form_matches_before"] == 1
    assert rows[1]["away_form_matches_before"] == 1
    assert rows[1]["home_form_signal"] < 0
    assert rows[1]["away_form_signal"] > 0


def test_brier_score_and_log_loss_for_known_probabilities():
    probabilities = {"home": 0.7, "draw": 0.2, "away": 0.1}

    assert round(brier_score(probabilities, "home"), 3) == 0.140
    assert round(log_loss(probabilities, "home"), 3) == 0.357


def test_draw_risk_label_classifies_clear_cases():
    assert draw_risk_label(0.08, 2.2, 35.0, 0.05, 0.08) == "high"
    assert draw_risk_label(0.28, 2.9, 120.0, 0.35, 0.20) == "medium"
    assert draw_risk_label(0.60, 3.4, 240.0, 0.80, 0.35) == "low"


def test_walk_forward_rows_include_draw_diagnostics():
    rows = walk_forward_backtest_rows(
        [
            fixture(
                1,
                "2026-06-12T10:00:00+00:00",
                10,
                "Alpha",
                20,
                "Beta",
                1,
                1,
            )
        ]
    )
    row = rows[0]

    assert row["actual_is_draw"] is True
    assert row["our_draw_miss"] == (row["predicted"] != "draw")
    assert row["expected_goal_gap"] >= 0
    assert row["total_expected_goals"] > 0
    assert row["draw_rank"] in {1, 2, 3}
    assert row["draw_risk_label"] in {"low", "medium", "high"}
    assert "top_vs_draw_margin" in row


def test_fair_comparison_uses_only_shared_api_available_fixtures_for_summary():
    rows = [
        {
            "fixture_id": 1,
            "actual": "home",
            "correct": True,
            "confidence": 0.60,
            "brier_score": 0.14,
            "log_loss": 0.35,
        },
        {
            "fixture_id": 2,
            "actual": "away",
            "correct": False,
            "confidence": 0.55,
            "brier_score": 1.10,
            "log_loss": 1.60,
        },
    ]
    enriched = fair_api_comparison_rows(
        rows,
        {
            1: api_prediction(home=0.55, draw=0.25, away=0.20),
            2: api_prediction(available=False),
        },
    )
    summary = fair_benchmark_summary(enriched)

    assert enriched[0]["api_football_predicted"] == "home"
    assert enriched[0]["api_football_correct"] is True
    assert enriched[1]["api_football_predicted"] == "-"
    assert enriched[1]["api_football_correct"] is None
    assert summary["completed"] == 2
    assert summary["shared_evaluated"] == 1
    assert summary["api_unavailable"] == 1
    assert summary["our_shared_accuracy"] == 1.0
    assert summary["api_accuracy"] == 1.0


def test_draw_diagnostic_summary_counts_draw_misses_and_api_draw_detection():
    rows = [
        {
            "fixture_id": 1,
            "match": "Alpha vs Beta",
            "score": "1-1",
            "actual": "draw",
            "actual_is_draw": True,
            "predicted": "home",
            "our_draw_miss": True,
            "draw_probability": 0.28,
            "top_vs_draw_margin": 0.12,
            "expected_goal_gap": 0.1,
            "total_expected_goals": 2.2,
            "rating_gap": 40.0,
            "form_gap": 0.05,
            "draw_risk_label": "high",
        },
        {
            "fixture_id": 2,
            "match": "Gamma vs Delta",
            "score": "2-0",
            "actual": "home",
            "actual_is_draw": False,
            "predicted": "home",
            "our_draw_miss": False,
            "draw_probability": 0.22,
            "top_vs_draw_margin": 0.30,
        },
    ]
    enriched = fair_api_comparison_rows(
        rows,
        {
            1: api_prediction(home=0.35, draw=0.40, away=0.25),
            2: api_prediction(home=0.60, draw=0.20, away=0.20),
        },
    )
    summary = draw_diagnostic_summary(enriched)
    draw_rows = draw_miss_diagnostic_rows(enriched)

    assert enriched[0]["api_predicted_draw"] is True
    assert enriched[0]["api_draw_miss"] is False
    assert summary["actual_draws"] == 1
    assert summary["our_draw_misses"] == 1
    assert summary["api_draw_misses"] == 0
    assert summary["average_draw_probability_on_draws"] == 0.28
    assert summary["average_draw_probability_on_non_draws"] == 0.22
    assert summary["average_top_vs_draw_margin_on_draw_misses"] == 0.12
    assert draw_rows[0]["draw_risk_label"] == "high"


def test_diagnostic_counts_classify_api_and_model_disagreements():
    rows = [
        {
            "fixture_id": 1,
            "match_number": 1,
            "match": "Alpha vs Beta",
            "actual": "home",
            "predicted": "home",
            "correct": True,
            "home_probability": 0.60,
            "draw_probability": 0.25,
            "away_probability": 0.15,
        },
        {
            "fixture_id": 2,
            "match_number": 2,
            "match": "Gamma vs Delta",
            "actual": "away",
            "predicted": "home",
            "correct": False,
            "home_probability": 0.55,
            "draw_probability": 0.25,
            "away_probability": 0.20,
        },
        {
            "fixture_id": 3,
            "match_number": 3,
            "match": "Echo vs Foxtrot",
            "actual": "draw",
            "predicted": "home",
            "correct": False,
            "home_probability": 0.45,
            "draw_probability": 0.30,
            "away_probability": 0.25,
        },
        {
            "fixture_id": 4,
            "match_number": 4,
            "match": "Hotel vs India",
            "actual": "home",
            "predicted": "home",
            "correct": True,
            "home_probability": 0.52,
            "draw_probability": 0.27,
            "away_probability": 0.21,
        },
    ]
    enriched = fair_api_comparison_rows(
        rows,
        {
            1: api_prediction(home=0.70, draw=0.20, away=0.10),
            2: api_prediction(home=0.20, draw=0.20, away=0.60),
            3: api_prediction(home=0.50, draw=0.30, away=0.20),
            4: api_prediction(home=0.20, draw=0.20, away=0.60),
        },
    )
    counts = benchmark_diagnostic_counts(enriched)
    diagnostics = benchmark_diagnostic_rows(enriched)

    assert counts["both_correct"] == 1
    assert counts["both_wrong"] == 1
    assert counts["api_only_wins"] == 1
    assert counts["our_only_wins"] == 1
    assert counts["draw_misses"] == 1
    assert counts["away_underdog_misses"] == 1
    assert {row["category"] for row in diagnostics} == {
        "api_only_win",
        "both_wrong",
        "our_only_win",
    }


def candidate_base_row() -> dict:
    return {
        "fixture_id": 9001,
        "kickoff_utc": "2026-06-14T10:00:00+00:00",
        "actual": "home",
        "predicted": "draw",
        "correct": False,
        "home_expected_goals": 1.0,
        "away_expected_goals": 1.0,
        "home_probability": 0.32,
        "draw_probability": 0.36,
        "away_probability": 0.32,
        "brier_score": 0.92,
        "log_loss": 1.15,
        "rating_gap": 0.0,
        "form_gap": 0.0,
    }


def sportmonks_enrichment(**overrides) -> dict:
    enrichment = {
        "sportmonks_fixture_id": 19609156,
        "mapping_confidence": "exact",
        "mapping_score": 1.0,
        "home_sportmonks_xg": 2.1,
        "away_sportmonks_xg": 0.4,
        "xg_pair_available": True,
        "captured_at": "2026-06-14T09:00:00+00:00",
        "availability": "pre_match_xg",
        "available_before_kickoff": True,
        "news_count": 1,
    }
    enrichment.update(overrides)
    return enrichment


def test_sportmonks_non_leaky_guard_requires_pre_kickoff_availability():
    assert sportmonks_enrichment_is_non_leaky(
        "2026-06-14T10:00:00+00:00",
        "2026-06-14T09:59:00+00:00",
        available_before_kickoff=True,
        availability="pre_match_xg",
    )
    assert not sportmonks_enrichment_is_non_leaky(
        "2026-06-14T10:00:00+00:00",
        "2026-06-14T11:00:00+00:00",
        available_before_kickoff=True,
        availability="pre_match_xg",
    )
    assert not sportmonks_enrichment_is_non_leaky(
        "2026-06-14T10:00:00+00:00",
        "2026-06-14T09:00:00+00:00",
        available_before_kickoff=False,
        availability="post_match_only",
    )


def test_sportmonks_candidate_blocks_post_match_or_late_xg():
    rows = sportmonks_candidate_rows(
        [candidate_base_row()],
        {
            9001: sportmonks_enrichment(
                captured_at="2026-06-14T11:00:00+00:00",
                availability="post_match_only",
                available_before_kickoff=False,
            )
        },
    )
    summary = sportmonks_candidate_summary(rows)

    assert rows[0]["sportmonks_candidate_status"] == "unavailable"
    assert rows[0]["sportmonks_candidate_reason"] == "post_match_or_late_enrichment"
    assert rows[0]["sportmonks_candidate_correct"] is None
    assert summary["eligible"] == 0
    assert summary["headline_recommendation"] == "keep_current_model"


def test_sportmonks_candidate_scores_eligible_pre_match_xg_separately():
    rows = sportmonks_candidate_rows(
        [candidate_base_row()],
        {9001: sportmonks_enrichment()},
    )
    summary = sportmonks_candidate_summary(rows, min_evaluation_rows=1)

    assert rows[0]["sportmonks_candidate_status"] == "eligible"
    assert rows[0]["sportmonks_candidate_correct"] is True
    assert rows[0]["sportmonks_candidate_brier_score"] is not None
    assert rows[0]["sportmonks_candidate_log_loss"] is not None
    assert summary["eligible"] == 1
    assert summary["candidate_proves_improvement"] is True
