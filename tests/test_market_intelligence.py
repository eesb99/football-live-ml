from __future__ import annotations

from datetime import timedelta, timezone

from src.market_intelligence import (
    benchmark_market_gate,
    bookmaker_probability_rows,
    closing_line_for_outcome,
    consensus_market_snapshot,
    market_edge_rows_for_fixtures,
    normalize_outcome_label,
    odds_movement_for_outcome,
    pre_kickoff_snapshots,
)


def fixture() -> dict:
    return {
        "fixture": {
            "id": 9001,
            "date": "2026-06-14T10:00:00+00:00",
            "status": {"short": "NS"},
        },
        "teams": {
            "home": {"id": 10, "name": "Alpha"},
            "away": {"id": 20, "name": "Beta"},
        },
        "goals": {"home": None, "away": None},
    }


def odds_record(bookmaker_id: int, label: str, value: str) -> dict:
    return {
        "fixture_id": 19609127,
        "market_id": 1,
        "market_description": "Fulltime Result",
        "bookmaker_id": bookmaker_id,
        "label": label,
        "value": value,
        "latest_bookmaker_update": "2026-06-14 08:00:00",
        "stopped": False,
    }


def odds_cache() -> dict:
    return {
        "cache_key": "19609127",
        "captured_at": "2026-06-14T09:00:00+00:00",
        "payload": {
            "data": [
                odds_record(1, "Home", "2.20"),
                odds_record(1, "Draw", "3.40"),
                odds_record(1, "Away", "3.20"),
                odds_record(2, "Home", "2.10"),
                odds_record(2, "Draw", "3.50"),
                odds_record(2, "Away", "3.40"),
            ]
        },
    }


def test_normalize_outcome_label_handles_common_1x2_labels():
    assert normalize_outcome_label("Home") == "home"
    assert normalize_outcome_label("X") == "draw"
    assert normalize_outcome_label("2") == "away"
    assert normalize_outcome_label("Over") is None


def test_bookmaker_probability_rows_remove_overround_by_bookmaker():
    rows = bookmaker_probability_rows(odds_cache()["payload"]["data"])

    assert len(rows) == 2
    assert round(
        rows[0]["home_probability"]
        + rows[0]["draw_probability"]
        + rows[0]["away_probability"],
        6,
    ) == 1.0
    assert rows[0]["overround"] > 1.0


def test_consensus_market_snapshot_averages_complete_bookmaker_triplets():
    snapshot = consensus_market_snapshot(odds_cache())

    assert snapshot is not None
    assert snapshot["fixture_id"] == 19609127
    assert snapshot["bookmaker_count"] == 2
    assert snapshot["home_best_decimal"] == 2.2
    assert snapshot["away_best_decimal"] == 3.4
    assert round(
        snapshot["home_probability"]
        + snapshot["draw_probability"]
        + snapshot["away_probability"],
        6,
    ) == 1.0


def test_benchmark_market_gate_requires_enough_clean_brier_and_log_loss():
    blocked = benchmark_market_gate(
        {
            "shared_evaluated": 8,
            "our_brier_score": 0.62,
            "api_brier_score": 0.60,
            "our_log_loss": 1.03,
            "api_log_loss": 0.95,
        },
        {"candidate_proves_improvement": False},
    )
    passed = benchmark_market_gate(
        {
            "shared_evaluated": 12,
            "our_brier_score": 0.55,
            "api_brier_score": 0.60,
            "our_log_loss": 0.90,
            "api_log_loss": 0.95,
        },
        {"candidate_proves_improvement": False},
    )

    assert blocked["passed"] is False
    assert blocked["reason"] == "insufficient_backtest_rows"
    assert passed["passed"] is True


def test_market_edge_rows_block_flags_until_benchmark_gate_passes():
    snapshot = {
        "fixture_id": 19609127,
        "captured_at": "2026-06-14T09:00:00+00:00",
        "bookmaker_count": 4,
        "latest_bookmaker_update": "2026-06-14 08:00:00",
        "average_overround": 1.05,
        "home_probability": 0.30,
        "draw_probability": 0.32,
        "away_probability": 0.38,
        "home_best_decimal": 3.8,
        "draw_best_decimal": 3.4,
        "away_best_decimal": 2.8,
    }
    coverage = [{"api_fixture_id": 9001, "sportmonks_fixture_id": 19609127}]

    blocked = market_edge_rows_for_fixtures(
        [fixture()],
        {},
        coverage,
        benchmark_gate={"passed": False, "reason": "insufficient_backtest_rows"},
        snapshots_by_fixture={19609127: [snapshot]},
        min_edge=0.01,
        min_expected_value=0.01,
    )
    passed = market_edge_rows_for_fixtures(
        [fixture()],
        {},
        coverage,
        benchmark_gate={"passed": True, "reason": "passed_brier_log_loss_gate"},
        snapshots_by_fixture={19609127: [snapshot]},
        min_edge=0.01,
        min_expected_value=0.01,
    )

    assert blocked[0]["status"] == "blocked_benchmark_gate"
    assert blocked[0]["edge_flag"] is False
    assert passed[0]["status"] == "paper_trade_candidate"
    assert passed[0]["edge_flag"] is True


def test_closing_line_for_outcome_tracks_probability_and_decimal_deltas():
    snapshots = [
        {
            "home_best_decimal": 3.8,
            "home_probability": 0.30,
        },
        {
            "home_best_decimal": 3.2,
            "home_probability": 0.36,
        },
    ]

    clv = closing_line_for_outcome(snapshots, "home")

    assert clv["clv_available"] is True
    assert round(clv["clv_decimal_delta"], 2) == 0.60
    assert round(clv["clv_probability_delta"], 2) == 0.06


def test_odds_movement_tracks_first_latest_best_worst_and_edge_change():
    snapshots = [
        {
            "captured_at": "2026-06-14T08:00:00+00:00",
            "home_best_decimal": 3.8,
            "home_probability": 0.30,
        },
        {
            "captured_at": "2026-06-14T09:00:00+00:00",
            "home_best_decimal": 3.2,
            "home_probability": 0.36,
        },
    ]

    movement = odds_movement_for_outcome(
        snapshots,
        "home",
        model_probability=0.42,
        kickoff_utc="2026-06-14T10:00:00+00:00",
    )

    assert movement["first_decimal"] == 3.8
    assert movement["latest_decimal"] == 3.2
    assert movement["best_seen_decimal"] == 3.8
    assert movement["worst_seen_decimal"] == 3.2
    assert round(movement["first_edge"], 2) == 0.12
    assert round(movement["latest_edge"], 2) == 0.06
    assert round(movement["edge_change"], 2) == -0.06
    assert round(movement["first_expected_value"], 3) == 0.596
    assert round(movement["latest_expected_value"], 3) == 0.344
    assert movement["clv_direction"] == "favorable"
    assert movement["first_hours_to_kickoff"] == 2.0
    assert movement["latest_hours_to_kickoff"] == 1.0


def test_pre_kickoff_snapshots_treat_timezone_naive_captures_as_local_time():
    snapshots = [{"captured_at": "2026-06-14T23:45:54"}]

    filtered = pre_kickoff_snapshots(
        snapshots,
        "2026-06-14T20:00:00+00:00",
        local_capture_timezone=timezone(timedelta(hours=8)),
    )

    assert filtered == snapshots
