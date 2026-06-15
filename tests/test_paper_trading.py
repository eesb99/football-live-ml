from src.paper_trading import (
    capped_fractional_kelly,
    kelly_fraction,
    paper_research_candidate,
    paper_trade_rows,
    paper_trade_summary,
)


def fixture(status_short="NS", home_goals=None, away_goals=None) -> dict:
    return {
        "fixture": {
            "id": 9001,
            "date": "2026-06-14T10:00:00+00:00",
            "status": {"short": status_short},
        },
        "teams": {
            "home": {"id": 10, "name": "Alpha"},
            "away": {"id": 20, "name": "Beta"},
        },
        "goals": {"home": home_goals, "away": away_goals},
    }


def market_row(**overrides) -> dict:
    row = {
        "fixture_id": 9001,
        "match": "Alpha vs Beta",
        "status": "blocked_benchmark_gate",
        "outcome": "draw",
        "model_probability": 0.36,
        "market_probability": 0.27,
        "edge": 0.09,
        "best_decimal": 3.60,
        "expected_value": 0.296,
        "bookmaker_count": 12,
        "average_overround": 1.06,
        "market_snapshots": 2,
        "benchmark_gate": "brier_log_loss_not_better",
        "first_decimal": 3.80,
        "latest_decimal": 3.60,
        "clv_direction": "favorable",
    }
    row.update(overrides)
    return row


def test_kelly_fraction_and_cap_are_conservative():
    raw = kelly_fraction(0.45, 2.50)
    capped = capped_fractional_kelly(
        0.45,
        2.50,
        kelly_multiplier=0.10,
        stake_cap_fraction=0.0025,
    )

    assert round(raw, 4) == 0.0833
    assert capped == 0.0025
    assert kelly_fraction(0.30, 2.0) == 0.0


def test_paper_research_candidate_filters_market_quality():
    assert paper_research_candidate(market_row())
    assert not paper_research_candidate(market_row(edge=0.03))
    assert not paper_research_candidate(market_row(expected_value=0.01))
    assert not paper_research_candidate(market_row(bookmaker_count=1))
    assert not paper_research_candidate(market_row(average_overround=1.20))


def test_paper_trade_rows_settle_pnl_and_keep_real_stake_zero():
    rows = paper_trade_rows(
        [fixture("FT", 1, 1)],
        [market_row()],
        paper_bankroll=1000,
        kelly_multiplier=0.10,
        stake_cap_fraction=0.0025,
    )

    assert rows[0]["paper_status"] == "won"
    assert rows[0]["paper_stake_units"] == 2.5
    assert rows[0]["paper_pnl_units"] == 6.5
    assert rows[0]["paper_pnl_entry_first_units"] == 7.0
    assert rows[0]["paper_pnl_entry_latest_units"] == 6.5
    assert rows[0]["paper_pnl_first_vs_latest_delta_units"] == 0.5
    assert rows[0]["real_stake_units"] == 0.0


def test_paper_trade_summary_tracks_open_exposure():
    rows = paper_trade_rows(
        [fixture()],
        [market_row(), market_row(fixture_id=9002, match="Gamma vs Delta")],
        paper_bankroll=1000,
        kelly_multiplier=0.10,
        stake_cap_fraction=0.0025,
    )
    summary = paper_trade_summary(rows)

    assert summary["research_candidates"] == 2
    assert summary["settled"] == 0
    assert summary["open"] == 2
    assert summary["open_exposure_units"] == 5.0
    assert summary["open_possible_profit_entry_first_units"] == 14.0
    assert summary["open_possible_profit_entry_latest_units"] == 13.0
    assert summary["clv_tracked"] == 2
    assert summary["clv_favorable"] == 2
    assert summary["real_stake_units"] == 0.0
