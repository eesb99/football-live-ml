from __future__ import annotations

from typing import Any

import numpy as np

from src.benchmark import actual_outcome
from src.ratings import fixture_result


DEFAULT_PAPER_BANKROLL = 1000.0
DEFAULT_KELLY_MULTIPLIER = 0.10
DEFAULT_STAKE_CAP_FRACTION = 0.0025
DEFAULT_RESEARCH_MIN_EDGE = 0.08
DEFAULT_RESEARCH_MIN_EXPECTED_VALUE = 0.06
DEFAULT_MIN_BOOKMAKERS = 3
DEFAULT_MAX_OVERROUND = 1.08


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0 or model_probability <= 0.0:
        return 0.0
    fraction = (decimal_odds * model_probability - 1.0) / (decimal_odds - 1.0)
    return float(max(0.0, fraction))


def capped_fractional_kelly(
    model_probability: float,
    decimal_odds: float,
    *,
    kelly_multiplier: float = DEFAULT_KELLY_MULTIPLIER,
    stake_cap_fraction: float = DEFAULT_STAKE_CAP_FRACTION,
) -> float:
    raw = kelly_fraction(model_probability, decimal_odds)
    return float(
        np.clip(raw * kelly_multiplier, 0.0, max(stake_cap_fraction, 0.0))
    )


def paper_research_candidate(
    row: dict[str, Any],
    *,
    min_edge: float = DEFAULT_RESEARCH_MIN_EDGE,
    min_expected_value: float = DEFAULT_RESEARCH_MIN_EXPECTED_VALUE,
    min_bookmakers: int = DEFAULT_MIN_BOOKMAKERS,
    max_overround: float = DEFAULT_MAX_OVERROUND,
) -> bool:
    if row.get("status") == "no_pre_kickoff_market":
        return False
    if row.get("best_decimal") is None:
        return False
    return (
        float(row.get("edge") or 0.0) >= min_edge
        and float(row.get("expected_value") or 0.0) >= min_expected_value
        and int(row.get("bookmaker_count") or 0) >= min_bookmakers
        and float(row.get("average_overround") or 99.0) <= max_overround
    )


def _fixture_by_id(fixtures: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int((fixture.get("fixture") or {}).get("id") or 0): fixture
        for fixture in fixtures
    }


def _paper_pnl_for_decimal(
    *,
    decimal_odds: float | None,
    stake: float,
    settled: bool,
    won: bool,
) -> float:
    if not settled or stake <= 0.0:
        return 0.0
    if decimal_odds is None or decimal_odds <= 1.0:
        return -stake
    return stake * (decimal_odds - 1.0) if won else -stake


def paper_trade_rows(
    fixtures: list[dict[str, Any]],
    market_rows: list[dict[str, Any]],
    *,
    paper_bankroll: float = DEFAULT_PAPER_BANKROLL,
    kelly_multiplier: float = DEFAULT_KELLY_MULTIPLIER,
    stake_cap_fraction: float = DEFAULT_STAKE_CAP_FRACTION,
    min_edge: float = DEFAULT_RESEARCH_MIN_EDGE,
    min_expected_value: float = DEFAULT_RESEARCH_MIN_EXPECTED_VALUE,
    min_bookmakers: int = DEFAULT_MIN_BOOKMAKERS,
    max_overround: float = DEFAULT_MAX_OVERROUND,
) -> list[dict[str, Any]]:
    fixtures_by_id = _fixture_by_id(fixtures)
    rows: list[dict[str, Any]] = []
    for row in market_rows:
        if row.get("status") == "no_pre_kickoff_market":
            continue
        fixture_id = int(row.get("fixture_id") or 0)
        fixture = fixtures_by_id.get(fixture_id, {})
        result = fixture_result(fixture)
        settled = result is not None
        settled_outcome = actual_outcome(*result) if result is not None else None
        decimal_odds = float(row.get("best_decimal") or 0.0)
        model_probability = float(row.get("model_probability") or 0.0)
        raw_kelly = kelly_fraction(model_probability, decimal_odds)
        paper_fraction = capped_fractional_kelly(
            model_probability,
            decimal_odds,
            kelly_multiplier=kelly_multiplier,
            stake_cap_fraction=stake_cap_fraction,
        )
        research_candidate = paper_research_candidate(
            row,
            min_edge=min_edge,
            min_expected_value=min_expected_value,
            min_bookmakers=min_bookmakers,
            max_overround=max_overround,
        )
        paper_stake = paper_bankroll * paper_fraction if research_candidate else 0.0
        outcome = str(row.get("outcome") or "-")
        won = settled and settled_outcome == outcome and research_candidate
        if not research_candidate:
            settlement_status = "filtered"
            paper_pnl = 0.0
        elif not settled:
            settlement_status = "open"
            paper_pnl = 0.0
        elif won:
            settlement_status = "won"
            paper_pnl = paper_stake * (decimal_odds - 1.0)
        else:
            settlement_status = "lost"
            paper_pnl = -paper_stake
        first_decimal = (
            float(row["first_decimal"]) if row.get("first_decimal") is not None else None
        )
        latest_decimal = (
            float(row["latest_decimal"]) if row.get("latest_decimal") is not None else None
        )
        first_entry_pnl = _paper_pnl_for_decimal(
            decimal_odds=first_decimal,
            stake=paper_stake,
            settled=settled and research_candidate,
            won=bool(won),
        )
        latest_entry_pnl = _paper_pnl_for_decimal(
            decimal_odds=latest_decimal or decimal_odds,
            stake=paper_stake,
            settled=settled and research_candidate,
            won=bool(won),
        )

        rows.append(
            {
                **row,
                "paper_status": settlement_status,
                "research_candidate": research_candidate,
                "settled": settled,
                "settled_outcome": settled_outcome or "open",
                "paper_bankroll": paper_bankroll,
                "raw_kelly_fraction": raw_kelly,
                "fractional_kelly_multiplier": kelly_multiplier,
                "paper_stake_fraction": paper_fraction if research_candidate else 0.0,
                "paper_stake_units": paper_stake,
                "real_stake_units": 0.0,
                "paper_pnl_units": paper_pnl,
                "paper_pnl_entry_first_units": first_entry_pnl,
                "paper_pnl_entry_latest_units": latest_entry_pnl,
                "paper_pnl_first_vs_latest_delta_units": first_entry_pnl - latest_entry_pnl,
                "paper_possible_profit_units": (
                    paper_stake * (decimal_odds - 1.0)
                    if research_candidate and not settled
                    else 0.0
                ),
                "paper_possible_profit_entry_first_units": (
                    paper_stake * ((first_decimal or decimal_odds) - 1.0)
                    if research_candidate and not settled and (first_decimal or decimal_odds) > 1.0
                    else 0.0
                ),
                "paper_possible_profit_entry_latest_units": (
                    paper_stake * ((latest_decimal or decimal_odds) - 1.0)
                    if research_candidate and not settled and (latest_decimal or decimal_odds) > 1.0
                    else 0.0
                ),
                "paper_possible_loss_units": (
                    paper_stake if research_candidate and not settled else 0.0
                ),
                "bookie_friendly_cap_units": paper_bankroll * stake_cap_fraction,
            }
        )
    return rows


def paper_trade_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("research_candidate")]
    settled = [row for row in candidates if row.get("settled")]
    open_rows = [
        row
        for row in candidates
        if row.get("paper_status") == "open"
    ]
    wins = [row for row in settled if row.get("paper_status") == "won"]
    losses = [row for row in settled if row.get("paper_status") == "lost"]
    settled_stake = float(sum(float(row.get("paper_stake_units") or 0.0) for row in settled))
    realized_pnl = float(sum(float(row.get("paper_pnl_units") or 0.0) for row in settled))
    open_exposure = float(
        sum(float(row.get("paper_possible_loss_units") or 0.0) for row in open_rows)
    )
    clv_rows = [row for row in candidates if row.get("clv_direction") != "unavailable"]
    favorable_clv = [row for row in clv_rows if row.get("clv_direction") == "favorable"]
    unfavorable_clv = [row for row in clv_rows if row.get("clv_direction") == "unfavorable"]
    flat_clv = [row for row in clv_rows if row.get("clv_direction") == "flat"]
    return {
        "market_rows": len(rows),
        "research_candidates": len(candidates),
        "settled": len(settled),
        "open": len(open_rows),
        "wins": len(wins),
        "losses": len(losses),
        "settled_stake_units": settled_stake,
        "realized_pnl_units": realized_pnl,
        "roi_on_settled": realized_pnl / settled_stake if settled_stake else None,
        "open_exposure_units": open_exposure,
        "open_possible_profit_units": float(
            sum(float(row.get("paper_possible_profit_units") or 0.0) for row in open_rows)
        ),
        "settled_first_entry_pnl_units": float(
            sum(float(row.get("paper_pnl_entry_first_units") or 0.0) for row in settled)
        ),
        "settled_latest_entry_pnl_units": float(
            sum(float(row.get("paper_pnl_entry_latest_units") or 0.0) for row in settled)
        ),
        "open_possible_profit_entry_first_units": float(
            sum(
                float(row.get("paper_possible_profit_entry_first_units") or 0.0)
                for row in open_rows
            )
        ),
        "open_possible_profit_entry_latest_units": float(
            sum(
                float(row.get("paper_possible_profit_entry_latest_units") or 0.0)
                for row in open_rows
            )
        ),
        "clv_tracked": len(clv_rows),
        "clv_favorable": len(favorable_clv),
        "clv_unfavorable": len(unfavorable_clv),
        "clv_flat": len(flat_clv),
        "real_stake_units": float(sum(float(row.get("real_stake_units") or 0.0) for row in rows)),
    }
