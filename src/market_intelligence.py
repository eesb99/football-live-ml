from __future__ import annotations

import argparse
import json
from datetime import datetime, tzinfo, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Settings, load_settings
from src.predictor import prematch_prediction
from src.ratings import RatingMap
from src.sportmonks_client import SportMonksClient, SportMonksError, sportmonks_records
from src.sportmonks_enrichment import (
    WORLD_CUP_2026_SPORTMONKS_SEASON_ID,
    load_latest_world_cup_enrichment,
)
from src.sportmonks_mapping import parse_fixture_datetime
from src.storage import (
    list_sportmonks_cache_files,
    save_sportmonks_odds_cache,
)


OUTCOMES = ("home", "draw", "away")
FULL_TIME_MARKET_ID = 1


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def cache_payload(cache_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cache_record, dict):
        return {}
    payload = cache_record.get("payload")
    return payload if isinstance(payload, dict) else {}


def parse_decimal_odds(value: Any) -> float | None:
    try:
        decimal = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(decimal) or decimal <= 1.0:
        return None
    return decimal


def normalize_outcome_label(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in {"home", "1"}:
        return "home"
    if text in {"draw", "x", "tie"}:
        return "draw"
    if text in {"away", "2"}:
        return "away"
    return None


def is_full_time_result_record(record: dict[str, Any]) -> bool:
    try:
        market_id = int(record.get("market_id") or 0)
    except (TypeError, ValueError):
        market_id = 0
    if market_id != FULL_TIME_MARKET_ID:
        return False
    market = str(record.get("market_description") or "").casefold()
    return (
        ("full" in market and "result" in market)
        or "match winner" in market
        or market == "1x2"
    )


def _record_update_key(record: dict[str, Any]) -> str:
    return str(
        record.get("latest_bookmaker_update")
        or record.get("created_at")
        or ""
    )


def bookmaker_probability_rows(
    records: list[dict[str, Any]],
    *,
    include_stopped: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not is_full_time_result_record(record):
            continue
        if bool(record.get("stopped")) and not include_stopped:
            continue
        outcome = normalize_outcome_label(
            record.get("label")
            or record.get("name")
            or record.get("original_label")
        )
        decimal = parse_decimal_odds(record.get("value") or record.get("dp3"))
        if outcome is None or decimal is None:
            continue
        try:
            bookmaker_id = int(record.get("bookmaker_id") or 0)
        except (TypeError, ValueError):
            continue
        if bookmaker_id <= 0:
            continue
        bookmaker = grouped.setdefault(bookmaker_id, {})
        current = bookmaker.get(outcome)
        if current is None or _record_update_key(record) >= _record_update_key(current):
            bookmaker[outcome] = {**record, "decimal_odds": decimal}

    rows: list[dict[str, Any]] = []
    for bookmaker_id, outcomes in grouped.items():
        if any(outcome not in outcomes for outcome in OUTCOMES):
            continue
        raw_probabilities = {
            outcome: 1.0 / float(outcomes[outcome]["decimal_odds"])
            for outcome in OUTCOMES
        }
        overround = sum(raw_probabilities.values())
        if overround <= 0.0:
            continue
        rows.append(
            {
                "bookmaker_id": bookmaker_id,
                "home_probability": raw_probabilities["home"] / overround,
                "draw_probability": raw_probabilities["draw"] / overround,
                "away_probability": raw_probabilities["away"] / overround,
                "home_decimal": outcomes["home"]["decimal_odds"],
                "draw_decimal": outcomes["draw"]["decimal_odds"],
                "away_decimal": outcomes["away"]["decimal_odds"],
                "overround": overround,
                "latest_bookmaker_update": max(
                    _record_update_key(outcomes[outcome]) for outcome in OUTCOMES
                ),
            }
        )
    return sorted(rows, key=lambda row: row["bookmaker_id"])


def consensus_market_snapshot(
    odds_cache: dict[str, Any] | None,
    *,
    include_stopped: bool = False,
) -> dict[str, Any] | None:
    payload = cache_payload(odds_cache)
    records = sportmonks_records(payload)
    bookmaker_rows = bookmaker_probability_rows(
        records,
        include_stopped=include_stopped,
    )
    if not bookmaker_rows:
        return None
    fixture_ids = {
        int(record.get("fixture_id"))
        for record in records
        if record.get("fixture_id") is not None
    }
    fixture_id = fixture_ids.pop() if len(fixture_ids) == 1 else odds_cache.get("cache_key")
    probabilities = {
        outcome: float(
            np.mean([row[f"{outcome}_probability"] for row in bookmaker_rows])
        )
        for outcome in OUTCOMES
    }
    best_decimal = {
        outcome: float(max(row[f"{outcome}_decimal"] for row in bookmaker_rows))
        for outcome in OUTCOMES
    }
    return {
        "fixture_id": int(fixture_id),
        "captured_at": str((odds_cache or {}).get("captured_at") or ""),
        "cache_file": str((odds_cache or {}).get("cache_file") or ""),
        "bookmaker_count": len(bookmaker_rows),
        "latest_bookmaker_update": max(
            str(row.get("latest_bookmaker_update") or "") for row in bookmaker_rows
        ),
        "average_overround": float(
            np.mean([row["overround"] for row in bookmaker_rows])
        ),
        "home_probability": probabilities["home"],
        "draw_probability": probabilities["draw"],
        "away_probability": probabilities["away"],
        "home_best_decimal": best_decimal["home"],
        "draw_best_decimal": best_decimal["draw"],
        "away_best_decimal": best_decimal["away"],
    }


def market_snapshots_by_sportmonks_fixture() -> dict[int, list[dict[str, Any]]]:
    snapshots: dict[int, list[dict[str, Any]]] = {}
    for path in list_sportmonks_cache_files(prefix="odds"):
        cache = _load_json(path)
        if not cache:
            continue
        cache["cache_file"] = str(path)
        snapshot = consensus_market_snapshot(cache)
        if not snapshot:
            continue
        fixture_id = int(snapshot["fixture_id"])
        snapshots.setdefault(fixture_id, []).append(snapshot)
    for fixture_snapshots in snapshots.values():
        fixture_snapshots.sort(key=lambda snapshot: str(snapshot.get("captured_at") or ""))
    return snapshots


def parse_capture_datetime(
    value: Any,
    *,
    local_timezone: tzinfo | None = None,
) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        capture_timezone = local_timezone or datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=capture_timezone or timezone.utc)
    return parsed.astimezone(timezone.utc)


def pre_kickoff_snapshots(
    snapshots: list[dict[str, Any]],
    kickoff_utc: Any,
    *,
    local_capture_timezone: tzinfo | None = None,
) -> list[dict[str, Any]]:
    kickoff = parse_fixture_datetime(kickoff_utc)
    if kickoff is None:
        return []
    filtered = []
    for snapshot in snapshots:
        captured = parse_capture_datetime(
            snapshot.get("captured_at"),
            local_timezone=local_capture_timezone,
        )
        if captured is not None and captured <= kickoff:
            filtered.append(snapshot)
    return filtered


def closing_line_for_outcome(
    snapshots: list[dict[str, Any]],
    outcome: str,
) -> dict[str, Any]:
    if not snapshots or outcome not in OUTCOMES:
        return {
            "entry_decimal": None,
            "closing_decimal": None,
            "entry_market_probability": None,
            "closing_market_probability": None,
            "clv_decimal_delta": None,
            "clv_probability_delta": None,
            "clv_available": False,
        }
    entry = snapshots[0]
    close = snapshots[-1]
    entry_decimal = entry.get(f"{outcome}_best_decimal")
    closing_decimal = close.get(f"{outcome}_best_decimal")
    entry_probability = entry.get(f"{outcome}_probability")
    closing_probability = close.get(f"{outcome}_probability")
    return {
        "entry_decimal": entry_decimal,
        "closing_decimal": closing_decimal,
        "entry_market_probability": entry_probability,
        "closing_market_probability": closing_probability,
        "clv_decimal_delta": (
            float(entry_decimal) - float(closing_decimal)
            if entry_decimal is not None and closing_decimal is not None
            else None
        ),
        "clv_probability_delta": (
            float(closing_probability) - float(entry_probability)
            if entry_probability is not None and closing_probability is not None
            else None
        ),
        "clv_available": len(snapshots) >= 2,
    }


def benchmark_market_gate(
    fair_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    min_rows: int = 10,
) -> dict[str, Any]:
    shared_rows = int(fair_summary.get("shared_evaluated") or 0)
    our_brier = fair_summary.get("our_brier_score")
    api_brier = fair_summary.get("api_brier_score")
    our_log_loss = fair_summary.get("our_log_loss")
    api_log_loss = fair_summary.get("api_log_loss")
    api_gate = (
        shared_rows >= min_rows
        and our_brier is not None
        and api_brier is not None
        and our_log_loss is not None
        and api_log_loss is not None
        and float(our_brier) < float(api_brier)
        and float(our_log_loss) < float(api_log_loss)
    )
    candidate_gate = bool(candidate_summary.get("candidate_proves_improvement"))
    passed = api_gate or candidate_gate
    if passed:
        reason = "passed_brier_log_loss_gate"
    elif shared_rows < min_rows:
        reason = "insufficient_backtest_rows"
    else:
        reason = "brier_log_loss_not_better"
    return {
        "passed": passed,
        "reason": reason,
        "min_rows": min_rows,
        "shared_evaluated": shared_rows,
        "api_gate": api_gate,
        "sportmonks_candidate_gate": candidate_gate,
    }


def _probabilities_from_prediction(prediction: dict[str, Any]) -> dict[str, float]:
    return {
        "home": float(prediction.get("home_win_probability", 0.0) or 0.0),
        "draw": float(prediction.get("draw_probability", 0.0) or 0.0),
        "away": float(prediction.get("away_win_probability", 0.0) or 0.0),
    }


def _best_edge_outcome(
    model_probabilities: dict[str, float],
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for outcome in OUTCOMES:
        model_probability = model_probabilities[outcome]
        market_probability = float(market_snapshot.get(f"{outcome}_probability") or 0.0)
        best_decimal = market_snapshot.get(f"{outcome}_best_decimal")
        expected_value = (
            model_probability * float(best_decimal) - 1.0
            if best_decimal is not None
            else None
        )
        candidates.append(
            {
                "outcome": outcome,
                "model_probability": model_probability,
                "market_probability": market_probability,
                "edge": model_probability - market_probability,
                "best_decimal": best_decimal,
                "expected_value": expected_value,
            }
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate["expected_value"] if candidate["expected_value"] is not None else -99.0,
            candidate["edge"],
        ),
    )


def market_edge_rows_for_fixtures(
    fixtures: list[dict[str, Any]],
    ratings: RatingMap,
    coverage_rows: list[dict[str, Any]],
    *,
    benchmark_gate: dict[str, Any],
    snapshots_by_fixture: dict[int, list[dict[str, Any]]] | None = None,
    min_edge: float = 0.04,
    min_expected_value: float = 0.03,
) -> list[dict[str, Any]]:
    snapshots_by_fixture = snapshots_by_fixture or market_snapshots_by_sportmonks_fixture()
    sportmonks_by_api = {
        int(row["api_fixture_id"]): row
        for row in coverage_rows
        if row.get("api_fixture_id") is not None
    }
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_data = fixture.get("fixture") or {}
        fixture_id = int(fixture_data.get("id") or 0)
        coverage = sportmonks_by_api.get(fixture_id, {})
        sportmonks_fixture_id = coverage.get("sportmonks_fixture_id")
        if not sportmonks_fixture_id:
            continue
        kickoff_utc = fixture_data.get("date") or ""
        snapshots = pre_kickoff_snapshots(
            snapshots_by_fixture.get(int(sportmonks_fixture_id), []),
            kickoff_utc,
        )
        teams = fixture.get("teams") or {}
        match = (
            f"{(teams.get('home') or {}).get('name') or 'Home'} vs "
            f"{(teams.get('away') or {}).get('name') or 'Away'}"
        )
        base_row = {
            "fixture_id": fixture_id,
            "sportmonks_fixture_id": sportmonks_fixture_id,
            "match": match,
            "kickoff_utc": kickoff_utc,
            "market_snapshots": len(snapshots),
            "benchmark_gate": benchmark_gate.get("reason"),
            "edge_flag": False,
        }
        if not snapshots:
            rows.append(
                {
                    **base_row,
                    "status": "no_pre_kickoff_market",
                    "outcome": "-",
                }
            )
            continue

        market_snapshot = snapshots[-1]
        prediction = prematch_prediction(fixture, ratings=ratings)
        best = _best_edge_outcome(
            _probabilities_from_prediction(prediction),
            market_snapshot,
        )
        clv = closing_line_for_outcome(snapshots, best["outcome"])
        passes_market = (
            best["edge"] >= min_edge
            and best["expected_value"] is not None
            and best["expected_value"] >= min_expected_value
        )
        if not benchmark_gate.get("passed"):
            status = "blocked_benchmark_gate"
        elif not passes_market:
            status = "no_market_edge"
        else:
            status = "paper_trade_candidate"
        rows.append(
            {
                **base_row,
                **best,
                **clv,
                "status": status,
                "edge_flag": status == "paper_trade_candidate",
                "market_captured_at": market_snapshot.get("captured_at"),
                "latest_bookmaker_update": market_snapshot.get("latest_bookmaker_update"),
                "bookmaker_count": market_snapshot.get("bookmaker_count"),
                "average_overround": market_snapshot.get("average_overround"),
            }
        )
    return rows


def market_edge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("status") != "no_pre_kickoff_market"]
    flags = [row for row in rows if row.get("edge_flag")]
    clv_rows = [row for row in usable if row.get("clv_available")]
    return {
        "fixtures_with_market": len(usable),
        "paper_trade_candidates": len(flags),
        "clv_tracked": len(clv_rows),
        "average_edge": (
            float(np.mean([row["edge"] for row in usable if row.get("edge") is not None]))
            if usable
            else None
        ),
        "average_expected_value": (
            float(
                np.mean(
                    [
                        row["expected_value"]
                        for row in usable
                        if row.get("expected_value") is not None
                    ]
                )
            )
            if usable
            else None
        ),
    }


def _future_odds_fixture_ids(
    bundle: dict[str, Any],
    *,
    now: datetime,
) -> list[int]:
    fixture_ids: list[int] = []
    for fixture in bundle.get("sportmonks_fixtures") or []:
        if not (fixture.get("has_odds") or fixture.get("has_premium_odds")):
            continue
        kickoff = parse_fixture_datetime(fixture.get("starting_at"))
        if kickoff is None or kickoff <= now:
            continue
        try:
            fixture_ids.append(int(fixture["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return fixture_ids


def capture_pre_kickoff_odds(
    settings: Settings | None = None,
    client: SportMonksClient | None = None,
    *,
    season_id: int = WORLD_CUP_2026_SPORTMONKS_SEASON_ID,
    max_fixtures: int = 20,
) -> dict[str, Any]:
    settings = settings or load_settings(
        require_api_key=False,
        require_sportmonks_token=True,
    )
    client = client or SportMonksClient(settings)
    bundle = load_latest_world_cup_enrichment(season_id)
    now = datetime.now(timezone.utc)
    fixture_ids = _future_odds_fixture_ids(bundle, now=now)[:max_fixtures]
    cached_paths: list[str] = []
    empty: list[int] = []
    errors: dict[int, str] = {}
    token = settings.sportmonks_api_token
    for fixture_id in fixture_ids:
        try:
            payload = client.get_pre_match_odds(fixture_id)
        except SportMonksError as exc:
            errors[fixture_id] = str(exc)[:240]
            continue
        records = sportmonks_records(payload)
        if not records:
            empty.append(fixture_id)
        path = save_sportmonks_odds_cache(
            payload,
            fixture_id,
            token=token,
            metadata={
                "snapshot_type": "pre_kickoff_odds",
                "captured_before_kickoff": True,
                "season_id": int(season_id),
            },
        )
        cached_paths.append(str(path))
    return {
        "season_id": int(season_id),
        "fixtures_considered": len(fixture_ids),
        "odds_cached": len(cached_paths),
        "empty_odds": len(empty),
        "errors": len(errors),
        "cached_paths": cached_paths,
        "empty_fixture_ids": empty,
        "error_fixture_ids": sorted(errors),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture SportMonks pre-kickoff odds snapshots and market signals."
    )
    subparsers = parser.add_subparsers(dest="command")
    capture = subparsers.add_parser("capture", help="Capture pre-kickoff odds snapshots.")
    capture.add_argument("--season-id", type=int, default=WORLD_CUP_2026_SPORTMONKS_SEASON_ID)
    capture.add_argument("--max-fixtures", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command != "capture":
        parser.print_help()
        return 2
    summary = capture_pre_kickoff_odds(
        season_id=args.season_id,
        max_fixtures=args.max_fixtures,
    )
    public_summary = {
        key: value
        for key, value in summary.items()
        if key != "cached_paths"
    }
    public_summary["cached_paths"] = f"{len(summary['cached_paths'])} files"
    print(json.dumps(public_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
