from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import ApiFootballClient, ApiFootballError, ApiFootballRateLimitError
from src.benchmark import (
    api_football_predicted_outcome as benchmark_api_football_predicted_outcome,
    benchmark_diagnostic_counts,
    benchmark_diagnostic_rows,
    draw_diagnostic_summary,
    draw_miss_diagnostic_rows,
    fair_api_comparison_rows,
    fair_benchmark_summary,
    sportmonks_candidate_rows,
    sportmonks_candidate_summary,
    team_prior_ablation_summary,
    walk_forward_backtest_rows,
)
from src.config import MissingApiKeyError, MissingSportMonksTokenError, load_settings
from src.external_predictions import (
    normalize_api_football_prediction,
    unavailable_prediction,
)
from src.features import build_live_match_table, build_match_features, is_live_status
from src.market_intelligence import (
    benchmark_market_gate,
    capture_pre_kickoff_odds,
    market_edge_rows_for_fixtures,
    market_edge_summary,
)
from src.paper_trading import (
    DEFAULT_KELLY_MULTIPLIER,
    DEFAULT_PAPER_BANKROLL,
    DEFAULT_STAKE_CAP_FRACTION,
    paper_trade_rows,
    paper_trade_summary,
)
from src.predictor import predict_fixture, prediction_snapshot_row, prematch_prediction
from src.ratings import (
    HOME_ADVANTAGE_ELO,
    RatingMap,
    get_rating,
    load_ratings,
    save_rating_snapshot,
    save_ratings,
    update_ratings_from_results,
)
from src.schedule import fixture_myt_fields
from src.storage import (
    load_api_prediction_cache,
    load_latest_sportmonks_audit,
    list_prediction_snapshots,
    save_api_prediction_cache,
    save_prediction_snapshot,
    save_snapshot,
)
from src.team_priors import TEAM_PRIORS_PATH, load_team_priors, prior_schema_rows
from src.sportmonks_client import SportMonksError
from src.sportmonks_enrichment import (
    load_latest_world_cup_enrichment,
    sportmonks_cache_status_rows,
    sportmonks_candidate_enrichment_by_api_fixture,
    sportmonks_mapping_coverage_rows,
    sportmonks_mapping_coverage_summary,
)


st.set_page_config(page_title="World Cup Prediction Dashboard", layout="wide")

WORLD_CUP_DEFAULT_LEAGUE_ID = 1
WORLD_CUP_DEFAULT_SEASON = 2026
FREE_PLAN_FALLBACK_SEASONS = [2022, 2023, 2024]
WORLD_CUP_KEYWORDS = ("world cup", "fifa world cup")
FINAL_STATUS_SHORTS = {"FT", "AET", "PEN"}
PUBLIC_ODDS_REFRESH_MAX_FIXTURES = 20
PUBLIC_ODDS_REFRESH_COOLDOWN_SECONDS = 900
PUBLIC_ODDS_REFRESH_STATE_PATH = PROJECT_ROOT / "data" / "sportmonks" / "odds_refresh_state.json"


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fb !important;
            color: #172033 !important;
        }
        section[data-testid="stMain"] {
            background: #f7f8fb !important;
            color: #172033 !important;
        }
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2rem;
            max-width: 1240px;
            color: #172033 !important;
        }
        .match-title {
            color: #172033 !important;
            font-size: 1.45rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
        }
        .muted-line {
            color: #64748b !important;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }
        .snapshot-line {
            color: #475569 !important;
            font-size: 0.85rem;
            margin-top: 0.65rem;
        }
        section[data-testid="stMain"] h1,
        section[data-testid="stMain"] h2,
        section[data-testid="stMain"] h3,
        section[data-testid="stMain"] h4,
        section[data-testid="stMain"] h5,
        section[data-testid="stMain"] h6 {
            color: #172033 !important;
            letter-spacing: 0 !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] {
            color: #172033 !important;
        }
        section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] p {
            color: #475569 !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] label p {
            color: #334155 !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] div[data-testid="stCaptionContainer"],
        section[data-testid="stMain"] div[data-testid="stCaptionContainer"] * {
            color: #64748b !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] div[data-testid="stAlert"] p {
            color: #334155 !important;
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
        }
        section[data-testid="stMain"] div[data-testid="stTabs"] button p {
            color: #64748b !important;
            opacity: 1 !important;
        }
        section[data-testid="stMain"] div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: #ef4444 !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde3ec;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def optional_percentage(value: Any) -> str:
    if value is None:
        return "-"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(parsed):
        return "-"
    return percentage(parsed)


def numeric_value(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(parsed):
        return "-"
    return f"{parsed:.{decimals}f}{suffix}"


def sportmonks_token_configured() -> bool:
    return bool(os.getenv("SPORTMONKS_API_TOKEN", "").strip())


def public_odds_refresh_summary(summary: dict[str, Any]) -> dict[str, Any]:
    public_keys = [
        "season_id",
        "fixtures_considered",
        "odds_cached",
        "empty_odds",
        "errors",
    ]
    return {key: summary.get(key) for key in public_keys}


def load_public_odds_refresh_state(
    path: Path = PUBLIC_ODDS_REFRESH_STATE_PATH,
) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_public_odds_refresh_state(
    summary: dict[str, Any],
    *,
    now_epoch: float | None = None,
    path: Path = PUBLIC_ODDS_REFRESH_STATE_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_refresh_epoch": float(now_epoch if now_epoch is not None else time.time()),
        "summary": public_odds_refresh_summary(summary),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def public_odds_refresh_remaining_seconds(
    state: dict[str, Any],
    *,
    now_epoch: float | None = None,
    cooldown_seconds: int = PUBLIC_ODDS_REFRESH_COOLDOWN_SECONDS,
) -> int:
    try:
        last_refresh_epoch = float(state.get("last_refresh_epoch") or 0.0)
    except (TypeError, ValueError):
        last_refresh_epoch = 0.0
    elapsed = float(now_epoch if now_epoch is not None else time.time()) - last_refresh_epoch
    return max(0, int(cooldown_seconds - elapsed))


def api_display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def flatten_api_record(
    record: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in record.items():
        column = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            if value:
                flattened.update(flatten_api_record(value, column))
            else:
                flattened[column] = ""
        elif isinstance(value, list):
            flattened[column] = api_display_value(value)
        else:
            flattened[column] = api_display_value(value)
    return flattened


def arrow_safe_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    safe_rows = [
        {str(key): api_display_value(value) for key, value in row.items()}
        for row in rows
    ]
    return pd.DataFrame(safe_rows).astype("string").fillna("")


def fixture_statistics_display_rows(statistics: Any) -> list[dict[str, str]]:
    if not isinstance(statistics, list):
        return []

    rows: list[dict[str, str]] = []
    for team_stats in statistics:
        if not isinstance(team_stats, dict):
            rows.append(
                {
                    "team_id": "",
                    "team": "",
                    "statistic": "",
                    "value": api_display_value(team_stats),
                }
            )
            continue

        team = team_stats.get("team") if isinstance(team_stats.get("team"), dict) else {}
        team_id = api_display_value(team.get("id"))
        team_name = api_display_value(team.get("name") or team_stats.get("team"))
        stats = team_stats.get("statistics")
        if not isinstance(stats, list) or not stats:
            rows.append(
                {
                    "team_id": team_id,
                    "team": team_name,
                    "statistic": "",
                    "value": "",
                }
            )
            continue

        for stat in stats:
            if isinstance(stat, dict):
                row = {
                    "team_id": team_id,
                    "team": team_name,
                    "statistic": api_display_value(stat.get("type")),
                    "value": api_display_value(stat.get("value")),
                }
                for key, value in stat.items():
                    if key not in {"type", "value"}:
                        row[f"stat.{key}"] = api_display_value(value)
                rows.append(row)
            else:
                rows.append(
                    {
                        "team_id": team_id,
                        "team": team_name,
                        "statistic": "",
                        "value": api_display_value(stat),
                    }
                )
    return rows


def fixture_events_display_rows(events: Any) -> list[dict[str, str]]:
    if not isinstance(events, list):
        return []

    priority_columns = [
        "time.elapsed",
        "time.extra",
        "team.id",
        "team.name",
        "player.id",
        "player.name",
        "assist.id",
        "assist.name",
        "type",
        "detail",
        "comments",
    ]
    rows: list[dict[str, str]] = []
    for event in events:
        if not isinstance(event, dict):
            rows.append({"event": api_display_value(event)})
            continue
        flattened = flatten_api_record(event)
        ordered = {
            column: flattened.pop(column, "")
            for column in priority_columns
            if column in flattened
        }
        ordered.update(dict(sorted(flattened.items())))
        rows.append(ordered)
    return rows


def render_api_dataframe(rows: list[dict[str, Any]], empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(arrow_safe_dataframe(rows), width="stretch", hide_index=True)


def probability_rows(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "Home win",
            "probability": float(prediction["home_win_probability"]),
            "display": percentage(prediction["home_win_probability"]),
            "group": "match outcome",
        },
        {
            "metric": "Draw",
            "probability": float(prediction["draw_probability"]),
            "display": percentage(prediction["draw_probability"]),
            "group": "match outcome",
        },
        {
            "metric": "Away win",
            "probability": float(prediction["away_win_probability"]),
            "display": percentage(prediction["away_win_probability"]),
            "group": "match outcome",
        },
        {
            "metric": "Home scores next",
            "probability": float(prediction["home_scores_next_probability"]),
            "display": percentage(prediction["home_scores_next_probability"]),
            "group": "next goal",
        },
        {
            "metric": "Away scores next",
            "probability": float(prediction["away_scores_next_probability"]),
            "display": percentage(prediction["away_scores_next_probability"]),
            "group": "next goal",
        },
        {
            "metric": "No next goal",
            "probability": float(prediction["no_next_goal_probability"]),
            "display": percentage(prediction["no_next_goal_probability"]),
            "group": "next goal",
        },
    ]


def outcome_label(outcome: str) -> str:
    labels = {"home": "Home win", "draw": "Draw", "away": "Away win"}
    return labels.get(outcome, "-")


def actual_outcome_from_features(features: dict[str, Any]) -> str | None:
    if features.get("status_short") not in FINAL_STATUS_SHORTS:
        return None
    home_goals = int(features.get("home_goals", 0) or 0)
    away_goals = int(features.get("away_goals", 0) or 0)
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def predicted_outcome_from_probabilities(prediction: dict[str, Any]) -> str:
    probabilities = {
        "home": float(prediction.get("home_win_probability", 0.0) or 0.0),
        "draw": float(prediction.get("draw_probability", 0.0) or 0.0),
        "away": float(prediction.get("away_win_probability", 0.0) or 0.0),
    }
    return max(probabilities, key=probabilities.get)


def outcome_probability(prediction: dict[str, Any], outcome: str) -> float:
    keys = {
        "home": "home_win_probability",
        "draw": "draw_probability",
        "away": "away_win_probability",
    }
    return float(prediction.get(keys[outcome], 0.0) or 0.0)


def model_result_banner_data(
    fixture: dict[str, Any],
    features: dict[str, Any],
    selected_prediction: dict[str, Any],
    ratings: RatingMap,
) -> dict[str, Any]:
    actual = actual_outcome_from_features(features)
    if actual is None:
        live_prediction = predicted_outcome_from_probabilities(selected_prediction)
        return {
            "status": "pending",
            "headline": "Result pending",
            "detail": (
                "Final score is not available yet. Current model leans "
                f"{outcome_label(live_prediction)} at "
                f"{percentage(outcome_probability(selected_prediction, live_prediction))}."
            ),
            "predicted": live_prediction,
            "actual": None,
            "basis": selected_prediction.get("prediction_mode", "current"),
            "correct": None,
        }

    evaluation_prediction = selected_prediction
    basis = str(selected_prediction.get("prediction_mode") or "current")
    if basis == "final":
        evaluation_prediction = prematch_prediction(fixture, ratings=ratings)
        basis = "pre-match prior"

    predicted = predicted_outcome_from_probabilities(evaluation_prediction)
    correct = predicted == actual
    teams = fixture.get("teams", {})
    home_team = features.get("home_team") or teams.get("home", {}).get("name", "Home")
    away_team = features.get("away_team") or teams.get("away", {}).get("name", "Away")
    return {
        "status": "win" if correct else "loss",
        "headline": "Our model WIN" if correct else "Our model LOSS",
        "detail": (
            f"Predicted {outcome_label(predicted)} at "
            f"{percentage(outcome_probability(evaluation_prediction, predicted))}; "
            f"actual result was {outcome_label(actual)} "
            f"({home_team} {features.get('home_goals', 0)}-"
            f"{features.get('away_goals', 0)} {away_team}). "
            f"Evaluation basis: {basis}."
        ),
        "predicted": predicted,
        "actual": actual,
        "basis": basis,
        "correct": correct,
    }


def elo_prior_rows(
    fixture: dict[str, Any],
    ratings: RatingMap,
) -> list[dict[str, Any]]:
    teams = fixture.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_id = int(home.get("id") or 0)
    away_id = int(away.get("id") or 0)
    home_name = home.get("name") or "Home"
    away_name = away.get("name") or "Away"
    home_rating = get_rating(ratings, home_id, home_name)
    away_rating = get_rating(ratings, away_id, away_name)
    prior = prematch_prediction(fixture, ratings=ratings)
    raw_difference = home_rating.rating - away_rating.rating
    adjusted_difference = raw_difference + HOME_ADVANTAGE_ELO

    return [
        {
            "metric": "Home Elo",
            "value": numeric_value(home_rating.rating, 1),
            "detail": f"{home_name}, {home_rating.matches_played} rated matches",
        },
        {
            "metric": "Away Elo",
            "value": numeric_value(away_rating.rating, 1),
            "detail": f"{away_name}, {away_rating.matches_played} rated matches",
        },
        {
            "metric": "Raw Elo difference",
            "value": numeric_value(raw_difference, 1),
            "detail": "Home rating minus away rating",
        },
        {
            "metric": "Home-adjusted Elo difference",
            "value": numeric_value(adjusted_difference, 1),
            "detail": f"Includes {HOME_ADVANTAGE_ELO:.0f} Elo home advantage",
        },
        {
            "metric": "Pre-match home",
            "value": percentage(prior["home_win_probability"]),
            "detail": "Elo prior converted through Poisson expected goals",
        },
        {
            "metric": "Pre-match draw",
            "value": percentage(prior["draw_probability"]),
            "detail": "Elo prior converted through Poisson expected goals",
        },
        {
            "metric": "Pre-match away",
            "value": percentage(prior["away_win_probability"]),
            "detail": "Elo prior converted through Poisson expected goals",
        },
    ]


def extracted_feature_rows(features: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "signal": "Minute",
            "home": "",
            "away": "",
            "match": f"{features.get('minute', 0)}'",
        },
        {
            "signal": "Score",
            "home": numeric_value(features.get("home_goals"), 0),
            "away": numeric_value(features.get("away_goals"), 0),
            "match": numeric_value(features.get("score_difference"), 0),
        },
        {
            "signal": "Red cards",
            "home": numeric_value(features.get("home_red_cards"), 0),
            "away": numeric_value(features.get("away_red_cards"), 0),
            "match": numeric_value(features.get("red_card_difference"), 0),
        },
        {
            "signal": "Shots",
            "home": numeric_value(features.get("home_shots"), 0),
            "away": numeric_value(features.get("away_shots"), 0),
            "match": numeric_value(features.get("shot_difference"), 0),
        },
        {
            "signal": "Shots on target",
            "home": numeric_value(features.get("home_shots_on_target"), 0),
            "away": numeric_value(features.get("away_shots_on_target"), 0),
            "match": numeric_value(features.get("shots_on_target_difference"), 0),
        },
        {
            "signal": "Possession",
            "home": numeric_value(features.get("home_possession"), 1, "%"),
            "away": numeric_value(features.get("away_possession"), 1, "%"),
            "match": numeric_value(features.get("possession_difference"), 1),
        },
        {
            "signal": "Corners",
            "home": numeric_value(features.get("home_corners"), 0),
            "away": numeric_value(features.get("away_corners"), 0),
            "match": numeric_value(features.get("corner_difference"), 0),
        },
    ]

    if float(features.get("home_xg", 0.0) or 0.0) > 0.0 or float(
        features.get("away_xg", 0.0) or 0.0
    ) > 0.0:
        rows.append(
            {
                "signal": "Real expected goals",
                "home": numeric_value(features.get("home_xg"), 2),
                "away": numeric_value(features.get("away_xg"), 2),
                "match": numeric_value(features.get("xg_difference"), 2),
            }
        )

    rows.extend(
        [
            {
                "signal": "Proxy expected goals",
                "home": numeric_value(features.get("home_proxy_xg"), 2),
                "away": numeric_value(features.get("away_proxy_xg"), 2),
                "match": numeric_value(features.get("proxy_xg_difference"), 2),
            },
            {
                "signal": "Effective expected goals",
                "home": numeric_value(features.get("home_effective_xg"), 2),
                "away": numeric_value(features.get("away_effective_xg"), 2),
                "match": str(features.get("xg_source") or "proxy_xg"),
            },
        ]
    )

    rows.extend(
        [
            {
                "signal": "Shots inside box",
                "home": numeric_value(features.get("home_shots_inside_box"), 0),
                "away": numeric_value(features.get("away_shots_inside_box"), 0),
                "match": "",
            },
            {
                "signal": "Blocked shots",
                "home": numeric_value(features.get("home_blocked_shots"), 0),
                "away": numeric_value(features.get("away_blocked_shots"), 0),
                "match": "",
            },
            {
                "signal": "Pass accuracy",
                "home": numeric_value(features.get("home_pass_accuracy"), 1, "%"),
                "away": numeric_value(features.get("away_pass_accuracy"), 1, "%"),
                "match": "",
            },
        ]
    )
    return rows


def data_source_rows(
    features: dict[str, Any],
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    xg_source = str(prediction.get("xg_source") or features.get("xg_source") or "proxy_xg")
    real_xg_available = bool(
        prediction.get("real_xg_available") or features.get("real_xg_available")
    )
    return [
        {
            "source": "Model mode",
            "status": str(prediction.get("prediction_mode") or "-"),
            "detail": str(prediction.get("model_version") or "-"),
        },
        {
            "source": "xG",
            "status": "real xG" if real_xg_available else "proxy xG",
            "detail": xg_source,
        },
        {
            "source": "Odds adapter",
            "status": "available" if prediction.get("odds_available") else "missing",
            "detail": str(prediction.get("odds_source") or "not configured"),
        },
        {
            "source": "Real xG adapter",
            "status": "available" if prediction.get("real_xg_available") else "missing",
            "detail": str(prediction.get("real_xg_source") or "not configured"),
        },
        {
            "source": "Injuries adapter",
            "status": "available" if prediction.get("injuries_available") else "missing",
            "detail": str(prediction.get("injuries_source") or "not configured"),
        },
        {
            "source": "News adapter",
            "status": "available" if prediction.get("news_available") else "missing",
            "detail": str(prediction.get("news_source") or "not configured"),
        },
    ]


def sportmonks_provider_status_rows() -> list[dict[str, Any]]:
    settings = load_settings(require_api_key=False)
    latest_audit = load_latest_sportmonks_audit()
    rows = [
        {
            "source": "SportMonks token",
            "status": "present" if settings.sportmonks_api_token else "missing",
            "detail": "SPORTMONKS_API_TOKEN configured"
            if settings.sportmonks_api_token
            else "SPORTMONKS_API_TOKEN not configured",
        },
        {
            "source": "SportMonks base URL",
            "status": "configured",
            "detail": settings.sportmonks_base_url,
        },
    ]
    if not latest_audit:
        rows.append(
            {
                "source": "Last access audit",
                "status": "missing",
                "detail": "Run python3 -m src.sportmonks_audit",
            }
        )
        return rows

    summary = latest_audit.get("summary") or {}
    metadata = summary.get("metadata") or {}
    subscription = metadata.get("subscription") or {}
    rate_limit = metadata.get("rate_limit") or {}
    accessible = summary.get("accessible_categories") or []
    world_cup_seasons = summary.get("world_cup_2026_season_ids") or []
    rows.extend(
        [
            {
                "source": "Last access audit",
                "status": "available",
                "detail": str(latest_audit.get("audit_file") or "-"),
            },
            {
                "source": "Accessible categories",
                "status": str(len(accessible)),
                "detail": ", ".join(accessible) if accessible else "-",
            },
            {
                "source": "World Cup 2026 season IDs",
                "status": "found" if world_cup_seasons else "missing",
                "detail": ", ".join(str(value) for value in world_cup_seasons) or "-",
            },
            {
                "source": "Subscription metadata",
                "status": "available" if subscription else "not returned",
                "detail": json.dumps(subscription, sort_keys=True)[:240]
                if subscription
                else "-",
            },
            {
                "source": "Rate-limit metadata",
                "status": "available" if rate_limit else "not returned",
                "detail": json.dumps(rate_limit, sort_keys=True)[:240]
                if rate_limit
                else "-",
            },
        ]
    )
    return rows


def sportmonks_audit_check_rows(
    audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    audit = audit or load_latest_sportmonks_audit()
    if not audit:
        return []
    rows = []
    for name, check in (audit.get("checks") or {}).items():
        detail = check.get("error") or check.get("reason") or check.get("endpoint") or ""
        rows.append(
            {
                "category": name,
                "status": str(check.get("status") or "-"),
                "available": bool(check.get("available")),
                "records": int(check.get("record_count") or 0),
                "detail": str(detail)[:240],
            }
        )
    return rows


def sportmonks_mapping_metric_rows(
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = sportmonks_mapping_coverage_summary(coverage_rows)
    return [
        {
            "metric": "API-Football fixtures",
            "value": summary["api_fixtures"],
            "detail": "Fixture rows currently shown by API-Football",
        },
        {
            "metric": "Mapped to SportMonks",
            "value": summary["mapped"],
            "detail": optional_percentage(summary["mapping_rate"]),
        },
        {
            "metric": "Exact mappings",
            "value": summary["exact"],
            "detail": "Kickoff and teams match strongly",
        },
        {
            "metric": "Likely mappings",
            "value": summary["likely"],
            "detail": "Good but not exact team/name match",
        },
        {
            "metric": "Ambiguous mappings",
            "value": summary["ambiguous"],
            "detail": "Needs manual review before use",
        },
        {
            "metric": "No match",
            "value": summary["no_match"],
            "detail": "No safe SportMonks fixture mapping",
        },
        {
            "metric": "Detail cache",
            "value": summary["detail_available"],
            "detail": "Mapped fixtures with cached detail",
        },
        {
            "metric": "xG pair cache",
            "value": summary["xg_pair_available"],
            "detail": "Mapped fixtures with home and away SportMonks xG",
        },
        {
            "metric": "News cache",
            "value": summary["news_available"],
            "detail": "Mapped fixtures with SportMonks pre-match news",
        },
    ]


def strength_component_rows(
    features: dict[str, Any],
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "component": "Live strength score",
            "home": numeric_value(prediction.get("home_strength_score"), 2),
            "away": numeric_value(prediction.get("away_strength_score"), 2),
            "influence": "Combined pressure, shots, discipline, possession, recency, and score state",
        },
        {
            "component": "Effective xG",
            "home": numeric_value(prediction.get("home_effective_xg"), 2),
            "away": numeric_value(prediction.get("away_effective_xg"), 2),
            "influence": f"Uses {prediction.get('xg_source', 'proxy_xg')} for attacking quality",
        },
        {
            "component": "Pressure score",
            "home": numeric_value(features.get("home_pressure_score"), 2),
            "away": numeric_value(features.get("away_pressure_score"), 2),
            "influence": "Shots, shots on target, corners, possession, and recent events",
        },
        {
            "component": "Pressure share",
            "home": percentage(float(features.get("home_pressure_share", 0.0) or 0.0)),
            "away": percentage(float(features.get("away_pressure_share", 0.0) or 0.0)),
            "influence": "Moves attacking share toward the team applying more pressure",
        },
        {
            "component": "Shot share",
            "home": percentage(float(features.get("home_shot_share", 0.0) or 0.0)),
            "away": percentage(float(features.get("away_shot_share", 0.0) or 0.0)),
            "influence": "Volume signal for attacking territory",
        },
        {
            "component": "Shot-on-target share",
            "home": percentage(
                float(features.get("home_shots_on_target_share", 0.0) or 0.0)
            ),
            "away": percentage(
                float(features.get("away_shots_on_target_share", 0.0) or 0.0)
            ),
            "influence": "Higher weight than raw shots because target shots are stronger goal signals",
        },
        {
            "component": "Recent events",
            "home": numeric_value(features.get("home_recent_events"), 0),
            "away": numeric_value(features.get("away_recent_events"), 0),
            "influence": "Last 10 minutes of API event activity",
        },
        {
            "component": "Recent goals",
            "home": numeric_value(features.get("home_recent_goals"), 0),
            "away": numeric_value(features.get("away_recent_goals"), 0),
            "influence": "Small momentum input, not a full causal model",
        },
        {
            "component": "Red cards",
            "home": numeric_value(features.get("home_red_cards"), 0),
            "away": numeric_value(features.get("away_red_cards"), 0),
            "influence": "Red cards reduce the penalized team's live strength",
        },
    ]


def expected_goal_rows(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "Projected total expected goals",
            "home": numeric_value(prediction.get("home_expected_goals"), 2),
            "away": numeric_value(prediction.get("away_expected_goals"), 2),
        },
        {
            "metric": "Effective xG input",
            "home": numeric_value(prediction.get("home_effective_xg"), 2),
            "away": numeric_value(prediction.get("away_effective_xg"), 2),
        },
        {
            "metric": "Remaining expected goals",
            "home": numeric_value(prediction.get("home_expected_remaining_goals"), 2),
            "away": numeric_value(prediction.get("away_expected_remaining_goals"), 2),
        },
        {
            "metric": "Scores next",
            "home": percentage(
                float(prediction.get("home_scores_next_probability", 0.0) or 0.0)
            ),
            "away": percentage(
                float(prediction.get("away_scores_next_probability", 0.0) or 0.0)
            ),
        },
        {
            "metric": "No next goal",
            "home": "",
            "away": percentage(float(prediction.get("no_next_goal_probability", 0.0) or 0.0)),
        },
    ]


def comparison_status_rows(api_prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": "API-Football prediction endpoint",
            "status": str(api_prediction.get("status") or "missing"),
            "detail": str(api_prediction.get("endpoint") or "/predictions"),
        },
        {
            "source": "Availability",
            "status": "available" if api_prediction.get("available") else "unavailable",
            "detail": str(api_prediction.get("last_error") or ""),
        },
    ]


def model_comparison_rows(
    prediction: dict[str, Any],
    api_prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "Home win",
            "our_model": percentage(prediction["home_win_probability"]),
            "api_football": str(api_prediction.get("home_display") or "-"),
            "difference": probability_difference(
                prediction.get("home_win_probability"),
                api_prediction.get("home_probability"),
            ),
        },
        {
            "metric": "Draw",
            "our_model": percentage(prediction["draw_probability"]),
            "api_football": str(api_prediction.get("draw_display") or "-"),
            "difference": probability_difference(
                prediction.get("draw_probability"),
                api_prediction.get("draw_probability"),
            ),
        },
        {
            "metric": "Away win",
            "our_model": percentage(prediction["away_win_probability"]),
            "api_football": str(api_prediction.get("away_display") or "-"),
            "difference": probability_difference(
                prediction.get("away_win_probability"),
                api_prediction.get("away_probability"),
            ),
        },
        {
            "metric": "Home scores next",
            "our_model": optional_percentage(
                prediction.get("home_scores_next_probability")
            ),
            "api_football": "-",
            "difference": "-",
        },
        {
            "metric": "Away scores next",
            "our_model": optional_percentage(
                prediction.get("away_scores_next_probability")
            ),
            "api_football": "-",
            "difference": "-",
        },
    ]


def probability_difference(our_value: Any, external_value: Any) -> str:
    if external_value is None:
        return "-"
    try:
        diff = float(our_value) - float(external_value)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff * 100:.1f} pp"


def api_football_advice_rows(api_prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "field": "Advice",
            "value": str(api_prediction.get("advice") or "-"),
        },
        {
            "field": "Winner",
            "value": str(api_prediction.get("winner_name") or "-"),
        },
        {
            "field": "Winner comment",
            "value": str(api_prediction.get("winner_comment") or "-"),
        },
        {
            "field": "Win or draw",
            "value": api_display_value(api_prediction.get("win_or_draw")),
        },
        {
            "field": "Under/over",
            "value": str(api_prediction.get("under_over") or "-"),
        },
        {
            "field": "Predicted goals",
            "value": (
                f"{api_prediction.get('goals_home') or '-'} - "
                f"{api_prediction.get('goals_away') or '-'}"
            ),
        },
    ]


def is_world_cup_fixture(fixture: dict[str, Any], world_cup_league_id: int) -> bool:
    league = fixture.get("league", {})
    league_id = int(league.get("id") or 0)
    league_name = str(league.get("name") or "").lower()
    return league_id == world_cup_league_id or any(
        keyword in league_name for keyword in WORLD_CUP_KEYWORDS
    )


def score_text(fixture: dict[str, Any]) -> str:
    goals = fixture.get("goals", {})
    home = goals.get("home")
    away = goals.get("away")
    if home is None or away is None:
        return "-"
    return f"{home}-{away}"


def fixture_schedule_rows(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = build_live_match_table(fixtures)
    for row, fixture in zip(rows, fixtures, strict=True):
        row["score"] = score_text(fixture)
        row["fixture_date"] = fixture.get("fixture", {}).get("date") or ""
        row.update(fixture_myt_fields(fixture))
        row["round"] = fixture.get("league", {}).get("round") or ""
        row["venue"] = fixture.get("fixture", {}).get("venue", {}).get("name") or ""
    return rows


def should_try_free_plan_fallback(error: ApiFootballError) -> bool:
    message = str(error).lower()
    return "free plans do not have access" in message and "season" in message


@st.cache_data(ttl=300, show_spinner=False)
def fetch_world_cup_fixtures(world_cup_league_id: int, season: int) -> dict[str, Any]:
    settings = load_settings()
    client = ApiFootballClient(settings)
    fixtures = client.get_fixtures({"league": world_cup_league_id, "season": season})
    ratings = update_ratings_from_results(fixtures)
    save_ratings(ratings)
    rating_snapshot_path = save_rating_snapshot(ratings)

    snapshot_rows = fixture_schedule_rows(fixtures)
    for row in snapshot_rows:
        row["view_scope"] = "World Cup season"
    fixture_snapshot_path = save_snapshot(snapshot_rows, allow_empty=True)
    return {
        "fixtures": fixtures,
        "ratings": ratings,
        "requested_season": season,
        "resolved_season": season,
        "fallback_reason": "",
        "fixture_snapshot_path": str(fixture_snapshot_path) if fixture_snapshot_path else None,
        "rating_snapshot_path": str(rating_snapshot_path),
    }


def fetch_world_cup_fixtures_with_fallback(
    world_cup_league_id: int,
    season: int,
) -> dict[str, Any]:
    try:
        return fetch_world_cup_fixtures(world_cup_league_id, season)
    except ApiFootballError as exc:
        if not should_try_free_plan_fallback(exc):
            raise
        fallback_errors = [str(exc)]
        for fallback_season in FREE_PLAN_FALLBACK_SEASONS:
            if fallback_season == season:
                continue
            try:
                dataset = fetch_world_cup_fixtures(world_cup_league_id, fallback_season)
            except ApiFootballError as fallback_exc:
                fallback_errors.append(str(fallback_exc))
                continue
            if dataset["fixtures"]:
                dataset["requested_season"] = season
                dataset["resolved_season"] = fallback_season
                dataset["fallback_reason"] = (
                    f"API-Football rejected season {season} on the current plan. "
                    f"Showing accessible season {fallback_season} instead."
                )
                return dataset
        raise ApiFootballError(
            "API-Football rejected the selected season and no fallback World Cup "
            f"season returned fixtures. Errors: {' | '.join(fallback_errors)}"
        ) from exc


@st.cache_data(ttl=30, show_spinner=False)
def fetch_fixture_detail(fixture_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = load_settings()
    client = ApiFootballClient(settings)
    return client.get_fixture_statistics(fixture_id), client.get_fixture_events(fixture_id)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_api_football_prediction(fixture_id: int) -> dict[str, Any]:
    cached = load_api_prediction_cache(fixture_id)
    if cached is not None:
        return cached

    settings = load_settings()
    client = ApiFootballClient(settings)
    try:
        response = client.get_fixture_predictions(fixture_id)
    except ApiFootballError as exc:
        prediction = unavailable_prediction(fixture_id, "error", last_error=str(exc))
    else:
        prediction = normalize_api_football_prediction(response, fixture_id)

    save_api_prediction_cache(prediction, fixture_id)
    return load_api_prediction_cache(fixture_id) or prediction


@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_prediction_dataset(
    view_scope: str,
    world_cup_league_id: int,
) -> dict[str, Any]:
    settings = load_settings()
    client = ApiFootballClient(settings)
    fixtures = client.get_live_fixtures()
    if view_scope == "World Cup live":
        fixtures = [
            fixture
            for fixture in fixtures
            if is_world_cup_fixture(fixture, world_cup_league_id)
        ]

    ratings = load_ratings()
    matches = []
    raw_snapshot_rows = []
    prediction_snapshot_rows = []
    for fixture in fixtures:
        fixture_id = int(fixture["fixture"]["id"])
        statistics = client.get_fixture_statistics(fixture_id)
        events = client.get_fixture_events(fixture_id)
        features = build_match_features(fixture, statistics=statistics, events=events)
        prediction = predict_fixture(
            fixture,
            statistics=statistics,
            events=events,
            ratings=ratings,
        )
        matches.append(
            {
                "fixture": fixture,
                "features": features,
                "prediction": prediction,
                "statistics": statistics,
                "events": events,
            }
        )
        raw_snapshot_rows.append({**features, **prediction, "view_scope": view_scope})
        prediction_snapshot_rows.append(prediction_snapshot_row(features, prediction))

    raw_snapshot_path = save_snapshot(raw_snapshot_rows, allow_empty=True)
    prediction_snapshot_path = save_prediction_snapshot(
        prediction_snapshot_rows,
        allow_empty=True,
    )
    return {
        "matches": matches,
        "raw_snapshot_path": str(raw_snapshot_path) if raw_snapshot_path else None,
        "prediction_snapshot_path": (
            str(prediction_snapshot_path) if prediction_snapshot_path else None
        ),
    }


def prediction_columns(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": prediction["prediction_mode"],
        "home_win": percentage(prediction["home_win_probability"]),
        "draw": percentage(prediction["draw_probability"]),
        "away_win": percentage(prediction["away_win_probability"]),
        "confidence": percentage(prediction["model_confidence"]),
    }


def fixture_board_rows(
    fixtures: list[dict[str, Any]],
    ratings: RatingMap,
) -> list[dict[str, Any]]:
    rows = []
    for fixture in fixtures:
        features = build_match_features(fixture)
        prediction = predict_fixture(fixture, ratings=ratings)
        rows.append(
            {
                "fixture_id": features["fixture_id"],
                "api_date": features["fixture_date"],
                **fixture_myt_fields(fixture),
                "round": features["league_round"],
                "status": features["status"],
                "match": f"{features['home_team']} vs {features['away_team']}",
                "score": score_text(fixture),
                **prediction_columns(prediction),
            }
        )
    return rows


def live_board_rows(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        features = match["features"]
        prediction = match["prediction"]
        rows.append(
            {
                "fixture_id": features["fixture_id"],
                "league": features["league_name"],
                **fixture_myt_fields(match["fixture"]),
                "minute": features["minute"],
                "status": features["status"],
                "match": f"{features['home_team']} vs {features['away_team']}",
                "score": f"{features['home_goals']}-{features['away_goals']}",
                "next_goal": percentage(prediction["next_goal_probability"]),
                **prediction_columns(prediction),
            }
        )
    return rows


def render_probability_metrics(prediction: dict[str, Any]) -> None:
    first = st.columns(3)
    first[0].metric("Home win", percentage(prediction["home_win_probability"]))
    first[1].metric("Draw", percentage(prediction["draw_probability"]))
    first[2].metric("Away win", percentage(prediction["away_win_probability"]))

    second = st.columns(4)
    second[0].metric("Prediction mode", prediction["prediction_mode"])
    second[1].metric("Model confidence", percentage(prediction["model_confidence"]))
    second[2].metric("Home expected goals", f"{prediction['home_expected_goals']:.2f}")
    second[3].metric("Away expected goals", f"{prediction['away_expected_goals']:.2f}")


def render_live_metrics(prediction: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Any next goal", percentage(prediction["next_goal_probability"]))
    cols[1].metric(
        "Home scores next",
        percentage(prediction["home_scores_next_probability"]),
    )
    cols[2].metric(
        "Away scores next",
        percentage(prediction["away_scores_next_probability"]),
    )
    cols[3].metric(
        "Home xG remaining",
        f"{prediction['home_expected_remaining_goals']:.2f}",
    )
    cols[4].metric(
        "Away xG remaining",
        f"{prediction['away_expected_remaining_goals']:.2f}",
    )


def render_prediction_bars(prediction: dict[str, Any]) -> None:
    cols = st.columns(3)
    for column, label, key in [
        (cols[0], "Home win", "home_win_probability"),
        (cols[1], "Draw", "draw_probability"),
        (cols[2], "Away win", "away_win_probability"),
    ]:
        with column:
            st.write(f"{label}: **{percentage(prediction[key])}**")
            st.progress(prediction[key])


def render_probability_cards(rows: list[dict[str, Any]], group: str) -> None:
    grouped = [row for row in rows if row["group"] == group]
    if not grouped:
        return
    cols = st.columns(len(grouped))
    for column, row in zip(cols, grouped, strict=True):
        with column:
            st.metric(row["metric"], row["display"])
            st.progress(float(row["probability"]))


def render_selected_header(features: dict[str, Any]) -> None:
    st.markdown(
        f"<div class='match-title'>{features['home_team']} "
        f"{features['home_goals']}-{features['away_goals']} "
        f"{features['away_team']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='muted-line'>{features['league_name']} | "
        f"{features['league_round']} | {features['status']} | "
        f"Fixture {features['fixture_id']}</div>",
        unsafe_allow_html=True,
    )


def render_model_result_banner(banner: dict[str, Any]) -> None:
    message = f"**{banner['headline']}** - {banner['detail']}"
    if banner["status"] == "win":
        st.success(message)
    elif banner["status"] == "loss":
        st.error(message)
    else:
        st.info(message)


def render_model_drivers(prediction: dict[str, Any]) -> None:
    drivers = prediction.get("model_drivers") or []
    if not drivers:
        st.info("No model drivers are available for this prediction.")
        return
    for driver in drivers:
        st.write(f"- {driver}")


def render_model_breakdown(
    fixture: dict[str, Any],
    features: dict[str, Any],
    prediction: dict[str, Any],
    ratings: RatingMap,
    selected_statistics: list[dict[str, Any]],
    selected_events: list[dict[str, Any]],
) -> None:
    render_selected_header(features)

    st.markdown("#### Data source status")
    st.dataframe(
        pd.DataFrame(data_source_rows(features, prediction)),
        width="stretch",
        hide_index=True,
    )

    rows = probability_rows(prediction)
    st.markdown("#### Final blended probabilities")
    render_probability_cards(rows, "match outcome")
    render_probability_cards(rows, "next goal")
    st.caption(
        f"Mode: {prediction['prediction_mode']} | "
        f"Confidence: {percentage(prediction['model_confidence'])} | "
        f"Version: {prediction['model_version']}"
    )

    st.markdown("#### Elo prior")
    st.dataframe(
        pd.DataFrame(elo_prior_rows(fixture, ratings)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### API-Football feature extraction")
    st.dataframe(
        pd.DataFrame(extracted_feature_rows(features)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Heuristic live strength score")
    st.dataframe(
        pd.DataFrame(strength_component_rows(features, prediction)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Poisson expected goals")
    st.dataframe(
        pd.DataFrame(expected_goal_rows(prediction)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Model drivers")
    render_model_drivers(prediction)

    with st.expander("Raw API statistics"):
        render_api_dataframe(
            fixture_statistics_display_rows(selected_statistics),
            "No statistics returned for this fixture.",
        )

    with st.expander("Raw API events"):
        render_api_dataframe(
            fixture_events_display_rows(selected_events),
            "No events returned for this fixture.",
        )


def render_model_comparison(
    features: dict[str, Any],
    prediction: dict[str, Any],
    api_prediction: dict[str, Any],
) -> None:
    render_selected_header(features)
    st.caption(
        "Benchmark/reference comparison only. API-Football predictions are not used "
        "inside our world-cup-rules-v2 probability calculation."
    )

    st.markdown("#### Data source status")
    st.dataframe(
        pd.DataFrame(comparison_status_rows(api_prediction)),
        width="stretch",
        hide_index=True,
    )

    if not api_prediction.get("available"):
        st.warning(
            "API-Football prediction data is unavailable for this fixture or plan. "
            "Our v2 rules model remains active."
        )

    st.markdown("#### Side-by-side probabilities")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### Our v2 rules model")
        render_probability_cards(probability_rows(prediction), "match outcome")
    with cols[1]:
        st.markdown("##### API-Football prediction")
        api_cols = st.columns(3)
        for column, label, key in [
            (api_cols[0], "Home win", "home_probability"),
            (api_cols[1], "Draw", "draw_probability"),
            (api_cols[2], "Away win", "away_probability"),
        ]:
            value = api_prediction.get(key)
            with column:
                st.metric(label, optional_percentage(value))
                if value is not None:
                    st.progress(float(value))

    st.dataframe(
        pd.DataFrame(model_comparison_rows(prediction, api_prediction)),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Next goal")
    next_goal_cols = st.columns(3)
    next_goal_cols[0].metric(
        "Our home scores next",
        optional_percentage(prediction.get("home_scores_next_probability")),
    )
    next_goal_cols[1].metric(
        "Our away scores next",
        optional_percentage(prediction.get("away_scores_next_probability")),
    )
    next_goal_cols[2].metric("API-Football next goal", "not provided")

    st.markdown("#### API-Football advice")
    st.dataframe(
        pd.DataFrame(api_football_advice_rows(api_prediction)),
        width="stretch",
        hide_index=True,
    )


def render_public_odds_refresh_panel() -> None:
    st.markdown("#### Refresh cached odds")
    st.caption(
        "Provider token status: "
        f"{'configured' if sportmonks_token_configured() else 'not configured'}."
    )
    st.caption(
        "Public refresh captures cached SportMonks pre-kickoff odds for the next "
        f"{PUBLIC_ODDS_REFRESH_MAX_FIXTURES} fixtures. It is app-wide cooldown "
        f"limited to {PUBLIC_ODDS_REFRESH_COOLDOWN_SECONDS // 60} minutes."
    )

    state = load_public_odds_refresh_state()
    latest_summary = state.get("summary")
    if isinstance(latest_summary, dict) and latest_summary:
        st.dataframe(pd.DataFrame([latest_summary]), width="stretch", hide_index=True)

    remaining_seconds = public_odds_refresh_remaining_seconds(state)
    if not sportmonks_token_configured():
        st.warning(
            "Public refresh is unavailable until SPORTMONKS_API_TOKEN is configured "
            "in Streamlit Secrets and the app is rebooted."
        )
        st.button("Refresh cached odds", disabled=True)
        return

    if remaining_seconds > 0:
        st.info(
            f"Refresh cooldown active. Try again in about {max(1, remaining_seconds // 60)} minutes."
        )
        st.button("Refresh cached odds", disabled=True)
        return

    if not st.button("Refresh cached odds"):
        return

    try:
        summary = capture_pre_kickoff_odds(
            max_fixtures=PUBLIC_ODDS_REFRESH_MAX_FIXTURES,
        )
    except MissingSportMonksTokenError:
        st.error("SPORTMONKS_API_TOKEN is not configured in Streamlit Secrets.")
        return
    except SportMonksError as exc:
        st.error(f"SportMonks odds refresh failed: {str(exc)[:180]}")
        return

    save_public_odds_refresh_state(summary)
    st.cache_data.clear()
    st.success("Cached odds refresh completed. Reloading the dashboard cache.")
    st.rerun()


def render_provider_status(fixtures: list[dict[str, Any]] | None = None) -> None:
    st.subheader("Provider Status")
    st.caption(
        "SportMonks is audited and cached as an external data provider. Cached "
        "enrichment is evaluated separately from the headline model until a "
        "non-leaky benchmark proves it improves scoring."
    )
    st.dataframe(
        pd.DataFrame(sportmonks_provider_status_rows()),
        width="stretch",
        hide_index=True,
    )
    check_rows = sportmonks_audit_check_rows()
    if check_rows:
        st.markdown("#### Latest SportMonks access audit")
        st.dataframe(pd.DataFrame(check_rows), width="stretch", hide_index=True)
    else:
        st.info("No SportMonks audit file is available yet.")

    bundle = load_latest_world_cup_enrichment()
    st.markdown("#### Local SportMonks cache")
    st.dataframe(
        pd.DataFrame(sportmonks_cache_status_rows(bundle)),
        width="stretch",
        hide_index=True,
    )

    if not fixtures:
        st.info("Load a schedule to calculate API-Football to SportMonks coverage.")
        return

    coverage_rows = sportmonks_mapping_coverage_rows(fixtures, bundle)
    st.markdown("#### API-Football to SportMonks coverage")
    st.dataframe(
        pd.DataFrame(sportmonks_mapping_metric_rows(coverage_rows)),
        width="stretch",
        hide_index=True,
    )
    if coverage_rows:
        display_columns = [
            "api_fixture_id",
            "api_match",
            "sportmonks_fixture_id",
            "sportmonks_match",
            "mapping_confidence",
            "mapping_score",
            "fixture_detail_available",
            "xg_pair_available",
            "news_count",
            "xg_availability",
        ]
        st.dataframe(
            pd.DataFrame(coverage_rows)[
                [column for column in display_columns if column in coverage_rows[0]]
            ],
            width="stretch",
            hide_index=True,
        )


def backtest_rows(
    fixtures: list[dict[str, Any]],
    ratings: RatingMap,
    *,
    team_priors: dict[int, Any] | None = None,
    model_label: str = "baseline",
) -> list[dict[str, Any]]:
    del ratings
    rows = walk_forward_backtest_rows(
        fixtures,
        team_priors=team_priors,
        model_label=model_label,
    )
    for row in rows:
        row["running_accuracy_display"] = percentage(float(row["running_accuracy"]))
        row["confidence_display"] = percentage(float(row["confidence"]))
        row["brier_score_display"] = numeric_value(row.get("brier_score"), 3)
        row["log_loss_display"] = numeric_value(row.get("log_loss"), 3)
        row["probability_margin_display"] = optional_percentage(
            row.get("probability_margin")
        )
        row["home_form_signal_display"] = numeric_value(row.get("home_form_signal"), 2)
        row["away_form_signal_display"] = numeric_value(row.get("away_form_signal"), 2)
        row["expected_goal_gap_display"] = numeric_value(row.get("expected_goal_gap"), 2)
        row["total_expected_goals_display"] = numeric_value(
            row.get("total_expected_goals"),
            2,
        )
        row["rating_gap_display"] = numeric_value(row.get("rating_gap"), 1)
        row["form_gap_display"] = numeric_value(row.get("form_gap"), 2)
        row["top_vs_draw_margin_display"] = optional_percentage(
            row.get("top_vs_draw_margin")
        )
        row["home_prior_adjustment_display"] = numeric_value(
            row.get("home_prior_adjustment"),
            1,
        )
        row["away_prior_adjustment_display"] = numeric_value(
            row.get("away_prior_adjustment"),
            1,
        )
    return rows


def running_accuracy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("kickoff_utc") or ""),
            str(row.get("match") or ""),
            int(row.get("fixture_id") or 0),
        ),
    )
    correct_count = 0
    for index, row in enumerate(sorted_rows, start=1):
        if bool(row.get("correct")):
            correct_count += 1
        running_accuracy = correct_count / index
        row["match_number"] = index
        row["cumulative_correct"] = correct_count
        row["running_accuracy"] = running_accuracy
        row["running_accuracy_display"] = percentage(running_accuracy)
    return sorted_rows


def api_football_predicted_outcome(api_prediction: dict[str, Any]) -> str | None:
    return benchmark_api_football_predicted_outcome(api_prediction)


def api_football_running_accuracy_rows(
    rows: list[dict[str, Any]],
    api_predictions_by_fixture: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_rows = fair_api_comparison_rows(rows, api_predictions_by_fixture)
    for row in enriched_rows:
        row["api_football_running_accuracy_display"] = optional_percentage(
            row.get("api_football_running_accuracy")
        )
        row["api_football_brier_score_display"] = numeric_value(
            row.get("api_football_brier_score"),
            3,
        )
        row["api_football_log_loss_display"] = numeric_value(
            row.get("api_football_log_loss"),
            3,
        )
    return enriched_rows


def api_football_accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_rows = [
        row
        for row in rows
        if row.get("api_football_correct") is not None
    ]
    if not evaluated_rows:
        return {
            "evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "unavailable": len(rows),
        }
    correct = sum(1 for row in evaluated_rows if row["api_football_correct"])
    return {
        "evaluated": len(evaluated_rows),
        "correct": correct,
        "accuracy": correct / len(evaluated_rows),
        "unavailable": len(rows) - len(evaluated_rows),
    }


def benchmark_market_context(
    fixtures: list[dict[str, Any]],
    ratings: RatingMap,
) -> dict[str, Any]:
    rows = backtest_rows(fixtures, ratings)
    api_predictions_by_fixture = {
        int(row["fixture_id"]): fetch_api_football_prediction(int(row["fixture_id"]))
        for row in rows
    }
    rows = api_football_running_accuracy_rows(rows, api_predictions_by_fixture)
    sportmonks_bundle = load_latest_world_cup_enrichment()
    sportmonks_coverage_rows = sportmonks_mapping_coverage_rows(
        fixtures,
        sportmonks_bundle,
    )
    rows = sportmonks_candidate_rows(
        rows,
        sportmonks_candidate_enrichment_by_api_fixture(sportmonks_coverage_rows),
    )
    summary = fair_benchmark_summary(rows)
    sportmonks_summary = sportmonks_candidate_summary(rows)
    market_gate = benchmark_market_gate(summary, sportmonks_summary)
    market_rows = market_edge_rows_for_fixtures(
        fixtures,
        ratings,
        sportmonks_coverage_rows,
        benchmark_gate=market_gate,
    )
    return {
        "rows": rows,
        "summary": summary,
        "sportmonks_summary": sportmonks_summary,
        "sportmonks_coverage_rows": sportmonks_coverage_rows,
        "market_gate": market_gate,
        "market_rows": market_rows,
        "market_summary": market_edge_summary(market_rows),
    }


def render_snapshot_list() -> None:
    paths = list_prediction_snapshots()[:10]
    if not paths:
        st.info("No prediction snapshots saved yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [{"file": str(path), "modified": path.stat().st_mtime} for path in paths]
        ),
        width="stretch",
        hide_index=True,
    )


def render_schedule_calendar(board_rows: list[dict[str, Any]]) -> None:
    if not board_rows:
        st.info("No fixtures are available for the calendar.")
        return

    frame = pd.DataFrame(board_rows)
    if "myt_date_label" not in frame.columns:
        st.info("No MYT schedule fields are available for these fixtures.")
        return

    st.caption("All fixture times are displayed in Malaysia Time (MYT, UTC+8).")
    date_rows = (
        frame[["myt_date", "myt_date_label"]]
        .drop_duplicates()
        .sort_values("myt_date")
        .to_dict("records")
    )
    date_options = [row["myt_date_label"] for row in date_rows]
    selected_date_label = st.selectbox("Match day (MYT)", date_options)
    selected_date = next(
        row["myt_date"]
        for row in date_rows
        if row["myt_date_label"] == selected_date_label
    )
    day_frame = frame[frame["myt_date"] == selected_date].copy()
    day_frame = day_frame.sort_values(["myt_time", "match"])
    display_columns = [
        column
        for column in [
            "myt_time",
            "match",
            "round",
            "status",
            "score",
            "home_win",
            "draw",
            "away_win",
            "confidence",
        ]
        if column in day_frame.columns
    ]
    st.dataframe(day_frame[display_columns], width="stretch", hide_index=True)

    with st.expander("Full MYT schedule"):
        full_columns = [
            column
            for column in [
                "myt_datetime",
                "match",
                "round",
                "status",
                "score",
                "mode",
                "home_win",
                "draw",
                "away_win",
            ]
            if column in frame.columns
        ]
        st.dataframe(
            frame.sort_values(["myt_date", "myt_time", "match"])[full_columns],
            width="stretch",
            hide_index=True,
        )


def render_schedule_fallback(
    world_cup_league_id: int,
    season: int,
    message: str,
) -> None:
    st.info(message)
    with st.spinner("Loading World Cup schedule in MYT..."):
        dataset = fetch_world_cup_fixtures_with_fallback(world_cup_league_id, season)

    fixtures = dataset["fixtures"]
    if not fixtures:
        st.warning(
            "No schedule was returned for this World Cup league ID and season. "
            "Try changing the league ID or season in the sidebar."
        )
        render_snapshot_list()
        return

    board_rows = fixture_board_rows(fixtures, dataset["ratings"])
    if dataset.get("fallback_reason"):
        st.warning(dataset["fallback_reason"])
    st.caption(
        f"Schedule season shown: {dataset.get('resolved_season', season)} "
        f"(requested {dataset.get('requested_season', season)})."
    )
    schedule_tab, snapshots_tab = st.tabs(["Schedule", "Snapshots"])
    with schedule_tab:
        calendar_tab, board_tab = st.tabs(["Calendar", "Match Board"])
        with calendar_tab:
            st.subheader("World Cup Schedule Calendar")
            render_schedule_calendar(board_rows)
        with board_tab:
            st.subheader("World Cup Match Schedule")
            st.dataframe(pd.DataFrame(board_rows), width="stretch", hide_index=True)
    with snapshots_tab:
        st.subheader("Snapshots")
        render_snapshot_list()

    st.markdown(
        f"<div class='snapshot-line'>Saved fixture snapshot: "
        f"{dataset['fixture_snapshot_path']}<br>Saved rating snapshot: "
        f"{dataset['rating_snapshot_path']}</div>",
        unsafe_allow_html=True,
    )


def selected_fixture_from_board(
    fixtures: list[dict[str, Any]],
    board_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    options = {
        f"{row['myt_datetime']} MYT | {row['match']} | {row['status']} | "
        f"{row['score']}": row["fixture_id"]
        for row in board_rows
    }
    selected_label = st.selectbox("Selected World Cup fixture", list(options.keys()))
    selected_fixture_id = options[selected_label]
    return next(
        fixture
        for fixture in fixtures
        if int(fixture.get("fixture", {}).get("id") or 0) == selected_fixture_id
    )


def render_prediction_dashboard(
    fixtures: list[dict[str, Any]],
    ratings: RatingMap,
    board_rows: list[dict[str, Any]],
    selected_fixture: dict[str, Any],
    selected_statistics: list[dict[str, Any]],
    selected_events: list[dict[str, Any]],
) -> None:
    features = build_match_features(
        selected_fixture,
        statistics=selected_statistics,
        events=selected_events,
    )
    prediction = predict_fixture(
        selected_fixture,
        statistics=selected_statistics,
        events=selected_events,
        ratings=ratings,
    )
    result_banner = model_result_banner_data(
        selected_fixture,
        features,
        prediction,
        ratings,
    )
    api_prediction = fetch_api_football_prediction(features["fixture_id"])
    save_prediction_snapshot([prediction_snapshot_row(features, prediction)])

    tabs = st.tabs(
        [
            "Schedule",
            "Predictions",
            "Model Breakdown",
            "Model Comparison",
            "Backtest",
            "Paper Trading",
            "Provider Status",
            "Snapshots",
        ]
    )

    with tabs[0]:
        st.subheader("Schedule")
        calendar_tab, board_tab = st.tabs(["Calendar", "Match Board"])
        with calendar_tab:
            render_schedule_calendar(board_rows)
        with board_tab:
            st.dataframe(pd.DataFrame(board_rows), width="stretch", hide_index=True)

    with tabs[1]:
        st.subheader("Predictions")
        render_selected_header(features)
        render_model_result_banner(result_banner)
        render_probability_cards(probability_rows(prediction), "match outcome")
        st.markdown("#### Next goal")
        if prediction["prediction_mode"] == "prematch":
            st.info("This fixture is not live yet. Showing pre-match prediction only.")
        else:
            render_live_metrics(prediction)
        st.markdown("#### Outcome bars")
        render_prediction_bars(prediction)

    with tabs[2]:
        st.subheader("Model Breakdown")
        render_model_breakdown(
            selected_fixture,
            features,
            prediction,
            ratings,
            selected_statistics,
            selected_events,
        )

    with tabs[3]:
        st.subheader("Model Comparison")
        render_model_comparison(features, prediction, api_prediction)

    with tabs[4]:
        st.subheader("Fair Walk-Forward Benchmark")
        rows = backtest_rows(fixtures, ratings)
        if rows:
            team_priors = load_team_priors()
            include_team_priors = st.checkbox(
                "Include non-leaky team-prior ablation",
                value=bool(team_priors),
                help=(
                    "Uses data/team_priors/team_priors.csv only when rows pass "
                    "pre-kickoff source and timestamp checks."
                ),
            )
            prior_rows: list[dict[str, Any]] = []
            prior_summary: dict[str, Any] | None = None
            if include_team_priors and team_priors:
                prior_rows = backtest_rows(
                    fixtures,
                    ratings,
                    team_priors=team_priors,
                    model_label="team_priors",
                )
                prior_summary = team_prior_ablation_summary(rows, prior_rows)
            elif include_team_priors:
                st.info(
                    "No production team-prior source is loaded. Add a real "
                    f"pre-match prior file at {TEAM_PRIORS_PATH} with this schema."
                )
                st.dataframe(
                    pd.DataFrame(prior_schema_rows()),
                    width="stretch",
                    hide_index=True,
                )
            include_api_benchmark = st.checkbox(
                "Include API-Football prediction benchmark",
                value=True,
                help=(
                    "Uses cached /predictions calls for completed fixtures. "
                    "Unavailable fixtures are skipped from API-Football accuracy."
                ),
            )
            if include_api_benchmark:
                api_predictions_by_fixture = {
                    int(row["fixture_id"]): fetch_api_football_prediction(
                        int(row["fixture_id"])
                    )
                    for row in rows
                }
                rows = api_football_running_accuracy_rows(
                    rows,
                    api_predictions_by_fixture,
                )
            sportmonks_bundle = load_latest_world_cup_enrichment()
            sportmonks_coverage_rows = sportmonks_mapping_coverage_rows(
                fixtures,
                sportmonks_bundle,
            )
            rows = sportmonks_candidate_rows(
                rows,
                sportmonks_candidate_enrichment_by_api_fixture(
                    sportmonks_coverage_rows
                ),
            )
            for row in rows:
                row["sportmonks_candidate_running_accuracy_display"] = optional_percentage(
                    row.get("sportmonks_candidate_running_accuracy")
                )
                row["sportmonks_candidate_brier_score_display"] = numeric_value(
                    row.get("sportmonks_candidate_brier_score"),
                    3,
                )
                row["sportmonks_candidate_log_loss_display"] = numeric_value(
                    row.get("sportmonks_candidate_log_loss"),
                    3,
                )
                row["sportmonks_candidate_home_probability_display"] = (
                    optional_percentage(
                        row.get("sportmonks_candidate_home_probability")
                    )
                )
                row["sportmonks_candidate_draw_probability_display"] = (
                    optional_percentage(
                        row.get("sportmonks_candidate_draw_probability")
                    )
                )
                row["sportmonks_candidate_away_probability_display"] = (
                    optional_percentage(
                        row.get("sportmonks_candidate_away_probability")
                    )
                )
            frame = pd.DataFrame(rows)
            summary = fair_benchmark_summary(rows)
            sportmonks_summary = sportmonks_candidate_summary(rows)
            market_gate = benchmark_market_gate(summary, sportmonks_summary)
            market_rows = market_edge_rows_for_fixtures(
                fixtures,
                ratings,
                sportmonks_coverage_rows,
                benchmark_gate=market_gate,
            )
            market_summary = market_edge_summary(market_rows)
            diagnostic_counts = benchmark_diagnostic_counts(rows)
            draw_summary = draw_diagnostic_summary(rows)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Completed fixtures", summary["completed"])
            metric_cols[1].metric(
                "Our walk-forward accuracy",
                optional_percentage(summary["our_accuracy"]),
            )
            metric_cols[2].metric(
                "Our correct predictions",
                f"{summary['our_correct']}/{summary['completed']}",
            )
            if include_api_benchmark:
                metric_cols[3].metric(
                    "Shared evaluated fixtures",
                    summary["shared_evaluated"],
                )
                fair_cols = st.columns(4)
                fair_cols[0].metric(
                    "Our accuracy on shared",
                    optional_percentage(summary["our_shared_accuracy"]),
                )
                fair_cols[1].metric(
                    "API-Football accuracy",
                    optional_percentage(summary["api_accuracy"]),
                )
                fair_cols[2].metric(
                    "API-Football correct",
                    (
                        f"{summary['api_correct']}/{summary['shared_evaluated']}"
                        if summary["shared_evaluated"]
                        else "unavailable"
                    ),
                )
                fair_cols[3].metric(
                    "API unavailable",
                    summary["api_unavailable"],
                )

                score_cols = st.columns(4)
                score_cols[0].metric(
                    "Our Brier score",
                    numeric_value(summary["our_brier_score"], 3),
                )
                score_cols[1].metric(
                    "API Brier score",
                    numeric_value(summary["api_brier_score"], 3),
                )
                score_cols[2].metric(
                    "Our log loss",
                    numeric_value(summary["our_log_loss"], 3),
                )
                score_cols[3].metric(
                    "API log loss",
                    numeric_value(summary["api_log_loss"], 3),
                )
                confidence_cols = st.columns(2)
                confidence_cols[0].metric(
                    "Avg confidence when correct",
                    optional_percentage(summary["average_confidence_correct"]),
                )
                confidence_cols[1].metric(
                    "Avg confidence when wrong",
                    optional_percentage(summary["average_confidence_wrong"]),
                )
                diagnostic_cols = st.columns(6)
                diagnostic_cols[0].metric("Both correct", diagnostic_counts["both_correct"])
                diagnostic_cols[1].metric("Both wrong", diagnostic_counts["both_wrong"])
                diagnostic_cols[2].metric("Our-only wins", diagnostic_counts["our_only_wins"])
                diagnostic_cols[3].metric("API-only wins", diagnostic_counts["api_only_wins"])
                diagnostic_cols[4].metric("Draw misses", diagnostic_counts["draw_misses"])
                diagnostic_cols[5].metric(
                    "Away underdog misses",
                    diagnostic_counts["away_underdog_misses"],
                )
                draw_cols = st.columns(6)
                draw_cols[0].metric("Actual draws", draw_summary["actual_draws"])
                draw_cols[1].metric("Our draw misses", draw_summary["our_draw_misses"])
                draw_cols[2].metric("API draw misses", draw_summary["api_draw_misses"])
                draw_cols[3].metric(
                    "Avg draw prob on draws",
                    optional_percentage(
                        draw_summary["average_draw_probability_on_draws"]
                    ),
                )
                draw_cols[4].metric(
                    "Avg draw prob on non-draws",
                    optional_percentage(
                        draw_summary["average_draw_probability_on_non_draws"]
                    ),
                )
                draw_cols[5].metric(
                    "Avg top-vs-draw miss margin",
                    optional_percentage(
                        draw_summary["average_top_vs_draw_margin_on_draw_misses"]
                    ),
                )
                if summary["api_unavailable"]:
                    st.warning(
                        "API-Football predictions were unavailable for "
                        f"{summary['api_unavailable']} completed fixture(s)."
                    )
                if int(summary["shared_evaluated"] or 0) < 10:
                    st.info(
                        "Sample size is small. Treat stronger/weaker conclusions "
                        "as directional until more shared completed fixtures are available."
                    )
            st.markdown("#### Team-prior ablation")
            if prior_summary:
                prior_cols = st.columns(4)
                prior_cols[0].metric(
                    "Prior rows with signal",
                    prior_summary["prior_rows_with_signal"],
                )
                prior_cols[1].metric(
                    "Baseline accuracy",
                    optional_percentage(prior_summary["baseline_accuracy"]),
                )
                prior_cols[2].metric(
                    "Prior accuracy",
                    optional_percentage(prior_summary["prior_accuracy"]),
                )
                prior_cols[3].metric("Changed picks", prior_summary["changed_picks"])
                prior_score_cols = st.columns(4)
                prior_score_cols[0].metric(
                    "Baseline Brier",
                    numeric_value(prior_summary["baseline_brier_score"], 3),
                )
                prior_score_cols[1].metric(
                    "Prior Brier",
                    numeric_value(prior_summary["prior_brier_score"], 3),
                )
                prior_score_cols[2].metric(
                    "Baseline log loss",
                    numeric_value(prior_summary["baseline_log_loss"], 3),
                )
                prior_score_cols[3].metric(
                    "Prior log loss",
                    numeric_value(prior_summary["prior_log_loss"], 3),
                )
                if prior_summary["candidate_proves_improvement"]:
                    st.success(
                        "Team priors improve Brier score and log loss in the "
                        "walk-forward ablation. Keep this as research until "
                        "calibration gates are reviewed."
                    )
                else:
                    st.info(
                        "Team priors are not promoted. Continue using the current "
                        "model unless the ablation improves both Brier score and "
                        "log loss."
                    )
            elif not team_priors:
                st.info(
                    "Team-prior ablation is inactive because no real prior source "
                    "file is loaded. The model is not using fabricated team ratings."
                )
            st.markdown("#### SportMonks candidate experiment")
            sportmonks_cols = st.columns(4)
            sportmonks_cols[0].metric("Mapped completed fixtures", sportmonks_summary["mapped"])
            sportmonks_cols[1].metric("Eligible non-leaky fixtures", sportmonks_summary["eligible"])
            sportmonks_cols[2].metric(
                "Candidate Brier score",
                numeric_value(sportmonks_summary["candidate_brier_score"], 3),
            )
            sportmonks_cols[3].metric(
                "Candidate log loss",
                numeric_value(sportmonks_summary["candidate_log_loss"], 3),
            )
            sportmonks_score_cols = st.columns(4)
            sportmonks_score_cols[0].metric(
                "Baseline Brier on eligible",
                numeric_value(sportmonks_summary["baseline_brier_score"], 3),
            )
            sportmonks_score_cols[1].metric(
                "Brier delta",
                numeric_value(sportmonks_summary["brier_delta"], 3),
            )
            sportmonks_score_cols[2].metric(
                "Baseline log loss on eligible",
                numeric_value(sportmonks_summary["baseline_log_loss"], 3),
            )
            sportmonks_score_cols[3].metric(
                "Log-loss delta",
                numeric_value(sportmonks_summary["log_loss_delta"], 3),
            )
            if sportmonks_summary["candidate_proves_improvement"]:
                st.success(
                    "SportMonks candidate beats the current model on Brier score "
                    "and log loss with the minimum eligible sample. Review before promotion."
                )
            else:
                st.info(
                    "Headline model remains unchanged. SportMonks data must be "
                    "mapped, non-leaky, and better on Brier score plus log loss "
                    "before promotion."
                )
            reason_counts = sportmonks_summary.get("reason_counts") or {}
            if reason_counts:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"reason": key, "fixtures": value}
                            for key, value in sorted(reason_counts.items())
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            st.markdown("#### Market edge and CLV gate")
            market_cols = st.columns(4)
            market_cols[0].metric(
                "Benchmark gate",
                "passed" if market_gate["passed"] else "blocked",
            )
            market_cols[1].metric(
                "Fixtures with market",
                market_summary["fixtures_with_market"],
            )
            market_cols[2].metric(
                "Paper-trade candidates",
                market_summary["paper_trade_candidates"],
            )
            market_cols[3].metric("CLV tracked", market_summary["clv_tracked"])
            market_score_cols = st.columns(4)
            market_score_cols[0].metric(
                "Gate reason",
                str(market_gate["reason"]).replace("_", " "),
            )
            market_score_cols[1].metric(
                "Avg model edge",
                optional_percentage(market_summary["average_edge"]),
            )
            market_score_cols[2].metric(
                "Avg EV at best price",
                optional_percentage(market_summary["average_expected_value"]),
            )
            market_score_cols[3].metric(
                "Gate sample",
                f"{market_gate['shared_evaluated']}/{market_gate['min_rows']}",
            )
            st.info(
                "Market rows are research signals only. A row is only flagged "
                "when the benchmark gate passes and the model has both positive "
                "edge and positive expected value versus market-implied probability."
            )
            if market_rows:
                market_frame = pd.DataFrame(market_rows)
                for column in [
                    "model_probability",
                    "market_probability",
                    "edge",
                    "expected_value",
                    "entry_market_probability",
                    "closing_market_probability",
                    "clv_probability_delta",
                    "average_overround",
                ]:
                    if column in market_frame.columns:
                        market_frame[f"{column}_display"] = market_frame[column].map(
                            optional_percentage
                        )
                for column in [
                    "best_decimal",
                    "entry_decimal",
                    "closing_decimal",
                    "clv_decimal_delta",
                ]:
                    if column in market_frame.columns:
                        market_frame[f"{column}_display"] = market_frame[column].map(
                            lambda value: numeric_value(value, 2)
                        )
                market_display_columns = [
                    "status",
                    "edge_flag",
                    "match",
                    "outcome",
                    "model_probability_display",
                    "market_probability_display",
                    "edge_display",
                    "best_decimal_display",
                    "expected_value_display",
                    "market_snapshots",
                    "clv_probability_delta_display",
                    "clv_decimal_delta_display",
                    "bookmaker_count",
                    "average_overround_display",
                    "market_captured_at",
                    "benchmark_gate",
                ]
                st.dataframe(
                    market_frame[
                        [
                            column
                            for column in market_display_columns
                            if column in market_frame.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No SportMonks market snapshots are available yet.")
            st.caption(
                "Walk-forward scoring predicts each completed fixture before its "
                "result updates Elo ratings. API-Football metrics use only shared "
                "fixtures with available home/draw/away probabilities. SportMonks "
                "candidate metrics use only mapped enrichment captured before kickoff. "
                "Market edge rows use pre-kickoff odds snapshots only."
            )
            chart_columns = ["match_number", "running_accuracy"]
            if include_api_benchmark and "api_football_running_accuracy" in frame.columns:
                chart_columns.append("api_football_running_accuracy")
            if (
                "sportmonks_candidate_running_accuracy" in frame.columns
                and frame["sportmonks_candidate_running_accuracy"].notna().any()
            ):
                chart_columns.append("sportmonks_candidate_running_accuracy")
            chart_frame = frame[chart_columns].rename(
                columns={
                    "running_accuracy": "our_model",
                    "api_football_running_accuracy": "api_football",
                    "sportmonks_candidate_running_accuracy": "sportmonks_candidate",
                }
            )
            st.line_chart(chart_frame.set_index("match_number"))
            display_columns = [
                "match_number",
                "myt_datetime",
                "match",
                "score",
                "predicted",
                "actual",
                "correct",
                "running_accuracy_display",
                "api_football_status",
                "api_football_predicted",
                "api_football_correct",
                "api_football_running_accuracy_display",
                "sportmonks_candidate_status",
                "sportmonks_candidate_reason",
                "sportmonks_mapping_confidence",
                "sportmonks_candidate_predicted",
                "sportmonks_candidate_correct",
                "sportmonks_candidate_running_accuracy_display",
                "brier_score_display",
                "log_loss_display",
                "api_football_brier_score_display",
                "api_football_log_loss_display",
                "sportmonks_candidate_brier_score_display",
                "sportmonks_candidate_log_loss_display",
                "confidence_display",
                "probability_margin_display",
                "home_rating_before",
                "away_rating_before",
                "team_prior_available",
                "team_prior_source",
                "home_prior_adjustment_display",
                "away_prior_adjustment_display",
                "home_form_matches_before",
                "away_form_matches_before",
                "home_form_signal_display",
                "away_form_signal_display",
                "draw_rank",
                "draw_risk_label",
                "top_vs_draw_margin_display",
                "expected_goal_gap_display",
                "total_expected_goals_display",
                "rating_gap_display",
                "form_gap_display",
            ]
            st.dataframe(
                frame[[column for column in display_columns if column in frame.columns]],
                width="stretch",
                hide_index=True,
            )
            diagnostics = benchmark_diagnostic_rows(rows)
            if diagnostics:
                st.markdown("#### Miss and disagreement diagnostics")
                diagnostic_frame = pd.DataFrame(diagnostics)
                for column in [
                    "our_home",
                    "our_draw",
                    "our_away",
                    "api_home",
                    "api_draw",
                    "api_away",
                    "probability_margin",
                    "confidence",
                ]:
                    if column in diagnostic_frame.columns:
                        diagnostic_frame[f"{column}_display"] = diagnostic_frame[
                            column
                        ].map(optional_percentage)
                for column in [
                    "home_rating_before",
                    "away_rating_before",
                    "home_form_signal",
                    "away_form_signal",
                    "expected_goal_gap",
                    "total_expected_goals",
                    "rating_gap",
                    "form_gap",
                    "brier_score",
                    "log_loss",
                ]:
                    if column in diagnostic_frame.columns:
                        diagnostic_frame[f"{column}_display"] = diagnostic_frame[
                            column
                        ].map(lambda value: numeric_value(value, 3))
                diagnostic_display_columns = [
                    "category",
                    "match_number",
                    "match",
                    "score",
                    "actual",
                    "our_predicted",
                    "api_predicted",
                    "our_home_display",
                    "our_draw_display",
                    "our_away_display",
                    "api_home_display",
                    "api_draw_display",
                    "api_away_display",
                    "probability_margin_display",
                    "confidence_display",
                    "home_rating_before_display",
                    "away_rating_before_display",
                    "home_form_signal_display",
                    "away_form_signal_display",
                    "expected_goal_gap_display",
                    "total_expected_goals_display",
                    "rating_gap_display",
                    "form_gap_display",
                    "draw_risk_label",
                    "brier_score_display",
                    "log_loss_display",
                ]
                st.dataframe(
                    diagnostic_frame[
                        [
                            column
                            for column in diagnostic_display_columns
                            if column in diagnostic_frame.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            draw_diagnostics = draw_miss_diagnostic_rows(rows)
            if draw_diagnostics:
                st.markdown("#### Draw miss diagnostics")
                draw_frame = pd.DataFrame(draw_diagnostics)
                for column in [
                    "our_draw_probability",
                    "api_draw_probability",
                    "top_vs_draw_margin",
                ]:
                    if column in draw_frame.columns:
                        draw_frame[f"{column}_display"] = draw_frame[column].map(
                            optional_percentage
                        )
                for column in [
                    "expected_goal_gap",
                    "total_expected_goals",
                    "rating_gap",
                    "form_gap",
                ]:
                    if column in draw_frame.columns:
                        draw_frame[f"{column}_display"] = draw_frame[column].map(
                            lambda value: numeric_value(value, 3)
                        )
                draw_display_columns = [
                    "match",
                    "score",
                    "our_predicted",
                    "api_predicted",
                    "our_draw_probability_display",
                    "api_draw_probability_display",
                    "top_vs_draw_margin_display",
                    "expected_goal_gap_display",
                    "total_expected_goals_display",
                    "rating_gap_display",
                    "form_gap_display",
                    "draw_risk_label",
                    "our_draw_miss",
                    "api_draw_miss",
                ]
                st.dataframe(
                    draw_frame[
                        [
                            column
                            for column in draw_display_columns
                            if column in draw_frame.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("No completed fixtures are available for backtesting yet.")

    with tabs[5]:
        st.subheader("Paper Trading")
        context = benchmark_market_context(fixtures, ratings)
        market_gate = context["market_gate"]
        market_summary = context["market_summary"]
        paper_rows = paper_trade_rows(
            fixtures,
            context["market_rows"],
            paper_bankroll=DEFAULT_PAPER_BANKROLL,
            kelly_multiplier=DEFAULT_KELLY_MULTIPLIER,
            stake_cap_fraction=DEFAULT_STAKE_CAP_FRACTION,
        )
        paper_summary = paper_trade_summary(paper_rows)
        paper_cols = st.columns(4)
        paper_cols[0].metric(
            "Benchmark gate",
            "passed" if market_gate["passed"] else "blocked",
        )
        paper_cols[1].metric("Odds rows", market_summary["fixtures_with_market"])
        paper_cols[2].metric("Research candidates", paper_summary["research_candidates"])
        paper_cols[3].metric("Real stake", numeric_value(paper_summary["real_stake_units"], 2))
        pnl_cols = st.columns(4)
        pnl_cols[0].metric("Settled", paper_summary["settled"])
        pnl_cols[1].metric("Open", paper_summary["open"])
        pnl_cols[2].metric(
            "Realized paper P&L",
            numeric_value(paper_summary["realized_pnl_units"], 2),
        )
        pnl_cols[3].metric(
            "ROI on settled",
            optional_percentage(paper_summary["roi_on_settled"]),
        )
        exposure_cols = st.columns(4)
        exposure_cols[0].metric(
            "Paper bankroll",
            numeric_value(DEFAULT_PAPER_BANKROLL, 2),
        )
        exposure_cols[1].metric(
            "Stake cap",
            optional_percentage(DEFAULT_STAKE_CAP_FRACTION),
        )
        exposure_cols[2].metric(
            "Open exposure",
            numeric_value(paper_summary["open_exposure_units"], 2),
        )
        exposure_cols[3].metric(
            "Open possible profit",
            numeric_value(paper_summary["open_possible_profit_units"], 2),
        )
        st.markdown("#### Odds Movement")
        movement_cols = st.columns(4)
        movement_cols[0].metric("CLV tracked", paper_summary["clv_tracked"])
        movement_cols[1].metric("Favorable CLV", paper_summary["clv_favorable"])
        movement_cols[2].metric("Unfavorable CLV", paper_summary["clv_unfavorable"])
        movement_cols[3].metric("Flat CLV", paper_summary["clv_flat"])
        entry_cols = st.columns(4)
        entry_cols[0].metric(
            "Settled P&L at first odds",
            numeric_value(paper_summary["settled_first_entry_pnl_units"], 2),
        )
        entry_cols[1].metric(
            "Settled P&L at latest odds",
            numeric_value(paper_summary["settled_latest_entry_pnl_units"], 2),
        )
        entry_cols[2].metric(
            "Open profit at first odds",
            numeric_value(paper_summary["open_possible_profit_entry_first_units"], 2),
        )
        entry_cols[3].metric(
            "Open profit at latest odds",
            numeric_value(paper_summary["open_possible_profit_entry_latest_units"], 2),
        )
        st.info(
            "This tab is a research ledger only. Real stake remains zero while "
            "the benchmark gate is blocked. Odds come from cached SportMonks "
            "pre-kickoff full-time-result snapshots."
        )
        st.caption(
            "The public dashboard reads cached odds only. Refresh is available "
            "when SPORTMONKS_API_TOKEN is configured in Streamlit Secrets."
        )
        render_public_odds_refresh_panel()
        if paper_rows:
            paper_frame = pd.DataFrame(paper_rows)
            for column in [
                "model_probability",
                "market_probability",
                "edge",
                "expected_value",
                "raw_kelly_fraction",
                "paper_stake_fraction",
                "clv_probability_delta",
                "average_overround",
                "first_market_probability",
                "latest_market_probability",
                "first_edge",
                "latest_edge",
                "edge_change",
                "first_expected_value",
                "latest_expected_value",
                "expected_value_change",
            ]:
                if column in paper_frame.columns:
                    paper_frame[f"{column}_display"] = paper_frame[column].map(
                        optional_percentage
                    )
            for column in [
                "best_decimal",
                "paper_stake_units",
                "paper_pnl_units",
                "paper_possible_profit_units",
                "paper_possible_profit_entry_first_units",
                "paper_possible_profit_entry_latest_units",
                "paper_possible_loss_units",
                "bookie_friendly_cap_units",
                "first_decimal",
                "latest_decimal",
                "best_seen_decimal",
                "worst_seen_decimal",
                "paper_pnl_entry_first_units",
                "paper_pnl_entry_latest_units",
                "paper_pnl_first_vs_latest_delta_units",
                "first_hours_to_kickoff",
                "latest_hours_to_kickoff",
            ]:
                if column in paper_frame.columns:
                    paper_frame[f"{column}_display"] = paper_frame[column].map(
                        lambda value: numeric_value(value, 2)
                    )
            paper_display_columns = [
                "paper_status",
                "research_candidate",
                "match",
                "outcome",
                "settled_outcome",
                "best_decimal_display",
                "paper_stake_units_display",
                "paper_pnl_units_display",
                "paper_possible_profit_units_display",
                "paper_possible_profit_entry_first_units_display",
                "paper_possible_profit_entry_latest_units_display",
                "model_probability_display",
                "market_probability_display",
                "edge_display",
                "expected_value_display",
                "raw_kelly_fraction_display",
                "paper_stake_fraction_display",
                "bookmaker_count",
                "average_overround_display",
                "market_snapshots",
                "first_decimal_display",
                "latest_decimal_display",
                "best_seen_decimal_display",
                "worst_seen_decimal_display",
                "clv_direction",
                "clv_probability_delta_display",
                "first_edge_display",
                "latest_edge_display",
                "edge_change_display",
                "first_expected_value_display",
                "latest_expected_value_display",
                "expected_value_change_display",
                "first_hours_to_kickoff_display",
                "latest_hours_to_kickoff_display",
                "paper_pnl_entry_first_units_display",
                "paper_pnl_entry_latest_units_display",
                "paper_pnl_first_vs_latest_delta_units_display",
                "benchmark_gate",
                "market_captured_at",
            ]
            st.dataframe(
                paper_frame[
                    [
                        column
                        for column in paper_display_columns
                        if column in paper_frame.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "No cached pre-kickoff odds rows are available for paper trading. "
                "Use Refresh cached odds above after SPORTMONKS_API_TOKEN is configured "
                "in Streamlit Secrets."
            )

    with tabs[6]:
        render_provider_status(fixtures)

    with tabs[7]:
        st.subheader("Snapshots")
        render_snapshot_list()


def render_world_cup_season(world_cup_league_id: int, season: int) -> None:
    with st.spinner("Fetching World Cup fixtures..."):
        dataset = fetch_world_cup_fixtures_with_fallback(world_cup_league_id, season)

    fixtures = dataset["fixtures"]
    ratings = dataset["ratings"]
    if not fixtures:
        st.info("No World Cup fixtures were returned for this league and season.")
        return

    board_rows = fixture_board_rows(fixtures, ratings)
    if dataset.get("fallback_reason"):
        st.warning(dataset["fallback_reason"])
    st.caption(
        f"Schedule season shown: {dataset.get('resolved_season', season)} "
        f"(requested {dataset.get('requested_season', season)})."
    )
    selected_fixture = selected_fixture_from_board(fixtures, board_rows)
    statistics, events = fetch_fixture_detail(
        int(selected_fixture.get("fixture", {}).get("id") or 0)
    )
    render_prediction_dashboard(
        fixtures,
        ratings,
        board_rows,
        selected_fixture,
        statistics,
        events,
    )

    st.markdown(
        f"<div class='snapshot-line'>Saved fixture snapshot: "
        f"{dataset['fixture_snapshot_path']}<br>Saved rating snapshot: "
        f"{dataset['rating_snapshot_path']}</div>",
        unsafe_allow_html=True,
    )


def render_live_dashboard(view_scope: str, world_cup_league_id: int, season: int) -> None:
    with st.spinner("Fetching live fixtures, statistics, and events..."):
        dataset = fetch_live_prediction_dataset(view_scope, world_cup_league_id)

    matches = dataset["matches"]
    if not matches:
        if view_scope == "World Cup live":
            render_schedule_fallback(
                world_cup_league_id,
                season,
                "No World Cup matches are live right now. Showing the World Cup schedule in MYT.",
            )
        else:
            st.info("No live matches are available right now.")
            render_snapshot_list()
        return

    live_rows = live_board_rows(matches)
    options = {
        f"{row['minute']}' | {row['match']} | {row['score']}": row["fixture_id"]
        for row in live_rows
    }
    selected_label = st.selectbox("Selected live match", list(options.keys()))
    selected_fixture_id = options[selected_label]
    selected_match = next(
        match
        for match in matches
        if match["features"]["fixture_id"] == selected_fixture_id
    )

    render_prediction_dashboard(
        [match["fixture"] for match in matches],
        load_ratings(),
        live_rows,
        selected_match["fixture"],
        selected_match["statistics"],
        selected_match["events"],
    )

    st.markdown(
        f"<div class='snapshot-line'>Saved raw snapshot: "
        f"{dataset['raw_snapshot_path']}<br>Saved prediction snapshot: "
        f"{dataset['prediction_snapshot_path']}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_page_styles()
    st.title("World Cup Prediction Dashboard")
    st.caption("API-Football World Cup fixtures with local Elo and live match-state updates.")
    st.warning("Educational model only. This dashboard is not betting advice.")

    with st.sidebar:
        st.header("World Cup Focus")
        view_scope = st.radio(
            "View",
            ["World Cup live", "World Cup season", "All live"],
            index=0,
        )
        world_cup_league_id = st.number_input(
            "World Cup league ID",
            min_value=1,
            value=WORLD_CUP_DEFAULT_LEAGUE_ID,
            step=1,
        )
        season = st.number_input(
            "Season",
            min_value=2000,
            max_value=2100,
            value=WORLD_CUP_DEFAULT_SEASON,
            step=1,
        )
        refresh_clicked = st.button("Refresh", type="primary")

    if refresh_clicked:
        fetch_live_prediction_dataset.clear()
        fetch_world_cup_fixtures.clear()
        fetch_fixture_detail.clear()
        fetch_api_football_prediction.clear()

    try:
        if view_scope == "World Cup season":
            render_world_cup_season(int(world_cup_league_id), int(season))
        else:
            render_live_dashboard(view_scope, int(world_cup_league_id), int(season))
    except MissingApiKeyError as exc:
        st.error(str(exc))
    except ApiFootballRateLimitError as exc:
        st.error(str(exc))
    except ApiFootballError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
