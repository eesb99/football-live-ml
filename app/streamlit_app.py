from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import ApiFootballClient, ApiFootballError, ApiFootballRateLimitError
from src.config import MissingApiKeyError, load_settings
from src.features import build_live_match_table, build_match_features, is_live_status
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
    list_prediction_snapshots,
    save_prediction_snapshot,
    save_snapshot,
)


st.set_page_config(page_title="World Cup Prediction Dashboard", layout="wide")

WORLD_CUP_DEFAULT_LEAGUE_ID = 1
WORLD_CUP_DEFAULT_SEASON = 2026
FREE_PLAN_FALLBACK_SEASONS = [2022, 2023, 2024]
WORLD_CUP_KEYWORDS = ("world cup", "fifa world cup")


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fb; }
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        .match-title {
            color: #172033;
            font-size: 1.45rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
        }
        .muted-line {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }
        .snapshot-line {
            color: #475569;
            font-size: 0.85rem;
            margin-top: 0.65rem;
        }
        h1, h2, h3 { color: #172033; letter-spacing: 0; }
        div[data-testid="stTabs"] button {
            font-weight: 650;
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


def numeric_value(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


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
                "signal": "API expected goals",
                "home": numeric_value(features.get("home_xg"), 2),
                "away": numeric_value(features.get("away_xg"), 2),
                "match": numeric_value(features.get("xg_difference"), 2),
            }
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


def backtest_rows(fixtures: list[dict[str, Any]], ratings: RatingMap) -> list[dict[str, Any]]:
    rows = []
    for fixture in fixtures:
        goals = fixture.get("goals", {})
        if goals.get("home") is None or goals.get("away") is None:
            continue
        status = fixture.get("fixture", {}).get("status", {}).get("short")
        if status not in {"FT", "AET", "PEN"}:
            continue
        features = build_match_features(fixture)
        prediction = predict_fixture(fixture, ratings=ratings)
        probabilities = {
            "home": prediction["home_win_probability"],
            "draw": prediction["draw_probability"],
            "away": prediction["away_win_probability"],
        }
        predicted = max(probabilities, key=probabilities.get)
        actual = "draw"
        if features["home_goals"] > features["away_goals"]:
            actual = "home"
        elif features["home_goals"] < features["away_goals"]:
            actual = "away"
        rows.append(
            {
                "fixture_id": features["fixture_id"],
                "match": f"{features['home_team']} vs {features['away_team']}",
                "score": f"{features['home_goals']}-{features['away_goals']}",
                "predicted": predicted,
                "actual": actual,
                "correct": predicted == actual,
                "confidence": percentage(prediction["model_confidence"]),
            }
        )
    return rows


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
    save_prediction_snapshot([prediction_snapshot_row(features, prediction)])

    tabs = st.tabs(
        [
            "Schedule",
            "Predictions",
            "Model Breakdown",
            "Backtest",
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
        st.subheader("Backtest")
        rows = backtest_rows(fixtures, ratings)
        if rows:
            frame = pd.DataFrame(rows)
            st.metric("Completed fixtures", len(frame))
            st.metric("Simple hit rate", percentage(float(frame["correct"].mean())))
            st.dataframe(frame, width="stretch", hide_index=True)
        else:
            st.info("No completed fixtures are available for backtesting yet.")

    with tabs[4]:
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
