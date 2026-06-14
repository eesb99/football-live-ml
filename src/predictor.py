from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np

from src.adapters import PaidDataSnapshot, default_paid_data_snapshot
from src.features import build_match_features, is_live_status
from src.model import predict_match_probabilities
from src.ratings import (
    DEFAULT_RATING,
    HOME_ADVANTAGE_ELO,
    RatingMap,
    expected_score,
    get_rating,
)


MODEL_VERSION = "world-cup-rules-v2"
FINAL_STATUSES = {"FT", "AET", "PEN"}
PREMATCH_STATUSES = {"TBD", "NS", "PST", "CANC", "ABD", "AWD", "WO"}


@dataclass(frozen=True)
class PredictionResult:
    prediction_mode: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    next_goal_probability: float
    home_scores_next_probability: float
    away_scores_next_probability: float
    no_next_goal_probability: float
    home_strength_score: float
    away_strength_score: float
    home_proxy_xg: float
    away_proxy_xg: float
    home_effective_xg: float
    away_effective_xg: float
    xg_source: str
    home_expected_goals: float
    away_expected_goals: float
    home_expected_remaining_goals: float
    away_expected_remaining_goals: float
    model_confidence: float
    model_version: str
    odds_available: bool
    real_xg_available: bool
    injuries_available: bool
    news_available: bool
    odds_source: str
    real_xg_source: str
    injuries_source: str
    news_source: str
    paid_data_availability: str
    model_drivers: list[str]

    def as_dict(self) -> dict[str, Any]:
        data = {
            "prediction_mode": self.prediction_mode,
            "home_win_probability": self.home_win_probability,
            "draw_probability": self.draw_probability,
            "away_win_probability": self.away_win_probability,
            "next_goal_probability": self.next_goal_probability,
            "home_scores_next_probability": self.home_scores_next_probability,
            "away_scores_next_probability": self.away_scores_next_probability,
            "no_next_goal_probability": self.no_next_goal_probability,
            "home_strength_score": self.home_strength_score,
            "away_strength_score": self.away_strength_score,
            "home_proxy_xg": self.home_proxy_xg,
            "away_proxy_xg": self.away_proxy_xg,
            "home_effective_xg": self.home_effective_xg,
            "away_effective_xg": self.away_effective_xg,
            "xg_source": self.xg_source,
            "home_expected_goals": self.home_expected_goals,
            "away_expected_goals": self.away_expected_goals,
            "home_expected_remaining_goals": self.home_expected_remaining_goals,
            "away_expected_remaining_goals": self.away_expected_remaining_goals,
            "model_confidence": self.model_confidence,
            "model_version": self.model_version,
            "odds_available": self.odds_available,
            "real_xg_available": self.real_xg_available,
            "injuries_available": self.injuries_available,
            "news_available": self.news_available,
            "odds_source": self.odds_source,
            "real_xg_source": self.real_xg_source,
            "injuries_source": self.injuries_source,
            "news_source": self.news_source,
            "paid_data_availability": self.paid_data_availability,
            "model_drivers": self.model_drivers,
            "model_driver_summary": "; ".join(self.model_drivers),
        }
        return data


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize_three(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return home / total, draw / total, away / total


def _data_source_fields(
    paid_data: PaidDataSnapshot,
    *,
    real_xg_available: bool = False,
    real_xg_source: str | None = None,
) -> dict[str, Any]:
    fields = paid_data.as_prediction_fields()
    fields["real_xg_available"] = bool(real_xg_available or paid_data.real_xg_available)
    if real_xg_source:
        fields["real_xg_source"] = real_xg_source
    fields["paid_data_availability"] = (
        f"odds={'available' if fields['odds_available'] else 'missing'}, "
        f"real_xg={'available' if fields['real_xg_available'] else 'missing'}, "
        f"injuries={'available' if fields['injuries_available'] else 'missing'}, "
        f"news={'available' if fields['news_available'] else 'missing'}"
    )
    return fields


def _blend_with_odds_prior(
    model_probabilities: tuple[float, float, float],
    paid_data: PaidDataSnapshot,
    weight: float = 0.2,
) -> tuple[float, float, float]:
    odds_values = (
        paid_data.home_odds_implied_probability,
        paid_data.draw_odds_implied_probability,
        paid_data.away_odds_implied_probability,
    )
    if not paid_data.odds_available or any(value is None for value in odds_values):
        return model_probabilities
    home_odds, draw_odds, away_odds = _normalize_three(
        float(odds_values[0]),
        float(odds_values[1]),
        float(odds_values[2]),
    )
    home, draw, away = model_probabilities
    return _normalize_three(
        home * (1.0 - weight) + home_odds * weight,
        draw * (1.0 - weight) + draw_odds * weight,
        away * (1.0 - weight) + away_odds * weight,
    )


def apply_paid_data_to_features(
    features: dict[str, Any],
    paid_data: PaidDataSnapshot,
) -> dict[str, Any]:
    enriched = dict(features)
    enriched["home_injury_impact"] = paid_data.home_injury_impact
    enriched["away_injury_impact"] = paid_data.away_injury_impact
    enriched["home_news_impact"] = paid_data.home_news_impact
    enriched["away_news_impact"] = paid_data.away_news_impact
    if (
        paid_data.real_xg_available
        and paid_data.home_real_xg is not None
        and paid_data.away_real_xg is not None
    ):
        enriched["home_xg"] = float(paid_data.home_real_xg)
        enriched["away_xg"] = float(paid_data.away_real_xg)
        enriched["xg_difference"] = enriched["home_xg"] - enriched["away_xg"]
        enriched["home_effective_xg"] = enriched["home_xg"]
        enriched["away_effective_xg"] = enriched["away_xg"]
        enriched["effective_xg_difference"] = (
            enriched["home_effective_xg"] - enriched["away_effective_xg"]
        )
        enriched["xg_source"] = paid_data.real_xg_source or "paid_real_xg"
        enriched["real_xg_available"] = True
    return enriched


def _poisson_pmf(count: int, lambda_value: float) -> float:
    return float(np.exp(-lambda_value) * (lambda_value ** count) / factorial(count))


def outcome_probabilities_from_expected_goals(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 8,
) -> tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_goals in range(max_goals + 1):
        home_prob = _poisson_pmf(home_goals, home_expected_goals)
        for away_goals in range(max_goals + 1):
            probability = home_prob * _poisson_pmf(away_goals, away_expected_goals)
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
    return _normalize_three(home_win, draw, away_win)


def prematch_expected_goals(home_rating: float, away_rating: float) -> tuple[float, float]:
    rating_gap = (home_rating + HOME_ADVANTAGE_ELO) - away_rating
    home_expected = 1.35 * np.exp(rating_gap / 900.0)
    away_expected = 1.10 * np.exp(-rating_gap / 900.0)
    return float(np.clip(home_expected, 0.25, 3.8)), float(np.clip(away_expected, 0.25, 3.8))


def prematch_prediction(
    fixture: dict[str, Any],
    ratings: RatingMap | None = None,
    paid_data: PaidDataSnapshot | None = None,
) -> dict[str, Any]:
    ratings = ratings or {}
    paid_data = paid_data or default_paid_data_snapshot()
    home = fixture.get("teams", {}).get("home", {})
    away = fixture.get("teams", {}).get("away", {})
    home_id = int(home.get("id") or 0)
    away_id = int(away.get("id") or 0)
    home_name = home.get("name") or "Home"
    away_name = away.get("name") or "Away"
    home_rating = get_rating(ratings, home_id, home_name)
    away_rating = get_rating(ratings, away_id, away_name)
    home_xg, away_xg = prematch_expected_goals(home_rating.rating, away_rating.rating)
    home_win, draw, away_win = outcome_probabilities_from_expected_goals(home_xg, away_xg)
    home_win, draw, away_win = _blend_with_odds_prior(
        (home_win, draw, away_win),
        paid_data,
    )
    confidence = 0.45 + min(home_rating.matches_played + away_rating.matches_played, 12) * 0.025
    drivers = prematch_drivers(home_name, away_name, home_rating.rating, away_rating.rating)
    if paid_data.odds_available:
        drivers.append("Optional odds adapter is available and lightly informs the pre-match prior.")
    else:
        drivers.append("Optional odds adapter is not configured; using Elo-only pre-match prior.")
    source_fields = _data_source_fields(paid_data)

    return PredictionResult(
        prediction_mode="prematch",
        home_win_probability=_bounded(home_win),
        draw_probability=_bounded(draw),
        away_win_probability=_bounded(away_win),
        next_goal_probability=0.0,
        home_scores_next_probability=0.0,
        away_scores_next_probability=0.0,
        no_next_goal_probability=1.0,
        home_strength_score=home_rating.rating,
        away_strength_score=away_rating.rating,
        home_proxy_xg=home_xg,
        away_proxy_xg=away_xg,
        home_effective_xg=home_xg,
        away_effective_xg=away_xg,
        xg_source="prematch_elo_expected_goals",
        home_expected_goals=home_xg,
        away_expected_goals=away_xg,
        home_expected_remaining_goals=home_xg,
        away_expected_remaining_goals=away_xg,
        model_confidence=float(np.clip(confidence, 0.35, 0.75)),
        model_version=MODEL_VERSION,
        odds_available=bool(source_fields["odds_available"]),
        real_xg_available=bool(source_fields["real_xg_available"]),
        injuries_available=bool(source_fields["injuries_available"]),
        news_available=bool(source_fields["news_available"]),
        odds_source=str(source_fields["odds_source"]),
        real_xg_source=str(source_fields["real_xg_source"]),
        injuries_source=str(source_fields["injuries_source"]),
        news_source=str(source_fields["news_source"]),
        paid_data_availability=str(source_fields["paid_data_availability"]),
        model_drivers=drivers,
    ).as_dict()


def prematch_drivers(
    home_name: str,
    away_name: str,
    home_rating: float,
    away_rating: float,
) -> list[str]:
    rating_gap = (home_rating + HOME_ADVANTAGE_ELO) - away_rating
    drivers = [f"Home advantage adds {HOME_ADVANTAGE_ELO:.0f} Elo points to {home_name}."]
    if abs(home_rating - DEFAULT_RATING) < 0.01 and abs(away_rating - DEFAULT_RATING) < 0.01:
        drivers.append("Both teams use fallback 1500 ratings until enough results are loaded.")
    if rating_gap > 80:
        drivers.append(f"{home_name} has the stronger pre-match rating after home advantage.")
    elif rating_gap < -80:
        drivers.append(f"{away_name} has the stronger pre-match rating despite home advantage.")
    else:
        drivers.append("Ratings are close, so draw probability remains material.")
    return drivers


def final_prediction(features: dict[str, Any], prematch: dict[str, Any]) -> dict[str, Any]:
    home_goals = int(features.get("home_goals", 0) or 0)
    away_goals = int(features.get("away_goals", 0) or 0)
    home_win, draw, away_win = 0.0, 0.0, 0.0
    if home_goals > away_goals:
        home_win = 1.0
    elif home_goals < away_goals:
        away_win = 1.0
    else:
        draw = 1.0
    result = dict(prematch)
    result.update(
        {
            "prediction_mode": "final",
            "home_win_probability": home_win,
            "draw_probability": draw,
            "away_win_probability": away_win,
            "next_goal_probability": 0.0,
            "home_scores_next_probability": 0.0,
            "away_scores_next_probability": 0.0,
            "no_next_goal_probability": 1.0,
            "home_proxy_xg": features.get("home_proxy_xg", result.get("home_proxy_xg", 0.0)),
            "away_proxy_xg": features.get("away_proxy_xg", result.get("away_proxy_xg", 0.0)),
            "home_effective_xg": features.get(
                "home_effective_xg",
                result.get("home_effective_xg", 0.0),
            ),
            "away_effective_xg": features.get(
                "away_effective_xg",
                result.get("away_effective_xg", 0.0),
            ),
            "xg_source": features.get("xg_source", result.get("xg_source", "proxy_xg")),
            "real_xg_available": bool(
                features.get("real_xg_available", result.get("real_xg_available", False))
            ),
            "real_xg_source": str(
                features.get("xg_source", result.get("real_xg_source", "not configured"))
            ),
            "model_confidence": 1.0,
            "model_drivers": [f"Final score is {home_goals}-{away_goals}; outcome is known."],
            "model_driver_summary": f"Final score is {home_goals}-{away_goals}; outcome is known.",
        }
    )
    return result


def live_blend_weight(features: dict[str, Any]) -> float:
    elapsed = float(features.get("elapsed_fraction", 0.0) or 0.0)
    completeness = float(features.get("data_completeness_score", 0.0) or 0.0)
    return float(np.clip(0.25 + elapsed * 0.45 + completeness * 0.20, 0.25, 0.85))


def blend_live_prediction(
    features: dict[str, Any],
    prematch: dict[str, Any],
    live: dict[str, Any],
    paid_data: PaidDataSnapshot,
) -> dict[str, Any]:
    live_weight = live_blend_weight(features)
    prior_weight = 1.0 - live_weight
    home, draw, away = _normalize_three(
        prematch["home_win_probability"] * prior_weight
        + live["home_win_probability"] * live_weight,
        prematch["draw_probability"] * prior_weight + live["draw_probability"] * live_weight,
        prematch["away_win_probability"] * prior_weight
        + live["away_win_probability"] * live_weight,
    )
    drivers = live_drivers(features, live_weight, paid_data)
    real_xg_available = bool(features.get("real_xg_available"))
    source_fields = _data_source_fields(
        paid_data,
        real_xg_available=real_xg_available,
        real_xg_source=str(features.get("xg_source") or ""),
    )
    return {
        **prematch,
        "prediction_mode": "live",
        "home_win_probability": _bounded(home),
        "draw_probability": _bounded(draw),
        "away_win_probability": _bounded(away),
        "next_goal_probability": live["next_goal_probability"],
        "home_scores_next_probability": live["home_scores_next_probability"],
        "away_scores_next_probability": live["away_scores_next_probability"],
        "no_next_goal_probability": live["no_next_goal_probability"],
        "home_strength_score": live["home_strength_score"],
        "away_strength_score": live["away_strength_score"],
        "home_proxy_xg": live["home_proxy_xg"],
        "away_proxy_xg": live["away_proxy_xg"],
        "home_effective_xg": live["home_effective_xg"],
        "away_effective_xg": live["away_effective_xg"],
        "xg_source": live["xg_source"],
        "home_expected_remaining_goals": live["home_expected_remaining_goals"],
        "away_expected_remaining_goals": live["away_expected_remaining_goals"],
        "model_confidence": float(
            np.clip((prematch["model_confidence"] * prior_weight) + (live["model_confidence"] * live_weight), 0.0, 0.98)
        ),
        **source_fields,
        "model_drivers": drivers,
        "model_driver_summary": "; ".join(drivers),
    }


def live_drivers(
    features: dict[str, Any],
    live_weight: float,
    paid_data: PaidDataSnapshot,
) -> list[str]:
    drivers = [f"Live match state weight is {live_weight * 100:.0f}% based on minute and data completeness."]
    score_difference = int(features.get("score_difference", 0) or 0)
    if score_difference > 0:
        drivers.append(f"{features['home_team']} lead by {score_difference}, lifting home win probability.")
    elif score_difference < 0:
        drivers.append(f"{features['away_team']} lead by {abs(score_difference)}, lifting away win probability.")
    else:
        drivers.append("Score is level, so draw probability remains important.")
    red_difference = float(features.get("red_card_difference", 0.0) or 0.0)
    if red_difference > 0:
        drivers.append(f"{features['home_team']} have more red cards, reducing their live strength.")
    elif red_difference < 0:
        drivers.append(f"{features['away_team']} have more red cards, reducing their live strength.")
    pressure_difference = float(features.get("pressure_difference", 0.0) or 0.0)
    if pressure_difference > 0.25:
        drivers.append(f"{features['home_team']} show stronger live pressure from shots, corners, and possession.")
    elif pressure_difference < -0.25:
        drivers.append(f"{features['away_team']} show stronger live pressure from shots, corners, and possession.")
    xg_source = str(features.get("xg_source") or "proxy_xg")
    if xg_source == "proxy_xg":
        drivers.append("Real xG is unavailable; proxy xG is estimated from shots, shots on target, box shots, corners, possession, and recent events.")
    else:
        drivers.append(f"Real xG source is available from {xg_source}; effective xG uses that feed.")
    if paid_data.injuries_available or paid_data.news_available:
        drivers.append("Optional injuries/news adapters are available and applied as small context multipliers.")
    else:
        drivers.append("Optional injuries/news adapters are not configured; no off-pitch context adjustment is applied.")
    return drivers


def predict_fixture(
    fixture: dict[str, Any],
    statistics: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    ratings: RatingMap | None = None,
    paid_data: PaidDataSnapshot | None = None,
) -> dict[str, Any]:
    paid_data = paid_data or default_paid_data_snapshot()
    pre = prematch_prediction(fixture, ratings=ratings, paid_data=paid_data)
    features = apply_paid_data_to_features(
        build_match_features(fixture, statistics=statistics, events=events),
        paid_data,
    )
    status = features.get("status_short")
    if status in FINAL_STATUSES:
        return final_prediction(features, pre)
    if is_live_status(fixture):
        live = predict_match_probabilities(features)
        return blend_live_prediction(features, pre, live, paid_data)
    return pre


def prediction_snapshot_row(features: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": features["fixture_id"],
        "fixture_date": features["fixture_date"],
        "league_id": features["league_id"],
        "league_name": features["league_name"],
        "league_season": features["league_season"],
        "league_round": features["league_round"],
        "home_team": features["home_team"],
        "away_team": features["away_team"],
        "status": features["status"],
        "status_short": features["status_short"],
        "home_goals": features["home_goals"],
        "away_goals": features["away_goals"],
        **prediction,
        "model_drivers": prediction.get("model_driver_summary", ""),
    }
