from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np

from src.adapters import PaidDataSnapshot, default_paid_data_snapshot
from src.competition_context import (
    effective_home_advantage_elo,
    is_neutral_world_cup_fixture,
)
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


def calibrate_draw_probability(
    probabilities: tuple[float, float, float],
    home_expected_goals: float,
    away_expected_goals: float,
    rating_gap: float = 0.0,
    form_gap: float = 0.0,
    neutral_site: bool = False,
    home_matches_played: int = 0,
    away_matches_played: int = 0,
) -> tuple[float, float, float]:
    home, draw, away = _normalize_three(*probabilities)
    expected_gap = abs(home_expected_goals - away_expected_goals)
    total_expected = home_expected_goals + away_expected_goals
    top_probability = max(home, draw, away)
    top_vs_draw_margin = max(0.0, top_probability - draw)
    rating_close = abs(rating_gap) <= 115.0
    form_close = abs(form_gap) <= 0.30
    if (
        expected_gap >= 0.42
        or total_expected > 3.15
        or top_vs_draw_margin > 0.22
        or not rating_close
        or not form_close
    ):
        return home, draw, away

    xg_closeness = 1.0 - min(expected_gap / 0.42, 1.0)
    rating_closeness = 1.0 - min(abs(rating_gap) / 115.0, 1.0)
    form_closeness = 1.0 - min(abs(form_gap) / 0.30, 1.0)
    tempo_fit = 1.0 - min(abs(total_expected - 2.25) / 1.15, 1.0)
    margin_fit = 1.0 - min(top_vs_draw_margin / 0.22, 1.0)
    boost = 0.055 * (
        xg_closeness * 0.34
        + rating_closeness * 0.22
        + form_closeness * 0.18
        + max(tempo_fit, 0.25) * 0.16
        + margin_fit * 0.10
    )
    if neutral_site and home_matches_played + away_matches_played <= 2:
        leading_win_probability = max(home, away)
        cold_start_draw_target = min(
            0.36,
            max(draw, leading_win_probability - 0.02),
        )
        boost = max(boost, max(0.0, cold_start_draw_target - draw))
    if boost <= 0.0:
        return home, draw, away

    draw = min(draw + boost, 0.44)
    remaining = max(1.0 - draw, 0.0)
    win_total = home + away
    if win_total <= 0.0:
        return _normalize_three(remaining / 2.0, draw, remaining / 2.0)
    return _normalize_three(
        remaining * (home / win_total),
        draw,
        remaining * (away / win_total),
    )


def probability_entropy(probabilities: tuple[float, float, float]) -> float:
    entropy = 0.0
    for probability in probabilities:
        if probability > 0.0:
            entropy -= probability * float(np.log(probability))
    return float(entropy / np.log(3.0))


def model_confidence_from_probabilities(
    probabilities: tuple[float, float, float],
    home_matches_played: int,
    away_matches_played: int,
) -> float:
    ordered = sorted(probabilities, reverse=True)
    top_probability = ordered[0]
    margin = ordered[0] - ordered[1]
    certainty = 1.0 - probability_entropy(probabilities)
    depth = min(home_matches_played + away_matches_played, 12) / 12.0
    fallback_penalty = 0.12 if home_matches_played == 0 or away_matches_played == 0 else 0.0
    confidence = (
        0.26
        + top_probability * 0.20
        + margin * 0.32
        + certainty * 0.18
        + depth * 0.18
        - fallback_penalty
    )
    return float(np.clip(confidence, 0.22, 0.86))


def apply_form_adjustment_to_expected_goals(
    home_expected_goals: float,
    away_expected_goals: float,
    form_adjustment: dict[str, Any] | None,
) -> tuple[float, float, list[str]]:
    if not form_adjustment:
        return home_expected_goals, away_expected_goals, []

    home_signal = float(form_adjustment.get("home_form_signal", 0.0) or 0.0)
    away_signal = float(form_adjustment.get("away_form_signal", 0.0) or 0.0)
    edge = float(np.clip(home_signal - away_signal, -0.35, 0.35))
    home_multiplier = float(np.clip(1.0 + edge * 0.08, 0.94, 1.06))
    away_multiplier = float(np.clip(1.0 - edge * 0.08, 0.94, 1.06))
    adjusted_home = float(np.clip(home_expected_goals * home_multiplier, 0.25, 3.8))
    adjusted_away = float(np.clip(away_expected_goals * away_multiplier, 0.25, 3.8))
    drivers = []
    if abs(edge) >= 0.05:
        leader = form_adjustment.get("home_team") if edge > 0 else form_adjustment.get("away_team")
        drivers.append(
            f"Recent form lightly favors {leader}; pre-match expected goals are adjusted by prior completed fixtures only."
        )
    else:
        drivers.append("Recent form is balanced; no material form adjustment is applied.")
    return adjusted_home, adjusted_away, drivers


def prematch_expected_goals(
    home_rating: float,
    away_rating: float,
    *,
    home_advantage_elo: float = HOME_ADVANTAGE_ELO,
    neutral_site: bool = False,
) -> tuple[float, float]:
    rating_gap = (home_rating + home_advantage_elo) - away_rating
    home_base = 1.30 if neutral_site else 1.35
    away_base = 1.05 if neutral_site else 1.10
    home_expected = home_base * np.exp(rating_gap / 900.0)
    away_expected = away_base * np.exp(-rating_gap / 900.0)
    return float(np.clip(home_expected, 0.25, 3.8)), float(np.clip(away_expected, 0.25, 3.8))


def prematch_prediction(
    fixture: dict[str, Any],
    ratings: RatingMap | None = None,
    paid_data: PaidDataSnapshot | None = None,
    form_adjustment: dict[str, Any] | None = None,
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
    home_advantage_elo = effective_home_advantage_elo(
        fixture,
        HOME_ADVANTAGE_ELO,
    )
    neutral_site = is_neutral_world_cup_fixture(fixture, HOME_ADVANTAGE_ELO)
    home_xg, away_xg = prematch_expected_goals(
        home_rating.rating,
        away_rating.rating,
        home_advantage_elo=home_advantage_elo,
        neutral_site=neutral_site,
    )
    form_adjustment = dict(form_adjustment or {})
    form_adjustment.setdefault("home_team", home_name)
    form_adjustment.setdefault("away_team", away_name)
    home_xg, away_xg, form_drivers = apply_form_adjustment_to_expected_goals(
        home_xg,
        away_xg,
        form_adjustment,
    )
    rating_gap = (home_rating.rating + home_advantage_elo) - away_rating.rating
    form_gap = float(
        form_adjustment.get("home_form_signal", 0.0)
        - form_adjustment.get("away_form_signal", 0.0)
    )
    home_win, draw, away_win = outcome_probabilities_from_expected_goals(home_xg, away_xg)
    raw_draw = draw
    home_win, draw, away_win = calibrate_draw_probability(
        (home_win, draw, away_win),
        home_xg,
        away_xg,
        rating_gap=rating_gap,
        form_gap=form_gap,
        neutral_site=neutral_site,
        home_matches_played=home_rating.matches_played,
        away_matches_played=away_rating.matches_played,
    )
    home_win, draw, away_win = _blend_with_odds_prior(
        (home_win, draw, away_win),
        paid_data,
    )
    confidence = model_confidence_from_probabilities(
        (home_win, draw, away_win),
        home_rating.matches_played,
        away_rating.matches_played,
    )
    drivers = prematch_drivers(
        home_name,
        away_name,
        home_rating.rating,
        away_rating.rating,
        home_advantage_elo=home_advantage_elo,
        neutral_site=neutral_site,
    )
    drivers.extend(form_drivers)
    if draw > raw_draw + 0.001:
        drivers.append(
            "Draw calibration is active because expected goals, ratings, form, venue context, and top-vs-draw margin are close."
        )
    ordered_probabilities = sorted((home_win, draw, away_win), reverse=True)
    margin = ordered_probabilities[0] - ordered_probabilities[1]
    drivers.append(
        f"Confidence reflects top probability, {margin * 100:.1f} percentage-point margin, rating depth, and probability uncertainty."
    )
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
    *,
    home_advantage_elo: float = HOME_ADVANTAGE_ELO,
    neutral_site: bool = False,
) -> list[str]:
    rating_gap = (home_rating + home_advantage_elo) - away_rating
    if neutral_site:
        drivers = [
            "Neutral World Cup venue removes the standard home-advantage Elo boost."
        ]
    else:
        drivers = [
            f"Home or host advantage adds {home_advantage_elo:.0f} Elo points to {home_name}."
        ]
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
    real_xg_available = bool(
        features.get("real_xg_available", result.get("real_xg_available", False))
    )
    result["paid_data_availability"] = (
        f"odds={'available' if result.get('odds_available') else 'missing'}, "
        f"real_xg={'available' if real_xg_available else 'missing'}, "
        f"injuries={'available' if result.get('injuries_available') else 'missing'}, "
        f"news={'available' if result.get('news_available') else 'missing'}"
    )
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
                real_xg_available
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
