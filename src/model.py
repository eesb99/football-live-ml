from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ProbabilityPrediction:
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
    home_expected_remaining_goals: float
    away_expected_remaining_goals: float
    model_confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
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
            "home_expected_remaining_goals": self.home_expected_remaining_goals,
            "away_expected_remaining_goals": self.away_expected_remaining_goals,
            "model_confidence": self.model_confidence,
        }


def _bounded_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _remaining_minutes(minute: float) -> float:
    return float(np.clip(95.0 - minute, 0.0, 95.0))


def _feature_float(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    return float(features.get(name, default) or default)


def _opponent(side: str) -> str:
    return "away" if side == "home" else "home"


def _side_sign(side: str) -> int:
    return 1 if side == "home" else -1


def team_strength_score(features: dict[str, Any], side: str) -> float:
    opponent = _opponent(side)
    pressure_share = _feature_float(features, f"{side}_pressure_share", 0.5)
    shot_share = _feature_float(features, f"{side}_shot_share", 0.5)
    sot_share = _feature_float(features, f"{side}_shots_on_target_share", 0.5)
    possession = _feature_float(features, f"{side}_possession", 50.0)
    shots_inside_box = _feature_float(features, f"{side}_shots_inside_box")
    recent_events = _feature_float(features, f"{side}_recent_events")
    recent_goals = _feature_float(features, f"{side}_recent_goals")
    goals = _feature_float(features, f"{side}_goals")
    red_cards = _feature_float(features, f"{side}_red_cards")
    opponent_red_cards = _feature_float(features, f"{opponent}_red_cards")
    pass_accuracy = _feature_float(features, f"{side}_pass_accuracy")
    effective_xg = _feature_float(features, f"{side}_effective_xg")

    raw_score = (
        1.0
        + pressure_share * 2.4
        + shot_share * 0.9
        + sot_share * 1.3
        + effective_xg * 0.34
        + ((possession - 50.0) / 50.0) * 0.45
        + shots_inside_box * 0.05
        + recent_events * 0.08
        + recent_goals * 0.16
        + goals * 0.18
        + pass_accuracy * 0.003
        - red_cards * 0.7
        + opponent_red_cards * 0.4
    )
    return float(max(raw_score, 0.1))


def _game_state_multiplier(features: dict[str, Any], side: str) -> float:
    score_difference = _feature_float(features, "score_difference")
    side_score_margin = score_difference * _side_sign(side)
    minute = _feature_float(features, "minute")
    late_match_factor = min(max((minute - 45.0) / 50.0, 0.0), 1.0)

    if side_score_margin < 0:
        return 1.0 + min(abs(side_score_margin), 3.0) * (0.14 + late_match_factor * 0.14)
    if side_score_margin > 0:
        return 1.0 - min(side_score_margin, 3.0) * (0.06 + late_match_factor * 0.05)
    return 1.0


def _discipline_multiplier(features: dict[str, Any], side: str) -> float:
    opponent = _opponent(side)
    red_cards = _feature_float(features, f"{side}_red_cards")
    opponent_red_cards = _feature_float(features, f"{opponent}_red_cards")
    yellow_cards = _feature_float(features, f"{side}_yellow_cards")
    return float(
        np.clip(
            1.0 - red_cards * 0.22 + opponent_red_cards * 0.16 - yellow_cards * 0.015,
            0.45,
            1.35,
        )
    )


def _home_advantage_multiplier(side: str) -> float:
    return 1.06 if side == "home" else 0.98


def _xg_quality_multiplier(features: dict[str, Any], side: str) -> float:
    minute = max(_feature_float(features, "minute"), 1.0)
    elapsed_fraction = max(minute / 95.0, 0.18)
    effective_xg = _feature_float(features, f"{side}_effective_xg")
    xg_pace = effective_xg / elapsed_fraction
    return float(np.clip(0.72 + xg_pace / 3.2, 0.68, 1.55))


def _paid_context_multiplier(features: dict[str, Any], side: str) -> float:
    injury_impact = _feature_float(features, f"{side}_injury_impact")
    news_impact = _feature_float(features, f"{side}_news_impact")
    return float(np.clip(1.0 - injury_impact + news_impact, 0.75, 1.18))


def expected_remaining_goals(features: dict[str, Any], side: str) -> float:
    minute = _feature_float(features, "minute")
    remaining_fraction = _remaining_minutes(minute) / 95.0
    strength = team_strength_score(features, side)
    opponent = _opponent(side)
    opponent_strength = team_strength_score(features, opponent)
    strength_share = strength / max(strength + opponent_strength, 0.1)
    pressure_share = _feature_float(features, f"{side}_pressure_share", strength_share)
    attack_share = (strength_share * 0.65) + (pressure_share * 0.35)

    total_goal_environment = 2.55
    match_tempo = 0.85 + (
        _feature_float(features, "home_shots") + _feature_float(features, "away_shots")
    ) / 28.0
    target_tempo = 0.85 + (
        _feature_float(features, "home_shots_on_target")
        + _feature_float(features, "away_shots_on_target")
    ) / 12.0
    tempo_multiplier = float(np.clip((match_tempo + target_tempo) / 2.0, 0.65, 1.45))

    side_lambda = (
        total_goal_environment
        * remaining_fraction
        * attack_share
        * tempo_multiplier
        * _game_state_multiplier(features, side)
        * _discipline_multiplier(features, side)
        * _home_advantage_multiplier(side)
        * _xg_quality_multiplier(features, side)
        * _paid_context_multiplier(features, side)
    )
    return float(max(side_lambda, 0.0))


def model_confidence(features: dict[str, Any]) -> float:
    completeness = _feature_float(features, "data_completeness_score", 0.0)
    minute = _feature_float(features, "minute")
    minute_signal = min(max(minute / 35.0, 0.15), 1.0)
    return float(np.clip(0.25 + completeness * 0.55 + minute_signal * 0.20, 0.15, 0.95))


def poisson_live_probabilities(
    features: dict[str, Any],
    max_extra_goals: int = 8,
) -> ProbabilityPrediction:
    home_goals = int(features.get("home_goals", 0) or 0)
    away_goals = int(features.get("away_goals", 0) or 0)
    home_lambda = expected_remaining_goals(features, "home")
    away_lambda = expected_remaining_goals(features, "away")

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    home_goal_counts = np.arange(max_extra_goals + 1)
    away_goal_counts = np.arange(max_extra_goals + 1)
    home_probs = _poisson_pmf(home_goal_counts, home_lambda)
    away_probs = _poisson_pmf(away_goal_counts, away_lambda)

    for home_extra, home_prob in zip(home_goal_counts, home_probs, strict=True):
        for away_extra, away_prob in zip(away_goal_counts, away_probs, strict=True):
            probability = float(home_prob * away_prob)
            projected_home = home_goals + int(home_extra)
            projected_away = away_goals + int(away_extra)
            if projected_home > projected_away:
                home_win += probability
            elif projected_home == projected_away:
                draw += probability
            else:
                away_win += probability

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    next_total = home_lambda + away_lambda
    no_next_goal = float(np.exp(-next_total))
    next_goal = 1.0 - no_next_goal
    if next_total <= 0:
        home_next = 0.0
        away_next = 0.0
    else:
        home_next = next_goal * (home_lambda / next_total)
        away_next = next_goal * (away_lambda / next_total)

    return ProbabilityPrediction(
        home_win_probability=_bounded_probability(home_win),
        draw_probability=_bounded_probability(draw),
        away_win_probability=_bounded_probability(away_win),
        next_goal_probability=_bounded_probability(next_goal),
        home_scores_next_probability=_bounded_probability(home_next),
        away_scores_next_probability=_bounded_probability(away_next),
        no_next_goal_probability=_bounded_probability(no_next_goal),
        home_strength_score=team_strength_score(features, "home"),
        away_strength_score=team_strength_score(features, "away"),
        home_proxy_xg=_feature_float(features, "home_proxy_xg"),
        away_proxy_xg=_feature_float(features, "away_proxy_xg"),
        home_effective_xg=_feature_float(features, "home_effective_xg"),
        away_effective_xg=_feature_float(features, "away_effective_xg"),
        xg_source=str(features.get("xg_source") or "proxy_xg"),
        home_expected_remaining_goals=home_lambda,
        away_expected_remaining_goals=away_lambda,
        model_confidence=model_confidence(features),
    )


def _poisson_pmf(counts: np.ndarray, lambda_value: float) -> np.ndarray:
    factorials = np.array([factorial(int(count)) for count in counts], dtype=float)
    return np.exp(-lambda_value) * np.power(lambda_value, counts) / factorials


class SklearnProbabilityModel:
    """Placeholder adapter for a future trained scikit-learn model."""

    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        self.estimator = estimator

    @property
    def is_trained(self) -> bool:
        return self.estimator is not None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.estimator is None:
            raise ValueError("No trained scikit-learn model is loaded.")
        predictions = self.estimator.predict_proba(features)
        return pd.DataFrame(predictions)


def predict_match_probabilities(features: dict[str, Any]) -> dict[str, float]:
    return poisson_live_probabilities(features).as_dict()
