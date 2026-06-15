from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np

from src.competition_context import effective_home_advantage_elo
from src.features import build_match_features
from src.predictor import (
    calibrate_draw_probability,
    outcome_probabilities_from_expected_goals,
    prematch_prediction,
)
from src.ratings import (
    DEFAULT_K_FACTOR,
    HOME_ADVANTAGE_ELO,
    RatingMap,
    get_rating,
    update_ratings_for_fixture,
)
from src.schedule import fixture_myt_fields
from src.sportmonks_mapping import parse_fixture_datetime


OUTCOMES = ("home", "draw", "away")
FINAL_STATUS_SHORTS = {"FT", "AET", "PEN"}
FORM_WINDOW = 5


@dataclass
class TeamFormState:
    results: list[dict[str, float]] = field(default_factory=list)

    def add(self, points: float, goals_for: float, goals_against: float) -> None:
        self.results.append(
            {
                "points": points,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goals_for - goals_against,
            }
        )
        if len(self.results) > FORM_WINDOW:
            self.results = self.results[-FORM_WINDOW:]

    @property
    def matches_played(self) -> int:
        return len(self.results)

    @property
    def points_per_match(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([result["points"] for result in self.results]))

    @property
    def goal_difference_per_match(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([result["goal_difference"] for result in self.results]))

    @property
    def signal(self) -> float:
        if not self.results:
            return 0.0
        points_component = (self.points_per_match - 1.0) / 2.0
        goal_component = self.goal_difference_per_match / 3.0
        return float(np.clip(points_component + goal_component, -1.0, 1.0))


def completed_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        fixture
        for fixture in fixtures
        if fixture.get("fixture", {}).get("status", {}).get("short")
        in FINAL_STATUS_SHORTS
        and fixture.get("goals", {}).get("home") is not None
        and fixture.get("goals", {}).get("away") is not None
    ]


def sorted_completed_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        completed_fixtures(fixtures),
        key=lambda fixture: (
            fixture.get("fixture", {}).get("date") or "",
            fixture.get("teams", {}).get("home", {}).get("name") or "",
            int(fixture.get("fixture", {}).get("id") or 0),
        ),
    )


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def fixture_points(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 3.0, 0.0
    if home_goals < away_goals:
        return 0.0, 3.0
    return 1.0, 1.0


def probabilities_from_prediction(prediction: dict[str, Any]) -> dict[str, float]:
    return {
        "home": float(prediction.get("home_win_probability", 0.0) or 0.0),
        "draw": float(prediction.get("draw_probability", 0.0) or 0.0),
        "away": float(prediction.get("away_win_probability", 0.0) or 0.0),
    }


def predicted_outcome(probabilities: dict[str, float | None]) -> str | None:
    usable = {
        outcome: float(value)
        for outcome, value in probabilities.items()
        if outcome in OUTCOMES and value is not None
    }
    if not usable:
        return None
    return max(usable, key=usable.get)


def probability_rank(probabilities: dict[str, float], outcome: str) -> int:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for index, (candidate, _) in enumerate(ordered, start=1):
        if candidate == outcome:
            return index
    return len(ordered)


def top_two_probabilities(probabilities: dict[str, float]) -> tuple[float, float]:
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return (ordered[0], 0.0) if ordered else (0.0, 0.0)
    return ordered[0], ordered[1]


def draw_risk_label(
    expected_goal_gap: float,
    total_expected_goals: float,
    rating_gap: float,
    form_gap: float,
    top_vs_draw_margin: float,
) -> str:
    if (
        expected_goal_gap <= 0.18
        and 1.6 <= total_expected_goals <= 2.85
        and abs(rating_gap) <= 80.0
        and abs(form_gap) <= 0.20
        and top_vs_draw_margin <= 0.14
    ):
        return "high"
    if (
        expected_goal_gap <= 0.35
        and total_expected_goals <= 3.10
        and abs(rating_gap) <= 140.0
        and abs(form_gap) <= 0.40
        and top_vs_draw_margin <= 0.24
    ):
        return "medium"
    return "low"


def brier_score(probabilities: dict[str, float | None], actual: str) -> float | None:
    if actual not in OUTCOMES:
        return None
    if any(probabilities.get(outcome) is None for outcome in OUTCOMES):
        return None
    return float(
        sum(
            (
                float(probabilities[outcome])
                - (1.0 if outcome == actual else 0.0)
            )
            ** 2
            for outcome in OUTCOMES
        )
    )


def log_loss(
    probabilities: dict[str, float | None],
    actual: str,
    epsilon: float = 1e-15,
) -> float | None:
    if actual not in OUTCOMES:
        return None
    value = probabilities.get(actual)
    if value is None:
        return None
    clipped = float(np.clip(float(value), epsilon, 1.0 - epsilon))
    return float(-log(clipped))


def api_football_probabilities(api_prediction: dict[str, Any]) -> dict[str, float | None]:
    if not api_prediction.get("available"):
        return {"home": None, "draw": None, "away": None}
    return {
        "home": api_prediction.get("home_probability"),
        "draw": api_prediction.get("draw_probability"),
        "away": api_prediction.get("away_probability"),
    }


def api_football_predicted_outcome(api_prediction: dict[str, Any]) -> str | None:
    return predicted_outcome(api_football_probabilities(api_prediction))


def form_adjustment_for_fixture(
    form_by_team: dict[int, TeamFormState],
    home_id: int,
    away_id: int,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    home_form = form_by_team.get(home_id, TeamFormState())
    away_form = form_by_team.get(away_id, TeamFormState())
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_form_signal": home_form.signal,
        "away_form_signal": away_form.signal,
        "home_form_matches": home_form.matches_played,
        "away_form_matches": away_form.matches_played,
        "home_points_per_match": home_form.points_per_match,
        "away_points_per_match": away_form.points_per_match,
        "home_goal_difference_per_match": home_form.goal_difference_per_match,
        "away_goal_difference_per_match": away_form.goal_difference_per_match,
    }


def update_form_for_fixture(
    form_by_team: dict[int, TeamFormState],
    fixture: dict[str, Any],
    home_id: int,
    away_id: int,
    home_goals: int,
    away_goals: int,
) -> dict[int, TeamFormState]:
    del fixture
    home_points, away_points = fixture_points(home_goals, away_goals)
    updated = dict(form_by_team)
    home_form = updated.get(home_id, TeamFormState())
    away_form = updated.get(away_id, TeamFormState())
    home_form.add(home_points, home_goals, away_goals)
    away_form.add(away_points, away_goals, home_goals)
    updated[home_id] = home_form
    updated[away_id] = away_form
    return updated


def walk_forward_backtest_rows(
    fixtures: list[dict[str, Any]],
    initial_ratings: RatingMap | None = None,
    k_factor: float = DEFAULT_K_FACTOR,
) -> list[dict[str, Any]]:
    ratings = dict(initial_ratings or {})
    form_by_team: dict[int, TeamFormState] = {}
    rows: list[dict[str, Any]] = []
    correct_count = 0
    brier_values: list[float] = []
    log_loss_values: list[float] = []

    for match_number, fixture in enumerate(sorted_completed_fixtures(fixtures), start=1):
        features = build_match_features(fixture)
        home_id = int(fixture.get("teams", {}).get("home", {}).get("id") or 0)
        away_id = int(fixture.get("teams", {}).get("away", {}).get("id") or 0)
        home_rating = get_rating(ratings, home_id, features["home_team"])
        away_rating = get_rating(ratings, away_id, features["away_team"])
        form_adjustment = form_adjustment_for_fixture(
            form_by_team,
            home_id,
            away_id,
            features["home_team"],
            features["away_team"],
        )

        prediction = prematch_prediction(
            fixture,
            ratings=ratings,
            form_adjustment=form_adjustment,
        )
        probabilities = probabilities_from_prediction(prediction)
        predicted = predicted_outcome(probabilities) or "-"
        actual = actual_outcome(features["home_goals"], features["away_goals"])
        top_probability, second_probability = top_two_probabilities(probabilities)
        home_advantage_elo = effective_home_advantage_elo(
            fixture,
            HOME_ADVANTAGE_ELO,
        )
        rating_gap = (home_rating.rating + home_advantage_elo) - away_rating.rating
        form_gap = float(
            form_adjustment["home_form_signal"] - form_adjustment["away_form_signal"]
        )
        expected_goal_gap = abs(
            float(prediction.get("home_expected_goals", 0.0) or 0.0)
            - float(prediction.get("away_expected_goals", 0.0) or 0.0)
        )
        total_expected_goals = float(prediction.get("home_expected_goals", 0.0) or 0.0) + float(
            prediction.get("away_expected_goals", 0.0) or 0.0
        )
        top_vs_draw_margin = top_probability - probabilities["draw"]
        draw_rank = probability_rank(probabilities, "draw")
        actual_is_draw = actual == "draw"
        our_draw_miss = actual_is_draw and predicted != "draw"
        draw_risk = draw_risk_label(
            expected_goal_gap,
            total_expected_goals,
            rating_gap,
            form_gap,
            top_vs_draw_margin,
        )
        correct = predicted == actual
        if correct:
            correct_count += 1

        row_brier = brier_score(probabilities, actual)
        row_log_loss = log_loss(probabilities, actual)
        if row_brier is not None:
            brier_values.append(row_brier)
        if row_log_loss is not None:
            log_loss_values.append(row_log_loss)

        myt_fields = fixture_myt_fields(fixture)
        kickoff = fixture.get("fixture", {}).get("date") or ""
        rows.append(
            {
                "fixture_id": features["fixture_id"],
                "kickoff_utc": kickoff,
                "myt_datetime": myt_fields.get("myt_datetime", ""),
                "match": f"{features['home_team']} vs {features['away_team']}",
                "score": f"{features['home_goals']}-{features['away_goals']}",
                "predicted": predicted,
                "actual": actual,
                "correct": correct,
                "match_number": match_number,
                "cumulative_correct": correct_count,
                "running_accuracy": correct_count / match_number,
                "home_probability": probabilities["home"],
                "draw_probability": probabilities["draw"],
                "away_probability": probabilities["away"],
                "home_expected_goals": float(
                    prediction.get("home_expected_goals", 0.0) or 0.0
                ),
                "away_expected_goals": float(
                    prediction.get("away_expected_goals", 0.0) or 0.0
                ),
                "confidence": float(prediction.get("model_confidence", 0.0) or 0.0),
                "brier_score": row_brier,
                "log_loss": row_log_loss,
                "running_brier_score": float(np.mean(brier_values)) if brier_values else None,
                "running_log_loss": float(np.mean(log_loss_values)) if log_loss_values else None,
                "home_rating_before": home_rating.rating,
                "away_rating_before": away_rating.rating,
                "home_matches_before": home_rating.matches_played,
                "away_matches_before": away_rating.matches_played,
                "home_form_matches_before": form_adjustment["home_form_matches"],
                "away_form_matches_before": form_adjustment["away_form_matches"],
                "home_form_signal": form_adjustment["home_form_signal"],
                "away_form_signal": form_adjustment["away_form_signal"],
                "home_points_per_match": form_adjustment["home_points_per_match"],
                "away_points_per_match": form_adjustment["away_points_per_match"],
                "home_goal_difference_per_match": form_adjustment[
                    "home_goal_difference_per_match"
                ],
                "away_goal_difference_per_match": form_adjustment[
                    "away_goal_difference_per_match"
                ],
                "probability_margin": top_probability - second_probability,
                "top_probability": top_probability,
                "second_probability": second_probability,
                "top_vs_draw_margin": top_vs_draw_margin,
                "draw_rank": draw_rank,
                "actual_is_draw": actual_is_draw,
                "our_draw_miss": our_draw_miss,
                "expected_goal_gap": expected_goal_gap,
                "total_expected_goals": total_expected_goals,
                "rating_gap": rating_gap,
                "form_gap": form_gap,
                "draw_risk_label": draw_risk,
            }
        )

        ratings = update_ratings_for_fixture(ratings, fixture, k_factor=k_factor)
        form_by_team = update_form_for_fixture(
            form_by_team,
            fixture,
            home_id,
            away_id,
            features["home_goals"],
            features["away_goals"],
        )

    return rows


def fair_api_comparison_rows(
    rows: list[dict[str, Any]],
    api_predictions_by_fixture: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_rows = [dict(row) for row in rows]
    api_evaluated_count = 0
    api_correct_count = 0

    for row in enriched_rows:
        fixture_id = int(row.get("fixture_id") or 0)
        api_prediction = api_predictions_by_fixture.get(fixture_id, {})
        api_probabilities = api_football_probabilities(api_prediction)
        api_predicted = api_football_predicted_outcome(api_prediction)
        actual = str(row.get("actual") or "")

        row["api_football_status"] = str(api_prediction.get("status") or "missing")
        row["api_football_predicted"] = api_predicted or "-"
        row["api_football_available"] = bool(api_predicted)
        row["api_home_probability"] = api_probabilities["home"]
        row["api_draw_probability"] = api_probabilities["draw"]
        row["api_away_probability"] = api_probabilities["away"]
        row["api_football_correct"] = None
        row["api_football_running_accuracy"] = None
        row["api_football_brier_score"] = None
        row["api_football_log_loss"] = None
        row["api_predicted_draw"] = api_predicted == "draw"
        row["api_draw_miss"] = (
            actual == "draw" and api_predicted is not None and api_predicted != "draw"
        )

        if api_predicted is None:
            continue

        api_evaluated_count += 1
        api_correct = api_predicted == actual
        if api_correct:
            api_correct_count += 1
        row["api_football_correct"] = api_correct
        row["api_football_evaluated_match_number"] = api_evaluated_count
        row["api_football_cumulative_correct"] = api_correct_count
        row["api_football_running_accuracy"] = api_correct_count / api_evaluated_count
        row["api_football_brier_score"] = brier_score(api_probabilities, actual)
        row["api_football_log_loss"] = log_loss(api_probabilities, actual)

    return enriched_rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def fair_benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shared_rows = [
        row
        for row in rows
        if row.get("api_football_correct") is not None
    ]
    completed_count = len(rows)
    our_correct_all = sum(1 for row in rows if row.get("correct"))
    our_correct_shared = sum(1 for row in shared_rows if row.get("correct"))
    api_correct_shared = sum(1 for row in shared_rows if row.get("api_football_correct"))

    correct_confidence = [
        float(row["confidence"])
        for row in rows
        if row.get("correct") and row.get("confidence") is not None
    ]
    wrong_confidence = [
        float(row["confidence"])
        for row in rows
        if row.get("correct") is False and row.get("confidence") is not None
    ]

    return {
        "completed": completed_count,
        "our_correct": our_correct_all,
        "our_accuracy": our_correct_all / completed_count if completed_count else None,
        "shared_evaluated": len(shared_rows),
        "api_unavailable": completed_count - len(shared_rows),
        "our_shared_correct": our_correct_shared,
        "our_shared_accuracy": (
            our_correct_shared / len(shared_rows) if shared_rows else None
        ),
        "api_correct": api_correct_shared,
        "api_accuracy": api_correct_shared / len(shared_rows) if shared_rows else None,
        "our_brier_score": _mean(
            [
                float(row["brier_score"])
                for row in shared_rows
                if row.get("brier_score") is not None
            ]
        ),
        "api_brier_score": _mean(
            [
                float(row["api_football_brier_score"])
                for row in shared_rows
                if row.get("api_football_brier_score") is not None
            ]
        ),
        "our_log_loss": _mean(
            [
                float(row["log_loss"])
                for row in shared_rows
                if row.get("log_loss") is not None
            ]
        ),
        "api_log_loss": _mean(
            [
                float(row["api_football_log_loss"])
                for row in shared_rows
                if row.get("api_football_log_loss") is not None
            ]
        ),
        "average_confidence_correct": _mean(correct_confidence),
        "average_confidence_wrong": _mean(wrong_confidence),
    }


def benchmark_diagnostic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    shared_rows = [
        row
        for row in rows
        if row.get("api_football_correct") is not None
    ]
    both_correct = sum(
        1
        for row in shared_rows
        if row.get("correct") is True and row.get("api_football_correct") is True
    )
    both_wrong = sum(
        1
        for row in shared_rows
        if row.get("correct") is False and row.get("api_football_correct") is False
    )
    our_only = sum(
        1
        for row in shared_rows
        if row.get("correct") is True and row.get("api_football_correct") is False
    )
    api_only = sum(
        1
        for row in shared_rows
        if row.get("correct") is False and row.get("api_football_correct") is True
    )
    draw_misses = sum(
        1
        for row in rows
        if row.get("actual") == "draw" and row.get("predicted") != "draw"
    )
    away_underdog_misses = sum(
        1
        for row in rows
        if row.get("actual") == "away"
        and row.get("correct") is False
        and float(row.get("away_probability", 0.0) or 0.0)
        < max(
            float(row.get("home_probability", 0.0) or 0.0),
            float(row.get("draw_probability", 0.0) or 0.0),
        )
    )
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "our_only_wins": our_only,
        "api_only_wins": api_only,
        "draw_misses": draw_misses,
        "away_underdog_misses": away_underdog_misses,
    }


def draw_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_draw_rows = [row for row in rows if row.get("actual_is_draw")]
    non_draw_rows = [row for row in rows if not row.get("actual_is_draw")]
    draw_miss_rows = [row for row in rows if row.get("our_draw_miss")]
    api_draw_miss_rows = [row for row in rows if row.get("api_draw_miss")]

    return {
        "actual_draws": len(actual_draw_rows),
        "our_draw_misses": len(draw_miss_rows),
        "api_draw_misses": len(api_draw_miss_rows),
        "average_draw_probability_on_draws": _mean(
            [
                float(row["draw_probability"])
                for row in actual_draw_rows
                if row.get("draw_probability") is not None
            ]
        ),
        "average_draw_probability_on_non_draws": _mean(
            [
                float(row["draw_probability"])
                for row in non_draw_rows
                if row.get("draw_probability") is not None
            ]
        ),
        "average_top_vs_draw_margin_on_draw_misses": _mean(
            [
                float(row["top_vs_draw_margin"])
                for row in draw_miss_rows
                if row.get("top_vs_draw_margin") is not None
            ]
        ),
    }


def draw_miss_diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = []
    for row in rows:
        if not row.get("actual_is_draw"):
            continue
        diagnostics.append(
            {
                "match": row.get("match"),
                "score": row.get("score"),
                "our_predicted": row.get("predicted"),
                "api_predicted": row.get("api_football_predicted", "-"),
                "our_draw_probability": row.get("draw_probability"),
                "api_draw_probability": row.get("api_draw_probability"),
                "top_vs_draw_margin": row.get("top_vs_draw_margin"),
                "expected_goal_gap": row.get("expected_goal_gap"),
                "total_expected_goals": row.get("total_expected_goals"),
                "rating_gap": row.get("rating_gap"),
                "form_gap": row.get("form_gap"),
                "draw_risk_label": row.get("draw_risk_label"),
                "our_draw_miss": row.get("our_draw_miss"),
                "api_draw_miss": row.get("api_draw_miss"),
            }
        )
    return diagnostics


def benchmark_diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostic_rows = []
    for row in rows:
        our_correct = row.get("correct")
        api_correct = row.get("api_football_correct")
        include = our_correct is False
        include = include or (api_correct is True and our_correct is False)
        include = include or (api_correct is False and our_correct is True)
        include = include or (api_correct is False and our_correct is False)
        if not include:
            continue

        category = "our_miss"
        if our_correct is False and api_correct is True:
            category = "api_only_win"
        elif our_correct is True and api_correct is False:
            category = "our_only_win"
        elif our_correct is False and api_correct is False:
            category = "both_wrong"

        diagnostic_rows.append(
            {
                "category": category,
                "match_number": row.get("match_number"),
                "match": row.get("match"),
                "score": row.get("score"),
                "actual": row.get("actual"),
                "our_predicted": row.get("predicted"),
                "api_predicted": row.get("api_football_predicted", "-"),
                "our_home": row.get("home_probability"),
                "our_draw": row.get("draw_probability"),
                "our_away": row.get("away_probability"),
                "api_home": row.get("api_home_probability"),
                "api_draw": row.get("api_draw_probability"),
                "api_away": row.get("api_away_probability"),
                "home_rating_before": row.get("home_rating_before"),
                "away_rating_before": row.get("away_rating_before"),
                "home_form_signal": row.get("home_form_signal"),
                "away_form_signal": row.get("away_form_signal"),
                "expected_goal_gap": row.get("expected_goal_gap"),
                "total_expected_goals": row.get("total_expected_goals"),
                "rating_gap": row.get("rating_gap"),
                "form_gap": row.get("form_gap"),
                "draw_risk_label": row.get("draw_risk_label"),
                "probability_margin": row.get("probability_margin"),
                "top_vs_draw_margin": row.get("top_vs_draw_margin"),
                "confidence": row.get("confidence"),
                "brier_score": row.get("brier_score"),
                "log_loss": row.get("log_loss"),
            }
        )
    return diagnostic_rows


def sportmonks_enrichment_is_non_leaky(
    kickoff_utc: Any,
    captured_at: Any,
    *,
    available_before_kickoff: Any = None,
    availability: Any = None,
) -> bool:
    if available_before_kickoff is not True:
        return False
    availability_text = str(availability or "").casefold()
    if availability_text in {"post_match_only", "post_match", "final_xg"}:
        return False
    kickoff = parse_fixture_datetime(kickoff_utc)
    captured = parse_fixture_datetime(captured_at)
    if kickoff is None or captured is None:
        return False
    return captured <= kickoff


def _candidate_reason_for_enrichment(
    row: dict[str, Any],
    enrichment: dict[str, Any] | None,
) -> str:
    if not enrichment:
        return "no_fixture_mapping"
    if enrichment.get("mapping_confidence") not in {"exact", "likely"}:
        return "mapping_not_confident"
    if not enrichment.get("xg_pair_available"):
        return "no_sportmonks_xg_pair"
    if not sportmonks_enrichment_is_non_leaky(
        row.get("kickoff_utc"),
        enrichment.get("captured_at"),
        available_before_kickoff=enrichment.get("available_before_kickoff"),
        availability=enrichment.get("availability"),
    ):
        return "post_match_or_late_enrichment"
    return "eligible"


def sportmonks_candidate_probabilities(
    row: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    xg_weight: float = 0.25,
) -> dict[str, float]:
    home_base = float(row.get("home_expected_goals", 0.0) or 0.0)
    away_base = float(row.get("away_expected_goals", 0.0) or 0.0)
    home_xg = float(enrichment.get("home_sportmonks_xg", home_base) or home_base)
    away_xg = float(enrichment.get("away_sportmonks_xg", away_base) or away_base)
    weight = float(np.clip(xg_weight, 0.0, 1.0))
    home_expected = float(np.clip(home_base * (1.0 - weight) + home_xg * weight, 0.25, 3.8))
    away_expected = float(np.clip(away_base * (1.0 - weight) + away_xg * weight, 0.25, 3.8))
    probabilities = outcome_probabilities_from_expected_goals(home_expected, away_expected)
    probabilities = calibrate_draw_probability(
        probabilities,
        home_expected,
        away_expected,
        rating_gap=float(row.get("rating_gap", 0.0) or 0.0),
        form_gap=float(row.get("form_gap", 0.0) or 0.0),
    )
    home, draw, away = probabilities
    return {"home": home, "draw": draw, "away": away}


def sportmonks_candidate_rows(
    rows: list[dict[str, Any]],
    enrichment_by_fixture: dict[int, dict[str, Any]],
    *,
    xg_weight: float = 0.25,
) -> list[dict[str, Any]]:
    candidate_rows = [dict(row) for row in rows]
    evaluated_count = 0
    correct_count = 0
    brier_values: list[float] = []
    log_loss_values: list[float] = []
    for row in candidate_rows:
        fixture_id = int(row.get("fixture_id") or 0)
        enrichment = enrichment_by_fixture.get(fixture_id)
        reason = _candidate_reason_for_enrichment(row, enrichment)
        row["sportmonks_candidate_status"] = "eligible" if reason == "eligible" else "unavailable"
        row["sportmonks_candidate_reason"] = reason
        row["sportmonks_fixture_id"] = (
            enrichment.get("sportmonks_fixture_id") if enrichment else None
        )
        row["sportmonks_mapping_confidence"] = (
            enrichment.get("mapping_confidence") if enrichment else "no_match"
        )
        row["sportmonks_news_count"] = int((enrichment or {}).get("news_count") or 0)
        row["sportmonks_candidate_predicted"] = "-"
        row["sportmonks_candidate_correct"] = None
        row["sportmonks_candidate_brier_score"] = None
        row["sportmonks_candidate_log_loss"] = None
        row["sportmonks_candidate_running_accuracy"] = None
        if reason != "eligible" or enrichment is None:
            continue

        probabilities = sportmonks_candidate_probabilities(
            row,
            enrichment,
            xg_weight=xg_weight,
        )
        predicted = predicted_outcome(probabilities) or "-"
        actual = str(row.get("actual") or "")
        correct = predicted == actual
        evaluated_count += 1
        if correct:
            correct_count += 1
        row_brier = brier_score(probabilities, actual)
        row_log_loss = log_loss(probabilities, actual)
        if row_brier is not None:
            brier_values.append(row_brier)
        if row_log_loss is not None:
            log_loss_values.append(row_log_loss)

        row["sportmonks_candidate_predicted"] = predicted
        row["sportmonks_candidate_correct"] = correct
        row["sportmonks_candidate_running_accuracy"] = correct_count / evaluated_count
        row["sportmonks_candidate_home_probability"] = probabilities["home"]
        row["sportmonks_candidate_draw_probability"] = probabilities["draw"]
        row["sportmonks_candidate_away_probability"] = probabilities["away"]
        row["sportmonks_candidate_brier_score"] = row_brier
        row["sportmonks_candidate_log_loss"] = row_log_loss
        row["sportmonks_candidate_running_brier_score"] = (
            float(np.mean(brier_values)) if brier_values else None
        )
        row["sportmonks_candidate_running_log_loss"] = (
            float(np.mean(log_loss_values)) if log_loss_values else None
        )
    return candidate_rows


def sportmonks_candidate_summary(
    rows: list[dict[str, Any]],
    *,
    min_evaluation_rows: int = 10,
) -> dict[str, Any]:
    mapped_rows = [
        row
        for row in rows
        if row.get("sportmonks_mapping_confidence") in {"exact", "likely", "ambiguous"}
    ]
    eligible_rows = [
        row
        for row in rows
        if row.get("sportmonks_candidate_correct") is not None
    ]
    baseline_brier = _mean(
        [
            float(row["brier_score"])
            for row in eligible_rows
            if row.get("brier_score") is not None
        ]
    )
    candidate_brier = _mean(
        [
            float(row["sportmonks_candidate_brier_score"])
            for row in eligible_rows
            if row.get("sportmonks_candidate_brier_score") is not None
        ]
    )
    baseline_log_loss = _mean(
        [
            float(row["log_loss"])
            for row in eligible_rows
            if row.get("log_loss") is not None
        ]
    )
    candidate_log_loss = _mean(
        [
            float(row["sportmonks_candidate_log_loss"])
            for row in eligible_rows
            if row.get("sportmonks_candidate_log_loss") is not None
        ]
    )
    candidate_correct = sum(1 for row in eligible_rows if row.get("sportmonks_candidate_correct"))
    proves_improvement = (
        len(eligible_rows) >= min_evaluation_rows
        and candidate_brier is not None
        and baseline_brier is not None
        and candidate_log_loss is not None
        and baseline_log_loss is not None
        and candidate_brier < baseline_brier
        and candidate_log_loss < baseline_log_loss
    )
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("sportmonks_candidate_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "completed": len(rows),
        "mapped": len(mapped_rows),
        "eligible": len(eligible_rows),
        "candidate_correct": candidate_correct,
        "candidate_accuracy": (
            candidate_correct / len(eligible_rows) if eligible_rows else None
        ),
        "baseline_brier_score": baseline_brier,
        "candidate_brier_score": candidate_brier,
        "brier_delta": (
            candidate_brier - baseline_brier
            if candidate_brier is not None and baseline_brier is not None
            else None
        ),
        "baseline_log_loss": baseline_log_loss,
        "candidate_log_loss": candidate_log_loss,
        "log_loss_delta": (
            candidate_log_loss - baseline_log_loss
            if candidate_log_loss is not None and baseline_log_loss is not None
            else None
        ),
        "candidate_proves_improvement": proves_improvement,
        "headline_recommendation": (
            "promote_candidate_after_review" if proves_improvement else "keep_current_model"
        ),
        "reason_counts": reason_counts,
    }
