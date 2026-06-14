from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT


DEFAULT_RATING = 1500.0
HOME_ADVANTAGE_ELO = 60.0
DEFAULT_K_FACTOR = 24.0
RATINGS_DIR = PROJECT_ROOT / "data" / "ratings"
RATINGS_PATH = RATINGS_DIR / "team_ratings.csv"


@dataclass(frozen=True)
class TeamRating:
    team_id: int
    team_name: str
    rating: float = DEFAULT_RATING
    matches_played: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "rating": self.rating,
            "matches_played": self.matches_played,
        }


RatingMap = dict[int, TeamRating]


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def actual_scores(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def fixture_result(fixture: dict[str, Any]) -> tuple[int, int] | None:
    goals = fixture.get("goals", {})
    home = goals.get("home")
    away = goals.get("away")
    if home is None or away is None:
        return None
    return int(home), int(away)


def is_completed_fixture(fixture: dict[str, Any]) -> bool:
    status = fixture.get("fixture", {}).get("status", {}).get("short")
    return status in {"FT", "AET", "PEN"} and fixture_result(fixture) is not None


def team_identity(fixture: dict[str, Any], side: str) -> tuple[int, str]:
    team = fixture.get("teams", {}).get(side, {})
    return int(team.get("id") or 0), team.get("name") or side.title()


def get_rating(ratings: RatingMap, team_id: int, team_name: str = "") -> TeamRating:
    if team_id in ratings:
        return ratings[team_id]
    return TeamRating(team_id=team_id, team_name=team_name or str(team_id))


def update_ratings_for_fixture(
    ratings: RatingMap,
    fixture: dict[str, Any],
    k_factor: float = DEFAULT_K_FACTOR,
) -> RatingMap:
    if not is_completed_fixture(fixture):
        return ratings

    home_id, home_name = team_identity(fixture, "home")
    away_id, away_name = team_identity(fixture, "away")
    if not home_id or not away_id:
        return ratings

    home_rating = get_rating(ratings, home_id, home_name)
    away_rating = get_rating(ratings, away_id, away_name)
    home_goals, away_goals = fixture_result(fixture) or (0, 0)
    home_actual, away_actual = actual_scores(home_goals, away_goals)

    home_expected = expected_score(
        home_rating.rating + HOME_ADVANTAGE_ELO,
        away_rating.rating,
    )
    away_expected = 1.0 - home_expected

    updated = dict(ratings)
    updated[home_id] = TeamRating(
        team_id=home_id,
        team_name=home_name,
        rating=home_rating.rating + k_factor * (home_actual - home_expected),
        matches_played=home_rating.matches_played + 1,
    )
    updated[away_id] = TeamRating(
        team_id=away_id,
        team_name=away_name,
        rating=away_rating.rating + k_factor * (away_actual - away_expected),
        matches_played=away_rating.matches_played + 1,
    )
    return updated


def update_ratings_from_results(
    fixtures: list[dict[str, Any]],
    initial_ratings: RatingMap | None = None,
    k_factor: float = DEFAULT_K_FACTOR,
) -> RatingMap:
    ratings = dict(initial_ratings or {})
    dated_fixtures = sorted(
        fixtures,
        key=lambda fixture: fixture.get("fixture", {}).get("date") or "",
    )
    for fixture in dated_fixtures:
        ratings = update_ratings_for_fixture(ratings, fixture, k_factor=k_factor)
    return ratings


def load_ratings(path: Path = RATINGS_PATH) -> RatingMap:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    ratings: RatingMap = {}
    for row in frame.to_dict("records"):
        team_id = int(row["team_id"])
        ratings[team_id] = TeamRating(
            team_id=team_id,
            team_name=str(row.get("team_name") or team_id),
            rating=float(row.get("rating", DEFAULT_RATING)),
            matches_played=int(row.get("matches_played", 0)),
        )
    return ratings


def save_ratings(ratings: RatingMap, path: Path = RATINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for rating in sorted(ratings.values(), key=lambda item: item.team_name):
        row = rating.as_dict()
        row["snapshot_captured_at"] = captured_at
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_rating_snapshot(ratings: RatingMap, snapshot_dir: Path = RATINGS_DIR) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
    path = snapshot_dir / f"{timestamp}_team_ratings.csv"
    save_ratings(ratings, path=path)
    return path
