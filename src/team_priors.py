from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT
from src.ratings import DEFAULT_RATING, TeamRating
from src.sportmonks_mapping import parse_fixture_datetime


TEAM_PRIORS_DIR = PROJECT_ROOT / "data" / "team_priors"
TEAM_PRIORS_PATH = TEAM_PRIORS_DIR / "team_priors.csv"
PRIOR_SCHEMA_COLUMNS = [
    "team_id",
    "team_name",
    "strength_rating",
    "source",
    "source_category",
    "as_of",
    "available_before_kickoff",
    "confederation",
    "confederation_strength",
    "host_adjustment_elo",
    "rank",
]
BLOCKED_SOURCE_CATEGORIES = {
    "api_prediction",
    "betting_market",
    "fixture_result",
    "in_tournament_result",
    "match_result",
    "odds",
    "post_match",
    "provider_prediction",
    "same_tournament_result",
    "same_tournament_results",
}


@dataclass(frozen=True)
class TeamPrior:
    team_id: int
    team_name: str
    strength_rating: float
    source: str
    source_category: str
    as_of: str
    available_before_kickoff: bool = True
    confederation: str = ""
    confederation_strength: float = 0.0
    host_adjustment_elo: float = 0.0
    rank: int | None = None

    @property
    def effective_rating(self) -> float:
        return float(
            np.clip(
                self.strength_rating
                + self.confederation_strength
                + self.host_adjustment_elo,
                1200.0,
                1900.0,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "strength_rating": self.strength_rating,
            "source": self.source,
            "source_category": self.source_category,
            "as_of": self.as_of,
            "available_before_kickoff": self.available_before_kickoff,
            "confederation": self.confederation,
            "confederation_strength": self.confederation_strength,
            "host_adjustment_elo": self.host_adjustment_elo,
            "rank": self.rank,
            "effective_rating": self.effective_rating,
        }


TeamPriorMap = dict[int, TeamPrior]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def team_prior_from_record(record: dict[str, Any]) -> TeamPrior:
    team_id = int(record.get("team_id") or 0)
    if team_id <= 0:
        raise ValueError("team prior requires a positive team_id")
    strength_rating = _as_float(record.get("strength_rating"), DEFAULT_RATING)
    return TeamPrior(
        team_id=team_id,
        team_name=str(record.get("team_name") or team_id),
        strength_rating=strength_rating,
        source=str(record.get("source") or "unknown"),
        source_category=str(record.get("source_category") or "pre_tournament_rating"),
        as_of=str(record.get("as_of") or ""),
        available_before_kickoff=_as_bool(record.get("available_before_kickoff", True)),
        confederation=str(record.get("confederation") or ""),
        confederation_strength=_as_float(record.get("confederation_strength"), 0.0),
        host_adjustment_elo=_as_float(record.get("host_adjustment_elo"), 0.0),
        rank=_as_optional_int(record.get("rank")),
    )


def load_team_priors(path: Path = TEAM_PRIORS_PATH) -> TeamPriorMap:
    if not path.exists() or not path.read_text().strip():
        return {}
    records: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            records = list(payload.get("teams") or [])
        elif isinstance(payload, list):
            records = payload
        else:
            return {}
    else:
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))

    priors: TeamPriorMap = {}
    for record in records:
        prior = team_prior_from_record(record)
        priors[prior.team_id] = prior
    return priors


def team_prior_is_non_leaky(prior: TeamPrior, kickoff_utc: Any) -> bool:
    if not prior.available_before_kickoff:
        return False
    if prior.source_category.strip().casefold() in BLOCKED_SOURCE_CATEGORIES:
        return False
    kickoff = parse_fixture_datetime(kickoff_utc)
    as_of = parse_fixture_datetime(prior.as_of)
    if kickoff is None or as_of is None:
        return False
    return as_of <= kickoff


def eligible_team_prior(
    priors: TeamPriorMap | None,
    team_id: int,
    kickoff_utc: Any,
) -> TeamPrior | None:
    prior = (priors or {}).get(team_id)
    if prior is None:
        return None
    if not team_prior_is_non_leaky(prior, kickoff_utc):
        return None
    return prior


def prior_adjusted_rating(
    rating: TeamRating,
    prior: TeamPrior | None,
    *,
    max_delta: float = 120.0,
) -> tuple[float, float]:
    if prior is None:
        return rating.rating, 0.0
    depth_weight = max(0.15, 0.45 - min(rating.matches_played, 6) * 0.05)
    delta = float(
        np.clip(
            (prior.effective_rating - DEFAULT_RATING) * depth_weight,
            -max_delta,
            max_delta,
        )
    )
    return rating.rating + delta, delta


def prior_schema_rows() -> list[dict[str, str]]:
    return [
        {
            "column": column,
            "purpose": purpose,
        }
        for column, purpose in [
            ("team_id", "API-Football team id"),
            ("team_name", "Readable team name"),
            ("strength_rating", "Pre-match Elo-equivalent team strength"),
            ("source", "Human-readable source name"),
            ("source_category", "Must not be result, odds, or provider prediction data"),
            ("as_of", "Timestamp when the prior was knowable"),
            ("available_before_kickoff", "true only when source was available pre-match"),
            ("confederation", "Optional region label"),
            ("confederation_strength", "Small Elo-equivalent regional adjustment"),
            ("host_adjustment_elo", "Small pre-known host-context adjustment"),
            ("rank", "Optional external rank for display and audit"),
        ]
    ]
