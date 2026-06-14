from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PaidDataSnapshot:
    odds_available: bool = False
    real_xg_available: bool = False
    injuries_available: bool = False
    news_available: bool = False
    odds_source: str = "not configured"
    real_xg_source: str = "not configured"
    injuries_source: str = "not configured"
    news_source: str = "not configured"
    home_real_xg: float | None = None
    away_real_xg: float | None = None
    home_odds_implied_probability: float | None = None
    draw_odds_implied_probability: float | None = None
    away_odds_implied_probability: float | None = None
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    home_news_impact: float = 0.0
    away_news_impact: float = 0.0

    @property
    def availability_summary(self) -> str:
        statuses = [
            f"odds={'available' if self.odds_available else 'missing'}",
            f"real_xg={'available' if self.real_xg_available else 'missing'}",
            f"injuries={'available' if self.injuries_available else 'missing'}",
            f"news={'available' if self.news_available else 'missing'}",
        ]
        return ", ".join(statuses)

    def as_prediction_fields(self) -> dict[str, float | str | bool | None]:
        return {
            "odds_available": self.odds_available,
            "real_xg_available": self.real_xg_available,
            "injuries_available": self.injuries_available,
            "news_available": self.news_available,
            "odds_source": self.odds_source,
            "real_xg_source": self.real_xg_source,
            "injuries_source": self.injuries_source,
            "news_source": self.news_source,
            "home_real_xg": self.home_real_xg,
            "away_real_xg": self.away_real_xg,
            "home_odds_implied_probability": self.home_odds_implied_probability,
            "draw_odds_implied_probability": self.draw_odds_implied_probability,
            "away_odds_implied_probability": self.away_odds_implied_probability,
            "home_injury_impact": self.home_injury_impact,
            "away_injury_impact": self.away_injury_impact,
            "home_news_impact": self.home_news_impact,
            "away_news_impact": self.away_news_impact,
            "paid_data_availability": self.availability_summary,
        }


class PaidDataAdapter(Protocol):
    def get_snapshot(self, fixture_id: int) -> PaidDataSnapshot:
        """Return optional paid-data signals for a fixture."""


class NullPaidDataAdapter:
    def get_snapshot(self, fixture_id: int) -> PaidDataSnapshot:
        return PaidDataSnapshot()


def default_paid_data_snapshot() -> PaidDataSnapshot:
    return NullPaidDataAdapter().get_snapshot(0)
