from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo


MYT_ZONE = ZoneInfo("Asia/Kuala_Lumpur")


def parse_api_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_myt(value: str | None) -> datetime | None:
    parsed = parse_api_datetime(value)
    if parsed is None:
        return None
    return parsed.astimezone(MYT_ZONE)


def fixture_myt_fields(fixture: dict[str, Any]) -> dict[str, str]:
    fixture_date = fixture.get("fixture", {}).get("date")
    myt = to_myt(fixture_date)
    if myt is None:
        return {
            "myt_datetime": "TBD",
            "myt_date": "TBD",
            "myt_date_label": "TBD",
            "myt_time": "TBD",
            "timezone": "MYT",
        }
    return {
        "myt_datetime": myt.strftime("%Y-%m-%d %H:%M"),
        "myt_date": myt.strftime("%Y-%m-%d"),
        "myt_date_label": myt.strftime("%a, %d %b %Y"),
        "myt_time": myt.strftime("%H:%M"),
        "timezone": "MYT",
    }
