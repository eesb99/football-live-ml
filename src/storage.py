from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, SNAPSHOT_DIR


PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
API_PREDICTION_DIR = PROJECT_ROOT / "data" / "api_predictions"
SPORTMONKS_DIR = PROJECT_ROOT / "data" / "sportmonks"
SPORTMONKS_AUDIT_DIR = SPORTMONKS_DIR / "audits"
SPORTMONKS_FIXTURE_DIR = SPORTMONKS_DIR / "fixtures"
SPORTMONKS_ODDS_DIR = SPORTMONKS_DIR / "odds"
SPORTMONKS_PREDICTION_DIR = SPORTMONKS_DIR / "predictions"
SPORTMONKS_XG_DIR = SPORTMONKS_DIR / "xg"
SPORTMONKS_NEWS_DIR = SPORTMONKS_DIR / "news"
SPORTMONKS_GENERATED_DIRS = [
    SPORTMONKS_AUDIT_DIR,
    SPORTMONKS_FIXTURE_DIR,
    SPORTMONKS_ODDS_DIR,
    SPORTMONKS_PREDICTION_DIR,
    SPORTMONKS_XG_DIR,
    SPORTMONKS_NEWS_DIR,
]
SPORTMONKS_CACHE_PREFIXES = {
    "fixtures": SPORTMONKS_FIXTURE_DIR,
    "fixture_detail": SPORTMONKS_FIXTURE_DIR,
    "odds": SPORTMONKS_ODDS_DIR,
    "xg": SPORTMONKS_XG_DIR,
    "news": SPORTMONKS_NEWS_DIR,
}
API_PREDICTION_CACHE_FIELDS = [
    "fixture_id",
    "captured_at",
    "status",
    "available",
    "endpoint",
    "raw_response_count",
    "home_probability",
    "draw_probability",
    "away_probability",
    "home_display",
    "draw_display",
    "away_display",
    "advice",
    "winner_id",
    "winner_name",
    "winner_comment",
    "win_or_draw",
    "under_over",
    "goals_home",
    "goals_away",
    "last_error",
]


def ensure_snapshot_dir(snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def snapshot_filename(snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
    return snapshot_dir / f"{timestamp}_live_matches.csv"


def prediction_snapshot_filename(snapshot_dir: Path = PREDICTION_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
    return snapshot_dir / f"{timestamp}_predictions.csv"


def save_snapshot(
    rows: list[dict[str, Any]],
    snapshot_dir: Path = SNAPSHOT_DIR,
    allow_empty: bool = False,
) -> Path | None:
    if not rows and not allow_empty:
        return None
    directory = ensure_snapshot_dir(snapshot_dir)
    path = snapshot_filename(directory)
    captured_at = datetime.now().isoformat(timespec="seconds")
    if rows:
        frame = pd.DataFrame(rows)
        frame.insert(0, "snapshot_captured_at", captured_at)
        frame.insert(1, "snapshot_row_count", len(rows))
    else:
        frame = pd.DataFrame(
            [{"snapshot_captured_at": captured_at, "snapshot_row_count": 0}]
        )
    frame.to_csv(path, index=False)
    return path


def save_prediction_snapshot(
    rows: list[dict[str, Any]],
    snapshot_dir: Path = PREDICTION_DIR,
    allow_empty: bool = False,
) -> Path | None:
    if not rows and not allow_empty:
        return None
    directory = ensure_snapshot_dir(snapshot_dir)
    path = prediction_snapshot_filename(directory)
    captured_at = datetime.now().isoformat(timespec="seconds")
    if rows:
        frame = pd.DataFrame(rows)
        frame.insert(0, "snapshot_captured_at", captured_at)
        frame.insert(1, "snapshot_row_count", len(rows))
    else:
        frame = pd.DataFrame(
            [{"snapshot_captured_at": captured_at, "snapshot_row_count": 0}]
        )
    frame.to_csv(path, index=False)
    return path


def list_prediction_snapshots(snapshot_dir: Path = PREDICTION_DIR) -> list[Path]:
    if not snapshot_dir.exists():
        return []
    return sorted(snapshot_dir.glob("*_predictions.csv"), reverse=True)


def ensure_sportmonks_dirs() -> list[Path]:
    for directory in SPORTMONKS_GENERATED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    return SPORTMONKS_GENERATED_DIRS


def sportmonks_audit_filename(audit_dir: Path = SPORTMONKS_AUDIT_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return audit_dir / f"{timestamp}_sportmonks_access_audit.json"


def save_sportmonks_audit(
    audit: dict[str, Any],
    audit_dir: Path = SPORTMONKS_AUDIT_DIR,
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = sportmonks_audit_filename(audit_dir)
    if path.exists():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
        path = audit_dir / f"{timestamp}_sportmonks_access_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return path


def list_sportmonks_audits(audit_dir: Path = SPORTMONKS_AUDIT_DIR) -> list[Path]:
    if not audit_dir.exists():
        return []
    return sorted(audit_dir.glob("*_sportmonks_access_audit.json"), reverse=True)


def load_latest_sportmonks_audit(
    audit_dir: Path = SPORTMONKS_AUDIT_DIR,
) -> dict[str, Any] | None:
    paths = list_sportmonks_audits(audit_dir)
    if not paths:
        return None
    try:
        audit = json.loads(paths[0].read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(audit, dict):
        return None
    audit["audit_file"] = str(paths[0])
    return audit


def sportmonks_safe_cache_key(value: Any) -> str:
    text = str(value or "latest").strip()
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")
    return text or "latest"


def sportmonks_cache_filename(
    prefix: str,
    cache_key: Any = "latest",
    cache_dir: Path | None = None,
) -> Path:
    directory = cache_dir or SPORTMONKS_CACHE_PREFIXES[prefix]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_key = sportmonks_safe_cache_key(cache_key)
    return directory / f"{timestamp}_{prefix}_{safe_key}.json"


def save_sportmonks_cache(
    payload: dict[str, Any],
    *,
    prefix: str,
    cache_key: Any = "latest",
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    from src.sportmonks_client import sanitize_sportmonks_payload

    directory = cache_dir or SPORTMONKS_CACHE_PREFIXES[prefix]
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "provider": "sportmonks",
        "cache_prefix": prefix,
        "cache_key": sportmonks_safe_cache_key(cache_key),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "payload": sanitize_sportmonks_payload(payload, token),
    }
    path = sportmonks_cache_filename(prefix, cache_key, directory)
    if path.exists():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M_%S_%f")
        path = directory / f"{timestamp}_{prefix}_{sportmonks_safe_cache_key(cache_key)}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def list_sportmonks_cache_files(
    *,
    prefix: str,
    cache_key: Any | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    directory = cache_dir or SPORTMONKS_CACHE_PREFIXES[prefix]
    if not directory.exists():
        return []
    key_glob = "*" if cache_key is None else sportmonks_safe_cache_key(cache_key)
    return sorted(directory.glob(f"*_{prefix}_{key_glob}.json"), reverse=True)


def load_latest_sportmonks_cache(
    *,
    prefix: str,
    cache_key: Any | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    paths = list_sportmonks_cache_files(
        prefix=prefix,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    if not paths:
        return None
    try:
        record = json.loads(paths[0].read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    record["cache_file"] = str(paths[0])
    return record


def save_sportmonks_fixtures_cache(
    payload: dict[str, Any],
    season_id: int,
    *,
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sportmonks_cache(
        payload,
        prefix="fixtures",
        cache_key=season_id,
        cache_dir=cache_dir,
        token=token,
        metadata=metadata,
    )


def load_latest_sportmonks_fixtures_cache(
    season_id: int,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    return load_latest_sportmonks_cache(
        prefix="fixtures",
        cache_key=season_id,
        cache_dir=cache_dir,
    )


def save_sportmonks_fixture_detail_cache(
    payload: dict[str, Any],
    fixture_id: int,
    *,
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sportmonks_cache(
        payload,
        prefix="fixture_detail",
        cache_key=fixture_id,
        cache_dir=cache_dir,
        token=token,
        metadata=metadata,
    )


def load_latest_sportmonks_fixture_detail_cache(
    fixture_id: int,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    return load_latest_sportmonks_cache(
        prefix="fixture_detail",
        cache_key=fixture_id,
        cache_dir=cache_dir,
    )


def save_sportmonks_odds_cache(
    payload: dict[str, Any],
    fixture_id: int,
    *,
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sportmonks_cache(
        payload,
        prefix="odds",
        cache_key=fixture_id,
        cache_dir=cache_dir,
        token=token,
        metadata=metadata,
    )


def load_latest_sportmonks_odds_cache(
    fixture_id: int,
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    return load_latest_sportmonks_cache(
        prefix="odds",
        cache_key=fixture_id,
        cache_dir=cache_dir,
    )


def save_sportmonks_xg_cache(
    payload: dict[str, Any],
    *,
    cache_key: Any = "latest",
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sportmonks_cache(
        payload,
        prefix="xg",
        cache_key=cache_key,
        cache_dir=cache_dir,
        token=token,
        metadata=metadata,
    )


def load_latest_sportmonks_xg_cache(
    *,
    cache_key: Any | None = "latest",
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    return load_latest_sportmonks_cache(
        prefix="xg",
        cache_key=cache_key,
        cache_dir=cache_dir,
    )


def save_sportmonks_news_cache(
    payload: dict[str, Any],
    *,
    cache_key: Any = "latest",
    cache_dir: Path | None = None,
    token: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_sportmonks_cache(
        payload,
        prefix="news",
        cache_key=cache_key,
        cache_dir=cache_dir,
        token=token,
        metadata=metadata,
    )


def load_latest_sportmonks_news_cache(
    *,
    cache_key: Any | None = "latest",
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    return load_latest_sportmonks_cache(
        prefix="news",
        cache_key=cache_key,
        cache_dir=cache_dir,
    )


def api_prediction_cache_path(
    fixture_id: int,
    cache_dir: Path = API_PREDICTION_DIR,
) -> Path:
    return cache_dir / f"{int(fixture_id)}.json"


def sanitize_api_prediction_error(value: Any) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)(x-apisports-key|api[_-]?football[_-]?key|api[_-]?key)(['\"]?\s*[:=]\s*)['\"]?[^,'\"\s}]+",
        r"\1\2[redacted]",
        text,
    )
    return text[:500]


def normalized_api_prediction_cache_record(
    prediction: dict[str, Any],
    fixture_id: int,
) -> dict[str, Any]:
    captured_at = str(
        prediction.get("captured_at") or datetime.now().isoformat(timespec="seconds")
    )
    source = dict(prediction)
    source["fixture_id"] = int(fixture_id)
    source["captured_at"] = captured_at
    source["last_error"] = sanitize_api_prediction_error(source.get("last_error", ""))
    return {field: source.get(field) for field in API_PREDICTION_CACHE_FIELDS}


def save_api_prediction_cache(
    prediction: dict[str, Any],
    fixture_id: int,
    cache_dir: Path = API_PREDICTION_DIR,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    record = normalized_api_prediction_cache_record(prediction, fixture_id)
    path = api_prediction_cache_path(fixture_id, cache_dir)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def load_api_prediction_cache(
    fixture_id: int,
    cache_dir: Path = API_PREDICTION_DIR,
) -> dict[str, Any] | None:
    path = api_prediction_cache_path(fixture_id, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return normalized_api_prediction_cache_record(data, fixture_id)
