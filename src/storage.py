from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, SNAPSHOT_DIR


PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"


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
