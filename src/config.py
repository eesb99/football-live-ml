from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_URL = "https://v3.football.api-sports.io"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"


class MissingApiKeyError(RuntimeError):
    """Raised when API_FOOTBALL_KEY is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = BASE_URL
    snapshot_dir: Path = SNAPSHOT_DIR
    request_timeout_seconds: int = 20


def load_settings(require_api_key: bool = True) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if require_api_key and not api_key:
        raise MissingApiKeyError(
            "Missing API_FOOTBALL_KEY. Add it to your environment or local .env file."
        )
    return Settings(api_key=api_key)
