from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_URL = "https://v3.football.api-sports.io"
SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"


class MissingApiKeyError(RuntimeError):
    """Raised when API_FOOTBALL_KEY is missing."""


class MissingSportMonksTokenError(RuntimeError):
    """Raised when SPORTMONKS_API_TOKEN is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = BASE_URL
    snapshot_dir: Path = SNAPSHOT_DIR
    request_timeout_seconds: int = 20
    sportmonks_api_token: str = ""
    sportmonks_base_url: str = SPORTMONKS_BASE_URL


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def load_settings(
    require_api_key: bool = True,
    require_sportmonks_token: bool = False,
    env_file: Path | None = PROJECT_ROOT / ".env",
) -> Settings:
    if env_file is not None:
        load_dotenv(env_file)
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    sportmonks_api_token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if require_api_key and not api_key:
        raise MissingApiKeyError(
            "Missing API_FOOTBALL_KEY. Add it to your environment or local .env file."
        )
    if require_sportmonks_token and not sportmonks_api_token:
        raise MissingSportMonksTokenError(
            "Missing SPORTMONKS_API_TOKEN. Add it to your environment or local .env file."
        )
    return Settings(
        api_key=api_key,
        base_url=os.getenv("API_FOOTBALL_BASE_URL", BASE_URL).strip() or BASE_URL,
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 20),
        sportmonks_api_token=sportmonks_api_token,
        sportmonks_base_url=(
            os.getenv("SPORTMONKS_BASE_URL", SPORTMONKS_BASE_URL).strip()
            or SPORTMONKS_BASE_URL
        ),
    )
