from __future__ import annotations

from typing import Any


PERCENT_FIELDS = {"Ball Possession"}
STAT_ALIASES = {
    "expected_goals": ("Expected Goals", "expected_goals", "xG", "XG"),
    "shots_on_goal": ("Shots on Goal", "Shots on Target"),
    "total_shots": ("Total Shots",),
    "blocked_shots": ("Blocked Shots",),
    "shots_inside_box": ("Shots insidebox", "Shots inside box"),
    "shots_outside_box": ("Shots outsidebox", "Shots outside box"),
    "fouls": ("Fouls",),
    "corners": ("Corner Kicks",),
    "offsides": ("Offsides",),
    "possession": ("Ball Possession",),
    "yellow_cards": ("Yellow Cards",),
    "goalkeeper_saves": ("Goalkeeper Saves",),
    "passes": ("Total passes", "Total Passes"),
    "passes_accurate": ("Passes accurate", "Accurate Passes"),
}


def parse_percentage(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def fixture_id(fixture: dict[str, Any]) -> int:
    return int(fixture.get("fixture", {}).get("id", 0) or 0)


def match_minute(fixture: dict[str, Any]) -> int:
    elapsed = fixture.get("fixture", {}).get("status", {}).get("elapsed")
    return int(elapsed or 0)


def score_line(fixture: dict[str, Any]) -> tuple[int, int]:
    goals = fixture.get("goals", {})
    return int(goals.get("home") or 0), int(goals.get("away") or 0)


def team_names(fixture: dict[str, Any]) -> tuple[str, str]:
    teams = fixture.get("teams", {})
    home = teams.get("home", {}).get("name", "Home")
    away = teams.get("away", {}).get("name", "Away")
    return home, away


def status_text(fixture: dict[str, Any]) -> str:
    status = fixture.get("fixture", {}).get("status", {})
    return status.get("long") or status.get("short") or "Unknown"


def status_short(fixture: dict[str, Any]) -> str:
    return fixture.get("fixture", {}).get("status", {}).get("short") or "UNK"


def fixture_date(fixture: dict[str, Any]) -> str:
    return fixture.get("fixture", {}).get("date") or ""


def league_info(fixture: dict[str, Any]) -> dict[str, Any]:
    league = fixture.get("league", {})
    return {
        "league_id": int(league.get("id") or 0),
        "league_name": league.get("name") or "Unknown",
        "league_country": league.get("country") or "",
        "league_season": int(league.get("season") or 0),
        "league_round": league.get("round") or "",
    }


def is_live_status(fixture: dict[str, Any]) -> bool:
    live_statuses = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
    return status_short(fixture) in live_statuses


def _team_side(team_id: int, home_team_id: int, away_team_id: int) -> str | None:
    if team_id == home_team_id:
        return "home"
    if team_id == away_team_id:
        return "away"
    return None


def event_summary(
    events: list[dict[str, Any]],
    home_team_id: int,
    away_team_id: int,
    current_minute: int,
) -> dict[str, float]:
    summary = {
        "home_red_cards": 0.0,
        "away_red_cards": 0.0,
        "home_yellow_cards": 0.0,
        "away_yellow_cards": 0.0,
        "home_goal_events": 0.0,
        "away_goal_events": 0.0,
        "home_penalty_goals": 0.0,
        "away_penalty_goals": 0.0,
        "home_recent_events": 0.0,
        "away_recent_events": 0.0,
        "home_recent_goals": 0.0,
        "away_recent_goals": 0.0,
    }

    for event in events:
        team_id = int(event.get("team", {}).get("id") or 0)
        side = _team_side(team_id, home_team_id, away_team_id)
        if side is None:
            continue

        event_type = str(event.get("type") or "")
        detail = str(event.get("detail") or "").lower()
        comments = str(event.get("comments") or "").lower()
        elapsed = int(event.get("time", {}).get("elapsed") or 0)
        is_recent = current_minute > 0 and elapsed >= max(current_minute - 10, 0)

        if is_recent:
            summary[f"{side}_recent_events"] += 1.0

        if event_type == "Card":
            if "red" in detail or "red" in comments:
                summary[f"{side}_red_cards"] += 1.0
            elif "yellow" in detail or "yellow" in comments:
                summary[f"{side}_yellow_cards"] += 1.0

        if event_type == "Goal" and "cancelled" not in comments and "var" not in detail:
            summary[f"{side}_goal_events"] += 1.0
            if "penalty" in detail:
                summary[f"{side}_penalty_goals"] += 1.0
            if is_recent:
                summary[f"{side}_recent_goals"] += 1.0

    return summary


def statistics_by_team(statistics: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    parsed: dict[int, dict[str, float]] = {}
    for team_stats in statistics:
        team_id = int(team_stats.get("team", {}).get("id") or 0)
        parsed[team_id] = {}
        for stat in team_stats.get("statistics", []):
            stat_type = stat.get("type")
            raw_value = stat.get("value")
            if stat_type in PERCENT_FIELDS:
                parsed[team_id][stat_type] = parse_percentage(raw_value)
            else:
                parsed[team_id][stat_type] = parse_number(raw_value)
    return parsed


def _stat_value(team_stats: dict[str, float], stat_key: str) -> float:
    for stat_name in STAT_ALIASES[stat_key]:
        if stat_name in team_stats:
            return float(team_stats[stat_name])
    return 0.0


def _safe_share(value: float, opponent_value: float, default: float = 0.5) -> float:
    total = value + opponent_value
    if total <= 0:
        return default
    return value / total


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_pass_accuracy(accurate: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return _bounded((accurate / total) * 100.0, 0.0, 100.0)


def proxy_expected_goals(
    shots: float,
    shots_on_target: float,
    shots_inside_box: float,
    corners: float,
    possession: float,
    recent_events: float,
    recent_goals: float,
) -> float:
    value = (
        shots * 0.045
        + shots_on_target * 0.115
        + shots_inside_box * 0.055
        + corners * 0.025
        + max(possession - 50.0, 0.0) * 0.004
        + recent_events * 0.025
        + recent_goals * 0.18
    )
    return round(_bounded(value, 0.0, 4.5), 3)


def build_match_features(
    fixture: dict[str, Any],
    statistics: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    statistics = statistics or []
    events = events or []

    teams = fixture.get("teams", {})
    home_team = teams.get("home", {})
    away_team = teams.get("away", {})
    home_team_id = int(home_team.get("id") or 0)
    away_team_id = int(away_team.get("id") or 0)

    home_name, away_name = team_names(fixture)
    home_goals, away_goals = score_line(fixture)
    minute = match_minute(fixture)
    event_features = event_summary(events, home_team_id, away_team_id, minute)
    stats = statistics_by_team(statistics)
    home_stats = stats.get(home_team_id, {})
    away_stats = stats.get(away_team_id, {})

    home_shots = _stat_value(home_stats, "total_shots")
    away_shots = _stat_value(away_stats, "total_shots")
    home_xg = _stat_value(home_stats, "expected_goals")
    away_xg = _stat_value(away_stats, "expected_goals")
    home_shots_on_target = _stat_value(home_stats, "shots_on_goal")
    away_shots_on_target = _stat_value(away_stats, "shots_on_goal")
    home_corners = _stat_value(home_stats, "corners")
    away_corners = _stat_value(away_stats, "corners")
    home_possession = _stat_value(home_stats, "possession")
    away_possession = _stat_value(away_stats, "possession")
    home_yellow_cards = max(
        event_features["home_yellow_cards"],
        _stat_value(home_stats, "yellow_cards"),
    )
    away_yellow_cards = max(
        event_features["away_yellow_cards"],
        _stat_value(away_stats, "yellow_cards"),
    )
    home_passes = _stat_value(home_stats, "passes")
    away_passes = _stat_value(away_stats, "passes")
    home_passes_accurate = _stat_value(home_stats, "passes_accurate")
    away_passes_accurate = _stat_value(away_stats, "passes_accurate")
    home_saves = _stat_value(home_stats, "goalkeeper_saves")
    away_saves = _stat_value(away_stats, "goalkeeper_saves")
    home_fouls = _stat_value(home_stats, "fouls")
    away_fouls = _stat_value(away_stats, "fouls")
    home_shots_inside_box = _stat_value(home_stats, "shots_inside_box")
    away_shots_inside_box = _stat_value(away_stats, "shots_inside_box")
    home_blocked_shots = _stat_value(home_stats, "blocked_shots")
    away_blocked_shots = _stat_value(away_stats, "blocked_shots")
    home_proxy_xg = proxy_expected_goals(
        home_shots,
        home_shots_on_target,
        home_shots_inside_box,
        home_corners,
        home_possession,
        event_features["home_recent_events"],
        event_features["home_recent_goals"],
    )
    away_proxy_xg = proxy_expected_goals(
        away_shots,
        away_shots_on_target,
        away_shots_inside_box,
        away_corners,
        away_possession,
        event_features["away_recent_events"],
        event_features["away_recent_goals"],
    )
    has_api_xg = home_xg > 0.0 or away_xg > 0.0
    home_effective_xg = home_xg if has_api_xg else home_proxy_xg
    away_effective_xg = away_xg if has_api_xg else away_proxy_xg

    home_pressure_score = (
        home_shots * 0.12
        + home_shots_on_target * 0.34
        + home_corners * 0.12
        + home_possession * 0.012
        + event_features["home_recent_events"] * 0.08
        + home_effective_xg * 0.22
    )
    away_pressure_score = (
        away_shots * 0.12
        + away_shots_on_target * 0.34
        + away_corners * 0.12
        + away_possession * 0.012
        + event_features["away_recent_events"] * 0.08
        + away_effective_xg * 0.22
    )

    league = league_info(fixture)
    features = {
        "fixture_id": fixture_id(fixture),
        "fixture_date": fixture_date(fixture),
        "minute": minute,
        "elapsed_fraction": _bounded(minute / 95.0, 0.0, 1.1),
        "remaining_fraction": _bounded((95.0 - minute) / 95.0, 0.0, 1.0),
        "status": status_text(fixture),
        "status_short": status_short(fixture),
        "is_live": is_live_status(fixture),
        **league,
        "home_team": home_name,
        "away_team": away_name,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "score_difference": home_goals - away_goals,
        **event_features,
        "home_red_cards": event_features["home_red_cards"],
        "away_red_cards": event_features["away_red_cards"],
        "red_card_difference": (
            event_features["home_red_cards"] - event_features["away_red_cards"]
        ),
        "home_yellow_cards": home_yellow_cards,
        "away_yellow_cards": away_yellow_cards,
        "yellow_card_difference": home_yellow_cards - away_yellow_cards,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "xg_difference": home_xg - away_xg,
        "home_proxy_xg": home_proxy_xg,
        "away_proxy_xg": away_proxy_xg,
        "proxy_xg_difference": home_proxy_xg - away_proxy_xg,
        "home_effective_xg": home_effective_xg,
        "away_effective_xg": away_effective_xg,
        "effective_xg_difference": home_effective_xg - away_effective_xg,
        "xg_source": "api_football_real_xg" if has_api_xg else "proxy_xg",
        "real_xg_available": has_api_xg,
        "proxy_xg_available": True,
        "shot_difference": home_shots - away_shots,
        "home_shot_share": _safe_share(home_shots, away_shots),
        "away_shot_share": _safe_share(away_shots, home_shots),
        "home_shots_on_target": home_shots_on_target,
        "away_shots_on_target": away_shots_on_target,
        "shots_on_target_difference": home_shots_on_target - away_shots_on_target,
        "home_shots_on_target_share": _safe_share(
            home_shots_on_target,
            away_shots_on_target,
        ),
        "away_shots_on_target_share": _safe_share(
            away_shots_on_target,
            home_shots_on_target,
        ),
        "home_shots_inside_box": home_shots_inside_box,
        "away_shots_inside_box": away_shots_inside_box,
        "home_blocked_shots": home_blocked_shots,
        "away_blocked_shots": away_blocked_shots,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "possession_difference": home_possession - away_possession,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "corner_difference": home_corners - away_corners,
        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "home_offsides": _stat_value(home_stats, "offsides"),
        "away_offsides": _stat_value(away_stats, "offsides"),
        "home_goalkeeper_saves": home_saves,
        "away_goalkeeper_saves": away_saves,
        "home_pass_accuracy": _safe_pass_accuracy(home_passes_accurate, home_passes),
        "away_pass_accuracy": _safe_pass_accuracy(away_passes_accurate, away_passes),
        "home_pressure_score": home_pressure_score,
        "away_pressure_score": away_pressure_score,
        "pressure_difference": home_pressure_score - away_pressure_score,
        "home_pressure_share": _safe_share(home_pressure_score, away_pressure_score),
        "away_pressure_share": _safe_share(away_pressure_score, home_pressure_score),
    }
    features["data_completeness_score"] = data_completeness_score(features)
    return features


def data_completeness_score(features: dict[str, Any]) -> float:
    signals = [
        features.get("home_shots", 0.0) + features.get("away_shots", 0.0),
        features.get("home_shots_on_target", 0.0)
        + features.get("away_shots_on_target", 0.0),
        features.get("home_possession", 0.0) + features.get("away_possession", 0.0),
        features.get("home_corners", 0.0) + features.get("away_corners", 0.0),
    ]
    available = sum(1 for value in signals if float(value or 0.0) > 0.0)
    return available / len(signals)


def build_live_match_table(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fixture in fixtures:
        home_goals, away_goals = score_line(fixture)
        home_team, away_team = team_names(fixture)
        rows.append(
            {
                "fixture_id": fixture_id(fixture),
                "fixture_date": fixture_date(fixture),
                "minute": match_minute(fixture),
                "status": status_text(fixture),
                "status_short": status_short(fixture),
                **league_info(fixture),
                "home_team": home_team,
                "away_team": away_team,
                "score": f"{home_goals}-{away_goals}",
            }
        )
    return rows
