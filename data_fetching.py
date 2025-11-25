"""
data_fetching.py
=================

Centralized data fetcher for Balldontlie, NBA.com (nba_api), and ESPN.
Balldontlie API key is now hard-coded for Streamlit Cloud compatibility.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------
# HARDCODED BALLDONTLIE API KEY (your key)
# ---------------------------------------------------------
BALLDONTLIE_API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"


# Optional external modules
try:
    import requests
except ImportError:
    requests = None

try:
    from nba_api.stats.endpoints import (
        playergamelog,
        leaguegamefinder,
        playercareerstats,
    )
    from nba_api.stats.static import players as nba_players
except ImportError:
    playergamelog = None
    leaguegamefinder = None
    playercareerstats = None
    nba_players = None


def _check_requests_available() -> None:
    if requests is None:
        raise RuntimeError(
            "The `requests` library is not installed. Please include it in requirements.txt."
        )


# ---------------------------------------------------------
# BALDONTLIE ENDPOINTS (USING HARDCODED API KEY)
# ---------------------------------------------------------

def get_active_players_balldontlie() -> pd.DataFrame:
    """Return list of active players from Balldontlie."""
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/active_players"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    players = []
    next_cursor = None

    while True:
        params = {}
        if next_cursor:
            params["cursor"] = next_cursor

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Could not fetch active players: {e}")

        players.extend(data.get("data", []))
        next_cursor = data.get("meta", {}).get("next_cursor")

        if not next_cursor:
            break

    return pd.DataFrame(players)


def search_player_balldontlie(name: str) -> pd.DataFrame:
    """Search for a player via Balldontlie."""
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/players"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    try:
        resp = requests.get(url, headers=headers, params={"search": name}, timeout=15)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get("data", []))
    except Exception as e:
        raise RuntimeError(f"Could not search players: {e}")


def get_season_averages_balldontlie(
    player_ids: List[int],
    season: int,
    category: str = "general",
    stat_type: str = "base",
    season_type: str = "regular",
) -> pd.DataFrame:

    _check_requests_available()
    url = f"https://api.balldontlie.io/v1/season_averages/{category}"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    params = {
        "season": season,
        "season_type": season_type,
        "type": stat_type,
    }

    for pid in player_ids:
        params.setdefault("player_ids[]", []).append(pid)

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch season averages: {e}")

    records = []
    for item in data.get("data", []):
        combined = {**item.get("player", {}), **item.get("stats", {})}
        combined["season"] = item.get("season")
        combined["season_type"] = item.get("season_type")
        records.append(combined)

    return pd.DataFrame(records)


def get_next_games_balldontlie(team_id: int, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Get games for a team between two dates."""
    _check_requests_available()

    url = "https://api.balldontlie.io/v1/games"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    params = {
        "team_ids[]": team_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "per_page": 100,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get("data", []))
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------
# NBA API ENDPOINTS
# ---------------------------------------------------------

def get_player_game_logs_nba(player_id: int, season: str, num_games: Optional[int] = None):
    if playergamelog is None:
        return pd.DataFrame()

    try:
        logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = logs.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

    if num_games:
        df = df.head(num_games)

    return df


def get_league_games_for_player_nba(player_id: int):
    if leaguegamefinder is None:
        return pd.DataFrame()
    try:
        finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        return finder.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


def get_player_career_stats_nba(player_id: int):
    if playercareerstats is None:
        return pd.DataFrame()
    try:
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        return stats.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


def get_player_list_nba() -> pd.DataFrame:
    if nba_players is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(nba_players.get_players())
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------
# ESPN HIDDEN API
# ---------------------------------------------------------

def get_espn_game_summary(event_id: str) -> Dict[str, Any]:
    _check_requests_available()
    url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    try:
        resp = requests.get(url, params={"event": event_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def get_espn_scoreboard(date: datetime.date) -> Dict[str, Any]:
    _check_requests_available()
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        resp = requests.get(url, params={"dates": date.strftime("%Y%m%d")}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}
