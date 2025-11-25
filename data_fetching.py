"""
data_fetching.py
=================

This module centralizes all of the functions used to collect and prepare
NBA data for the prop prediction dashboard. It wraps three different
sources of information (Balldontlie, ESPN and NBA.com) behind a set of
friendly functions. Whenever possible the functions return pandas
DataFrames to simplify downstream processing.

The Balldontlie API provides official data going back to 1946. It
contains endpoints for players, games, box scores and season
averages. According to the official documentation, the API requires
an API key and supports query parameters such as player IDs, game IDs
and date ranges. Season averages can be
requested by specifying a category (e.g. `general`) and a type (e.g.
`base`).

ESPN’s website exposes a set of hidden JSON endpoints that return
structured data about games, box scores, rosters and schedules. A
blog post describing the hidden API notes that these endpoints are
public, live under `site.web.api.espn.com/apis` and do not require
authentication. For example, the summary
endpoint for a specific event returns team and player statistics for
that game.

The `nba_api` Python package is an unofficial wrapper around the
statistics exposed on NBA.com.  It simplifies access to endpoints such
as `playergamelog`, `leaguegamefinder` and `playercareerstats`.  The
official documentation for the package explains that no
authentication is required and provides example usage.

Because this repository executes in a sandboxed environment without
internet access, these functions are written defensively. They
attempt to call the remote APIs when possible, but will return
empty DataFrames or raise meaningful exceptions if network calls
fail. When deploying the app on Streamlit Cloud or another
environment with external network access, these functions will work
as intended.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    # Import optional dependencies; if unavailable they will be
    # installed via requirements.txt and available at runtime.
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from nba_api.stats.endpoints import (
        playergamelog,
        leaguegamefinder,
        playercareerstats,
    )
    from nba_api.stats.static import players as nba_players
except ImportError:
    # In offline mode the nba_api might not be available. Placeholders
    # allow type checking but will raise at runtime if called.
    playergamelog = None  # type: ignore
    leaguegamefinder = None  # type: ignore
    playercareerstats = None  # type: ignore
    nba_players = None  # type: ignore


def _check_requests_available() -> None:
    """Internal helper to ensure the requests library is available."""
    if requests is None:
        raise RuntimeError(
            "The `requests` library is not installed. Please ensure your "
            "environment includes requests as specified in requirements.txt."
        )


def get_active_players_balldontlie(api_key: str) -> pd.DataFrame:
    """Return a list of active NBA players from the Balldontlie API."""
    _check_requests_available()
    base_url = "https://api.balldontlie.io/v1/active_players"
    headers = {"Authorization": api_key}
    players: List[Dict[str, Any]] = []
    next_cursor: Optional[int] = None
    while True:
        params: Dict[str, Any] = {}
        if next_cursor is not None:
            params["cursor"] = next_cursor
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            if resp.status_code == 401:
                raise RuntimeError(
                    "Unauthorized. Please verify that your Balldontlie API key is valid."
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch active players: {exc}. If running offline, "
                "this function cannot retrieve data."
            )
        players.extend(data.get("data", []))
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
    return pd.DataFrame(players)


def search_player_balldontlie(name: str, api_key: str) -> pd.DataFrame:
    """Search for a player by name using the Balldontlie API."""
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/players"
    headers = {"Authorization": api_key}
    params = {"search": name, "per_page": 25}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to search players: {exc}. Ensure network access is available."
        )
    return pd.DataFrame(data.get("data", []))


def get_season_averages_balldontlie(
    player_ids: List[int],
    season: int,
    category: str = "general",
    stat_type: str = "base",
    season_type: str = "regular",
    api_key: str = "",
) -> pd.DataFrame:
    """Retrieve season averages for one or more players from Balldontlie."""
    _check_requests_available()
    url = f"https://api.balldontlie.io/v1/season_averages/{category}"
    headers = {"Authorization": api_key}
    params: Dict[str, Any] = {
        "season": season,
        "season_type": season_type,
        "type": stat_type,
    }
    for pid in player_ids:
        params.setdefault("player_ids[]", []).append(pid)
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch season averages: {exc}. Check your API key and network."
        )
    records: List[Dict[str, Any]] = []
    for item in data.get("data", []):
        player_info = item.get("player", {})
        stats = item.get("stats", {})
        record = {**player_info, **stats}
        record["season"] = item.get("season")
        record["season_type"] = item.get("season_type")
        records.append(record)
    return pd.DataFrame(records)


def get_next_games_balldontlie(
    team_id: int,
    start_date: datetime.date,
    end_date: datetime.date,
    api_key: str,
) -> pd.DataFrame:
    """Return the list of games for a team within a date range."""
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/games"
    headers = {"Authorization": api_key}
    params = {
        "team_ids[]": team_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "per_page": 100,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch games: {exc}. Check network connectivity and API key."
        )
    return pd.DataFrame(data.get("data", []))


def get_player_game_logs_nba(
    player_id: int,
    season: str,
    num_games: Optional[int] = None,
    measure_type: str = "Base",
) -> pd.DataFrame:
    """Retrieve game logs for a given player using the NBA.com stats API."""
    if playergamelog is None:
        return pd.DataFrame()
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, measure_type_detailed=measure_type
        )
        df = logs.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    if num_games is not None and num_games > 0:
        df = df.head(num_games)
    return df


def get_league_games_for_player_nba(player_id: int) -> pd.DataFrame:
    """Return all games for a player using the league game finder."""
    if leaguegamefinder is None:
        return pd.DataFrame()
    try:
        finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        df = finder.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    return df


def get_player_career_stats_nba(player_id: int) -> pd.DataFrame:
    """Return career statistics for a player using the NBA.com stats API."""
    if playercareerstats is None:
        return pd.DataFrame()
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = career.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    return df


def get_player_list_nba() -> pd.DataFrame:
    """Return a DataFrame of all NBA players from the nba_api static module."""
    if nba_players is None:
        return pd.DataFrame()
    try:
        player_list = nba_players.get_players()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(player_list)


def get_espn_game_summary(event_id: str) -> Dict[str, Any]:
    """Fetch a game summary from ESPN's hidden API."""
    _check_requests_available()
    url = (
        "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    )
    params = {
        "region": "us",
        "lang": "en",
        "contentorigin": "espn",
        "event": event_id,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def get_espn_scoreboard(date: datetime.date) -> Dict[str, Any]:
    """Return the ESPN scoreboard for a specific date."""
    _check_requests_available()
    url = (
        "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    )
    params = {"dates": date.strftime("%Y%m%d")}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}
