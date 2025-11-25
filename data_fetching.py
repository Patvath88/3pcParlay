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
and date ranges【389649769524007†L670-L788】. Season averages can be
requested by specifying a category (e.g. `general`) and a type (e.g.
`base`)【389649769524007†L792-L826】.

ESPN’s website exposes a set of hidden JSON endpoints that return
structured data about games, box scores, rosters and schedules. A
blog post describing the hidden API notes that these endpoints are
public, live under `site.web.api.espn.com/apis` and do not require
authentication【918690647919691†L27-L45】. For example, the summary
endpoint for a specific event returns team and player statistics for
that game【918690647919691†L27-L74】.

The `nba_api` Python package is an unofficial wrapper around the
statistics exposed on NBA.com.  It simplifies access to endpoints such
as `playergamelog`, `leaguegamefinder` and `playercareerstats`.  The
official documentation for the package explains that no
authentication is required and provides example usage with
`playercareerstats`【466081084939129†L64-L83】.  A tutorial on
retrieving NBA statistics shows how to use `playercareerstats` and
`playergamelog` to pull game logs and career data for a player
【480705325969819†L37-L100】.

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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    # Import optional dependencies; if unavailable they will be
    # installed via requirements.txt and available at runtime.
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from nba_api.stats.endpoints import playergamelog, leaguegamefinder, playercareerstats
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
            "The `requests` library is not installed. Please ensure your"
            " environment includes requests as specified in requirements.txt."
        )


def get_active_players_balldontlie(api_key: str) -> pd.DataFrame:
    """Return a list of active NBA players from the Balldontlie API.

    Balldontlie exposes an `active players` endpoint (noted in their
    documentation) that returns currently active players. To use
    Balldontlie you must include your API key in the `Authorization`
    header of each request. The function paginates through the
    endpoint until all players have been collected.

    Parameters
    ----------
    api_key : str
        Your Balldontlie API key. According to the docs, the API key
        should be sent in the Authorization header.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per player. Columns include `id`,
        `first_name`, `last_name`, `team_id` and other meta-data.
    """
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
            # In offline mode, propagate a helpful error
            raise RuntimeError(
                f"Failed to fetch active players: {exc}. If running offline,"
                " this function cannot retrieve data."
            )
        players.extend(data.get("data", []))
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
    return pd.DataFrame(players)


def search_player_balldontlie(name: str, api_key: str) -> pd.DataFrame:
    """Search for a player by name using the Balldontlie API.

    This helper wraps the `/players` endpoint and accepts a partial
    name string. It returns up to 25 matching players per request.

    Parameters
    ----------
    name : str
        The partial or full player name to search for.
    api_key : str
        Your Balldontlie API key.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of matching players with their IDs and meta-data.
    """
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
    """Retrieve season averages for one or more players.

    Balldontlie's season averages endpoint requires a `category` and
    `type` parameter to specify the bucket of metrics (e.g. general
    base stats). You can specify multiple player IDs. The API returns
    a list of objects containing the player's info and a `stats` dict
    with various statistics【389649769524007†L792-L826】.

    Parameters
    ----------
    player_ids : List[int]
        A list of player IDs for whom to fetch averages.
    season : int
        The season year (e.g. 2024 for the 2024–25 season).
    category : str
        The category of statistics to request. See the docs for
        valid values (general, clutch, defense, shooting, etc.).
    stat_type : str
        The type within the category (base, advanced, misc, etc.).
    season_type : str
        "regular" or "playoffs". Only used for categories that
        support this distinction.
    api_key : str
        Your Balldontlie API key.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a player and
        columns correspond to the stats returned for the chosen
        category/type combination.
    """
    _check_requests_available()
    url = f"https://api.balldontlie.io/v1/season_averages/{category}"
    headers = {"Authorization": api_key}
    params: Dict[str, Any] = {
        "season": season,
        "season_type": season_type,
        "type": stat_type,
    }
    for pid in player_ids:
        # The API expects repeated query parameters like player_ids[]=ID
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
        # Add metadata
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
    """Return the list of games for a team within a date range.

    This helper queries the Balldontlie `/games` endpoint with
    `start_date` and `end_date` filters. The endpoint returns both
    home and away games. According to the docs the `start_date` and
    `end_date` parameters accept `YYYY-MM-DD` strings【389649769524007†L785-L788】.

    Parameters
    ----------
    team_id : int
        The team ID whose games you want to fetch.
    start_date : datetime.date
        The earliest date (inclusive) to look for games.
    end_date : datetime.date
        The latest date (inclusive) to look for games.
    api_key : str
        Your Balldontlie API key.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing games scheduled within the date range.
    """
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
    """Retrieve game logs for a given player using the NBA.com stats API.

    This function uses the `playergamelog` endpoint from the `nba_api`
    package. The endpoint returns a DataFrame with one row per game and
    includes statistics such as points, rebounds, assists, etc.

    The `season` argument should be formatted as `YYYY-YY` (e.g.
    `'2024-25'`).  If `num_games` is provided, the resulting
    DataFrame is truncated to the most recent `num_games` games.

    Parameters
    ----------
    player_id : int
        The NBA.com player ID.
    season : str
        The season string (e.g. '2024-25').
    num_games : Optional[int]
        Number of most recent games to return. If None, all games are
        returned.
    measure_type : str
        The measure type (e.g. 'Base', 'Advanced', etc.). See NBA.com
        stats documentation for supported values.

    Returns
    -------
    pandas.DataFrame
        The game logs as a DataFrame. If the nba_api library is not
        available or the call fails, an empty DataFrame is returned.
    """
    if playergamelog is None:
        # nba_api is not installed or cannot be imported.
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
    """Return all games for a player using the league game finder.

    The `leaguegamefinder` endpoint returns historical games for a
    player. It can be filtered by season, opponent and more. See the
    nba_api documentation for details【480705325969819†L74-L100】.

    Parameters
    ----------
    player_id : int
        NBA player ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame of games for the player, empty if unavailable.
    """
    if leaguegamefinder is None:
        return pd.DataFrame()
    try:
        finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        df = finder.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    return df


def get_player_career_stats_nba(player_id: int) -> pd.DataFrame:
    """Return career statistics for a player using the NBA.com stats API.

    This wraps the `playercareerstats` endpoint. The function returns
    the regular season totals by default【781714960155839†L87-L127】. The
    DataFrame can be further processed to compute season averages.

    Parameters
    ----------
    player_id : int
        NBA player ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame of career stats or empty if unavailable.
    """
    if playercareerstats is None:
        return pd.DataFrame()
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = career.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    return df


def get_player_list_nba() -> pd.DataFrame:
    """Return a DataFrame of all NBA players from the nba_api static module.

    According to the `nba_api` tutorial, you can retrieve player
    information and IDs by calling `players.get_players()`【480705325969819†L59-L70】.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns like `id`, `full_name`, `first_name`,
        `last_name`, etc. Returns empty DataFrame if the library
        cannot be imported.
    """
    if nba_players is None:
        return pd.DataFrame()
    try:
        player_list = nba_players.get_players()
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(player_list)


def get_espn_game_summary(event_id: str) -> Dict[str, Any]:
    """Fetch a game summary from ESPN's hidden API.

    The summary endpoint returns detailed JSON for a specific game,
    including team and player statistics【918690647919691†L61-L82】. The
    endpoint does not require authentication but network access is
    required.

    Parameters
    ----------
    event_id : str
        The ESPN event identifier for the game (e.g. 401382337).

    Returns
    -------
    dict
        The parsed JSON response. An empty dictionary is returned if
        the request fails.
    """
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
    """Return the ESPN scoreboard for a specific date.

    This function wraps the ESPN scoreboard endpoint. The
    `date` parameter is formatted as YYYYMMDD. The endpoint returns
    schedules and game identifiers which can be passed into
    `get_espn_game_summary`. The gist documenting hidden endpoints
    lists the scoreboard URL for NBA games【845929871356886†L169-L176】.

    Parameters
    ----------
    date : datetime.date
        The date of interest.

    Returns
    -------
    dict
        Parsed JSON of the scoreboard. Returns empty dict on failure.
    """
    _check_requests_available()
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {"dates": date.strftime("%Y%m%d")}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}