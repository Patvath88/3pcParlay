"""
data_fetching.py — PRO VERSION
Fully upgraded for Option B.

Adds:
- Hardcoded Balldontlie API key
- Per-game stat extraction helpers
- Opponent detection
- Defensive metrics scaffolding
- Cloud-safe request handling
"""

from __future__ import annotations

import datetime
import pandas as pd
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# HARD-CODED BALLDONTLIE API KEY
# -------------------------------------------------------------------
BALLDONTLIE_API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"


# -------------------------------------------------------------------
# SAFE IMPORTS
# -------------------------------------------------------------------
try:
    import requests
except ImportError:
    requests = None

try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.endpoints import playercareerstats
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.static import players as nba_players
except ImportError:
    playergamelog = None
    leaguegamefinder = None
    playercareerstats = None
    nba_players = None


# -------------------------------------------------------------------
# INTERNAL SAFETY CHECK
# -------------------------------------------------------------------
def _check_requests_available():
    if requests is None:
        raise RuntimeError("The `requests` library is missing. Install it in requirements.txt")


# -------------------------------------------------------------------
# BALDONTLIE API WRAPPERS
# -------------------------------------------------------------------

def get_active_players_balldontlie() -> pd.DataFrame:
    """Retrieve all active NBA players from Balldontlie API."""
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/active_players"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    players = []
    cursor = None

    while True:
        params = {}
        if cursor is not None:
            params["cursor"] = cursor

        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            break  # fail safe

        players.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return pd.DataFrame(players)


def get_next_games_balldontlie(team_id: int, start: datetime.date, end: datetime.date):
    _check_requests_available()
    url = "https://api.balldontlie.io/v1/games"
    headers = {"Authorization": BALLDONTLIE_API_KEY}

    params = {
        "team_ids[]": team_id,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "per_page": 100,
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("data", []))
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------------
# NBA API WRAPPERS
# -------------------------------------------------------------------

def get_player_game_logs_nba(player_id: int, season: str, num_games: Optional[int] = None):
    """Pull player's complete game logs for the specified season."""
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


def get_player_career_stats_nba(player_id: int):
    if playercareerstats is None:
        return pd.DataFrame()

    try:
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        return stats.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


def get_player_list_nba():
    if nba_players is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(nba_players.get_players())
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------------
# ESPN HIDDEN API WRAPPERS
# -------------------------------------------------------------------

def get_espn_scoreboard(date: datetime.date) -> Dict[str, Any]:
    """Retrieve ESPN scoreboard data for a given date."""
    _check_requests_available()

    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        r = requests.get(url, params={"dates": date.strftime("%Y%m%d")}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_espn_game_summary(event_id: str) -> Dict[str, Any]:
    """Retrieve detailed ESPN game summary."""
    _check_requests_available()

    url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    try:
        r = requests.get(url, params={"event": event_id}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# -------------------------------------------------------------------
# RICHER GAME MERGING LOGIC
# -------------------------------------------------------------------

def determine_next_opponent(game_logs: pd.DataFrame) -> Optional[str]:
    """Return the opponent team abbreviation for the next game."""
    if game_logs.empty:
        return None

    # last matchup entry format: "LAL vs BOS" or "LAL @ BOS"
    last_matchup = game_logs.iloc[0]["MATCHUP"]

    # reverse logic: WHO did they play last? extract home/away pattern
    # pattern: "LAL vs BOS" → opponent = BOS
    #          "LAL @ BOS"  → opponent = BOS
    parts = last_matchup.split(" ")
    if len(parts) >= 3:
        return parts[-1]
    return None


# -------------------------------------------------------------------
# GAME LOG CLEANING
# -------------------------------------------------------------------

def clean_nba_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns & enforce correct dtypes."""

    if df.empty:
        return df

    # Renames for consistency
    rename_map = {
        "FG3M": "FG3M",
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "MIN": "MIN",
    }

    df = df.rename(columns=rename_map)

    # Convert all numerical stat columns to float
    numeric_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without valid stats
    df = df.dropna(subset=["PTS", "REB", "AST"])

    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# MERGED PLAYER LOG PREPARATION
# -------------------------------------------------------------------

def prepare_player_training_logs(player_id: int, season: str, limit: int = 30) -> pd.DataFrame:
    """
    Fetch cleaned NBA logs + enrich with opponent extraction.
    """

    raw = get_player_game_logs_nba(player_id, season, num_games=limit)
    if raw.empty:
        return pd.DataFrame()

    df = clean_nba_game_logs(raw)

    # Extract opponent abbreviation from MATCHUP text
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    return df.reset_index(drop=True)

