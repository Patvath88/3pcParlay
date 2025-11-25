# app.py
# Streamlit NBA Player Prop Dashboard & Research (balldontlie)
# Run: streamlit run app.py
# Secrets: create .streamlit/secrets.toml with balldontlie_api_key="YOUR_KEY"

from __future__ import annotations

import os
import time
import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(
    page_title="NBA Player Prop Dashboard · balldontlie",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Constants ----------
API_BASE = "https://api.balldontlie.io/v1"
FAST_TTL = 10      # seconds: live calls
SLOW_TTL = 1800    # seconds: reference data

# ---------- API Key ----------
def resolve_api_key() -> str:
    # Prefer Streamlit secrets; allow env var fallback; final fallback: provided key (user supplied)
    key = None
    try:
        key = st.secrets.get("balldontlie_api_key")
    except Exception:
        pass
    key = key or os.getenv("BALLDONTLIE_API_KEY")
    key = key or "7f4db7a9-c34e-478d-a799-fef77b9d1f78"  # NOTE: move to secrets for deployment
    return str(key).strip()

API_KEY = resolve_api_key()

def build_headers() -> Dict[str, str]:
    # Some deployments expect Bearer; others accept raw token. We default to Bearer.
    token = API_KEY
    if token.lower().startswith("bearer "):
        bearer = token
    else:
        bearer = f"Bearer {token}"
    return {
        "Authorization": bearer,
        "Accept": "application/json",
        "User-Agent": "streamlit-nba-prop-dashboard/1.0",
    }

# ---------- Caching wrappers ----------
def _cache_key(path: str, params: Optional[Dict[str, Any]]) -> str:
    params = params or {}
    items = sorted([(k, str(v)) for k, v in params.items()])
    return f"{path}?{'&'.join([f'{k}={v}' for k, v in items])}"

@st.cache_data(ttl=FAST_TTL, show_spinner=False)
def api_get_fast(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _api_get(path, params)

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def api_get_slow(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _api_get(path, params)

def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = build_headers()
    params = params or {}
    # Basic retry (why: absorb transient 429/5xx during streams)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429 and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise e  # surfaced by caller
    return {"data": [], "meta": {}}

def fetch_all_pages(path: str, params: Optional[Dict[str, Any]] = None, fast: bool = False) -> List[Dict[str, Any]]:
    params = dict(params or {})
    params.setdefault("per_page", 100)
    page = 1
    out: List[Dict[str, Any]] = []
    while True:
        params["page"] = page
        try:
            js = api_get_fast(path, params) if fast else api_get_slow(path, params)
        except Exception as e:
            st.warning(f"API error on {path}: {e}")
            break
        data = js.get("data", [])
        out.extend(data)
        meta = js.get("meta") or {}
        next_page = meta.get("next_page")
        if not next_page:
            break
        page = next_page
        if page > 50:  # hard ceiling guard
            break
    return out

# ---------- Domain helpers ----------
@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def list_teams() -> pd.DataFrame:
    teams = fetch_all_pages("teams", fast=False)
    df = pd.DataFrame(teams)
    return df

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def search_players(query: str) -> pd.DataFrame:
    if not query:
        return pd.DataFrame()
    players = fetch_all_pages("players", params={"search": query, "per_page": 100}, fast=False)
    if not players:
        return pd.DataFrame()
    # Flatten team info
    rows = []
    for p in players:
        team = p.get("team") or {}
        rows.append({
            "id": p.get("id"),
            "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "position": p.get("position") or "",
            "team_id": team.get("id"),
            "team_abbr": team.get("abbreviation"),
            "height": p.get("height"),
            "weight": p.get("weight"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def get_player(player_id: int) -> Dict[str, Any]:
    try:
        js = api_get_slow(f"players/{player_id}")
    except Exception as e:
        st.error(f"Failed to load player {player_id}: {e}")
        return {}
    return js or {}

@st.cache_data(ttl=FAST_TTL, show_spinner=False)
def games_by_date(date_iso: str) -> pd.DataFrame:
    # API uses 'dates[]' param; returns games for exact date
    try:
        js = api_get_fast("games", params={"dates[]": date_iso, "per_page": 100})
    except Exception as e:
        st.warning(f"Failed to load games for {date_iso}: {e}")
        return pd.DataFrame()
    games = js.get("data", [])
    if not games:
        return pd.DataFrame()
    rows = []
    for g in games:
        rows.append({
            "id": g["id"],
            "date": g.get("date"),
            "status": g.get("status"),
            "period": g.get("period"),
            "time": g.get("time"),
            "season": g.get("season"),
            "home_id": (g.get("home_team") or {}).get("id"),
            "home_abbr": (g.get("home_team") or {}).get("abbreviation"),
            "visitor_id": (g.get("visitor_team") or {}).get("id"),
            "visitor_abbr": (g.get("visitor_team") or {}).get("abbreviation"),
            "home_score": g.get("home_team_score"),
            "visitor_score": g.get("visitor_team_score"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=FAST_TTL, show_spinner=False)
def game_stats(game_id: int) -> pd.DataFrame:
    stats = fetch_all_pages("stats", params={"game_ids[]": game_id, "per_page": 100}, fast=True)
    if not stats:
        return pd.DataFrame()
    rows = []
    for s in stats:
        p = s.get("player") or {}
        t = s.get("team") or {}
        g = s.get("game") or {}
        rows.append({
            "player_id": p.get("id"),
            "player_name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "team_id": t.get("id"),
            "team_abbr": t.get("abbreviation"),
            "game_id": g.get("id"),
            "min": s.get("min"),
            "pts": s.get("pts") or 0,
            "reb": s.get("reb") or 0,
            "ast": s.get("ast") or 0,
            "fg3m": s.get("fg3m") or 0,
            "stl": s.get("stl") or 0,
            "blk": s.get("blk") or 0,
            "tov": s.get("turnover") or s.get("turnovers") or 0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pr"] = df["pts"] + df["reb"]
        df["pa"] = df["pts"] + df["ast"]
        df["ra"] = df["reb"] + df["ast"]
        df["pra"] = df["pts"] + df["reb"] + df["ast"]
        df["min"] = df["min"].fillna("0:00")
    return df

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def player_game_logs(player_id: int, seasons: Optional[List[int]] = None, last_n: Optional[int] = None) -> pd.DataFrame:
    params = {"player_ids[]": player_id, "per_page": 100, "postseason": "false"}
    if seasons:
        for s in seasons:
            params.setdefault("seasons[]", [])
            params["seasons[]"].append(s)
    stats = fetch_all_pages("stats", params=params, fast=False)
    if not stats:
        return pd.DataFrame()
    rows = []
    for s in stats:
        g = s.get("game") or {}
        v = s.get("team") or {}
        home = (g.get("home_team_id") == v.get("id"))
        opp_id = g.get("home_team_id") if not home else g.get("visitor_team_id")
        rows.append({
            "game_id": g.get("id"),
            "date": g.get("date"),
            "season": g.get("season"),
            "is_home": home,
            "opp_team_id": opp_id,
            "min": s.get("min"),
            "pts": s.get("pts") or 0,
            "reb": s.get("reb") or 0,
            "ast": s.get("ast") or 0,
            "fg3m": s.get("fg3m") or 0,
            "stl": s.get("stl") or 0,
            "blk": s.get("blk") or 0,
            "tov": s.get("turnover") or s.get("turnovers") or 0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    if last_n:
        df = df.head(int(last_n))
    return df

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def season_averages(player_id: int, season: int) -> Dict[str, Any]:
    try:
        js = api_get_slow("season_averages", params={"season": season, "player_ids[]": player_id})
        data = (js or {}).get("data") or []
        return data[0] if data else {}
    except Exception:
        return {}

def compute_career_highs_lows(player_id: int, current_season: int) -> pd.DataFrame:
    # Iterate seasons backwards until 3 empty in a row (why: limit API load)
    highs: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    lows: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    tracked = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pr", "pa", "ra", "pra", "min"]
    empty_streak = 0
    for season in range(current_season, current_season - 30, -1):
        logs = player_game_logs(player_id, seasons=[season], last_n=None)
        if logs.empty:
            empty_streak += 1
            if empty_streak >= 3:
                break
            continue
        empty_streak = 0
        # Minutes to numeric for high/low; parse "mm:ss"
        mm = logs["min"].fillna("0:00").astype(str).str.split(":")
        logs = logs.assign(_min_num=mm.apply(lambda x: int(x[0]) + int(x[1]) / 60 if len(x) == 2 else 0.0))
        logs["pr"] = logs["pr"].astype(float)
        logs["pa"] = logs["pa"].astype(float)
        logs["ra"] = logs["ra"].astype(float)
        logs["pra"] = logs["pra"].astype(float)
        metric_map = {
            "pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m",
            "stl": "stl", "blk": "blk", "tov": "tov",
            "pr": "pr", "pa": "pa", "ra": "ra", "pra": "pra", "min": "_min_num"
        }
        for label, col in metric_map.items():
            # High
            idx_hi = logs[col].astype(float).idxmax()
            val_hi = float(logs.loc[idx_hi, col])
            rec_hi = logs.loc[idx_hi].to_dict()
            if (label not in highs) or (val_hi > highs[label][0]):
                highs[label] = (val_hi, rec_hi)
            # Low
            idx_lo = logs[col].astype(float).idxmin()
            val_lo = float(logs.loc[idx_lo, col])
            rec_lo = logs.loc[idx_lo].to_dict()
            if (label not in lows) or (val_lo < lows[label][0]):
                lows[label] = (val_lo, rec_lo)
    # Build DataFrame
    rows = []
    for label in tracked:
        hi_val, hi_row = highs.get(label, (float("nan"), {}))
        lo_val, lo_row = lows.get(label, (float("nan"), {}))
        rows.append({
            "stat": label,
            "career_high": round(hi_val, 2) if isinstance(hi_val, (int, float)) else None,
            "career_high_game_id": hi_row.get("game_id"),
            "career_high_date": hi_row.get("date"),
            "career_low": round(lo_val, 2) if isinstance(lo_val, (int, float)) else None,
            "career_low_game_id": lo_row.get("game_id"),
            "career_low_date": lo_row.get("date"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=FAST_TTL, show_spinner=False)
def upcoming_games_for_team(team_id: int, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    # Query by date window (API supports start_date/end_date)
    try:
        js = api_get_fast("games", params={
            "team_ids[]": team_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "per_page": 100
        })
    except Exception as e:
        st.warning(f"Failed to load upcoming games: {e}")
        return pd.DataFrame()
    games = js.get("data", [])
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "id": g["id"],
        "date": g.get("date"),
        "status": g.get("status"),
        "home_id": (g.get("home_team") or {}).get("id"),
        "visitor_id": (g.get("visitor_team") or {}).get("id"),
        "home_abbr": (g.get("home_team") or {}).get("abbreviation"),
        "visitor_abbr": (g.get("visitor_team") or {}).get("abbreviation"),
    } for g in games])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

@st.cache_data(ttl=SLOW_TTL, show_spinner=False)
def games_between_teams(team_a: int, team_b: int, before_date: dt.date, limit: int = 20) -> pd.DataFrame:
    # Pull a larger window back (1.5 seasons) and filter
    start = (before_date - dt.timedelta(days=500)).isoformat()
    end = before_date.isoformat()
    js = api_get_slow("games", params={
        "team_ids[]": [team_a, team_b],
        "start_date": start,
        "end_date": end,
        "per_page": 100
    })
    games = (js or {}).get("data", [])
    rows = []
    for g in games:
        home = (g.get("home_team") or {}).get("id")
        visitor = (g.get("visitor_team") or {}).get("id")
        if {home, visitor} == {team_a, team_b}:
            rows.append({
                "id": g["id"],
                "date": g.get("date"),
                "home_id": home,
                "visitor_id": visitor
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < pd.to_datetime(before_date)].sort_values("date", ascending=False).head(limit)
    return df.reset_index(drop=True)

def parse_min_to_float(min_str: str) -> float:
    if not isinstance(min_str, str) or ":" not in min_str:
        return 0.0
    mm, ss = min_str.split(":")
    try:
        return int(mm) + int(ss) / 60.0
    except Exception:
        return 0.0

def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for c in ["pts","reb","ast"]:
        if c not in df.columns: df[c] = 0
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    return df

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Settings")
    auto = st.toggle("Auto-refresh live data (15s)", value=True)
    refresh_interval = 15
    if auto:
        st.experimental_rerun  # hint for IDEs; real trigger below
    date_default = dt.date.today()
    pick_date = st.date_input("Game date", value=date_default)
    st.caption("Tip: Store API key in `.streamlit/secrets.toml` as `balldontlie_api_key`.")

# Auto-refresh
if "tick" not in st.session_state:
    st.session_state["tick"] = 0
if auto:
    st_autorefresh_count = st.experimental_memo.clear if False else st.experimental_rerun  # placeholder to please linters
    st.experimental_set_query_params(_ts=int(time.time() // refresh_interval))

# ---------- Title ----------
st.title("🏀 NBA Player Prop Dashboard")
st.caption("Live tracker + research using balldontlie. Mobile-friendly, cached, and fast.")

# ---------- Tabs ----------
tab_live, tab_research = st.tabs(["📡 Live Tracker", "🔎 Research"])

# ---------- Live Tracker ----------
with tab_live:
    date_iso = pick_date.isoformat()
    games_df = games_by_date(date_iso)
    if games_df.empty:
        st.info(f"No games on {date_iso}.")
    else:
        # Game picker
        left, right = st.columns([2, 3])
        with left:
            game_label = (games_df["visitor_abbr"] + " @ " + games_df["home_abbr"] + " · " + games_df["status"]).tolist()
            game_map = {lbl: gid for lbl, gid in zip(game_label, games_df["id"])}
            sel_lbl = st.selectbox("Select game", game_label, index=0)
            sel_game_id = game_map[sel_lbl]
        sel_game = games_df[games_df["id"] == sel_game_id].iloc[0]
        with right:
            h = f"{sel_game['visitor_abbr']} @ {sel_game['home_abbr']}"
            st.markdown(f"### {h}")
            st.metric("Score", f"{int(sel_game['visitor_score'])} - {int(sel_game['home_score'])}")
            st.caption(f"Status: {sel_game['status']} · Period: {sel_game['period']} · Time: {sel_game['time'] or '-'}")

        # Fetch current game stats
        live_df = game_stats(int(sel_game_id))
        known_players = []
        if not live_df.empty:
            known_players = sorted(live_df[["player_id", "player_name", "team_abbr"]].drop_duplicates().itertuples(index=False),
                                   key=lambda r: r.player_name)

        # Player selection
        st.markdown("#### Players to track")
        col_find, col_known = st.columns(2)
        with col_find:
            q = st.text_input("Search players (name)", value="", placeholder="e.g., Stephen Curry")
            found_df = search_players(q) if q else pd.DataFrame()
            choices = []
            if not found_df.empty:
                choices = [
                    f"{row.name} — {row.team_abbr or 'FA'} (#{row.id})"
                    for _, row in found_df.iterrows()
                ]
            selected_labels = st.multiselect("Add players", options=choices, default=[])
        with col_known:
            if known_players:
                quick = st.multiselect(
                    "Quick-pick (players with current stat rows)",
                    options=[f"{r.player_name} — {r.team_abbr} (#{r.player_id})" for r in known_players],
                    default=[]
                )
            else:
                st.write("No live box score yet.")
                quick = []

        def parse_selected(labels: List[str]) -> List[int]:
            out: List[int] = []
            for lab in labels:
                if "#"+"" in lab:
                    try:
                        pid = int(lab.split("#")[-1].strip(")"))
                        out.append(pid)
                    except Exception:
                        pass
            return sorted(set(out))

        player_ids = parse_selected(selected_labels + quick)

        # Prop lines inputs
        stat_labels = ["pts","reb","ast","fg3m","pr","pa","ra","pra","stl","blk","tov","min"]
        defaults = {k: 0.0 for k in stat_labels}
        st.divider()
        if not player_ids:
            st.info("Select players to start tracking.")
        else:
            # Build per-player panels
            for pid in player_ids:
                pinfo = get_player(pid) or {}
                pname = f"{(pinfo.get('first_name') or '')} {(pinfo.get('last_name') or '')}".strip() or f"Player {pid}"
                pteam = ((pinfo.get("team") or {}).get("abbreviation")) or "—"
                with st.container(border=True):
                    st.subheader(f"{pname} ({pteam})")

                    # Existing live row for this game
                    prow = pd.Series(dtype="float64")
                    if not live_df.empty:
                        m = live_df[live_df["player_id"] == pid]
                        if not m.empty:
                            prow = m.iloc[0]
                    col1, col2 = st.columns([2, 3])

                    with col1:
                        st.caption("Prop lines")
                        # Use session_state to remember lines per player
                        key_prefix = f"props_{pid}_"
                        for k in stat_labels:
                            st.session_state.setdefault(key_prefix + k, 0.0)
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            st.session_state[key_prefix + "pts"] = st.number_input("PTS", min_value=0.0, step=0.5, key=key_prefix + "pts")
                            st.session_state[key_prefix + "reb"] = st.number_input("REB", min_value=0.0, step=0.5, key=key_prefix + "reb")
                            st.session_state[key_prefix + "ast"] = st.number_input("AST", min_value=0.0, step=0.5, key=key_prefix + "ast")
                            st.session_state[key_prefix + "fg3m"] = st.number_input("3PM", min_value=0.0, step=0.5, key=key_prefix + "fg3m")
                        with g2:
                            st.session_state[key_prefix + "pr"] = st.number_input("PR", min_value=0.0, step=0.5, key=key_prefix + "pr")
                            st.session_state[key_prefix + "pa"] = st.number_input("PA", min_value=0.0, step=0.5, key=key_prefix + "pa")
                            st.session_state[key_prefix + "ra"] = st.number_input("RA", min_value=0.0, step=0.5, key=key_prefix + "ra")
                            st.session_state[key_prefix + "pra"] = st.number_input("PRA", min_value=0.0, step=0.5, key=key_prefix + "pra")
                        with g3:
                            st.session_state[key_prefix + "stl"] = st.number_input("STL", min_value=0.0, step=0.5, key=key_prefix + "stl")
                            st.session_state[key_prefix + "blk"] = st.number_input("BLK", min_value=0.0, step=0.5, key=key_prefix + "blk")
                            st.session_state[key_prefix + "tov"] = st.number_input("TOV", min_value=0.0, step=0.5, key=key_prefix + "tov")
                            st.session_state[key_prefix + "min"] = st.number_input("MIN", min_value=0.0, step=0.5, key=key_prefix + "min")

                        if st.button("↺ Recalc PR/PA/RA/PRA from PTS/REB/AST", key=f"recalc_{pid}"):
                            pts = float(st.session_state[key_prefix + "pts"])
                            reb = float(st.session_state[key_prefix + "reb"])
                            ast = float(st.session_state[key_prefix + "ast"])
                            st.session_state[key_prefix + "pr"] = pts + reb
                            st.session_state[key_prefix + "pa"] = pts + ast
                            st.session_state[key_prefix + "ra"] = reb + ast
                            st.session_state[key_prefix + "pra"] = pts + reb + ast
                            st.rerun()

                    with col2:
                        st.caption("Live progress")
                        # Build current values
                        current = {k: 0.0 for k in stat_labels}
                        if not prow.empty:
                            current.update({
                                "pts": float(prow.get("pts", 0)),
                                "reb": float(prow.get("reb", 0)),
                                "ast": float(prow.get("ast", 0)),
                                "fg3m": float(prow.get("fg3m", 0)),
                                "stl": float(prow.get("stl", 0)),
                                "blk": float(prow.get("blk", 0)),
                                "tov": float(prow.get("tov", 0)),
                            })
                            current["pr"] = current["pts"] + current["reb"]
                            current["pa"] = current["pts"] + current["ast"]
                            current["ra"] = current["reb"] + current["ast"]
                            current["pra"] = current["pts"] + current["reb"] + current["ast"]
                            current["min"] = parse_min_to_float(str(prow.get("min", "0:00")))
                        grid1, grid2, grid3 = st.columns(3)
                        def show_progress(container, label, cur, line):
                            with container:
                                line = float(line or 0.0)
                                cur = float(cur or 0.0)
                                pct = 0.0 if line <= 0 else min(2.0, cur / line)  # cap 200%
                                st.metric(label, f"{cur:.1f} / {line:.1f}")
                                st.progress(pct)

                        keyp = f"props_{pid}_"
                        # First row
                        show_progress(grid1, "PTS", current["pts"], st.session_state[keyp + "pts"])
                        show_progress(grid2, "REB", current["reb"], st.session_state[keyp + "reb"])
                        show_progress(grid3, "AST", current["ast"], st.session_state[keyp + "ast"])
                        grid4, grid5, grid6 = st.columns(3)
                        show_progress(grid4, "3PM", current["fg3m"], st.session_state[keyp + "fg3m"])
                        show_progress(grid5, "PR", current["pr"], st.session_state[keyp + "pr"])
                        show_progress(grid6, "PA", current["pa"], st.session_state[keyp + "pa"])
                        grid7, grid8, grid9 = st.columns(3)
                        show_progress(grid7, "RA", current["ra"], st.session_state[keyp + "ra"])
                        show_progress(grid8, "PRA", current["pra"], st.session_state[keyp + "pra"])
                        show_progress(grid9, "STL", current["stl"], st.session_state[keyp + "stl"])
                        grid10, grid11, grid12 = st.columns(3)
                        show_progress(grid10, "BLK", current["blk"], st.session_state[keyp + "blk"])
                        show_progress(grid11, "TOV", current["tov"], st.session_state[keyp + "tov"])
                        show_progress(grid12, "MIN", current["min"], st.session_state[keyp + "min"])

                    # Raw box (optional visibility)
                    if not prow.empty:
                        with st.expander("Box score row"):
                            st.dataframe(pd.DataFrame([prow]))

# ---------- Research ----------
with tab_research:
    st.markdown("#### Player lookup")
    q2 = st.text_input("Search player", value="", placeholder="e.g., Jayson Tatum", key="search_research")
    cand = search_players(q2) if q2 else pd.DataFrame()
    player_choice = None
    if not cand.empty:
        labels = [f"{r.name} — {r.team_abbr or 'FA'} (#{r.id})" for _, r in cand.iterrows()]
        lab = st.selectbox("Select", options=labels)
        try:
            player_choice = int(lab.split("#")[-1].strip(")"))
        except Exception:
            player_choice = None

    if not player_choice:
        st.info("Search and select a player.")
    else:
        pinfo = get_player(player_choice) or {}
        pname = f"{(pinfo.get('first_name') or '')} {(pinfo.get('last_name') or '')}".strip()
        pteam = ((pinfo.get("team") or {}).get("abbreviation")) or "—"
        st.subheader(f"{pname} ({pteam})")

        # Controls
        left, right = st.columns([2, 1])
        with left:
            mode = st.radio("Window", ["Last 5", "Last 10", "Last 15", "Last 20", "Current season", "Previous season", "Custom season", "Career"], horizontal=True)
        with right:
            custom_season = st.number_input("Custom season (YYYY)", min_value=1979, max_value=2100, value=dt.date.today().year)

        # Determine selection
        today = dt.date.today()
        cur_season = today.year if today.month >= 10 else today.year - 1  # approx NBA season cutoff Oct
        if mode.startswith("Last "):
            n = int(mode.split(" ")[1])
            logs = player_game_logs(player_choice, seasons=None, last_n=n)
        elif mode == "Current season":
            logs = player_game_logs(player_choice, seasons=[cur_season], last_n=None)
        elif mode == "Previous season":
            logs = player_game_logs(player_choice, seasons=[cur_season - 1], last_n=None)
        elif mode == "Custom season":
            logs = player_game_logs(player_choice, seasons=[int(custom_season)], last_n=None)
        else:  # Career
            seasons = list(range(cur_season, cur_season - 30, -1))
            logs = player_game_logs(player_choice, seasons=seasons, last_n=None)

        # Summaries
        if logs.empty:
            st.info("No logs found for selection.")
        else:
            # Attach opp abbr
            tdf = list_teams()
            abbr_map = {row.id: row.abbreviation for _, row in tdf.iterrows()} if not tdf.empty else {}
            logs = logs.copy()
            logs["opp"] = logs["opp_team_id"].map(abbr_map)
            logs["date"] = pd.to_datetime(logs["date"]).dt.date
            # Summary metrics
            num_cols = ["pts","reb","ast","fg3m","stl","blk","tov","pr","pa","ra","pra"]
            summary = logs[num_cols].mean().round(2)
            colA, colB, colC, colD = st.columns(4)
            with colA: st.metric("PTS", f"{summary['pts']:.2f}")
            with colB: st.metric("REB", f"{summary['reb']:.2f}")
            with colC: st.metric("AST", f"{summary['ast']:.2f}")
            with colD: st.metric("3PM", f"{summary['fg3m']:.2f}")
            colE, colF, colG, colH = st.columns(4)
            with colE: st.metric("PRA", f"{summary['pra']:.2f}")
            with colF: st.metric("PR", f"{summary['pr']:.2f}")
            with colG: st.metric("PA", f"{summary['pa']:.2f}")
            with colH: st.metric("RA", f"{summary['ra']:.2f}")

            # Table
            view_cols = ["date","is_home","opp","min","pts","reb","ast","fg3m","stl","blk","tov","pr","pa","ra","pra"]
            st.dataframe(logs[view_cols].rename(columns={
                "is_home": "home"
            }), use_container_width=True)
            csv = logs[view_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{pname.replace(' ','_')}_{mode.replace(' ','_')}.csv", mime="text/csv")

            # Season averages
            st.markdown("#### Season averages")
            sa_cur = season_averages(player_choice, cur_season)
            sa_prev = season_averages(player_choice, cur_season - 1)
            col1, col2 = st.columns(2)
            with col1:
                if sa_cur:
                    st.write(f"{cur_season} season")
                    st.json({k: sa_cur[k] for k in ["pts","reb","ast","fg3m","stl","blk","turnover","min"] if k in sa_cur})
                else:
                    st.write(f"{cur_season} season: n/a")
            with col2:
                if sa_prev:
                    st.write(f"{cur_season-1} season")
                    st.json({k: sa_prev[k] for k in ["pts","reb","ast","fg3m","stl","blk","turnover","min"] if k in sa_prev})
                else:
                    st.write(f"{cur_season-1} season: n/a")

            # Career highs/lows (on demand)
            with st.expander("Career highs & lows (compute)"):
                if st.button("Compute highs/lows"):
                    with st.spinner("Scanning seasons..."):
                        hi_lo = compute_career_highs_lows(player_choice, current_season=cur_season)
                    if not hi_lo.empty:
                        st.dataframe(hi_lo, use_container_width=True)
                    else:
                        st.write("No data.")

            # Next opponent + most recent vs them
            st.markdown("#### Next opponent & most recent vs them")
            p_team_id = ((pinfo.get("team") or {}).get("id"))
            if not p_team_id:
                st.info("No current team on record for this player.")
            else:
                upcoming = upcoming_games_for_team(int(p_team_id), today, today + dt.timedelta(days=7))
                if upcoming.empty:
                    st.write("No scheduled games in next 7 days.")
                else:
                    nxt = upcoming[upcoming["status"].str.contains("Scheduled", case=False, na=False)]
                    if nxt.empty:
                        nxt = upcoming.head(1)
                    nxt_game = nxt.iloc[0]
                    is_home = int(nxt_game["home_id"]) == int(p_team_id)
                    opp_id = int(nxt_game["visitor_id"] if is_home else nxt_game["home_id"])
                    opp_abbr = nxt_game["visitor_abbr"] if is_home else nxt_game["home_abbr"]
                    st.write(f"Next: {'HOME' if is_home else 'AWAY'} vs {opp_abbr} on {pd.to_datetime(nxt_game['date']).date()}")
                    prior = games_between_teams(int(p_team_id), opp_id, today, limit=10)
                    if prior.empty:
                        st.write("No prior matchups found in the last ~500 days.")
                    else:
                        # Player's last game vs that opponent
                        game_ids = prior["id"].tolist()
                        stats = fetch_all_pages("stats", params={"player_ids[]": player_choice, "game_ids[]": game_ids, "per_page": 100}, fast=False)
                        if not stats:
                            st.write("No player stat vs opponent found.")
                        else:
                            dfp = pd.DataFrame([{
                                "game_id": s["game"]["id"],
                                "date": s["game"]["date"],
                                "pts": s.get("pts") or 0,
                                "reb": s.get("reb") or 0,
                                "ast": s.get("ast") or 0,
                                "fg3m": s.get("fg3m") or 0,
                                "stl": s.get("stl") or 0,
                                "blk": s.get("blk") or 0,
                                "tov": s.get("turnover") or s.get("turnovers") or 0,
                            } for s in stats])
                            dfp["date"] = pd.to_datetime(dfp["date"])
                            dfp = dfp.sort_values("date", ascending=False)
                            last_row = dfp.iloc[0]
                            st.write(f"Most recent vs {opp_abbr} on {last_row['date'].date()}: "
                                     f"{int(last_row['pts'])}p / {int(last_row['reb'])}r / {int(last_row['ast'])}a, "
                                     f"{int(last_row['fg3m'])} threes, {int(last_row['stl'])} stl, {int(last_row['blk'])} blk, {int(last_row['tov'])} tov")

# ---------- Footer ----------
st.caption("Data via balldontlie • For best performance, keep auto-refresh on during live games. Move API key to secrets in production.")
