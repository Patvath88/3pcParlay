import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(
    page_title="NBA Player Research + Projection AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body, .stApp {
        background-color: #0d1117;
        color: white;
    }
    .big-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 20px;
        margin-top: 25px;
        margin-bottom: 5px;
    }
    .stDataFrame, .stTable {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# API WRAPPER
# --------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch(endpoint, params=None):
    try:
        r = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"data": [], "error": str(e)}

# --------------------------------------------------
# PLAYER SEARCH
# --------------------------------------------------
def find_player(name):
    res = fetch("players", {"search": name})
    players = res.get("data", [])
    return players

# --------------------------------------------------
# GAME LOGS
# --------------------------------------------------
def get_full_game_logs(player_id, seasons=range(2014, 2025)):
    frames = []
    for season in seasons:
        page = 1
        while True:
            payload = fetch(
                "stats",
                {
                    "player_ids[]": player_id,
                    "seasons[]": season,
                    "per_page": 100,
                    "page": page,
                }
            )
            data = payload.get("data", [])
            if not data:
                break
            frames.append(pd.DataFrame(data))
            if payload.get("meta", {}).get("next_page") is None:
                break
            page += 1

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()

    return df

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def build_features(df):
    stat_cols = ["pts", "reb", "ast", "stl", "blk", "turnover", "fg3m", "min"]
    results = []

    for _, row in df.iterrows():
        stats = row["stats"]
        game = row["game"]
        rec = {col: stats.get(col, 0) for col in stat_cols}
        rec["date"] = game.get("date")
        results.append(rec)

    df2 = pd.DataFrame(results)
    df2["min"] = df2["min"].astype(float)

    df2["pra"] = df2["pts"] + df2["reb"] + df2["ast"]
    df2["pr"] = df2["pts"] + df2["reb"]
    df2["pa"] = df2["pts"] + df2["ast"]
    df2["ra"] = df2["reb"] + df2["ast"]

    df2 = df2.sort_values("date")
    return df2.reset_index(drop=True)

# --------------------------------------------------
# NEXT MATCHUP
# --------------------------------------------------
def get_next_game(team_id):
    today = datetime.today().date()
    future = today + timedelta(days=7)

    payload = fetch("games", {
        "team_ids[]": team_id,
        "start_date": today,
        "end_date": future,
    })

    games = payload.get("data", [])
    if not games:
        return None

    return sorted(games, key=lambda g: g["date"])[0]

# --------------------------------------------------
# MODELING
# --------------------------------------------------
def project_next_game(df):
    X_cols = ["min"]
    targets = ["pts", "reb", "ast", "stl", "blk", "turnover", "fg3m",
               "pra", "pr", "pa", "ra"]

    results = {}

    for t in targets:
        good = df.dropna(subset=[t])
        if len(good) < 6:
            results[t] = None
            continue

        X = good[X_cols]
        y = good[t]

        model = XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror"
        )
        model.fit(X, y)

        next_min = df["min"].tail(5).mean()
        pred = model.predict(np.array([[next_min]]))[0]

        results[t] = round(float(pred), 2)

    return results

# --------------------------------------------------
# APP UI
# --------------------------------------------------
st.markdown("<div class='big-title'>NBA Player Research + Projection AI</div>", unsafe_allow_html=True)

player_name = st.text_input("Search Player Name", value="", placeholder="LeBron James")

if player_name.strip() == "":
    st.stop()

players = find_player(player_name)

if len(players) == 0:
    st.error("No players found.")
    st.stop()

player = players[0]
player_id = player["id"]
team_id = player["team"]["id"]

st.subheader(f"Player Selected: {player['first_name']} {player['last_name']} – {player['team']['full_name']}")

# --------------------------------------------------
# LOAD GAME LOGS
# --------------------------------------------------
with st.spinner("Loading full game logs..."):
    logs = get_full_game_logs(player_id)

if logs.empty:
    st.error("No historical data found for this player.")
    st.stop()

# Build features
df = build_features(logs)

# Recent form
st.markdown("<div class='subheader'>Recent Form</div>", unsafe_allow_html=True)
tabs = st.tabs(["Last 1", "Last 5", "Last 10", "Last 15", "Last 20"])

segments = {
    "Last 1": df.tail(1),
    "Last 5": df.tail(5),
    "Last 10": df.tail(10),
    "Last 15": df.tail(15),
    "Last 20": df.tail(20),
}

for t, key in zip(tabs, segments):
    t.dataframe(segments[key], use_container_width=True)

# Career highs/lows
st.markdown("<div class='subheader'>Career Highs & Lows</div>", unsafe_allow_html=True)
highs = df.describe().loc[["min", "50%", "max"]]
st.dataframe(highs, use_container_width=True)

# Next matchup
next_game = get_next_game(team_id)
if next_game:
    opp = next_game["visitor_team"] if next_game["home_team"]["id"] == team_id else next_game["home_team"]
    st.markdown("<div class='subheader'>Next Matchup</div>", unsafe_allow_html=True)
    st.write(f"Next Opponent: **{opp['full_name']}** on **{next_game['date']}**")

# Projections
st.markdown("<div class='subheader'>AI Projected Next Game Stats</div>", unsafe_allow_html=True)

with st.spinner("Building model and generating projections..."):
    projections = project_next_game(df)

proj_df = pd.DataFrame(projections.items(), columns=["Stat", "Projection"])
st.dataframe(proj_df, use_container_width=True)

st.success("Research + projections complete.")
