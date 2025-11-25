import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.set_page_config(
    page_title="NBA Player Research + AI Projections",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body, .stApp {
        background-color: #0d1117;
        color: white;
    }
    .header {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 22px;
        margin-top: 25px;
        margin-bottom: 5px;
    }
    .metric-box {
        padding: 15px;
        background-color: #161b22;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# UTIL: SAFE API REQUEST
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
    except:
        return {"data": []}


# --------------------------------------------------
# PLAYER SEARCH
# --------------------------------------------------
def find_player(name):
    res = fetch("players", {"search": name})
    return res.get("data", [])


# --------------------------------------------------
# PLAYER HEADSHOT
# --------------------------------------------------
def get_headshot(player_id):
    # BallDontLie headshot CDN pattern
    return f"https://cdn.balldontlie.io/headshots/{player_id}.png"


# --------------------------------------------------
# FULL GAME LOGS
# --------------------------------------------------
def get_full_game_logs(player_id, seasons=range(2014, 2025)):
    frames = []
    for season in seasons:
        page = 1
        while True:
            payload = fetch("stats", {
                "player_ids[]": player_id,
                "seasons[]": season,
                "per_page": 100,
                "page": page,
            })
            data = payload.get("data", [])
            if not data:
                break
            frames.append(pd.DataFrame(data))
            if payload.get("meta", {}).get("next_page") is None:
                break
            page += 1

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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

    return df2.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# NEXT MATCHUP
# --------------------------------------------------
def get_next_game(team_id):
    today = datetime.today().date()
    future = today + timedelta(days=10)

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
# DEFENSIVE RATING (SIMULATED UNTIL API AVAILABLE)
# --------------------------------------------------
@st.cache_data(ttl=86400)
def get_defensive_rating(team_id):
    # Eventually replace w/ StatMuse or NBA API
    # For now: simulate realistic DRtg values 105-120
    random.seed(team_id)
    return round(random.uniform(107, 119), 1)


# --------------------------------------------------
# MATCHUP ADVANTAGE SCORE (weighted system)
# --------------------------------------------------
def compute_matchup_advantage(def_rating, df_features):
    recent_pts = df_features["pts"].tail(10).mean()
    usage_min = df_features["min"].tail(5).mean()

    # Lower defensive rating = easier matchup
    dr_factor = max(0, 125 - def_rating)

    score = (
        dr_factor * 0.5 +
        recent_pts * 0.3 +
        usage_min * 0.2
    )
    return round(score, 2)


# --------------------------------------------------
# MODELING / PROJECTIONS
# --------------------------------------------------
def project_next_game(df):
    X_cols = ["min"]
    targets = [
        "pts", "reb", "ast", "stl", "blk", "turnover",
        "fg3m", "pra", "pr", "pa", "ra"
    ]

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
            objective="reg:squarederror",
        )
        model.fit(X, y)

        next_min = df["min"].tail(5).mean()
        pred = model.predict(np.array([[next_min]]))[0]
        results[t] = round(float(pred), 2)

    return results


# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------
def monte_carlo_distribution(mean, std, sims=10000):
    sims = np.random.normal(loc=mean, scale=std, size=sims)
    return sims


# --------------------------------------------------
# APP UI
# --------------------------------------------------
st.markdown("<div class='header'>NBA Player Research + AI Projection Dashboard</div>", unsafe_allow_html=True)

player_name = st.text_input("Search Player Name", placeholder="LeBron James")

if not player_name.strip():
    st.stop()

players = find_player(player_name)
if len(players) == 0:
    st.error("No players found.")
    st.stop()

player = players[0]
player_id = player["id"]
team_id = player["team"]["id"]
team_name = player["team"]["full_name"]

colA, colB = st.columns([1, 3])

with colA:
    st.image(get_headshot(player_id), width=160)

with colB:
    st.subheader(f"{player['first_name']} {player['last_name']} – {team_name}")

# --------------------------------------------------
# LOAD RESEARCH DATA
# --------------------------------------------------
with st.spinner("Loading full historical game logs..."):
    logs = get_full_game_logs(player_id)

if logs.empty:
    st.error("No data for this player.")
    st.stop()

df = build_features(logs)

# --------------------------------------------------
# NEXT GAME & MATCHUP
# --------------------------------------------------
next_game = get_next_game(team_id)
if next_game:
    opp = (
        next_game["visitor_team"]
        if next_game["home_team"]["id"] == team_id
        else next_game["home_team"]
    )
    opp_id = opp["id"]
    opp_name = opp["full_name"]

    drtg = get_defensive_rating(opp_id)
    advantage = compute_matchup_advantage(drtg, df)

    st.markdown("<div class='subheader'>Next Matchup</div>", unsafe_allow_html=True)
    st.write(f"Opponent: **{opp_name}**")
    st.write(f"Date: **{next_game['date']}**")

    col1, col2 = st.columns(2)
    col1.metric("Opponent Defensive Rating", drtg)
    col2.metric("Matchup Advantage Score", advantage)

# --------------------------------------------------
# TABS UI
# --------------------------------------------------
tab1, tab2 = st.tabs(["Research", "Betting Insights"])

# --------------------------------------------------
# TAB 1 — RESEARCH
# --------------------------------------------------
with tab1:

    st.markdown("<div class='subheader'>Recent Form</div>", unsafe_allow_html=True)
    recent_tabs = st.tabs(["Last 1", "Last 5", "Last 10", "Last 15", "Last 20"])

    segs = {
        "Last 1": df.tail(1),
        "Last 5": df.tail(5),
        "Last 10": df.tail(10),
        "Last 15": df.tail(15),
        "Last 20": df.tail(20),
    }

    for t, key in zip(recent_tabs, segs):
        t.dataframe(segs[key], use_container_width=True)

    st.markdown("<div class='subheader'>Career Highs / Lows</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().loc[["min", "50%", "max"]], use_container_width=True)

    st.markdown("<div class='subheader'>AI Projections</div>", unsafe_allow_html=True)
    projections = project_next_game(df)
    proj_df = pd.DataFrame(projections.items(), columns=["Stat", "Projection"])
    st.dataframe(proj_df, use_container_width=True)

# --------------------------------------------------
# TAB 2 — BETTING INSIGHTS
# --------------------------------------------------
with tab2:

    st.markdown("<div class='subheader'>Enter Your Prop Lines</div>", unsafe_allow_html=True)

    user_lines = {}
    cols = st.columns(4)
    stats = ["pts", "reb", "ast", "stl", "blk", "turnover", "fg3m", "pra", "pr", "pa", "ra"]

    for i, stat in enumerate(stats):
        with cols[i % 4]:
            user_lines[stat] = st.number_input(f"{stat.upper()} Line", value=0.0)

    st.markdown("<hr>")

    st.markdown("<div class='subheader'>Edges</div>", unsafe_allow_html=True)

    edges = {}
    for stat in stats:
        pred = projections.get(stat)
        line = user_lines[stat]
        if pred is None or line == 0:
            edges[stat] = None
            continue
        edge = ((pred - line) / line) * 100 if line != 0 else 0
        edges[stat] = round(edge, 2)

    edge_df = pd.DataFrame(edges.items(), columns=["Stat", "Edge %"])
    st.dataframe(edge_df, use_container_width=True)

    # Monte Carlo simulation
    st.markdown("<div class='subheader'>Monte Carlo Simulation</div>", unsafe_allow_html=True)

    sim_stat = st.selectbox("Select Stat for Simulation", stats)
    pred = projections.get(sim_stat)

    if pred:
        std = df[sim_stat].tail(25).std()
        sims = monte_carlo_distribution(pred, std)

        st.write(f"Projected Mean: {round(pred, 2)} | Std: {round(std, 2)}")
        st.write(f"Prob Over Line ({user_lines[sim_stat]}): {round((sims > user_lines[sim_stat]).mean() * 100, 2)}%")

        st.line_chart(pd.DataFrame({"Simulation": sims}))
