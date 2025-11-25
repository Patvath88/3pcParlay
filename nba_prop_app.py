import streamlit as st
import pandas as pd
import requests
import time
import pickle
from sklearn.ensemble import RandomForestRegressor  # we'll regress actual stat values

API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"

# sanitize API key to avoid Latin-1 header errors
API_KEY = API_KEY.encode("utf-8", "ignore").decode("utf-8").strip()

HEADERS = {"Authorization": API_KEY}


# Utility to fetch all pages
def fetch_all(endpoint, params=None, per_page=100, sleep_time=0.5):
    results = []
    page = 1
    while True:
        p = {} if params is None else dict(params)
        p.update({"page": page, "per_page": per_page})
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=p, headers=HEADERS)
        data = resp.json()
        if not data.get("data"):
            break
        results.extend(data["data"])
        if data.get("meta", {}).get("next_cursor") is None:
            break
        page += 1
        time.sleep(sleep_time)
    return pd.DataFrame(results)

## 1) Load or train model (for demo: simple model on PTS only)
@st.cache_resource
def load_model():
    try:
        with open("model_pts.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError:
        model = None
        feature_cols = []
    return model, feature_cols

model, feature_cols = load_model()

st.title("NBA Prop Predictor & Historical Stats")

mode = st.radio("Search by:", ("Player", "Team"))

if mode == "Player":
    player_name = st.text_input("Enter player name (first last):")
    if st.button("Search Player"):
        # fetch player info
        players = fetch_all("players", params={"search": player_name})
        if players.empty:
            st.write("No player found by that name.")
        else:
            player = players.iloc[0]
            pid = player["id"]
            st.write(f"Found player: {player['first_name']} {player['last_name']} (ID {pid})")

            # fetch historical stats (recent 20 games)
            stats = fetch_all("stats", params={"player_ids[]": pid, "per_page":100})
            stats["min"] = pd.to_numeric(stats["min"], errors="coerce")
            stats = stats.sort_values("game.date", ascending=False).head(20)
            hist = stats[["game.date","pts","ast","reb","fg3m","stl","blk","turnover","min"]]
            hist.rename(columns={"turnover":"tov"}, inplace=True)
            st.subheader("Recent Historical Stats")
            st.dataframe(hist)

            if model is not None:
                # for demo: predict pts only using last 3 game avg min & pts
                last3 = stats.head(3)
                feats = {
                    "avg_pts_last3": last3["pts"].mean(),
                    "avg_min_last3": last3["min"].mean()
                }
                X = pd.DataFrame([feats])[feature_cols]
                pred_pts = model.predict(X)[0]
                st.subheader("Predicted Stats")
                st.write(f"➡️ Predicted PTS: {pred_pts:.1f}")
                # you could extend to predict other stats similarly
            else:
                st.write("Model not trained yet — cannot generate predictions.")

if mode == "Team":
    team_name = st.text_input("Enter team name (e.g., Lakers):")
    if st.button("Search Team"):
        # fetch teams
        teams = fetch_all("teams")
        found = teams[teams["full_name"].str.contains(team_name, case=False, na=False)]
        if found.empty:
            st.write("No team found")
        else:
            team = found.iloc[0]
            tid = team["id"]
            st.write(f"Found team: {team['full_name']} (ID {tid})")

            # fetch players roster (filter by team)
            players = fetch_all("players", params={"team_ids[]": tid})
            st.subheader("Roster")
            roster = players[["id","first_name","last_name","position"]]
            roster["name"] = roster["first_name"] + " " + roster["last_name"]
            st.dataframe(roster[["name","position"]])

            # For each player: show simple hist & (if model) prediction
            results = []
            for idx, row in roster.iterrows():
                pid = row["id"]
                stats = fetch_all("stats", params={"player_ids[]": pid, "per_page":100})
                stats["min"] = pd.to_numeric(stats["min"], errors="coerce")
                last3 = stats.sort_values("game.date", ascending=False).head(3)
                feat = {
                    "player": row["first_name"]+" "+row["last_name"],
                    "avg_pts_last3": last3["pts"].mean(),
                    "avg_min_last3": last3["min"].mean()
                }
                if model is not None and feature_cols:
                    X = pd.DataFrame([feat])[feature_cols]
                    feat["pred_pts"] = model.predict(X)[0]
                results.append(feat)
            df_res = pd.DataFrame(results)
            st.subheader("Player Predictions & Recent Form")
            st.dataframe(df_res.sort_values("pred_pts", ascending=False if "pred_pts" in df_res else True))
