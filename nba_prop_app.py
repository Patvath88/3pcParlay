import streamlit as st
import pandas as pd
import requests
import time
import pickle
from sklearn.ensemble import RandomForestRegressor

# --- API Setup ---
BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
API_KEY = API_KEY.encode("utf-8", "ignore").decode("utf-8").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# --- Utils ---
def fetch_all(endpoint, params=None, per_page=100, sleep_time=0.5):
    results = []
    page = 1
    while True:
        try:
            p = dict(params) if params else {}
            p.update({"page": page, "per_page": per_page})
            url = f"{BASE_URL}/{endpoint}"
            resp = requests.get(url, params=p, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("data"):
                break
            results.extend(data["data"])
            if not data.get("meta") or data["meta"].get("next_cursor") is None:
                break
            page += 1
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error fetching {endpoint} (page {page}):", e)
            break
    return pd.DataFrame(results)

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

# --- Streamlit UI ---
st.title("🏀 NBA Prop Predictor + Historical Stats")

mode = st.radio("Search by:", ("Player", "Team"))

if mode == "Player":
    player_name = st.text_input("Enter player name (e.g. LeBron James):")
    if st.button("Search Player"):
        players = fetch_all("players", params={"search": player_name})
        if players.empty:
            st.warning("No player found.")
        else:
            player = players.iloc[0]
            pid = player["id"]
            st.success(f"Found: {player['first_name']} {player['last_name']} (ID {pid})")

            stats = fetch_all("stats", params={"player_ids[]": pid})
            stats["min"] = pd.to_numeric(stats["min"], errors="coerce")
            stats = stats.sort_values("game.date", ascending=False).head(20)

            hist = stats[["game.date","pts","ast","reb","fg3m","stl","blk","turnover","min"]].copy()
            hist.rename(columns={"turnover": "tov"}, inplace=True)
            hist["pr"] = hist["pts"] + hist["reb"]
            hist["pa"] = hist["pts"] + hist["ast"]
            hist["ra"] = hist["reb"] + hist["ast"]
            hist["pra"] = hist["pts"] + hist["reb"] + hist["ast"]

            st.subheader("Recent Historical Stats (Last 20 Games)")
            st.dataframe(hist)

            if model is not None:
                last3 = stats.head(3)
                feats = {
                    "avg_pts_last3": last3["pts"].mean(),
                    "avg_min_last3": last3["min"].mean()
                }
                X = pd.DataFrame([feats])[feature_cols]
                pred_pts = model.predict(X)[0]
                st.subheader("🔮 Predicted Stats")
                st.write(f"Predicted Points: **{pred_pts:.1f}**")
            else:
                st.info("Model not trained — using historical data only.")

if mode == "Team":
    team_query = st.text_input("Enter team name (e.g. Lakers):")
    if st.button("Search Team"):
        teams = fetch_all("teams")
        match = teams[teams["full_name"].str.contains(team_query, case=False, na=False)]
        if match.empty:
            st.warning("Team not found.")
        else:
            team = match.iloc[0]
            tid = team["id"]
            st.success(f"Team Found: {team['full_name']} (ID {tid})")

            players = fetch_all("players", params={"team_ids[]": tid})
            st.subheader("Roster")
            players["name"] = players["first_name"] + " " + players["last_name"]
            st.dataframe(players[["name", "position"]])

            results = []
            for idx, row in players.iterrows():
                pid = row["id"]
                stats = fetch_all("stats", params={"player_ids[]": pid})
                if stats.empty:
                    continue
                stats["min"] = pd.to_numeric(stats["min"], errors="coerce")
                last3 = stats.sort_values("game.date", ascending=False).head(3)
                feats = {
                    "player": row["first_name"] + " " + row["last_name"],
                    "avg_pts_last3": last3["pts"].mean(),
                    "avg_min_last3": last3["min"].mean()
                }
                if model is not None and feature_cols:
                    X = pd.DataFrame([feats])[feature_cols]
                    feats["pred_pts"] = model.predict(X)[0]
                results.append(feats)

            st.subheader("Team Predictions")
            df_res = pd.DataFrame(results)
            st.dataframe(df_res.sort_values("pred_pts", ascending=False if "pred_pts" in df_res else True))
