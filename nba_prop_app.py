import streamlit as st
import pandas as pd
import requests
import time
import pickle

# --- API Setup ---
BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
API_KEY = API_KEY.encode("utf-8", "ignore").decode("utf-8").strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# --- Utils ---
@st.cache_resource
def fetch_all_players(active_only=True):
    results = []
    page = 1
    while True:
        try:
            url = f"{BASE_URL}/players"
            resp = requests.get(url, params={"page": page, "per_page": 100}, headers=HEADERS)
            if resp.status_code == 429:
                time.sleep(10)
                continue
            data = resp.json()
            if not data.get("data"):
                break
            results.extend(data["data"])
            if data.get("meta", {}).get("next_cursor") is None:
                break
            page += 1
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Error fetching players: {e}")
            break
    df = pd.DataFrame(results)
    df["name"] = df["first_name"] + " " + df["last_name"]

    if active_only:
        # Filter out players with no team (i.e., team['id'] == None or 'G League' etc.)
        df = df[df["team"].apply(lambda x: x and x.get("id") is not None and "NBA" in x.get("full_name", "NBA"))]

    return df.sort_values("name").reset_index(drop=True)


def fetch_player_stats(player_id, n_games=20):
    results = []
    page = 1
    while len(results) < n_games:
        try:
            url = f"{BASE_URL}/stats"
            resp = requests.get(url, params={"player_ids[]": player_id, "page": page, "per_page": 100}, headers=HEADERS)
            if resp.status_code == 429:
                time.sleep(10)
                continue
            data = resp.json()
            if not data.get("data"):
                break
            results.extend(data["data"])
            if data.get("meta", {}).get("next_cursor") is None:
                break
            page += 1
            time.sleep(1.0)
        except Exception as e:
            st.error(f"Error fetching stats: {e}")
            break
    df = pd.DataFrame(results[:n_games])
    if not df.empty and "game" in df.columns:
        df["game_date"] = pd.to_datetime(df["game"].apply(lambda x: x.get("date") if x else None))
        df["min"] = pd.to_numeric(df["min"], errors="coerce")
    return df

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

# --- Load Resources ---
players_df = fetch_all_players()
model, feature_cols = load_model()

# --- UI ---
st.title("🏀 NBA Player Prop Predictor + Recent Stats")

selected_name = st.selectbox("Select a player:", players_df["name"].tolist())
player_row = players_df[players_df["name"] == selected_name].iloc[0]
pid = player_row["id"]

st.subheader(f"Recent Stats for {selected_name}")
stats_df = fetch_player_stats(pid)

if not stats_df.empty:
    df = stats_df[["game_date","pts","ast","reb","fg3m","stl","blk","turnover","min"]].copy()
    df.rename(columns={"turnover": "tov"}, inplace=True)
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]
    df = df.round(1)

    st.dataframe(df)

    if model is not None and len(df) >= 3:
        last3 = df.head(3)
        feats = {
            "avg_pts_last3": last3["pts"].mean(),
            "avg_min_last3": last3["min"].mean()
        }
        X = pd.DataFrame([feats])[feature_cols]
        pred_pts = model.predict(X)[0]
        st.subheader("🔮 Predicted Stats")
        st.write(f"Predicted Points: **{pred_pts:.1f}**")
    else:
        st.info("Model not trained or not enough recent games to predict.")
else:
    st.warning("No game data available for this player.")
