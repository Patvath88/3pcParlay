import requests
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ------------------------------------------------------------
# GENERIC API WRAPPER WITH SAFETY
# ------------------------------------------------------------
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Fetch error:", endpoint, e)
        return {"data": []}


# ------------------------------------------------------------
# PLAYER LOOKUP
# ------------------------------------------------------------
def find_player(player_name):
    res = fetch("players", {"search": player_name})
    if len(res.get("data", [])) == 0:
        raise ValueError(f"No player found for query: {player_name}")
    return res["data"][0]


# ------------------------------------------------------------
# FULL GAME LOGS
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# UPCOMING GAME FETCH (predict opponent)
# ------------------------------------------------------------
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

    # pick the closest game
    games_sorted = sorted(games, key=lambda g: g["date"])
    return games_sorted[0]


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def build_features(df):
    # Extract nested stat fields into columns cleanly
    stat_cols = [
        "pts", "reb", "ast", "stl", "blk", "turnover",
        "fg3m", "min"
    ]
    results = []
    for _, row in df.iterrows():
        stats = row["stats"]
        game = row["game"]

        # Many statistics live inside nested objects
        rec = {col: stats.get(col, 0) for col in stat_cols}
        rec["date"] = game.get("date")
        results.append(rec)

    df2 = pd.DataFrame(results)
    df2["min"] = df2["min"].astype(float)

    # Derivative props
    df2["pra"] = df2["pts"] + df2["reb"] + df2["ast"]
    df2["pr"] = df2["pts"] + df2["reb"]
    df2["pa"] = df2["pts"] + df2["ast"]
    df2["ra"] = df2["reb"] + df2["ast"]

    df2 = df2.sort_values("date")
    return df2.reset_index(drop=True)


# ------------------------------------------------------------
# TRAIN MODEL + PROJECT NEXT GAME
# ------------------------------------------------------------
def project_next_game(df_features):
    X_cols = ["min"]
    y_cols = ["pts", "reb", "ast", "stl", "blk", "turnover", "fg3m",
              "pra", "pr", "pa", "ra"]

    results = {}

    for target in y_cols:
        good = df_features.dropna(subset=[target])
        if len(good) < 6:
            results[target] = None
            continue

        X = good[X_cols]
        y = good[target]

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror"
        )
        model.fit(X, y)

        next_min = df_features["min"].tail(5).mean()
        pred = model.predict(np.array([[next_min]]))[0]
        results[target] = round(float(pred), 2)

    return results


# ------------------------------------------------------------
# CAREER HIGHS / LOWS
# ------------------------------------------------------------
def compute_high_lows(df):
    numeric = df.select_dtypes("number")
    return {
        col: {
            "career_high": float(numeric[col].max()),
            "career_low": float(numeric[col].min()),
        }
        for col in numeric.columns
    }


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def research_and_project(player_name):
    # Identify player
    player = find_player(player_name)
    player_id = player["id"]
    team_id = player["team"]["id"]

    # Full history
    logs = get_full_game_logs(player_id)
    if logs.empty:
        raise ValueError("No stats found for this player.")

    # Build features
    df_features = build_features(logs)

    # Highs + Lows
    high_low = compute_high_lows(df_features)

    # Recent form segments
    recent = {
        "last_1": df_features.tail(1).to_dict("records"),
        "last_5": df_features.tail(5).to_dict("records"),
        "last_10": df_features.tail(10).to_dict("records"),
        "last_15": df_features.tail(15).to_dict("records"),
        "last_20": df_features.tail(20).to_dict("records"),
    }

    # Next opponent
    next_game = get_next_game(team_id)
    if next_game:
        opp = next_game["visitor_team"] if next_game["home_team"]["id"] == team_id else next_game["home_team"]
        opp_name = opp["full_name"]
    else:
        opp = None
        opp_name = None

    # Projection
    projections = project_next_game(df_features)

    # OUTPUT PACKAGE
    return {
        "player_info": player,
        "summary_stats": df_features.describe().to_dict(),
        "career_highs_lows": high_low,
        "recent_form": recent,
        "next_game": next_game,
        "next_opponent": opp_name,
        "projections": projections,
    }


# ------------------------------------------------------------
# RUN EXAMPLE
# ------------------------------------------------------------
if __name__ == "__main__":
    player = "Stephen Curry"   # change to any player
    results = research_and_project(player)
    print("\n=== PLAYER RESEARCH REPORT ===")
    print(results)
