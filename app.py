"""
NBA Prop Predictor — Cached Multi-Model Version
Predicts: PTS, REB, AST, PRA, PR, PA, RA, 3PM, STL, BLK, TOV, MIN
All ML models are cached per-player for FAST performance.
"""

from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager


# ============================================================
# Feature Engineering
# ============================================================

STAT_COLUMNS = ["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]

PROP_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "PRA": ["PTS","REB","AST"],
    "PR": ["PTS","REB"],
    "PA": ["PTS","AST"],
    "RA": ["REB","AST"],
    "3PM": "FG3M",
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "Minutes": "MIN"
}


def compute_opponent_strength(df):
    opp = df.groupby("OPP_TEAM")[["PTS","REB","AST"]].mean().rename(columns={
        "PTS":"OPP_ALLOW_PTS",
        "REB":"OPP_ALLOW_REB",
        "AST":"OPP_ALLOW_AST"
    })
    return df.join(opp, on="OPP_TEAM")


def add_lag_features(df):
    for col in STAT_COLUMNS:
        for lag in [1,2,3,5]:
            df[f"{col}_L{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    for col in STAT_COLUMNS:
        df[f"{col}_AVG5"] = df[col].rolling(5).mean()
        df[f"{col}_AVG10"] = df[col].rolling(10).mean()
        df[f"{col}_STD5"] = df[col].rolling(5).std()
    return df


def add_trend(df):
    for col in STAT_COLUMNS:
        df[f"{col}_TREND"] = df[col].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==5 else np.nan,
            raw=True
        )
    return df


def add_context(df):
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


def build_training_dataset(logs):
    if logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    df = compute_opponent_strength(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend(df)
    df = add_context(df)

    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# Player List Cache
# ============================================================

@st.cache_data(show_spinner=False)
def load_player_list():
    try:
        p = dfetch.get_active_players_balldontlie()
        p["full_name"] = p["first_name"] + " " + p["last_name"]
        return p.sort_values("full_name")[["id","full_name","team_id"]]
    except:
        fallback = dfetch.get_player_list_nba()
        fallback["full_name"] = fallback["full_name"]
        fallback["team_id"] = None
        return fallback[["id","full_name","team_id"]]


# ============================================================
# MODEL CACHE — trains ONCE per player
# ============================================================

@st.cache_resource
def get_cached_models(player_id: int, stat_choice: str, X: pd.DataFrame, y: pd.Series):
    """Train ALL MODELS once per player + stat. Future predictions instant."""
    manager = ModelManager(random_state=42)
    manager.train(X, y)
    return manager


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor Pro", page_icon="🏀", layout="wide")
    st.title("NBA Prop Predictor — Cached Multi-Model Version")

    players = load_player_list()

    with st.sidebar:
        name = st.selectbox("Player", players["full_name"])
        row = players[players["full_name"] == name].iloc[0]
        player_id = int(row["id"])

        stat_choice = st.selectbox("Stat to Predict", list(PROP_MAP.keys()))

        run = st.button("Predict")

    if not run:
        st.info("Select player/stat and click Predict.")
        return

    # Load logs
    year = datetime.date.today().year
    season = f"{year-1}-{str(year)[-2:]}"
    logs = dfetch.get_player_game_logs_nba(player_id, season)

    if logs.empty:
        st.error("No data found.")
        return

    df = build_training_dataset(logs)

    # Build target
    target = PROP_MAP[stat_choice]
    if isinstance(target, list):
        df["TARGET"] = df[target].sum(axis=1)
    else:
        df["TARGET"] = df[target]

    y = df["TARGET"]

    # Build feature matrix (numeric only)
    remove_cols = [
        "TARGET","MATCHUP","GAME_DATE","OPP_TEAM",
        "SEASON_ID","TEAM_ABBREVIATION","WL","VIDEO_AVAILABLE"
    ]
    X = df.drop(columns=remove_cols, errors="ignore")
    X = X.select_dtypes(include=["float","int"])

    # TRAIN OR LOAD CACHED MODELS
    manager = get_cached_models(player_id, stat_choice, X, y)

    # Predict next game
    X_next = X.tail(1)
    predictions = manager.predict(X_next)
    best = manager.best_model()

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    st.subheader("Model Predictions")
    out = []
    for name, info in manager.models.items():
        out.append({
            "Model": name,
            "Prediction": info.prediction,
            "MAE": info.mae,
            "MSE": info.mse
        })

    st.dataframe(pd.DataFrame(out).sort_values("MAE"), use_container_width=True)

    st.success(
        f"Best Model: **{best.name}** → Prediction: **{best.prediction:.1f} {stat_choice}**"
    )

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE","MATCHUP","PTS","REB","AST",
              "FG3M","STL","BLK","TOV","MIN"]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
