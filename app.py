"""
NBA Prop Prediction Dashboard — FAST VERSION
Only RandomForest + XGBoost models
All heavy/slow models removed.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager

# -------------------------------
# Feature Engineering Constants
# -------------------------------
STAT_COLUMNS = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"
]

PROP_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
    "3PM": "FG3M",
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "Minutes": "MIN",
}


# -------------------------------
# Advanced Feature Engineering
# -------------------------------
def compute_opponent_strength(df):
    opp_stats = (
        df.groupby("OPP_TEAM")
        .agg({"PTS": "mean", "REB": "mean", "AST": "mean"})
        .rename(columns={
            "PTS": "OPP_ALLOW_PTS",
            "REB": "OPP_ALLOW_REB",
            "AST": "OPP_ALLOW_AST",
        })
    )
    return df.join(opp_stats, on="OPP_TEAM")


def add_lag_features(df):
    for col in STAT_COLUMNS:
        for lag in [1, 2, 3, 5]:
            df[f"{col}_L{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    for col in STAT_COLUMNS:
        df[f"{col}_AVG5"] = df[col].rolling(5).mean()
        df[f"{col}_AVG10"] = df[col].rolling(10).mean()
        df[f"{col}_STD5"] = df[col].rolling(5).std()
    return df


def add_trend_features(df):
    for col in STAT_COLUMNS:
        df[f"{col}_TREND"] = df[col].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
            if len(x) == 5 else np.nan,
            raw=True,
        )
    return df


def add_context_features(df):
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


def build_training_dataset(game_logs):
    if game_logs.empty:
        return pd.DataFrame()

    df = game_logs.copy()

    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    df = compute_opponent_strength(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_context_features(df)

    df = df.dropna().reset_index(drop=True)
    return df


# -------------------------------
# UI Helpers
# -------------------------------
def format_player_name(row):
    return f"{row['first_name']} {row['last_name']}"


@st.cache_data(show_spinner=False)
def load_player_list():
    try:
        players = dfetch.get_active_players_balldontlie()
        players["full_name"] = players.apply(format_player_name, axis=1)
        return players.sort_values("full_name")[["id", "full_name", "team_id"]]
    except:
        fallback = dfetch.get_player_list_nba()
        if fallback.empty:
            return pd.DataFrame(columns=["id", "full_name", "team_id"])
        fallback["full_name"] = fallback["full_name"]
        fallback["team_id"] = None
        return fallback[["id", "full_name", "team_id"]]


# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.set_page_config(
        page_title="NBA Prop Predictor — Fast",
        page_icon="🏀",
        layout="wide"
    )

    st.title("NBA Prop Prediction Dashboard — Fast Version")
    st.caption("Uses only RandomForest + XGBoost for maximum speed.")

    players_df = load_player_list()
    if players_df.empty:
        st.error("Failed to load players.")
        return

    with st.sidebar:
        selected_name = st.selectbox("Player", players_df["full_name"])
        selected_row = players_df[players_df["full_name"] == selected_name].iloc[0]
        player_id = int(selected_row["id"])

        stat_choice = st.selectbox("Stat to Predict", list(PROP_MAP.keys()))

        run_pred = st.button("Predict")

    if not run_pred:
        st.info("Select player/stat → Predict")
        return

    # Load game logs
    year = datetime.date.today().year
    season_str = f"{year-1}-{str(year)[-2:]}"

    logs = dfetch.get_player_game_logs_nba(player_id, season_str)
    if logs.empty:
        st.error("No game logs found.")
        return

    train_df = build_training_dataset(logs)

    # Build TARGET
    target_map = PROP_MAP[stat_choice]
    if isinstance(target_map, list):
        train_df["TARGET"] = train_df[target_map].sum(axis=1)
    else:
        train_df["TARGET"] = train_df[target_map]

    y = train_df["TARGET"]

    # Drop string columns
    X = train_df.drop(
        columns=["TARGET", "MATCHUP", "GAME_DATE", "TEAM_ABBREVIATION",
                 "SEASON_ID", "OPP_TEAM", "WL", "VIDEO_AVAILABLE"],
        errors="ignore"
    )

    X = X.select_dtypes(include=["float", "int"])

    model_manager = ModelManager(random_state=42)
    models = model_manager.train(X, y)

    st.subheader("Model Performance")
    perf = pd.DataFrame({
        "Model": [m.name for m in models.values()],
        "MAE": [m.mae for m in models.values()],
        "MSE": [m.mse for m in models.values()],
    }).sort_values("MAE")
    st.dataframe(perf, use_container_width=True)

    next_features = X.tail(1)
    preds = model_manager.predict(next_features)
    best = model_manager.best_model()

    st.success(f"{stat_choice} Prediction: **{preds[best.name][0]:.1f}** ({best.name})")

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "FG3M",
              "STL", "BLK", "TOV", "MIN"]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
