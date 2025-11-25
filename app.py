"""
NBA Prop Prediction Dashboard — Professional Version
Option B Upgrade: Full rolling features, opponent strength modeling,
trend metrics, lag features, 12+ props, chronological training.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
import data_fetching as dfetch
from models import ModelManager


###############################################################################
# FEATURE ENGINEERING PIPELINE
###############################################################################

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


def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute defensive strength: opponent-allowed PTS/REB/AST."""
    opp_stats = (
        df.groupby("OPP_TEAM")
        .agg({
            "PTS": "mean",
            "REB": "mean",
            "AST": "mean",
        })
        .rename(columns={
            "PTS": "OPP_ALLOW_PTS",
            "REB": "OPP_ALLOW_REB",
            "AST": "OPP_ALLOW_AST"
        })
    )
    df = df.join(opp_stats, on="OPP_TEAM")
    return df


def add_lag_features(df: pd.DataFrame, stat_cols=STAT_COLUMNS, lags=[1, 2, 3, 5]):
    """Add lag features L1, L2, L3, L5 for each stat."""
    for col in stat_cols:
        for lag in lags:
            df[f"{col}_L{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, stat_cols=STAT_COLUMNS):
    """Add rolling averages and rolling standard deviation."""
    for col in stat_cols:
        df[f"{col}_AVG5"] = df[col].rolling(5).mean()
        df[f"{col}_AVG10"] = df[col].rolling(10).mean()
        df[f"{col}_STD5"] = df[col].rolling(5).std()
        df[f"{col}_STD10"] = df[col].rolling(10).std()
    return df


def add_trend_features(df: pd.DataFrame, stat_cols=STAT_COLUMNS):
    """Trend slope using last 5 games."""
    for col in stat_cols:
        df[f"{col}_TREND"] = df[col].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan,
            raw=True,
        )
    return df


def add_context_features(df: pd.DataFrame):
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days
    df["REST_DAYS"] = df["REST_DAYS"].fillna(df["REST_DAYS"].median())

    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)

    return df


def build_training_dataset(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Construct the full per-game feature dataset for model training."""
    if game_logs.empty:
        return pd.DataFrame()

    df = game_logs.copy()

    # Rename VISITOR/HOME team columns
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    # Add all engineered features
    df = compute_opponent_strength(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_context_features(df)

    # Drop rows where lag features not available
    df = df.dropna().reset_index(drop=True)
    return df


###############################################################################
# UI HELPERS
###############################################################################

def format_player_name(row: pd.Series) -> str:
    return f"{row['first_name']} {row['last_name']}"


@st.cache_data(show_spinner=False)
def load_player_list():
    """Loads player list using Balldontlie."""
    try:
        players_df = dfetch.get_active_players_balldontlie()
        players_df["full_name"] = players_df.apply(format_player_name, axis=1)
        return players_df.sort_values("full_name")[["id", "full_name", "team_id"]]
    except Exception:
        fallback = dfetch.get_player_list_nba()
        if fallback.empty:
            return pd.DataFrame(columns=["id", "full_name", "team_id"])
        fallback["full_name"] = fallback["full_name"]
        fallback["team_id"] = None
        return fallback[["id", "full_name", "team_id"]]


###############################################################################
# STREAMLIT UI
###############################################################################

def main():
    st.set_page_config(
        page_title="NBA Prop Predictor (Pro)",
        page_icon="🏀",
        layout="wide"
    )

    st.title("NBA Prop Prediction Dashboard — Pro Model")

    st.caption("Predict PTS, REB, AST, PRA, PR, PA, RA, 3PM, STL, BLK, TOV, MIN using advanced ML models.")

    # Load players
    players_df = load_player_list()
    if players_df.empty:
        st.error("Player list could not be loaded.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Player Selection")
        selected_name = st.selectbox("Player", players_df["full_name"])
        selected_row = players_df[players_df["full_name"] == selected_name].iloc[0]
        player_id = int(selected_row["id"])

        st.header("Prediction Settings")
        stat_choice = st.selectbox(
            "Stat to Predict",
            list(PROP_MAP.keys())
        )

        lookback = st.slider("Games for Rolling Trends", 5, 20, 10, step=5)

        use_neural = st.checkbox("Use Neural Network?", False)

        run_pred = st.button("Predict Next Game")

    if not run_pred:
        st.info("Select options and click Predict.")
        return

    # Fetch game logs
    current_year = datetime.date.today().year
    season_str = f"{current_year-1}-{str(current_year)[-2:]}"
    logs = dfetch.get_player_game_logs_nba(player_id, season_str)

    if logs.empty:
        st.warning("No logs available for this player.")
        return

    # Build training dataset
    train_df = build_training_dataset(logs)
    if train_df.empty:
        st.error("Not enough data to build features.")
        return

    # Determine target variable
    prop = PROP_MAP[stat_choice]

    if isinstance(prop, list):
        train_df["TARGET"] = train_df[prop].sum(axis=1)
    else:
        train_df["TARGET"] = train_df[prop]

    y = train_df["TARGET"].astype(float)
    X = train_df.drop(
        columns=["TARGET", "SEASON_ID", "TEAM_ABBREVIATION", "MATCHUP", "GAME_DATE"],
        errors="ignore"
    )

    # Train models
    model_manager = ModelManager(use_neural=use_neural, random_state=42)
    models = model_manager.train(X, y)

    # Show model results
    st.subheader("Model Evaluation")
    perf = pd.DataFrame({
        "Model": [m.name for m in models.values()],
        "MAE": [m.mae for m in models.values()],
        "MSE": [m.mse for m in models.values()],
    }).sort_values("MAE")
    st.dataframe(perf, use_container_width=True)

    # Build next-game prediction features
    next_game_df = train_df.tail(1).drop(columns="TARGET")
    predictions = model_manager.predict(next_game_df)

    best_model = model_manager.best_model()
    final_pred = predictions[best_model.name][0]

    # Display prediction
    st.success(f"{stat_choice} Prediction: **{final_pred:.1f}** (Model = {best_model.name})")

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
