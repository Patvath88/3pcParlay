"""
NBA Prop Prediction Dashboard
"""

from __future__ import annotations

import datetime
import json
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# FIXED IMPORTS (no relative imports)
import data_fetching as dfetch
from models import ModelManager


def format_player_name(row: pd.Series) -> str:
    return f"{row['first_name']} {row['last_name']}"


@st.cache_data(show_spinner=False)
def load_player_list() -> pd.DataFrame:
    """Loads player list directly from Balldontlie (hardcoded API key)."""

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


def build_feature_frame(game_logs: pd.DataFrame) -> pd.DataFrame:
    if game_logs.empty:
        return pd.DataFrame()

    numeric_cols = [
        "PTS","REB","AST","STL","BLK","TOV",
        "FGA","FGM","FG3A","FG3M","FTA","FTM","MIN",
    ]

    for col in numeric_cols:
        if col in game_logs:
            game_logs[col] = pd.to_numeric(game_logs[col], errors="coerce")

    def rolling(col, w): return game_logs[col].head(w).mean()

    features = {}
    for stat in ["PTS","REB","AST","FGA","FG3A","FTA","MIN"]:
        for window in [5, 10, 20]:
            features[f"{stat}_avg_{window}"] = rolling(stat, window)

    return pd.DataFrame([features])


def main():
    st.set_page_config(
        page_title="NBA Prop Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("NBA Prop Prediction Dashboard")
    st.caption("Predict upcoming NBA player performance using machine learning.")

    players_df = load_player_list()
    if players_df.empty:
        st.error("Unable to load players.")
        return

    with st.sidebar:
        st.header("Select Player")
        selected_name = st.selectbox("Player", players_df["full_name"])
        selected_row = players_df[players_df["full_name"] == selected_name].iloc[0]
        player_id = int(selected_row["id"])
        team_id = selected_row["team_id"]

        stat_to_predict = st.radio(
            "Stat to predict", ["Points", "Rebounds", "Assists"]
        )
        lookback = st.slider("Games to analyze", 5, 20, 10, step=5)
        use_neural = st.checkbox("Use Neural Network?", value=False)

        predict_button = st.button("Predict Next Game")

    if predict_button:

        current_year = datetime.date.today().year
        season_str = f"{current_year-1}-{str(current_year)[-2:]}"
        game_logs = dfetch.get_player_game_logs_nba(
            player_id, season_str, num_games=lookback
        )

        if game_logs.empty:
            st.warning("No logs available for this player.")
            return

        features_df = build_feature_frame(game_logs)
        target_map = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}
        y = game_logs[target_map[stat_to_predict]].astype(float)

        X = pd.DataFrame([features_df.iloc[0].values] * len(y), columns=features_df.columns)

        model_manager = ModelManager(use_neural=use_neural, random_state=42)
        models = model_manager.train(X, y)

        st.subheader("Model Performance (MAE/MSE)")
        perf = pd.DataFrame({
            "Model": [m.name for m in models.values()],
            "MAE": [m.mae for m in models.values()],
            "MSE": [m.mse for m in models.values()],
        }).sort_values("MAE")

        st.dataframe(perf, use_container_width=True)

        best = model_manager.best_model()
        preds = model_manager.predict(features_df)
        best_pred = preds[best.name][0]

        st.success(f"{stat_to_predict} prediction: **{best_pred:.1f}** using {best.name}")

        st.subheader("Recent Games")
        st.dataframe(
            game_logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
