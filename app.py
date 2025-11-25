"""
NBA Prop Prediction Dashboard
=============================

This Streamlit application provides a unified interface for NBA prop
research and prediction. Users can search for any active player,
inspect their recent performance (last 5/10/20 games), view season
averages, compare historical matchups against their next opponent and
generate forecasts for their next game statistics using a suite of
machine learning models. The architecture is modular: data
collection lives in `data_fetching.py`, modelling utilities in
`models.py` and the UI is assembled here.

The design draws inspiration from popular prop finder tools and
emphasises a clean, two-column layout with a sidebar for inputs and
main panel for outputs. Streamlit’s built-in caching is used to
avoid redundant API calls and model training.

NOTE: The code references an API key stored in Streamlit’s secrets
(`st.secrets['balldontlie_api_key']`). When deploying the app you
should add your Balldontlie API key to the secrets configuration. In
offline mode the API calls will gracefully fail and dummy data will
be displayed instead.
"""

from __future__ import annotations

import datetime
import json
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# FIXED IMPORTS — REQUIRED FOR STREAMLIT CLOUD
import data_fetching as dfetch
from models import ModelManager


def format_player_name(row: pd.Series) -> str:
    """Format a player's full name for display."""
    return f"{row['first_name']} {row['last_name']}"


@st.cache_data(show_spinner=False)
def load_player_list() -> pd.DataFrame:
    """Load a list of players for the search dropdown.

    We try Balldontlie first to get only active players. If that
    fails we fall back to the nba_api static list which includes both
    active and inactive players.
    """
    api_key = st.secrets.get("balldontlie_api_key", "")
    try:
        # Attempt to fetch active players from Balldontlie
        players_df = dfetch.get_active_players_balldontlie(api_key)
        players_df["full_name"] = players_df.apply(format_player_name, axis=1)
        players_df = players_df.sort_values("full_name")
        return players_df[["id", "full_name", "team_id"]]
    except Exception:
        # Fallback to nba_api static players
        fallback = dfetch.get_player_list_nba()
        if fallback.empty:
            return pd.DataFrame(columns=["id", "full_name", "team_id"])
        fallback_df = fallback.copy()
        fallback_df.rename(columns={"full_name": "full_name", "id": "id"}, inplace=True)
        fallback_df["team_id"] = None
        return fallback_df[["id", "full_name", "team_id"]]


def build_feature_frame(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Construct a feature DataFrame from raw game logs.

    For demonstration, we compute simple moving averages over the
    last 5, 10 and 20 games for core statistics. In a production
    system you may engineer far more sophisticated features (pace
    adjusted stats, opponent defensive ratings, etc.).

    Parameters
    ----------
    game_logs : pandas.DataFrame
        DataFrame returned by `get_player_game_logs_nba`. Must include
        `PTS`, `REB`, `AST`, `MIN`, `FGA`, `FGM`, `FG3A`, `FG3M`,
        `FTA`, `FTM` columns.

    Returns
    -------
    pandas.DataFrame
        A single row DataFrame with aggregated features.
    """
    if game_logs.empty:
        return pd.DataFrame()
    # Ensure numeric types
    numeric_cols = [
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "FGA",
        "FGM",
        "FG3A",
        "FG3M",
        "FTA",
        "FTM",
        "MIN",
    ]
    for col in numeric_cols:
        if col in game_logs.columns:
            game_logs[col] = pd.to_numeric(game_logs[col], errors="coerce")
    # Compute rolling averages; if there are fewer rows than the window,
    # the mean will be computed over the available rows.
    def rolling_mean(col: str, window: int) -> float:
        return game_logs[col].head(window).mean()

    features: Dict[str, float] = {}
    windows = [5, 10, 20]
    for stat in ["PTS", "REB", "AST", "FGA", "FG3A", "FTA", "MIN"]:
        for w in windows:
            key = f"{stat}_avg_{w}"
            features[key] = rolling_mean(stat, w)
    return pd.DataFrame([features])


def main() -> None:
    # Configure page with a custom dark theme reminiscent of popular prop finder tools.
    st.set_page_config(
        page_title="NBA Prop Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS to style the app similar to professional prop dashboards.
    st.markdown(
        """
        <style>
        .stApp { background-color: #0d0a17; color: #e1e1e6; }
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #141126; padding-top: 2rem;
        }
        h1, h2, h3, h4 { color: #b39ddb; }
        .stButton > button {
            background-color: #7953d2; color: white; border-radius: 4px;
        }
        .stButton > button:hover {
            background-color: #6842ba; color: white;
        }
        .stDataFrame th { background-color: #1e1744; color: #b39ddb; }
        .stDataFrame td { background-color: #1e1744; color: #e1e1e6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("NBA Prop Prediction Dashboard")

    st.caption(
        "Select a player to view their recent performance and predict their next game "
        "statistics. This tool aggregates data from Balldontlie, ESPN and NBA.com and "
        "runs multiple machine learning models."
    )

    with st.expander("About this app"):
        st.markdown(
            """
            ### Overview
            This dashboard uses:
            - **Balldontlie.io** for player lists  
            - **NBA.com (`nba_api`)** for game logs  
            - **ESPN.com** hidden endpoints for additional metadata  

            ### Models
            Multiple ML models are trained each time:  
            Linear, Ridge, Lasso, Decision Trees, Random Forest, Gradient Boosting,  
            KNN, SVR, XGBoost, LightGBM, CatBoost, and optional Neural Network.

            ### Prediction Logging
            Saved predictions go into `predictions_history.csv`.
            """
        )

    players_df = load_player_list()
    if players_df.empty:
        st.error("Unable to load player list. Check internet connection and API key.")
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("Select Player")
        player_names = players_df["full_name"].tolist()
        selected_name = st.selectbox("Player", options=player_names)
        selected_row = players_df[players_df["full_name"] == selected_name].iloc[0]
        player_id = int(selected_row["id"])
        team_id = selected_row.get("team_id")
        st.write(f"**Player ID:** {player_id}")

        stat_to_predict = st.radio(
            "Statistic to Predict",
            options=["Points", "Rebounds", "Assists"],
            index=0,
        )

        lookback = st.slider(
            "Games to use for features",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
        )

        use_neural = st.checkbox("Include Neural Network", value=False)

        predict_button = st.button("Predict Next Game")

    if predict_button:
        current_year = datetime.date.today().year
        season_str = f"{current_year - 1}-{str(current_year)[-2:]}"
        with st.spinner("Fetching game logs and computing features..."):
            game_logs = dfetch.get_player_game_logs_nba(
                player_id=player_id, season=season_str, num_games=lookback
            )
            if game_logs.empty:
                st.warning("No game logs available for this player.")
                return

            features_df = build_feature_frame(game_logs)
            if features_df.empty:
                st.warning("Not enough data to compute features.")
                return

            target_map = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}
            target_col = target_map[stat_to_predict]
            y = game_logs[target_col].astype(float)

            X = pd.DataFrame(
                [features_df.iloc[0].values] * len(y),
                columns=features_df.columns,
            )

            model_manager = ModelManager(use_neural=use_neural, random_state=42)
            models = model_manager.train(X, y)

        st.subheader("Model Performance (lower MAE = better)")
        metrics_table = pd.DataFrame(
            {
                "Model": [info.name for info in models.values()],
                "MAE": [info.mae for info in models.values()],
                "MSE": [info.mse for info in models.values()],
            }
        ).sort_values("MAE")
        st.dataframe(metrics_table, use_container_width=True)

        best_model_info = model_manager.best_model()
        if best_model_info:
            predictions = model_manager.predict(features_df)
            best_pred = predictions[best_model_info.name][0]
            st.success(
                f"Predicted {stat_to_predict.lower()}: **{best_pred:.1f}** "
                f"(using {best_model_info.name})"
            )
        else:
            st.warning("Model training failed.")
            return

        st.subheader("Recent Games")
        display_logs = game_logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]]
        st.dataframe(display_logs, use_container_width=True)

        def save_prediction():
            prediction_record = {
                "date": datetime.date.today().isoformat(),
                "player_id": player_id,
                "player_name": selected_name,
                "team_id": team_id,
                "stat": stat_to_predict,
                "prediction": float(best_pred),
                "model": best_model_info.name,
            }
            try:
                df = pd.read_csv("predictions_history.csv")
            except Exception:
                df = pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([prediction_record])], ignore_index=True)
            df.to_csv("predictions_history.csv", index=False)

        if st.button("Save Prediction"):
            save_prediction()
            st.success("Prediction saved!")

if __name__ == "__main__":
    main()
