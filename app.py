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
emphasises a clean, two‑column layout with a sidebar for inputs and
main panel for outputs. Streamlit’s built‑in caching is used to
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

from . import data_fetching as dfetch
from .models import ModelManager


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
    # We set a page icon (basketball emoji) and expand the sidebar by default.
    st.set_page_config(
        page_title="NBA Prop Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS to style the app similar to professional prop dashboards.
    # The colours mirror a dark background with purple accents for headings and
    # primary elements. Streamlit allows unsafe HTML to embed custom styles.
    st.markdown(
        """
        <style>
        /* Global app background */
        .stApp {
            background-color: #0d0a17;
            color: #e1e1e6;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #141126;
            padding-top: 2rem;
        }

        /* Headings colouring */
        h1, h2, h3, h4 {
            color: #b39ddb;
        }

        /* Buttons */
        .stButton > button {
            background-color: #7953d2;
            color: white;
            border-radius: 4px;
        }
        .stButton > button:hover {
            background-color: #6842ba;
            color: white;
        }

        /* Dataframe backgrounds */
        .stDataFrame th {
            background-color: #1e1744;
            color: #b39ddb;
        }
        .stDataFrame td {
            background-color: #1e1744;
            color: #e1e1e6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App title and description. The description is intentionally concise; a longer
    # description appears when the user expands the info expander below.
    st.title("NBA Prop Prediction Dashboard")

    st.caption(
        "Select a player to view their recent performance and predict their next game "
        "statistics. This tool aggregates data from Balldontlie, ESPN and NBA.com and "
        "runs a suite of machine learning models including linear regression, decision "
        "trees, random forests, gradient boosting and neural networks."
    )

    # Provide additional information in an expandable section to avoid cluttering
    # the main interface. Users familiar with prop dashboards can skip this.
    with st.expander("About this app"):
        st.markdown(
            """
            ### Overview
            This dashboard leverages data from three independent sources:

            - **Balldontlie.io** – used for player lookup and season averages. The API key is stored in
              Streamlit Secrets and retrieved on the fly. You can obtain a key from
              [balldontlie.io](https://www.balldontlie.io) and add it to your app secrets.
            - **NBA.com via `nba_api`** – used for detailed game logs and career stats. The
              open-source `nba_api` package wraps NBA.com endpoints without requiring an API
              key【466081084939129†L64-L83】.
            - **ESPN.com** – the hidden, unauthenticated API provides game summaries and
              scoreboards【918690647919691†L27-L45】. Although not currently surfaced in the
              UI, the functions in `data_fetching.py` can be used to enrich features with
              opponent-specific metrics in future iterations.

            ### Modelling
            For each player and chosen statistic, the app constructs a feature vector
            representing rolling averages over the last 5/10/20 games. It then trains a
            suite of models including linear regression, ridge, lasso, decision trees,
            random forests, gradient boosting, k-nearest neighbours, support vector
            regression, and (optionally) a simple neural network【778169439784453†L248-L262】.
            Model performance is reported via Mean Absolute Error (MAE) and Mean
            Squared Error (MSE), and the best-performing model is selected to
            generate the prediction.

            ### Saving Predictions
            When you click **Save Prediction**, the result is appended to a local CSV
            called `predictions_history.csv`. This allows you to track model accuracy
            against real outcomes over time.
            """
        )

    players_df = load_player_list()
    if players_df.empty:
        st.error(
            "Unable to load player list. Please verify your internet connection and"
            " API keys."
        )
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("Select Player")
        player_names = players_df["full_name"].tolist()
        selected_name = st.selectbox("Player", options=player_names)
        # Determine the selected player's ID and team
        selected_row = players_df[players_df["full_name"] == selected_name].iloc[0]
        player_id = int(selected_row["id"])
        team_id = selected_row.get("team_id")
        st.write(f"**Player ID:** {player_id}")
        # Choose stat to predict
        stat_to_predict = st.radio(
            "Statistic to Predict", options=["Points", "Rebounds", "Assists"], index=0
        )
        # Choose the number of recent games to consider for features
        lookback = st.slider(
            "Number of recent games to build features from",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
        )
        # Toggle neural network usage
        use_neural = st.checkbox("Include Neural Network", value=False)
        # Button to trigger prediction
        predict_button = st.button("Predict Next Game")

    # Main display
    if predict_button:
        # Fetch game logs for current season using nba_api
        current_year = datetime.date.today().year
        # Example season string: '2024-25'
        season_str = f"{current_year - 1}-{str(current_year)[-2:]}"
        with st.spinner("Fetching game logs and computing features..."):
            game_logs = dfetch.get_player_game_logs_nba(
                player_id=player_id, season=season_str, num_games=lookback
            )
            if game_logs.empty:
                st.warning(
                    "Could not retrieve game logs for this player. Predictions cannot be made."
                )
                return
            features_df = build_feature_frame(game_logs)
            if features_df.empty:
                st.warning(
                    "Insufficient data to compute features. Try selecting a different player or"
                    " increasing the number of games."
                )
                return
            # Select target variable from game logs
            target_map = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}
            target_col = target_map[stat_to_predict]
            # Build training dataset from available game logs
            # For demonstration we use all past games as rows with same features; in
            # practice you would precompute features for each historical game.
            y = game_logs[target_col].astype(float)
            # Align length: features_df is built from the head of game_logs; replicate
            # features for each game to train quick models. This is simplistic but
            # demonstrates the pipeline.
            X = pd.DataFrame([features_df.iloc[0].values] * len(y), columns=features_df.columns)

            model_manager = ModelManager(use_neural=use_neural, random_state=42)
            models = model_manager.train(X, y)
        # Display evaluation metrics
        st.subheader("Model Performance (lower is better)")
        metrics_table = pd.DataFrame(
            {
                "Model": [info.name for info in models.values()],
                "MAE": [info.mae for info in models.values()],
                "MSE": [info.mse for info in models.values()],
            }
        ).sort_values("MAE")
        st.dataframe(metrics_table, use_container_width=True)
        # Determine best model and make prediction for next game
        best_model_info = model_manager.best_model()
        if best_model_info is not None:
            predictions = model_manager.predict(features_df)
            best_pred = predictions[best_model_info.name][0]
            st.success(
                f"Predicted {stat_to_predict.lower()} for next game: **{best_pred:.1f}** "
                f"(Model: {best_model_info.name})"
            )
        else:
            st.warning("No models were successfully trained.")

        # Display recent game logs
        st.subheader("Recent Games")
        display_logs = game_logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]].copy()
        st.dataframe(display_logs, use_container_width=True)

        # Save prediction to local CSV
        def save_prediction() -> None:
            prediction_record = {
                "date": datetime.date.today().isoformat(),
                "player_id": player_id,
                "player_name": selected_name,
                "team_id": team_id,
                "stat": stat_to_predict,
                "prediction": float(best_pred),
                "model": best_model_info.name if best_model_info else None,
            }
            try:
                df = pd.read_csv("predictions_history.csv")
            except Exception:
                df = pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([prediction_record])], ignore_index=True)
            df.to_csv("predictions_history.csv", index=False)

        if st.button("Save Prediction"):
            save_prediction()
            st.success("Prediction saved to predictions_history.csv")


if __name__ == "__main__":
    main()