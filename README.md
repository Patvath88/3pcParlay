# NBA Prop Prediction Dashboard

The **NBA Prop Prediction Dashboard** is a Streamlit‑powered web application for research and forecasting of NBA player performance.  Inspired by professional prop finder tools, this app aggregates player statistics from **Balldontlie.io**, **NBA.com** (via the `nba_api` Python package) and **ESPN.com** to deliver a unified experience.  Users can inspect recent game logs, season averages and matchup trends, then train a suite of machine‑learning models to predict a player’s next‑game statistics.  Predictions can be saved locally for later accuracy analysis.

## Features

* **Unified data sources** – Utilises Balldontlie’s API to look up active players and fetch season averages.  Detailed game logs come from NBA.com through the `nba_api` wrapper, which does not require authentication【466081084939129†L64-L83】.  ESPN’s hidden JSON endpoints can be used to gather scoreboard and opponent information【918690647919691†L27-L45】.  If the network fails, the app gracefully degrades with informative warnings.
* **Rich player research** – View the last 5, 10 or 20 games for any NBA player, including points, rebounds and assists.  Season averages and historical matchups can be integrated by extending the provided functions in `data_fetching.py`.
* **Multiple ML models** – A `ModelManager` class trains an array of regression models—linear regression, ridge, lasso, decision trees, random forests, gradient boosting, k‑nearest neighbours, support vector regression, XGBoost, LightGBM, CatBoost and an optional neural network【778169439784453†L248-L262】.  Each model is evaluated via Mean Absolute Error (MAE) and Mean Squared Error (MSE); the best performer is chosen for prediction.
* **Modern UI** – The dashboard adopts a dark theme with purple accents reminiscent of popular prop finder dashboards.  A collapsible *About* section explains the data sources, modelling strategy and saving mechanism.
* **Prediction history** – Clicking **Save Prediction** appends the result to a local `predictions_history.csv` file.  This lets you track how well the models perform against real game outcomes.

## Getting Started

### Prerequisites

This repository is designed to run on a Python environment with the following dependencies.  Many packages are heavy (e.g., TensorFlow, CatBoost), so ensure sufficient disk space and memory.

* Python 3.8+
* pandas
* numpy
* scikit‑learn
* xgboost
* lightgbm
* catboost
* TensorFlow (if using the neural network)
* `nba_api` (for NBA.com data)
* Streamlit

All required packages are listed in `requirements.txt`.  You can install them via pip:

```bash
pip install -r requirements.txt
```

If you encounter issues installing certain libraries (e.g., LightGBM or CatBoost), you may comment them out of `requirements.txt` or adjust the `ModelManager` to skip those models.

### API Keys

Balldontlie requires an API key for some endpoints.  Sign up at [balldontlie.io](https://www.balldontlie.io) to obtain a key.  In Streamlit Cloud or your local environment, place the key in `.streamlit/secrets.toml` like so:

```toml
[general]
balldontlie_api_key = "YOUR_BALLDONTLIE_API_KEY"
```

Alternatively, you can pass the key directly into functions in `data_fetching.py`.

### Running the App

After installing dependencies and configuring your API key, launch the dashboard with:

```bash
streamlit run nba_prop_dashboard/app.py
```

The app will open in your default web browser.  From the sidebar you can select a player, choose the statistic to predict (points, rebounds or assists), pick how many recent games to use for feature engineering, and decide whether to include a neural network model.  After clicking **Predict Next Game**, the models are trained and the forecast is displayed along with a comparison of model errors.

### Extending the App

The current implementation uses simple rolling averages for feature construction.  To create a more sophisticated model:

* Enhance `build_feature_frame` to include pace‑adjusted stats, opponent defensive ratings, usage rate, true shooting percentage, etc.  A Medium post on NBA player scoring prediction illustrates using minutes per game, field goal attempts, shooting percentages and other features【668753447422985†L73-L151】.
* Incorporate opponent data from ESPN’s scoreboard endpoint or NBA.com’s matchup statistics.  The `data_fetching.get_espn_scoreboard` and `get_espn_game_summary` functions provide a starting point for extracting team defence metrics.
* Implement time‑series models (e.g., ARIMA, Prophet) for forecasting, or combine ensemble methods to further improve accuracy【778169439784453†L248-L262】.

### Disclaimer

This project is intended for educational purposes and should not be used for actual betting.  Model outputs are predictions with inherent uncertainty.

---
Feel free to fork and adapt the dashboard to your specific research needs.  Contributions are welcome!