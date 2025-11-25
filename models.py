"""
models.py
==========

This module defines a `ModelManager` class responsible for training and
using multiple regression models to predict player statistics. The
objective of the dashboard is to forecast points, rebounds, assists
and other box score metrics for a player's next game. To that end
we implement a suite of algorithms drawn from scikit-learn and other
popular libraries that are compatible with Streamlit.

Neural networks, XGBoost, and LightGBM remain optional.
CatBoost is *disabled entirely* due to incompatibility with Python 3.13
and Streamlit Cloud deployment environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# -----------------------------
# Core scikit-learn models
# -----------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Optional models
# -----------------------------
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ---- IMPORTANT ----
# CatBoost is manually disabled because it crashes on Streamlit Cloud & Python 3.13.
cb = None
# --------------------

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    Sequential = None
    Dense = None
    Dropout = None
    Adam = None


@dataclass
class ModelInfo:
    """Container to store a trained model and its evaluation metrics."""
    name: str
    model: Any
    mae: Optional[float] = None
    mse: Optional[float] = None
    fit_time: Optional[float] = None


@dataclass
class ModelManager:
    """Manage training and predictions for multiple regression models."""
    use_neural: bool = False
    random_state: Optional[int] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict, init=False)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, ModelInfo]:
        """Train a suite of regression models and evaluate them."""
        self.models.clear()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Helper to evaluate and store results
        def _evaluate(name: str, model: Any) -> None:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            self.models[name] = ModelInfo(name=name, model=model, mae=mae, mse=mse)

        # -----------------------------
        # Standard models
        # -----------------------------
        _evaluate("LinearRegression", LinearRegression())
        _evaluate("Ridge", Ridge(random_state=self.random_state))
        _evaluate("Lasso", Lasso(random_state=self.random_state))

        _evaluate("DecisionTree", DecisionTreeRegressor(random_state=self.random_state))
        _evaluate("RandomForest", RandomForestRegressor(
            n_estimators=200, random_state=self.random_state, n_jobs=-1))

        _evaluate("GradientBoosting", GradientBoostingRegressor(
            random_state=self.random_state))

        _evaluate("KNN", KNeighborsRegressor())
        _evaluate("SVR", SVR(kernel="rbf"))

        # -----------------------------
        # XGBoost (optional)
        # -----------------------------
        if xgb is not None:
            params = {
                "n_estimators": 300,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state or 0,
            }
            _evaluate("XGBoost", xgb.XGBRegressor(**params))

        # -----------------------------
        # LightGBM (optional)
        # -----------------------------
        if lgb is not None:
            params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "objective": "regression",
                "random_state": self.random_state or 0,
            }
            _evaluate("LightGBM", lgb.LGBMRegressor(**params))

        # -----------------------------
        # CatBoost — DISABLED
        # -----------------------------
        # (Do nothing)

        # -----------------------------
        # Neural Network (optional)
        # -----------------------------
        if self.use_neural and Sequential is not None:
            # Normalize
            X_train_nn = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
            X_test_nn = (X_test - X_train.mean()) / (X_train.std() + 1e-8)

            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(X_train_nn.shape[1],)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            model.fit(
                X_train_nn,
                y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
            )

            preds = model.predict(X_test_nn).flatten()
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))

            self.models["NeuralNetwork"] = ModelInfo(
                name="NeuralNetwork", model=model, mae=mae, mse=mse
            )

        return self.models

    def predict(self, X_new: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions for new observations."""
        results: Dict[str, np.ndarray] = {}
        for name, info in self.models.items():
            model = info.model
            try:
                if name == "NeuralNetwork" and self.use_neural and Sequential is not None:
                    X_norm = (X_new - X_new.mean()) / (X_new.std() + 1e-8)
                    preds = model.predict(X_norm).flatten()
                else:
                    preds = model.predict(X_new)

                results[name] = preds
            except Exception:
                continue
        return results

    def best_model(self) -> Optional[ModelInfo]:
        """Return model with lowest MAE."""
        if not self.models:
            return None

        return min(
            self.models.values(),
            key=lambda m: m.mae if m.mae is not None else float("inf"),
        )
