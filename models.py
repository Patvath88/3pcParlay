"""
models.py — Cached Multi-Model Engine
Fast + Streamlit-Safe + Shows All Predictions + MAE/MSE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Optional XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ModelInfo:
    name: str
    model: Any
    mae: float
    mse: float
    prediction: Optional[float] = None


@dataclass
class ModelManager:
    random_state: Optional[int] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict, init=False)

    # ============================================================
    # TRAIN ALL MODELS
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.models.clear()

        # Protect against extremely small datasets
        if len(X) < 3:
            raise ValueError(
                f"Not enough data to train models. Need >= 3 rows, have {len(X)}."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )

        def _evaluate(name, model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)

            self.models[name] = ModelInfo(
                name=name,
                model=model,
                mae=float(mae),
                mse=float(mse)
            )

        # ============================================================
        # MODELS (KNN REMOVED)
        # ============================================================
        _evaluate("LinearRegression", LinearRegression())
        _evaluate("Ridge", Ridge(random_state=self.random_state))
        _evaluate("Lasso", Lasso(random_state=self.random_state))
        _evaluate("DecisionTree", DecisionTreeRegressor(random_state=self.random_state))
        _evaluate(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=250,
                random_state=self.random_state,
                n_jobs=-1
            )
        )
        _evaluate("GradientBoosting", GradientBoostingRegressor(random_state=self.random_state))
        _evaluate("SVR", SVR(kernel="rbf"))

        # Optional XGBoost if available
        if xgb is not None:
            _evaluate(
                "XGBoost",
                xgb.XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.07,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state or 0,
                    tree_method="hist",
                )
            )

        return self.models

    # ============================================================
    # PREDICT ON NEW INPUT
    # ============================================================
    def predict(self, X_new: pd.DataFrame) -> Dict[str, float]:
        preds = {}
        for name, info in self.models.items():
            try:
                pred = float(info.model.predict(X_new)[0])
                info.prediction = pred
                preds[name] = pred
            except:
                continue
        return preds

    # ============================================================
    # GET BEST MODEL
    # ============================================================
    def best_model(self) -> Optional[ModelInfo]:
        if not self.models:
            return None
        return min(self.models.values(), key=lambda m: m.mae)
