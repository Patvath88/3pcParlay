"""
models.py — Safe & Cloud-Compatible Version
Includes:
- CatBoost disabled
- LightGBM optional
- XGBoost optional
- Neural net optional
- Full error protection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional Models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# CatBoost is disabled due to incompatibility
cb = None

# Neural net
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
    name: str
    model: Any
    mae: Optional[float] = None
    mse: Optional[float] = None


@dataclass
class ModelManager:
    use_neural: bool = False
    random_state: Optional[int] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict, init=False)

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        self.models.clear()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        def _evaluate(name, model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            self.models[name] = ModelInfo(name, model, mae, mse)

        # Base Models
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

        # Optional Models
        if xgb is not None:
            _evaluate("XGBoost", xgb.XGBRegressor(
                n_estimators=250,
                learning_rate=0.06,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state or 0
            ))

        if lgb is not None:
            _evaluate("LightGBM", lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                objective="regression",
                random_state=self.random_state or 0
            ))

        # Neural Net
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
            model.fit(X_train_nn, y_train, epochs=40, batch_size=32, verbose=0)

            preds = model.predict(X_test_nn).flatten()
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))

            self.models["NeuralNetwork"] = ModelInfo("NeuralNetwork", model, mae, mse)

        return self.models

    def predict(self, X_new: pd.DataFrame) -> Dict[str, np.ndarray]:
        results = {}
        for name, info in self.models.items():
            try:
                results[name] = info.model.predict(X_new)
            except Exception:
                continue
        return results

    def best_model(self) -> Optional[ModelInfo]:
        if not self.models:
            return None
        return min(self.models.values(), key=lambda m: m.mae)
