"""
models.py — Fast Version
Only XGBoost + RandomForest are used for maximum speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None


@dataclass
class ModelInfo:
    name: str
    model: Any
    mae: Optional[float] = None
    mse: Optional[float] = None


@dataclass
class ModelManager:
    random_state: Optional[int] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict, init=False)

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train only RandomForest and XGBoost."""
        self.models.clear()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        def _evaluate(name, model):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            self.models[name] = ModelInfo(
                name=name,
                model=model,
                mae=float(mean_absolute_error(y_test, preds)),
                mse=float(mean_squared_error(y_test, preds)),
            )

        # --------------------------
        # Random Forest (FAST)
        # --------------------------
        _evaluate("RandomForest", RandomForestRegressor(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        ))

        # --------------------------
        # XGBoost (FAST & ACCURATE)
        # --------------------------
        if xgb is not None:
            _evaluate("XGBoost", xgb.XGBRegressor(
                n_estimators=250,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state or 0,
                tree_method="hist"  # FASTER
            ))

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
