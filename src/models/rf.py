"""Random Forest wrapper with channel importance tracking."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


class RandomForestWrapper:
    """
    Thin wrapper around sklearn RandomForest that:
      - stores per-channel importances after fit (needed for electrode sweep)
      - provides save/load with both model and importances
    """

    def __init__(self, n_estimators: int = 200, max_depth=None,
                 min_samples_split: int = 2, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        self.channel_importances: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        # Infer shape: features = n_channels * n_bands
        # We try n_bands=5 first (standard bandpower), then fall back to flat
        n_features = X.shape[1]
        if n_features % 5 == 0:
            n_ch = n_features // 5
            self.channel_importances = (
                self.model.feature_importances_.reshape(n_ch, 5).mean(axis=1)
            )
        else:
            # Non-bandpower features: store flat importances
            self.channel_importances = self.model.feature_importances_
        return self

    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)

    @property
    def feature_importances_(self): return self.model.feature_importances_

    def save(self, path: Path):
        path = Path(path)
        joblib.dump(self.model, path)
        if self.channel_importances is not None:
            np.save(path.with_suffix(".channel_imp.npy"), self.channel_importances)

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        obj = cls()
        obj.model = joblib.load(path)
        imp_path = path.with_suffix(".channel_imp.npy")
        if imp_path.exists():
            obj.channel_importances = np.load(imp_path)
        obj.is_fitted = True
        return obj