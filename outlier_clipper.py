"""Transformador usado no pipeline salvo (modelo_evasao_pipeline.joblib).

Precisa ser idêntico à classe definida no notebook, porque o joblib guarda
apenas uma referência ao nome da classe.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorização baseada em 1.5 * IQR (limites calculados só no treino)."""

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            if X_df[col].nunique() > 2:
                q25 = X_df[col].quantile(0.25)
                q75 = X_df[col].quantile(0.75)
                iqr = q75 - q25
                self.lower_bounds_[col] = q25 - self.factor * iqr
                self.upper_bounds_[col] = q75 + self.factor * iqr
            else:
                self.lower_bounds_[col] = -np.inf
                self.upper_bounds_[col] = np.inf
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df.values
