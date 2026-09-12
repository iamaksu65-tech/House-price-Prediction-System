"""
preprocessing.py
Reusable, picklable Scikit-Learn custom transformers for the
House Price Prediction project.

Keeping transformers in a standalone module (not defined inline
inside a Jupyter notebook) is the only reliable way to serialise
them with joblib/pickle, because pickle needs to look up the class
by its fully-qualified module path at load-time.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create engineered features from raw housing columns.

    New features added:
    - area_per_bedroom  : Total area divided by bedroom count (luxury indicator).
    - bath_to_bed_ratio : Bathrooms-to-bedrooms ratio (comfort indicator).
    - total_rooms       : Sum of bedrooms + bathrooms + stories.
    - luxury_score      : Count of premium amenities (0-6).
    """

    # Binary columns that feed the luxury score
    _BINARY_COLS = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea",
    ]

    def fit(self, X, y=None):          # noqa: D401
        """No fitting required; transformer is stateless."""
        return self

    def transform(self, X):
        X_eng = X.copy()

        # 1. Area per bedroom
        X_eng["area_per_bedroom"] = (
            X_eng["area"] / (X_eng["bedrooms"] + 1e-5)
        )

        # 2. Bathroom-to-bedroom ratio
        X_eng["bath_to_bed_ratio"] = (
            X_eng["bathrooms"] / (X_eng["bedrooms"] + 1e-5)
        )

        # 3. Total rooms
        X_eng["total_rooms"] = (
            X_eng["bedrooms"] + X_eng["bathrooms"] + X_eng["stories"]
        )

        # 4. Luxury score – handle both string ('yes'/'no') and int (1/0)
        temp_bin = pd.DataFrame(index=X_eng.index)
        for col in self._BINARY_COLS:
            if pd.api.types.is_numeric_dtype(X_eng[col]):
                temp_bin[col] = X_eng[col]
            else:
                temp_bin[col] = X_eng[col].map({"yes": 1, "no": 0}).fillna(0)

        X_eng["luxury_score"] = temp_bin.sum(axis=1)
        return X_eng


class HousingCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical string columns to integers.

    - Binary yes/no columns  -> 1 / 0
    - furnishingstatus (ordinal): unfurnished=0, semi-furnished=1, furnished=2
    """

    _BINARY_COLS = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea",
    ]

    _FURNISH_MAP = {
        "unfurnished": 0,
        "semi-furnished": 1,
        "furnished": 2,
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_enc = X.copy()

        # Binary mappings
        for col in self._BINARY_COLS:
            if not pd.api.types.is_numeric_dtype(X_enc[col]):
                X_enc[col] = (
                    X_enc[col].map({"yes": 1, "no": 0})
                    .fillna(0)
                    .astype(int)
                )
            else:
                X_enc[col] = X_enc[col].astype(int)

        # Furnishing status
        if "furnishingstatus" in X_enc.columns:
            if not pd.api.types.is_numeric_dtype(X_enc["furnishingstatus"]):
                X_enc["furnishingstatus"] = (
                    X_enc["furnishingstatus"]
                    .map(self._FURNISH_MAP)
                    .fillna(1)
                    .astype(int)
                )
            else:
                X_enc["furnishingstatus"] = X_enc["furnishingstatus"].astype(int)

        return X_enc
