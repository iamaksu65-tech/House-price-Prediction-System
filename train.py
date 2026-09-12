"""
train_and_save.py
Standalone script to train the House Price Prediction pipeline,
evaluate all models, and serialise the best pipeline to Model/best_model.pkl.

Run from the project root:
    venv\\Scripts\\python Model\\train_and_save.py
"""

import sys
import os

# Make sure the project root is on the Python path so we can import preprocessing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Import our picklable custom transformers
from Model.preprocessing import HousingFeatureEngineer, HousingCategoricalEncoder

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "Housing.csv")
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Train / Test Split on RAW data (before any transformation)
# ---------------------------------------------------------------------------
X_raw = df.drop(columns=["price"])
y = df["price"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ---------------------------------------------------------------------------
# 3. Outlier removal on TRAINING set only (after split, before pipeline)
# ---------------------------------------------------------------------------
def remove_outliers_iqr(X_tr, y_tr):
    """Remove outlier rows from the training set using the IQR method."""

    def iqr_mask(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)

    mask = iqr_mask(X_tr["area"]) & iqr_mask(y_tr)
    removed = (~mask).sum()
    print(f"Outlier removal: {removed} rows removed from training set "
          f"({removed / len(X_tr) * 100:.2f}% of training data).")
    return X_tr[mask], y_tr[mask]


X_train_clean, y_train_clean = remove_outliers_iqr(X_train_raw, y_train)

# ---------------------------------------------------------------------------
# 4. Quick Manual Preprocessing for Baseline Evaluation
#    (to compare models before wrapping in the production pipeline)
# ---------------------------------------------------------------------------
fe = HousingFeatureEngineer()
enc = HousingCategoricalEncoder()

NUM_COLS = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "area_per_bedroom", "bath_to_bed_ratio", "total_rooms", "luxury_score",
]

# Fit/transform training; transform-only on test
X_tr_eng  = enc.transform(fe.transform(X_train_clean))
X_te_eng  = enc.transform(fe.transform(X_test_raw))

scaler = StandardScaler()
X_tr_scaled = X_tr_eng.copy()
X_te_scaled  = X_te_eng.copy()
X_tr_scaled[NUM_COLS] = scaler.fit_transform(X_tr_eng[NUM_COLS])
X_te_scaled[NUM_COLS]  = scaler.transform(X_te_eng[NUM_COLS])

# ---------------------------------------------------------------------------
# 5. Baseline Model Comparison
# ---------------------------------------------------------------------------
baseline_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(random_state=42),
    "XGBoost":           xgb.XGBRegressor(random_state=42, n_jobs=-1),
}

results = {}
print("\n--- Baseline Model Comparison ---")
for name, model in baseline_models.items():
    model.fit(X_tr_scaled, y_train_clean)
    y_pred = model.predict(X_te_scaled)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"  {name:22s}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R2={r2:.4f}")

best_name = max(results, key=lambda n: results[n]["R2"])
print(f"\nBest baseline model: {best_name}  (R2={results[best_name]['R2']:.4f})")

# ---------------------------------------------------------------------------
# 6. Hyperparameter Tuning on best model
# ---------------------------------------------------------------------------
print("\n--- Hyperparameter Tuning ---")
if best_name == "Random Forest":
    param_grid = {
        "n_estimators":    [100, 150, 200],
        "max_depth":       [8, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    tuned_estimator = RandomForestRegressor(random_state=42)
elif best_name == "XGBoost":
    param_grid = {
        "n_estimators":  [100, 150, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    tuned_estimator = xgb.XGBRegressor(random_state=42, n_jobs=-1)
else:
    # Linear Regression doesn't benefit from grid search; fall back to RF
    print("  (Linear Regression selected – defaulting to tune Random Forest)")
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth":    [8, 12, None],
    }
    tuned_estimator = RandomForestRegressor(random_state=42)

gs = GridSearchCV(
    estimator=tuned_estimator,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_tr_scaled, y_train_clean)
print(f"  Best params : {gs.best_params_}")
print(f"  CV R2       : {gs.best_score_:.4f}")

y_tuned_pred = gs.best_estimator_.predict(X_te_scaled)
print(f"  Test R2     : {r2_score(y_test, y_tuned_pred):.4f}")
print(f"  Test RMSE   : {np.sqrt(mean_squared_error(y_test, y_tuned_pred)):,.0f}")

# ---------------------------------------------------------------------------
# 7. Build & Fit Production Pipeline
#    Accepts RAW input -> engineer -> encode -> scale -> predict
# ---------------------------------------------------------------------------
print("\n--- Building Production Pipeline ---")

scaling_ct = ColumnTransformer(
    transformers=[("scale", StandardScaler(), NUM_COLS)],
    remainder="passthrough",
)

production_pipeline = Pipeline(steps=[
    ("feat_engineer", HousingFeatureEngineer()),
    ("cat_encoder",   HousingCategoricalEncoder()),
    ("scaler",        scaling_ct),
    ("model",         gs.best_estimator_),
])

# Retrain the full pipeline on raw outlier-cleaned training data
production_pipeline.fit(X_train_clean, y_train_clean)

# Evaluate on raw test data
pipe_preds = production_pipeline.predict(X_test_raw)
pipe_r2    = r2_score(y_test, pipe_preds)
pipe_rmse  = np.sqrt(mean_squared_error(y_test, pipe_preds))
print(f"  Pipeline Test R2   : {pipe_r2:.4f}")
print(f"  Pipeline Test RMSE : {pipe_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 8. Save Pipeline
# ---------------------------------------------------------------------------
SAVE_PATH = os.path.join(PROJECT_ROOT, "Model", "best_model.pkl")
joblib.dump(production_pipeline, SAVE_PATH)
print(f"\nPipeline saved -> {SAVE_PATH}")
print("Training complete.")
