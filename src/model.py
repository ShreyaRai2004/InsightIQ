from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

MODEL = Path("models/best_model.joblib")

# These columns are outcomes or direct accounting components of Profit.
# Keeping them out of the prediction model prevents target leakage.
LEAKAGE_COLUMNS = ["Profit", "Revenue", "Cost", "Operating Cost", "Gross_Margin"]


def prep(df):
    """Build prediction-time features using information available before profit is known."""
    x = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore").copy()

    if "Date" in x.columns:
        d = pd.to_datetime(x.pop("Date"), errors="coerce")
        x["Year"] = d.dt.year
        x["Month"] = d.dt.month
        x["Quarter"] = d.dt.quarter
        x["DayOfWeek"] = d.dt.dayofweek

    return x


def _pipeline(x, estimator):
    cats = x.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in x.columns if c not in cats]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cats)
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def train(df):
    if "Profit" not in df.columns:
        raise ValueError("A Profit column is required for supervised training.")

    work = df.copy()
    if "Date" in work.columns:
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.sort_values("Date", kind="stable").reset_index(drop=True)

    x = prep(work)
    y = pd.to_numeric(work["Profit"], errors="coerce")
    valid = y.notna()
    x, y = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    if len(x) < 30:
        raise ValueError("At least 30 valid rows are recommended for predictive modeling.")

    # Chronological split: 60% train, 20% validation, 20% final test.
    # The validation set selects the model; the final test set remains untouched
    # until after model selection, reducing optimistic evaluation.
    n = len(x)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    if train_end < 10 or val_end <= train_end or val_end >= n:
        raise ValueError("Not enough rows for chronological train/validation/test splits.")

    x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
    x_val, y_val = x.iloc[train_end:val_end], y.iloc[train_end:val_end]
    x_test, y_test = x.iloc[val_end:], y.iloc[val_end:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=350, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            n_jobs=2, objective="reg:squarederror"
        )
    }

    rows = []
    fitted = {}
    validation_r2 = {}
    final_metrics = {}

    for name, estimator in models.items():
        pipe = _pipeline(x_train, estimator)
        pipe.fit(x_train, y_train)

        val_pred = pipe.predict(x_val)
        test_pred = pipe.predict(x_test)
        train_pred = pipe.predict(x_train)

        val_r2 = r2_score(y_val, val_pred)
        validation_r2[name] = val_r2
        fitted[name] = pipe

        final_metrics[name] = {
            "Train MAE": mean_absolute_error(y_train, train_pred),
            "Test MAE": mean_absolute_error(y_test, test_pred),
            "Train RMSE": mean_squared_error(y_train, train_pred) ** 0.5,
            "Test RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
            "Train R2": r2_score(y_train, train_pred),
            "Test R2": r2_score(y_test, test_pred),
            "Validation R2": val_r2,
        }

    # Select using validation performance, not the final test set.
    best = max(validation_r2, key=validation_r2.get)

    # Refit every model on train + validation for comparable final metrics.
    x_dev = pd.concat([x_train, x_val], ignore_index=True)
    y_dev = pd.concat([y_train, y_val], ignore_index=True)
    for name, estimator in models.items():
        final_pipe = _pipeline(x_dev, estimator)
        final_pipe.fit(x_dev, y_dev)
        fitted[name] = final_pipe
        dev_pred = final_pipe.predict(x_dev)
        test_pred = final_pipe.predict(x_test)
        final_metrics[name] = {
            "Train MAE": mean_absolute_error(y_dev, dev_pred),
            "Test MAE": mean_absolute_error(y_test, test_pred),
            "Train RMSE": mean_squared_error(y_dev, dev_pred) ** 0.5,
            "Test RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
            "Train R2": r2_score(y_dev, dev_pred),
            "Test R2": r2_score(y_test, test_pred),
            "Validation R2": validation_r2[name],
        }
        final_metrics[name]["Overfit Warning"] = (
            final_metrics[name]["Train R2"] - final_metrics[name]["Test R2"]
        ) > 0.12

    final_pipe = fitted[best]

    rows = [{"Model": name, **final_metrics[name]} for name in models]
    comparison = pd.DataFrame(rows)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_pipe,
        "best_model_name": best,
        "metrics": final_metrics,
        "prediction_features": list(x.columns),
        "excluded_target_leakage_columns": LEAKAGE_COLUMNS,
        "split": "chronological 60% train / 20% validation / 20% test",
    }, MODEL)
    comparison.to_csv("models/model_comparison.csv", index=False)

    return {
        "model": final_pipe,
        "best_model_name": best,
        "metrics": final_metrics,
        "comparison": comparison,
        "prediction_features": list(x.columns),
        "excluded_target_leakage_columns": LEAKAGE_COLUMNS,
    }


def load_saved():
    if not MODEL.exists():
        return None
    data = joblib.load(MODEL)
    comparison_path = Path("models/model_comparison.csv")
    if comparison_path.exists():
        data["comparison"] = pd.read_csv(comparison_path)
    return data
