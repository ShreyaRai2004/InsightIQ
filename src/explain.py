import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.model import prep


def explain(df, result):
    """
    Generate global SHAP feature importance.

    For XGBoost, this uses XGBoost's native TreeSHAP
    implementation through pred_contribs=True.
    """

    pipe = result["model"]

    preprocessor = pipe.named_steps["pre"]
    estimator = pipe.named_steps["model"]

    # --------------------------------------------------
    # Prepare the exact same features used during training
    # --------------------------------------------------

    x = prep(df)

    # --------------------------------------------------
    # Apply the fitted preprocessing pipeline
    # --------------------------------------------------

    transformed = preprocessor.transform(x)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    X = np.asarray(
        transformed,
        dtype=np.float64
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got shape {X.shape}"
        )

    # --------------------------------------------------
    # Feature names
    # --------------------------------------------------

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Feature mismatch: "
            f"model input has {X.shape[1]} columns, "
            f"but preprocessing produced "
            f"{len(feature_names)} feature names."
        )

    # --------------------------------------------------
    # Use a reasonable sample size
    # --------------------------------------------------

    sample = X[:300]

    if len(sample) == 0:
        raise ValueError(
            "No rows available for SHAP explanation."
        )

    # ==================================================
    # XGBOOST
    # ==================================================

    if isinstance(estimator, xgb.XGBRegressor):

        booster = estimator.get_booster()

        # XGBoost's native TreeSHAP.
        #
        # pred_contribs returns:
        #   feature_1
        #   feature_2
        #   ...
        #   feature_n
        #   bias/base value
        #
        # The final column is NOT a feature, so remove it.

        contributions = booster.predict(
            xgb.DMatrix(sample),
            pred_contribs=True
        )

        contributions = np.asarray(
            contributions,
            dtype=np.float64
        )

        if contributions.ndim != 2:
            raise ValueError(
                "Unexpected XGBoost SHAP output shape: "
                f"{contributions.shape}"
            )

        # Remove the bias/base-value column
        shap_values = contributions[:, :-1]

    # ==================================================
    # OTHER TREE MODELS
    # ==================================================

    elif hasattr(estimator, "feature_importances_"):

        explainer = shap.TreeExplainer(
            estimator
        )

        shap_values = explainer.shap_values(
            sample
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values,
            dtype=np.float64
        )

    # ==================================================
    # LINEAR MODEL
    # ==================================================

    else:

        explainer = shap.Explainer(
            estimator,
            sample
        )

        shap_values = explainer(
            sample
        ).values

        shap_values = np.asarray(
            shap_values,
            dtype=np.float64
        )

    # --------------------------------------------------
    # Validate SHAP dimensions
    # --------------------------------------------------

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(
            1, -1
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP output does not match feature names: "
            f"{shap_values.shape[1]} SHAP features vs "
            f"{len(feature_names)} feature names."
        )

    # --------------------------------------------------
    # Global SHAP importance
    # --------------------------------------------------

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    result_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean |SHAP|": importance
        }
    )

    result_df = (
        result_df
        .sort_values(
            "Mean |SHAP|",
            ascending=False
        )
        .head(20)
        .reset_index(drop=True)
    )

    return result_df