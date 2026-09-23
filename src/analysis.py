import pandas as pd
from sklearn.ensemble import IsolationForest


def anomalies(df):
    """
    Detect unusual business records using Isolation Forest.
    """

    cols = [
        c
        for c in [
            "Revenue",
            "Cost",
            "Profit",
            "Quantity",
            "Discount",
            "Marketing Spend",
            "Operating Cost",
        ]
        if c in df.columns
    ]

    if not cols:
        raise ValueError(
            "No suitable numerical columns were found for anomaly detection."
        )

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median())

    if len(x) < 20:
        raise ValueError(
            "At least 20 rows are recommended for anomaly detection."
        )

    detector = IsolationForest(
        contamination=0.02,
        random_state=42,
        n_estimators=200,
    )

    predictions = detector.fit_predict(x)

    out = df.copy()

    # 1 = normal, -1 = anomaly
    out["Anomaly"] = predictions

    # Easier for the UI to use
    out["Is Anomaly"] = out["Anomaly"] == -1

    return out