# InsightIQ — Predictive Business Analytics Platform

InsightIQ is an end-to-end business analytics and machine learning platform for analyzing business performance, estimating future profit, explaining model predictions, detecting anomalies, forecasting trends, assessing risk, and testing what-if scenarios.

## What makes the prediction setup realistic?

Profit is not predicted from values that directly calculate the same profit record. The ML pipeline excludes `Revenue`, `Cost`, `Operating Cost`, `Gross_Margin`, and the target `Profit` from prediction features. Instead, it uses operational inputs such as quantity, discount, marketing spend, product/category, region, customer type, and date-derived seasonality features.

The training workflow also uses a chronological **60% train / 20% validation / 20% test** split. The validation period selects the model, while the final test period is kept for the final evaluation. This better reflects a business scenario where future observations should not be used to train a model for the past.

## ML models

- Linear Regression — interpretable baseline
- Random Forest Regressor — nonlinear ensemble model
- XGBoost Regressor — gradient-boosted model

Models are compared using **MAE, RMSE, and R²**. The selected model is chosen using validation R², not the final test set.

## Other modules

- CSV / Excel upload and validation
- Business KPI dashboard and interactive charts
- SHAP-based prediction explainability
- Isolation Forest anomaly detection
- Monthly profit forecasting
- Rule-based profitability risk analysis
- What-if simulation without retraining the model

## Run

```cmd
cd C:\ShreyaResumeProjects\AIML\InsightIQ\InsightIQ
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py --file data\sample\InsightIQ_Business_Data.xlsx
streamlit run app.py
```

If `.venv` already exists, activate it and run the last two commands.

The included Excel file is synthetic demonstration data. Use real company data only when you are authorized to use it, and retrain the model on that company's historical data.

Model metrics in the README or resume should always come from an actual training run; do not present example metrics as measured results.
