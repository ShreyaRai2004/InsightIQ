# InsightIQ — Predictive Business Analytics Platform

InsightIQ is a business analytics and machine learning platform for analyzing business performance, predicting profit, explaining predictions, detecting anomalies, forecasting future profit, and evaluating what-if scenarios.

**Live Demo:** https://insightiq-kdeewsolmwwxuduitkdrwe.streamlit.app/  
**GitHub:** https://github.com/ShreyaRai2004/InsightIQ

## Features

- Business performance analysis and KPI dashboard
- CSV and Excel data upload
- Data validation and preprocessing
- Profit prediction using machine learning
- Linear Regression, Random Forest, and XGBoost comparison
- SHAP-based model explainability
- Isolation Forest anomaly detection
- Monthly profit forecasting
- Risk analysis
- What-if scenario simulation
- Interactive Plotly visualizations

## Machine Learning

The project uses a chronological **60% training / 20% validation / 20% test** split. Models are evaluated using **MAE, RMSE, and R²**, with validation performance used for model selection.

To reduce target leakage, `Profit`, `Revenue`, `Cost`, `Operating Cost`, and `Gross_Margin` are excluded from prediction inputs.

### Model Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | ₹30,141.09 | ₹48,437.57 | 0.7548 |
| Random Forest | ₹13,578.71 | ₹21,573.81 | 0.9514 |
| XGBoost | ₹12,035.95 | ₹19,654.00 | 0.9596 |

**Selected Model:** XGBoost  
**Validation R²:** 0.9377

## Explainability

SHAP/TreeSHAP is used to identify the features that contribute most to model predictions and display global feature importance.

## Technology Stack

**Python · Streamlit · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Plotly · Statsmodels · Joblib · OpenPyXL**

## Project Structure

```text
InsightIQ/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample/
│       └── InsightIQ_Business_Data.xlsx
├── models/
│   ├── best_model.joblib
│   └── model_comparison.csv
└── src/
    ├── data_loader.py
    ├── validation.py
    ├── features.py
    ├── model.py
    ├── analysis.py
    └── explain.py
```

## Dataset

The repository includes a synthetic business dataset for demonstration with fields such as Date, Product, Category, Region, Quantity, Revenue, Cost, Discount, Marketing Spend, Customer Type, Operating Cost, and Profit.

Custom CSV and Excel datasets are also supported.

## License

MIT License

## Author

**Shreya R Sai**

GitHub: https://github.com/ShreyaRai2004
