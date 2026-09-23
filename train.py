import argparse
from src.data_loader import load_data
from src.features import add_features
from src.model import train
p=argparse.ArgumentParser(); p.add_argument("--file",required=True); a=p.parse_args()
r=train(add_features(load_data(a.file)))
print("\nINSIGHTIQ MODEL RESULTS\n")
print(r["comparison"].round(4).to_string(index=False))
print(f"\nBest Model: {r['best_model_name']}")
m=r["metrics"][r["best_model_name"]]
print(f"Test MAE : {m['Test MAE']:.2f}")
print(f"Test RMSE: {m['Test RMSE']:.2f}")
print(f"Test R2  : {m['Test R2']:.4f}")
print("\nModel saved to models/best_model.joblib")
