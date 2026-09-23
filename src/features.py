import pandas as pd, numpy as np
def add_features(df):
    x=df.copy()
    if "Date" in x:
        x["Date"]=pd.to_datetime(x["Date"],errors="coerce")
        x["Year"]=x.Date.dt.year; x["Month"]=x.Date.dt.month
        x["Quarter"]=x.Date.dt.quarter; x["DayOfWeek"]=x.Date.dt.dayofweek
    if "Revenue" in x and "Quantity" in x:
        x["Revenue_per_Unit"]=x.Revenue/x.Quantity.replace(0,np.nan)
    if "Revenue" in x and "Cost" in x:
        x["Cost_to_Revenue"]=x.Cost/x.Revenue.replace(0,np.nan)
    if "Marketing Spend" in x and "Revenue" in x:
        x["Marketing_to_Revenue"]=x["Marketing Spend"]/x.Revenue.replace(0,np.nan)
    return x.replace([np.inf,-np.inf],np.nan)
