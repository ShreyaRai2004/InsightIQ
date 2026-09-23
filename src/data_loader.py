import pandas as pd
def load_data(file):
    name=str(getattr(file,"name",file)).lower()
    if name.endswith(".csv"): df=pd.read_csv(file)
    elif name.endswith((".xlsx",".xls")): df=pd.read_excel(file)
    else: raise ValueError("Use CSV, XLSX or XLS.")
    df.columns=[str(c).strip() for c in df.columns]
    return df
