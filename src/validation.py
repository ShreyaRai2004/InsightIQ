def validate(df):
    issues=[]
    if df.empty: issues.append("Dataset is empty.")
    if df.columns.duplicated().any(): issues.append("Duplicate column names found.")
    for c in ["Revenue","Profit"]:
        if c not in df.columns: issues.append(f"Required column missing: {c}")
    if len(df)<50: issues.append("Use at least 50 rows for model training.")
    return issues
