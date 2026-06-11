import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
)
from sklearn.impute import SimpleImputer
import pickle

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a pandas DataFrame from Parquet or Pickle format."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Save a pandas DataFrame in Parquet or Pickle format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        with open(path, "wb") as f:
            pickle.dump(df, f)

def get_dataframe_preview(df: pd.DataFrame, n: int = 10) -> list:
    """Get JSON-serializable list of dictionaries representing the preview of a DataFrame."""
    # Convert NaNs and infinite values to None for clean JSON serialization
    df_clean = df.head(n).replace({np.nan: None, np.inf: None, -np.inf: None})
    return df_clean.to_dict(orient="records")

def apply_preprocessing_operations(df: pd.DataFrame, operations: list) -> pd.DataFrame:
    """
    Apply a list of preprocessing operations to a DataFrame sequentially.
    Operations format:
    [
        {"op": "drop_cols", "columns": ["id", "name"]},
        {"op": "fill_missing", "strategy": "mean/median/mode/drop", "columns": ["age"]},
        {"op": "scale", "method": "standard/minmax/robust", "columns": ["age", "fare"]},
        {"op": "encode", "method": "label/onehot", "columns": ["gender"]},
        {"op": "remove_outliers", "column": "fare"}
    ]
    """
    df = df.copy()
    
    for op_data in operations:
        op = op_data.get("op")
        
        if op == "drop_cols":
            cols = op_data.get("columns", [])
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                
        elif op == "fill_missing":
            strategy = op_data.get("strategy", "mean")
            cols = op_data.get("columns", [])
            existing_cols = [c for c in cols if c in df.columns]
            
            if existing_cols:
                if strategy == "drop":
                    df = df.dropna(subset=existing_cols)
                else:
                    # Map SimpleImputer strategies
                    imputer_strategy = "most_frequent" if strategy == "mode" else strategy
                    imputer = SimpleImputer(strategy=imputer_strategy)
                    df[existing_cols] = imputer.fit_transform(df[existing_cols])
                    
        elif op == "scale":
            method = op_data.get("method", "standard")
            cols = op_data.get("columns", [])
            existing_cols = [c for c in cols if c in df.columns]
            
            if existing_cols:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                elif method == "maxabs":
                    scaler = MaxAbsScaler()
                elif method == "normalizer":
                    scaler = Normalizer()
                elif method == "quantile":
                    scaler = QuantileTransformer(random_state=42)
                elif method == "power":
                    scaler = PowerTransformer()
                else:
                    continue
                df[existing_cols] = scaler.fit_transform(df[existing_cols])
                
        elif op == "encode":
            method = op_data.get("method", "label")
            cols = op_data.get("columns", [])
            existing_cols = [c for c in cols if c in df.columns]
            
            if existing_cols:
                if method == "label":
                    for col in existing_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                elif method == "onehot":
                    df = pd.get_dummies(df, columns=existing_cols)
                    
        elif op == "remove_outliers":
            method = op_data.get("method", "iqr")
            col = op_data.get("column")
            cols = op_data.get("columns", [])
            
            if method == "iqr":
                if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == "zscore":
                threshold = op_data.get("threshold", 3.0)
                if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                    if col_std > 0:
                        z_scores = (df[col] - col_mean) / col_std
                        df = df[z_scores.abs() <= threshold]
            elif method in ["isolation_forest", "lof"]:
                numeric_cols = [c for c in cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
                if not numeric_cols:
                    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                
                if numeric_cols:
                    clean_data = df[numeric_cols].dropna()
                    if len(clean_data) > 10:
                        contamination = op_data.get("contamination", 0.05)
                        if method == "isolation_forest":
                            from sklearn.ensemble import IsolationForest
                            iso = IsolationForest(contamination=contamination, random_state=42)
                            preds = iso.fit_predict(clean_data)
                        else:
                            from sklearn.neighbors import LocalOutlierFactor
                            n_neighbors = min(20, len(clean_data) - 1)
                            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                            preds = lof.fit_predict(clean_data)
                        
                        inlier_index = clean_data.index[preds == 1]
                        nan_index = df.index[df[numeric_cols].isna().any(axis=1)]
                        df = df.loc[inlier_index.union(nan_index)].sort_index()
                
    return df
