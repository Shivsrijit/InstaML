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
                
        elif op == "drop_duplicates":
            df = df.drop_duplicates()
                
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
                        
        elif op == "pca":
            cols = op_data.get("columns", [])
            n_components = int(op_data.get("n_components", 2))
            existing_cols = [c for c in cols if c in df.columns]
            
            if existing_cols and len(existing_cols) >= n_components:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Extract columns to reduce
                X = df[existing_cols]
                # Impute missing values if any to prevent PCA crashing
                if X.isnull().any().any():
                    from sklearn.impute import SimpleImputer
                    X = SimpleImputer(strategy="median").fit_transform(X)
                else:
                    X = X.values
                    
                # Scale features
                X_scaled = StandardScaler().fit_transform(X)
                
                # Fit PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(X_scaled)
                
                # Create PC column names
                pca_cols = [f"PC{i+1}" for i in range(n_components)]
                pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df.index)
                
                # Drop original columns and concatenate PCA columns
                df = df.drop(columns=existing_cols)
                df = pd.concat([df, pca_df], axis=1)
                
        elif op == "feature_eng":
            fe_type = op_data.get("fe_type")
            if fe_type == "math_op":
                col1 = op_data.get("col1")
                col2 = op_data.get("col2")
                operator = op_data.get("operator")
                new_col = op_data.get("new_col")
                if col1 in df.columns and col2 in df.columns and operator and new_col:
                    if operator == "+":
                        df[new_col] = df[col1] + df[col2]
                    elif operator == "-":
                        df[new_col] = df[col1] - df[col2]
                    elif operator == "*":
                        df[new_col] = df[col1] * df[col2]
                    elif operator == "/":
                        df[new_col] = df[col1] / (df[col2] + 1e-9)
            elif fe_type == "math_transform":
                col = op_data.get("column")
                transform = op_data.get("transform")
                new_col = op_data.get("new_col")
                if col in df.columns and transform and new_col:
                    if transform == "log":
                        df[new_col] = np.log(np.maximum(df[col], 1e-9))
                    elif transform == "sqrt":
                        df[new_col] = np.sqrt(np.maximum(df[col], 0))
                    elif transform == "square":
                        df[new_col] = df[col] ** 2
                    elif transform == "abs":
                        df[new_col] = df[col].abs()
            elif fe_type == "binning":
                col = op_data.get("column")
                bins = int(op_data.get("bins", 4))
                new_col = op_data.get("new_col")
                if col in df.columns and new_col:
                    try:
                        df[new_col] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')
                    except Exception:
                        df[new_col] = pd.cut(df[col], bins=bins, labels=False)
                        
        elif op == "feature_select":
            method = op_data.get("method")
            target_col = op_data.get("target_col")
            
            # 1. Variance Threshold
            if method == "variance_threshold":
                threshold = float(op_data.get("threshold", 0.0))
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=threshold)
                    selector.fit(df[num_cols].fillna(df[num_cols].median()))
                    features_keep = df[num_cols].columns[selector.get_support()].tolist()
                    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    all_keep = cat_cols + features_keep
                    if target_col and target_col in df.columns and target_col not in all_keep:
                        all_keep.append(target_col)
                    df = df[[c for c in df.columns if c in all_keep]]
                    
            # 2. SelectKBest
            elif method == "select_k_best":
                k = op_data.get("k", 5)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in num_cols:
                    num_cols.remove(target_col)
                
                if num_cols and target_col in df.columns:
                    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                    task = op_data.get("task", "classification").lower()
                    score_func = f_classif if "class" in task else f_regression
                    
                    X = df[num_cols].fillna(df[num_cols].median())
                    y = df[target_col]
                    if not np.issubdtype(y.dtype, np.number):
                        y = LabelEncoder().fit_transform(y.astype(str))
                    else:
                        y = y.fillna(y.median())
                    
                    k_val = min(int(k), len(num_cols))
                    if k_val > 0:
                        selector = SelectKBest(score_func=score_func, k=k_val)
                        selector.fit(X, y)
                        features_keep = [num_cols[i] for i in selector.get_support(indices=True)]
                        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                        all_keep = cat_cols + features_keep
                        if target_col not in all_keep:
                            all_keep.append(target_col)
                        df = df[[c for c in df.columns if c in all_keep]]
                        
            # 3. RFE / RFECV
            elif method in ["rfe", "rfecv"]:
                n_features = int(op_data.get("n_features", 5))
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in num_cols:
                    num_cols.remove(target_col)
                    
                if num_cols and target_col in df.columns:
                    X = df[num_cols].fillna(df[num_cols].median())
                    y = df[target_col]
                    if not np.issubdtype(y.dtype, np.number):
                        y = LabelEncoder().fit_transform(y.astype(str))
                    else:
                        y = y.fillna(y.median())
                        
                    task = op_data.get("task", "classification").lower()
                    if "class" in task:
                        from sklearn.ensemble import RandomForestClassifier
                        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                        
                    if method == "rfe":
                        from sklearn.feature_selection import RFE
                        n_feat_val = min(n_features, len(num_cols))
                        selector = RFE(estimator=estimator, n_features_to_select=n_feat_val, step=1)
                    else:
                        from sklearn.feature_selection import RFECV
                        selector = RFECV(estimator=estimator, step=1, cv=3)
                        
                    selector.fit(X, y)
                    features_keep = [num_cols[i] for i in selector.get_support(indices=True)]
                    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    all_keep = cat_cols + features_keep
                    if target_col not in all_keep:
                        all_keep.append(target_col)
                    df = df[[c for c in df.columns if c in all_keep]]
                    
            # 4. Correlation Threshold
            elif method == "correlation_threshold":
                threshold = float(op_data.get("threshold", 0.85))
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in num_cols:
                    num_cols.remove(target_col)
                    
                if len(num_cols) > 1:
                    corr_matrix = df[num_cols].corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
                    df = df.drop(columns=to_drop)
                    
    return df
