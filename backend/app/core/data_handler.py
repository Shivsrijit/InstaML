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

class TargetLeakageError(ValueError):
    """Exception raised when target-derived features leak into input features X."""
    pass


class InvalidTargetLineageError(ValueError):
    """Exception raised when the selected active target is not in the target lineage graph."""
    pass


class TransformRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, forward_fn, inverse_fn, is_invertible=True):
        cls._registry[name] = {
            "forward": forward_fn,
            "inverse": inverse_fn,
            "is_invertible": is_invertible
        }

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)


# Register default transformations with explicit invertibility metadata
# log is uniquely invertible (using exp)
TransformRegistry.register(
    "log",
    lambda x: np.log(np.maximum(x, 1e-9)),
    lambda x: np.exp(x),
    is_invertible=True
)
# sqrt is uniquely invertible (using square)
TransformRegistry.register(
    "sqrt",
    lambda x: np.sqrt(np.maximum(x, 0)),
    lambda x: x ** 2,
    is_invertible=True
)
# square is uniquely invertible (using sqrt)
TransformRegistry.register(
    "square",
    lambda x: x ** 2,
    lambda x: np.sqrt(np.maximum(x, 0)),
    is_invertible=True
)
# abs is non-invertible since abs(-10) = abs(10) = 10 (predecessor sign is lost)
TransformRegistry.register(
    "abs",
    lambda x: x.abs(),
    None,
    is_invertible=False
)
# binning, bucketing, categorization, thresholding, rounding, and math operators are non-invertible
TransformRegistry.register("binning", None, None, is_invertible=False)
TransformRegistry.register("bucketing", None, None, is_invertible=False)
TransformRegistry.register("categorization", None, None, is_invertible=False)
TransformRegistry.register("thresholding", None, None, is_invertible=False)
TransformRegistry.register("rounding", None, None, is_invertible=False)

TransformRegistry.register("+", None, None, is_invertible=False)
TransformRegistry.register("-", None, None, is_invertible=False)
TransformRegistry.register("*", None, None, is_invertible=False)
TransformRegistry.register("/", None, None, is_invertible=False)


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
                
                # Extract columns to reduce
                X = df[existing_cols]
                # Impute missing values if any to prevent PCA crashing
                if X.isnull().any().any():
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
                operator = op_data.get("operator")
                new_col = op_data.get("new_col")
                columns = op_data.get("columns", [])
                
                if not columns:
                    col1 = op_data.get("col1")
                    col2 = op_data.get("col2")
                    if col1 and col2:
                        columns = [col1, col2]
                        
                existing_cols = [c for c in columns if c in df.columns]
                if len(existing_cols) >= 2 and operator and new_col:
                    if operator == "/" and len(existing_cols) != 2:
                        existing_cols = existing_cols[:2]
                        
                    result = df[existing_cols[0]].copy()
                    for next_col in existing_cols[1:]:
                        if operator == "+":
                            result = result + df[next_col]
                        elif operator == "-":
                            result = result - df[next_col]
                        elif operator == "*":
                            result = result * df[next_col]
                        elif operator == "/":
                            # Replace 0 values with NaN to prevent division by zero
                            denominator = df[next_col].replace(0, np.nan)
                            result = result / denominator
                    df[new_col] = result.round(4)
            elif fe_type == "math_transform":
                col = op_data.get("column")
                transform = op_data.get("transform")
                new_col = op_data.get("new_col")
                if col in df.columns and transform and new_col:
                    registry_entry = TransformRegistry.get(transform)
                    if registry_entry and registry_entry.get("forward"):
                        df[new_col] = registry_entry["forward"](df[col]).round(4)
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


def get_operations_for_version(project_id: int, version_id: int) -> list:
    """Collect all operations applied sequentially to reach the given dataset version ID."""
    operations_list = []
    current_id = version_id
    import json
    from backend.app.db.session import query_sync
    while True:
        rows = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? AND id = ?", [project_id, current_id])
        if not rows:
            break
        version = rows[0]
        
        metadata = {}
        if version.metadata_json:
            try:
                metadata = json.loads(version.metadata_json)
            except:
                pass
                
        if version.step_name == "Restore Checkpoint":
            restored_from_vid = metadata.get("restored_from")
            if restored_from_vid:
                r_rows = query_sync("SELECT id FROM dataset_versions WHERE project_id = ? AND version_id = ?", [project_id, restored_from_vid])
                if r_rows:
                    current_id = r_rows[0].id
                    continue
            prev_rows = query_sync("SELECT id FROM dataset_versions WHERE project_id = ? AND id < ? ORDER BY id DESC LIMIT 1", [project_id, version.id])
            if prev_rows:
                current_id = prev_rows[0].id
            else:
                break
        elif version.step_name in ["Preprocessing", "Feature Engineering"]:
            ops = metadata.get("operations", [])
            operations_list = ops + operations_list
            prev_rows = query_sync("SELECT id FROM dataset_versions WHERE project_id = ? AND id < ? ORDER BY id DESC LIMIT 1", [project_id, version.id])
            if prev_rows:
                current_id = prev_rows[0].id
            else:
                break
        elif version.step_name == "Data Upload":
            break
        else:
            ops = metadata.get("operations", [])
            operations_list = ops + operations_list
            prev_rows = query_sync("SELECT id FROM dataset_versions WHERE project_id = ? AND id < ? ORDER BY id DESC LIMIT 1", [project_id, version.id])
            if prev_rows:
                current_id = prev_rows[0].id
            else:
                break
    return operations_list


def trace_target_operations(operations: list, active_target_col: str):
    """
    Traces all operations to build a target lineage dependency graph starting
    from the training/model active target column by walking backwards to the
    original target root(s) and then traversing forward to find all target-derived columns.
    
    Returns a TargetLineage dict:
    {
        "original_target": original_target_col,
        "target_derived_cols": list of target-derived column names,
        "lineage_graph": dict mapping a column to its downstream dependency columns,
        "creation_steps": dict mapping an output column to its creation step dict
    }
    """
    from collections import defaultdict
    lineage_graph = defaultdict(set)
    parent_graph = defaultdict(set)
    creation_steps = {}
    
    for op_data in operations:
        op = op_data.get("op")
        if op == "feature_eng":
            fe_type = op_data.get("fe_type")
            new_col = op_data.get("new_col")
            if not new_col:
                continue
                
            inputs = []
            transform_key = None
            metadata = {}
            
            if fe_type == "math_op":
                cols = op_data.get("columns", [])
                if not cols:
                    col1 = op_data.get("col1")
                    col2 = op_data.get("col2")
                    if col1 and col2:
                        cols = [col1, col2]
                inputs = cols
                transform_key = op_data.get("operator")
                metadata = {"operator": transform_key}
            elif fe_type == "math_transform":
                col = op_data.get("column")
                if col:
                    inputs = [col]
                transform_key = op_data.get("transform")
                metadata = {"transform": transform_key}
            elif fe_type == "binning":
                col = op_data.get("column")
                if col:
                    inputs = [col]
                transform_key = "binning"
                metadata = {"bins": op_data.get("bins", 4)}
                
            for inp in inputs:
                lineage_graph[inp].add(new_col)
                parent_graph[new_col].add(inp)
                
            registry_entry = TransformRegistry.get(transform_key)
            is_invertible = registry_entry.get("is_invertible", False) if registry_entry else False
            
            creation_steps[new_col] = {
                "operation": op_data,
                "input_columns": inputs,
                "output_column": new_col,
                "is_invertible": is_invertible,
                "transform_key": transform_key,
                "metadata": metadata
            }
            
        elif op == "pca":
            cols = op_data.get("columns", [])
            n_components = int(op_data.get("n_components", 2))
            for i in range(n_components):
                new_col = f"PC{i+1}"
                for inp in cols:
                    lineage_graph[inp].add(new_col)
                    parent_graph[new_col].add(inp)
                creation_steps[new_col] = {
                    "operation": op_data,
                    "input_columns": cols,
                    "output_column": new_col,
                    "is_invertible": False,
                    "transform_key": "pca",
                    "metadata": {"n_components": n_components}
                }
                
    # Validate DAG/cycle detection on the lineage graph
    visited_nodes = set()
    rec_stack = set()
    
    def check_cycle(node):
        if node in rec_stack:
            raise ValueError(f"Cycle detected in target lineage graph at node '{node}'")
        if node in visited_nodes:
            return
        rec_stack.add(node)
        for neighbor in lineage_graph[node]:
            check_cycle(neighbor)
        rec_stack.remove(node)
        visited_nodes.add(node)
        
    for node in list(lineage_graph.keys()):
        check_cycle(node)
        
    # Find all root ancestors of active_target_col by walking backwards
    roots = set()
    visited_back = set()
    
    def find_roots(node):
        if node in visited_back:
            return
        visited_back.add(node)
        parents = parent_graph[node]
        if not parents:
            roots.add(node)
        else:
            for p in parents:
                find_roots(p)
                
    find_roots(active_target_col)
    
    if not roots:
        roots.add(active_target_col)
        
    # Walk forward from all roots to find all target-derived columns
    target_derived = set()
    visited_forward = set()
    
    def dfs(node):
        if node in visited_forward:
            return
        visited_forward.add(node)
        target_derived.add(node)
        for child in lineage_graph[node]:
            dfs(child)
            
    for r in roots:
        dfs(r)
        
    # original_target is the primary root
    original_target_col = sorted(list(roots))[0] if roots else active_target_col
    
    # Convert sets to lists in the lineage graph for JSON serializability
    serializable_graph = {k: list(v) for k, v in lineage_graph.items()}
    
    return {
        "original_target": original_target_col,
        "target_derived_cols": list(target_derived),
        "lineage_graph": serializable_graph,
        "creation_steps": creation_steps
    }


def reconstruct_original_target(predictions, lineage: dict, active_target: str, original_target: str):
    """
    Graph-driven reconstruction engine.
    Solves the lineage graph backwards from active_target to original_target.
    
    Returns:
        preds: predictions in original space (or active space if not reconstructable)
        reconstructable (bool): whether reconstruction succeeded
        reason (str/None): explanation if not reconstructable
        non_invertible_step (str/None): name of the step that blocked invertibility
    """
    creation_steps = lineage.get("creation_steps", {})
    target_derived_cols = set(lineage.get("target_derived_cols", []))
    
    # 1. Validation: selected active target must belong to the target lineage!
    if active_target not in target_derived_cols:
        reason = f"Selected active target '{active_target}' is not in the target lineage graph."
        return predictions, False, reason, "invalid_active_target"
        
    if active_target == original_target:
        return predictions, True, None, None
        
    # Initialize the solved nodes with the active target predictions
    import numpy as np
    
    is_series = isinstance(predictions, (pd.Series, np.ndarray))
    current_values = {active_target: predictions.copy() if is_series else predictions}
    solved = {active_target}
    
    # Track executed output columns to avoid redundant inverse attempts
    executed_steps = set()
    
    # Keep resolving until original_target is in solved
    while original_target not in solved:
        progress = False
        
        # Search for a step where the output is solved, but target-derived inputs are not
        for out_col, step in creation_steps.items():
            if out_col not in solved or out_col in executed_steps:
                continue
                
            inputs = step.get("input_columns", [])
            target_inputs = [inp for inp in inputs if inp in target_derived_cols]
            unsolved_target_inputs = [inp for inp in target_inputs if inp not in solved]
            
            if not unsolved_target_inputs:
                continue
                
            transform_key = step.get("transform_key")
            registry_entry = TransformRegistry.get(transform_key)
            
            # Check if this transformation step is invertible in the registry
            if not step.get("is_invertible", False) or not registry_entry or not registry_entry.get("inverse"):
                reason = f"Non-invertible transformation '{transform_key}' detected at column '{out_col}'"
                return predictions, False, reason, transform_key
                
            # Execute inverse operation
            inverse_fn = registry_entry["inverse"]
            out_val = current_values[out_col]
            
            try:
                in_val = inverse_fn(out_val)
            except Exception as err:
                reason = f"Failed to execute inverse transform '{transform_key}': {str(err)}"
                return predictions, False, reason, transform_key
                
            # Assign computed input values to unsolved targets
            if len(target_inputs) > 1:
                # Merge operation: inverse must uniquely reconstruct all predecessor values
                if isinstance(in_val, (tuple, list)) and len(in_val) == len(target_inputs):
                    for idx, inp_col in enumerate(target_inputs):
                        current_values[inp_col] = in_val[idx]
                        solved.add(inp_col)
                else:
                    reason = f"Merge operation '{transform_key}' did not return unique predecessor values for all target inputs"
                    return predictions, False, reason, transform_key
            else:
                # Single input
                inp_col = target_inputs[0]
                current_values[inp_col] = in_val
                solved.add(inp_col)
                
            executed_steps.add(out_col)
            progress = True
            break # Re-evaluate while loop
            
        if not progress:
            # We reached a dead end, cannot solve original target
            reason = f"Reconstruction path blocked: unable to solve original target '{original_target}' from active target '{active_target}'"
            return predictions, False, reason, "blocked"
            
    return current_values[original_target], True, None, None
