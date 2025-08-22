import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from core.preprocess import create_preprocessing_pipeline
from core.evaluator import evaluate_classification

def train_tabular_model(df, target_col, model_type="random_forest", scaling="standard", test_size=0.2, random_state=42):
    """Train a tabular model (sklearn/XGBoost)."""
    preprocessor, _, _ = create_preprocessing_pipeline(df, target_col, scaling)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    from sklearn.pipeline import Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    metrics = evaluate_classification(clf, X_test, y_test)

    return clf, metrics
