# core/model_registry.py
import joblib
import json
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models")

def save_model(model, metrics, model_name="model"):
    MODEL_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"{model_name}_{timestamp}.pkl"
    meta_path = MODEL_DIR / f"{model_name}_{timestamp}.json"

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return str(model_path), str(meta_path)

def load_model(model_path):
    return joblib.load(model_path)
