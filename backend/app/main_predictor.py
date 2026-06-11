import sys
import io
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import Dict, Any

# Setup pathing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.db.session import query_sync

app = FastAPI(
    title="InstaML Predictor Service",
    description="Microservice dedicated to executing model predictions.",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/projects/{project_id}/predict")
def predict_realtime(
    project_id: int,
    req: PredictRequest
):
    """Execute real-time prediction using the active model for the project."""
    active_models = query_sync("SELECT * FROM models WHERE project_id = ? AND is_deployed = 1 LIMIT 1", [project_id])
    active_model = active_models[0] if active_models else None
    if not active_model:
        raise HTTPException(
            status_code=404,
            detail="No active model deployed for this project. Please deploy a model first."
        )
        
    try:
        # Load the pipeline (.pkl file) from disk (shared volume)
        pipeline = joblib.load(active_model.file_path)
        
        # Check project modality
        projects = query_sync("SELECT * FROM projects WHERE id = ?", [project_id])
        project = projects[0] if projects else None
        
        if project and project.data_type == "text":
            # Extract the raw text string value
            text_val = ""
            if req.features:
                if "text" in req.features:
                    text_val = req.features["text"]
                else:
                    text_val = list(req.features.values())[0]
            predictions = pipeline.predict([text_val])
        else:
            # Convert input features to a DataFrame with one row
            input_df = pd.DataFrame([req.features])
            predictions = pipeline.predict(input_df)
            
        prediction_val = predictions[0]
        
        if isinstance(prediction_val, (np.integer, np.floating)):
            prediction_val = prediction_val.item()
            
        result = {
            "prediction": prediction_val,
            "model_type": active_model.model_type,
            "target_col": active_model.target_col
        }
        
        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(input_df)
                max_prob = float(np.max(probabilities))
                result["confidence"] = max_prob
                result["class_probabilities"] = {
                    str(pipeline.classes_[i]): float(probabilities[0, i])
                    for i in range(len(pipeline.classes_))
                }
            except:
                pass
                
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/projects/{project_id}/predict-batch")
def predict_batch(
    project_id: int,
    file: UploadFile = File(...)
):
    """Execute batch predictions on a CSV file and return the same CSV with predictions."""
    active_models = query_sync("SELECT * FROM models WHERE project_id = ? AND is_deployed = 1 LIMIT 1", [project_id])
    active_model = active_models[0] if active_models else None
    if not active_model:
        raise HTTPException(
            status_code=404,
            detail="No active model deployed for this project."
        )
        
    try:
        df = pd.read_csv(file.file)
        pipeline = joblib.load(active_model.file_path)
        
        predict_df = df.copy()
        if active_model.target_col in predict_df.columns:
            predict_df = predict_df.drop(columns=[active_model.target_col])
            
        preds = pipeline.predict(predict_df)
        df["prediction"] = preds
        
        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(predict_df)
                df["prediction_confidence"] = np.max(probabilities, axis=1)
            except:
                pass
                
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=batch_predictions_{file.filename}"
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    """Simple verification route."""
    return {"status": "healthy", "service": "instaml-predictor"}
