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

def preprocess_raw_input(project_id: int, active_model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw input DataFrame using the project's historical preprocessing operations."""
    if not active_model.dataset_version_id:
        return input_df
        
    upload_versions = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? AND step_name = 'Data Upload' ORDER BY id ASC LIMIT 1",
        [project_id]
    )
    if not upload_versions:
        return input_df
        
    upload_version = upload_versions[0]
    try:
        from backend.app.core.data_handler import load_dataframe, apply_preprocessing_operations
        raw_df = load_dataframe(upload_version.file_path)
        
        # Collect operations back to upload version
        operations_list = []
        current_id = active_model.dataset_version_id
        import json
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
                    
        # Apply operations on combined dataframe
        if operations_list:
            # Align input_df columns with raw_df columns
            for col in raw_df.columns:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[raw_df.columns]
            
            # Combine
            combined_df = pd.concat([raw_df, input_df], ignore_index=True)
            
            # Run operations
            preprocessed_combined = apply_preprocessing_operations(combined_df, operations_list)
            
            # Extract input rows back
            input_df = preprocessed_combined.iloc[-len(input_df):]
            
        # Align final input columns with model expected columns
        model_version_rows = query_sync("SELECT columns_json FROM dataset_versions WHERE id = ?", [active_model.dataset_version_id])
        if model_version_rows:
            model_cols = json.loads(model_version_rows[0].columns_json)
            projects = query_sync("SELECT * FROM projects WHERE id = ?", [project_id])
            project = projects[0] if projects else None
            
            if project and project.data_type == "tabular":
                from backend.app.core.data_handler import get_operations_for_version, trace_target_operations
                ops = get_operations_for_version(project_id, active_model.dataset_version_id)
                original_target_col = project.target_col or active_model.target_col
                lineage = trace_target_operations(ops, original_target_col)
                derived_cols = lineage.get("target_derived_cols", [])
                expected_features = [col for col in model_cols if col not in derived_cols]
            else:
                expected_features = [col for col in model_cols if col != active_model.target_col]
                
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[expected_features]
            
    except Exception as prep_err:
        print(f"Error applying predict-time preprocessing: {prep_err}")
        
    return input_df

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
            input_df = preprocess_raw_input(project_id, active_model, input_df)
            predictions = pipeline.predict(input_df)
            
        reconstructable = True
        reason = None
        non_invertible_step = None
        active_target = active_model.target_col
        original_target = active_model.target_col
        
        if project and project.data_type == "tabular":
            from backend.app.core.data_handler import get_operations_for_version, trace_target_operations, reconstruct_original_target
            ops = get_operations_for_version(project_id, active_model.dataset_version_id)
            model_target_rows = query_sync("SELECT target_col FROM models WHERE id = ?", [active_model.id])
            training_target = model_target_rows[0].target_col if model_target_rows else active_model.target_col
            
            original_target = project.target_col or active_model.target_col
            lineage = trace_target_operations(ops, original_target)
            active_target = training_target
            
            reconstructed_preds, reconstructable, reason, non_invertible_step = reconstruct_original_target(
                predictions, lineage, active_target, original_target
            )
            predictions = reconstructed_preds

        prediction_val = predictions[0]
        
        if isinstance(prediction_val, (np.integer, np.floating)):
            prediction_val = prediction_val.item()
            
        result = {
            "prediction": prediction_val,
            "model_type": active_model.model_type,
            "target_col": active_model.target_col,
            "prediction_space": original_target if reconstructable else active_target,
            "reconstructable": reconstructable,
            "reason": reason or "Success",
            "active_target": active_target,
            "original_target": original_target,
            "non_invertible_step": non_invertible_step or "None"
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
            
        predict_df = preprocess_raw_input(project_id, active_model, predict_df)
        preds = pipeline.predict(predict_df)
        
        projects = query_sync("SELECT * FROM projects WHERE id = ?", [project_id])
        project = projects[0] if projects else None
        
        if project and project.data_type == "tabular":
            from backend.app.core.data_handler import get_operations_for_version, trace_target_operations, reconstruct_original_target
            ops = get_operations_for_version(project_id, active_model.dataset_version_id)
            model_target_rows = query_sync("SELECT target_col FROM models WHERE id = ?", [active_model.id])
            training_target = model_target_rows[0].target_col if model_target_rows else active_model.target_col
            
            lineage = trace_target_operations(ops, active_model.target_col)
            preds, reconstructable, reason, non_invertible_step = reconstruct_original_target(
                preds, lineage, training_target, active_model.target_col
            )
            
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
