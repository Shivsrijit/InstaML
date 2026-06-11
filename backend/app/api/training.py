from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import List
import json
import os
import requests

from backend.app.db.session import query_sync, execute_sync
from backend.app.db.schemas import TrainRequest, ModelResponse
from backend.app.api.deps import get_current_user
from backend.app.api.data import get_user_project

TRAINER_SERVICE_URL = os.getenv("TRAINER_SERVICE_URL", "http://localhost:8001")

router = APIRouter(prefix="/projects/{project_id}", tags=["training"])

@router.get("/training/options")
def get_training_options(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve available algorithms for classification and regression tasks based on project data type."""
    project = get_user_project(project_id, current_user.id)
    
    if project.data_type == "tabular":
        return {
            "classification": [
                "Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression",
                "SVM", "KNN", "Decision Tree", "Naive Bayes"
            ],
            "regression": [
                "Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression",
                "Ridge", "Lasso", "SVR", "KNN", "Decision Tree"
            ]
        }
    elif project.data_type == "text":
        return {
            "classification": [
                "Logistic Regression", "Random Forest", "Naive Bayes", "XGBoost"
            ]
        }
    elif project.data_type == "image":
        return {
            "classification": ["resnet18", "resnet50", "vgg16", "mobilenet"],
            "detection": ["yolo"]
        }
    elif project.data_type == "audio":
        return {
            "classification": ["cnn", "lstm"]
        }
    elif project.data_type == "multi_dimensional":
        return {
            "classification": ["MLP", "CNN", "RNN"],
            "regression": ["MLP", "CNN", "RNN"]
        }
    else:
        return {}

@router.post("/training/train", response_model=ModelResponse)
def trigger_training(
    project_id: int,
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Start asynchronous background model training on the latest dataset version."""
    project = get_user_project(project_id, current_user.id)
    
    # Resolve active dataset version
    latest_versions = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        [project_id]
    )
        
    if not latest_versions:
        raise HTTPException(
            status_code=400,
            detail="No dataset version uploaded. Please upload a dataset before training."
        )
    latest_version = latest_versions[0]
        
    # Check if a training job is already running
    try:
        res = requests.get(f"{TRAINER_SERVICE_URL}/status/{project_id}")
        if res.status_code == 200:
            current_status = res.json()
        else:
            current_status = {"status": "idle"}
    except Exception:
        current_status = {"status": "idle"}
        
    if current_status["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="A training job is already running for this project. Please wait for it to complete."
        )
        
    # Create DB entry for Model (placeholder with empty metrics and path)
    execute_sync(
        """INSERT INTO models (
            name, model_type, target_col, metrics_json, file_path, is_deployed, project_id, dataset_version_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            f"{req.model_name} Run",
            req.model_name,
            req.target_col,
            json.dumps({"status": "training"}),
            "",
            0,
            project_id,
            latest_version.id
        ]
    )
    
    # Retrieve newly created model
    models = query_sync(
        "SELECT * FROM models WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        [project_id]
    )
    db_model = models[0]
    
    # Delegate training to the trainer microservice
    try:
        res = requests.post(
            f"{TRAINER_SERVICE_URL}/train",
            json={
                "project_id": project_id,
                "model_id": db_model.id,
                "dataset_version_id": latest_version.id,
                "model_name": req.model_name,
                "target_col": req.target_col,
                "use_hyperparameter_tuning": req.use_hyperparameter_tuning,
                "user_id": current_user.id,
                "text_col": req.text_col,
                "trials": req.trials,
                "validation_split": req.validation_split
            }
        )
        if res.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Trainer microservice returned status {res.status_code}: {res.text}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trainer microservice is unreachable: {str(e)}"
        )
    
    return db_model

@router.get("/training/status")
def get_job_status(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve execution logs and status of the current training run."""
    get_user_project(project_id, current_user.id)
    try:
        res = requests.get(f"{TRAINER_SERVICE_URL}/status/{project_id}")
        if res.status_code == 200:
            return res.json()
        return {"status": "idle", "logs": [f"Trainer service returned status {res.status_code}"], "model_id": None}
    except Exception as e:
        return {"status": "idle", "logs": [f"Trainer service unreachable: {str(e)}"], "model_id": None}

@router.get("/models", response_model=List[ModelResponse])
def list_trained_models(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve all trained models for the project."""
    get_user_project(project_id, current_user.id)
    
    models = query_sync(
        "SELECT * FROM models WHERE project_id = ? ORDER BY created_at DESC",
        [project_id]
    )
    return models

@router.get("/models/{model_id}/evaluation")
def get_model_evaluation(
    project_id: int,
    model_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve detailed metrics, confusion matrix, and feature importances for a model."""
    get_user_project(project_id, current_user.id)
    
    models = query_sync(
        "SELECT * FROM models WHERE id = ? AND project_id = ?",
        [model_id, project_id]
    )
    if not models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[0]
        
    try:
        metrics_data = json.loads(model.metrics_json) if model.metrics_json else {}
        return metrics_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model metrics: {str(e)}")
