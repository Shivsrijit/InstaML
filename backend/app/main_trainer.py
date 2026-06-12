import sys
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import os

# Setup pathing so core is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

print(f"DEBUG: Trainer Service DATABASE_URL = {os.getenv('DATABASE_URL')}")

from backend.app.core.training_worker import run_training_task, get_training_status

app = FastAPI(
    title="InstaML Trainer Service",
    description="Microservice dedicated to running CPU/GPU training tasks.",
    version="1.0.0"
)

class TrainerRequest(BaseModel):
    project_id: int
    model_id: int
    dataset_version_id: int
    model_name: str
    target_col: str
    use_hyperparameter_tuning: bool
    user_id: int
    text_col: Optional[str] = None
    trials: Optional[int] = 10
    validation_split: Optional[float] = 0.2
    k_folds: Optional[int] = None

@app.post("/train")
def train_model_job(req: TrainerRequest, background_tasks: BackgroundTasks):
    """Start model training in a background thread."""
    background_tasks.add_task(
        run_training_task,
        project_id=req.project_id,
        model_id=req.model_id,
        dataset_version_id=req.dataset_version_id,
        model_name=req.model_name,
        target_col=req.target_col,
        use_hyperparameter_tuning=req.use_hyperparameter_tuning,
        user_id=req.user_id,
        text_col=req.text_col,
        trials=req.trials or 10,
        validation_split=req.validation_split or 0.2,
        k_folds=req.k_folds
    )
    return {"status": "started", "project_id": req.project_id, "model_id": req.model_id}

@app.get("/status/{project_id}")
def get_job_status(project_id: int):
    """Check in-memory logs/status of a running training thread."""
    return get_training_status(project_id)

@app.get("/health")
def health_check():
    """Simple verification route."""
    return {"status": "healthy", "service": "instaml-trainer"}
