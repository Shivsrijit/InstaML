import unittest
from fastapi.testclient import TestClient
import os
import sys
import time
import json
import io
import pandas as pd
from pathlib import Path

# Setup pathing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Define test database URL and service URLs BEFORE importing any backend components
db_path = Path(__file__).resolve().parent / "test_cv_and_mlp.db"
os.environ["TURSO_DATABASE_URL"] = f"file:{db_path.resolve().as_posix()}"
os.environ["TRAINER_SERVICE_URL"] = "http://localhost:8013"
os.environ["PREDICTOR_SERVICE_URL"] = "http://localhost:8014"

from backend.app.main import app
from backend.app.db.session import run_migrations_sync

class TestInstaMLCVAndMLP(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        db_path = (Path(__file__).resolve().parent / "test_cv_and_mlp.db").resolve()
        os.environ["TURSO_DATABASE_URL"] = f"file:{db_path.as_posix()}"
        os.environ["TRAINER_SERVICE_URL"] = "http://localhost:8013"
        os.environ["PREDICTOR_SERVICE_URL"] = "http://localhost:8014"
        
        # Clean slate
        if db_path.exists():
            try:
                db_path.unlink()
            except:
                pass
        
        run_migrations_sync()
        cls.client = TestClient(app)
        
        import subprocess
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cls.trainer_proc = subprocess.Popen([
            "uvicorn", "backend.app.main_trainer:app", "--host", "127.0.0.1", "--port", "8013"
        ], shell=True, env=env)
        
        cls.predictor_proc = subprocess.Popen([
            "uvicorn", "backend.app.main_predictor:app", "--host", "127.0.0.1", "--port", "8014"
        ], shell=True, env=env)
        
        time.sleep(3) # Wait for uvicorn instances to boot up
        
    @classmethod
    def tearDownClass(cls):
        # Terminate background services
        if hasattr(cls, 'trainer_proc'):
            cls.trainer_proc.terminate()
            cls.trainer_proc.wait()
        if hasattr(cls, 'predictor_proc'):
            cls.predictor_proc.terminate()
            cls.predictor_proc.wait()
            
        db_path = (Path(__file__).resolve().parent / "test_cv_and_mlp.db").resolve()
        db_journal = db_path.with_name(db_path.name + "-journal")
        for db_file in [db_path, db_journal]:
            if db_file.exists():
                try:
                    os.remove(db_file)
                except:
                    pass
        # Clean storage
        import shutil
        test_storage = Path("storage")
        if test_storage.exists():
            try:
                shutil.rmtree(test_storage)
            except:
                pass

    def test_mlp_classification_with_cv(self):
        # Register & Login
        self.client.post("/api/auth/register", json={"username": "mldev", "email": "dev@instaml.com", "password": "securepass"})
        token = self.client.post("/api/auth/login", json={"username": "mldev", "password": "securepass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create Project
        res_proj = self.client.post(
            "/api/projects", 
            json={"name": "Classification Project", "description": "Tabular MLP Classifier", "data_type": "tabular"},
            headers=headers
        )
        self.assertEqual(res_proj.status_code, 200)
        proj_id = res_proj.json()["id"]

        # Upload classification data (30 rows, 3 features, 1 target label)
        csv_data = (
            "feat1,feat2,feat3,label\n"
            "0.1,0.2,0.3,0\n"
            "0.4,0.5,0.6,1\n"
            "0.2,0.1,0.4,0\n"
            "0.5,0.4,0.7,1\n"
            "0.1,0.3,0.2,0\n"
            "0.6,0.5,0.8,1\n"
            "0.2,0.2,0.3,0\n"
            "0.7,0.6,0.9,1\n"
            "0.3,0.1,0.2,0\n"
            "0.8,0.7,1.0,1\n"
            "0.11,0.21,0.31,0\n"
            "0.41,0.51,0.61,1\n"
            "0.21,0.11,0.41,0\n"
            "0.51,0.41,0.71,1\n"
            "0.11,0.31,0.21,0\n"
            "0.61,0.51,0.81,1\n"
            "0.21,0.21,0.31,0\n"
            "0.71,0.61,0.91,1\n"
            "0.31,0.11,0.21,0\n"
            "0.81,0.71,1.01,1\n"
            "0.12,0.22,0.32,0\n"
            "0.42,0.52,0.62,1\n"
            "0.22,0.12,0.42,0\n"
            "0.52,0.42,0.72,1\n"
            "0.12,0.32,0.22,0\n"
            "0.62,0.52,0.82,1\n"
            "0.22,0.22,0.32,0\n"
            "0.72,0.62,0.92,1\n"
            "0.32,0.12,0.22,0\n"
            "0.82,0.72,1.02,1\n"
        )
        file_like = io.BytesIO(csv_data.encode('utf-8'))
        res_upload = self.client.post(
            f"/api/projects/{proj_id}/upload",
            files={"file": ("classification.csv", file_like, "text/csv")},
            headers=headers
        )
        self.assertEqual(res_upload.status_code, 200)

        # Trigger model training with MLP & 3-Fold Cross-Validation & hyperparameter tuning
        train_payload = {
            "model_name": "MLP",
            "target_col": "label",
            "use_hyperparameter_tuning": True,
            "trials": 5,
            "validation_split": 0.2,
            "k_folds": 3
        }
        res_train = self.client.post(f"/api/projects/{proj_id}/training/train", json=train_payload, headers=headers)
        self.assertEqual(res_train.status_code, 200)
        model_id = res_train.json()["id"]

        # Poll training task
        completed = False
        for _ in range(30):
            res_status = self.client.get(f"/api/projects/{proj_id}/training/status", headers=headers)
            status_val = res_status.json()["status"]
            if status_val == "completed":
                completed = True
                break
            elif status_val == "failed":
                print("\n=== BACKGROUND TRAINING JOB FAILED! LOGS: ===")
                print(json.dumps(res_status.json(), indent=2))
                self.fail(f"Background training job failed: {res_status.json()}")
            time.sleep(1)
            
        self.assertTrue(completed, "Training task timed out")

        # Fetch evaluation metrics
        res_eval = self.client.get(f"/api/projects/{proj_id}/models/{model_id}/evaluation", headers=headers)
        self.assertEqual(res_eval.status_code, 200)
        eval_json = res_eval.json()
        
        self.assertEqual(eval_json["task_type"], "classification")
        self.assertIn("cv_metrics", eval_json["metrics"])
        cv_metrics = eval_json["metrics"]["cv_metrics"]
        self.assertIn("accuracy", cv_metrics)
        self.assertEqual(len(cv_metrics["accuracy"]["scores"]), 3)

    def test_mlp_regression_with_cv(self):
        # Login
        token = self.client.post("/api/auth/login", json={"username": "mldev", "password": "securepass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create Project
        res_proj = self.client.post(
            "/api/projects", 
            json={"name": "Regression Project", "description": "Tabular MLP Regressor", "data_type": "tabular"},
            headers=headers
        )
        self.assertEqual(res_proj.status_code, 200)
        proj_id = res_proj.json()["id"]

        # Upload regression data (30 rows, 3 features, 1 continuous target variable)
        csv_data = (
            "feat1,feat2,feat3,target\n"
            "0.1,0.2,0.3,1.5\n"
            "0.4,0.5,0.6,2.5\n"
            "0.2,0.1,0.4,1.8\n"
            "0.5,0.4,0.7,2.9\n"
            "0.1,0.3,0.2,1.2\n"
            "0.6,0.5,0.8,3.1\n"
            "0.2,0.2,0.3,1.6\n"
            "0.7,0.6,0.9,3.5\n"
            "0.3,0.1,0.2,1.3\n"
            "0.8,0.7,1.0,3.8\n"
            "0.11,0.21,0.31,1.51\n"
            "0.41,0.51,0.61,2.51\n"
            "0.21,0.11,0.41,1.81\n"
            "0.51,0.41,0.71,2.91\n"
            "0.11,0.31,0.21,1.21\n"
            "0.61,0.51,0.81,3.11\n"
            "0.21,0.21,0.31,1.61\n"
            "0.71,0.61,0.91,3.51\n"
            "0.31,0.11,0.21,1.31\n"
            "0.81,0.71,1.01,3.81\n"
            "0.12,0.22,0.32,1.52\n"
            "0.42,0.52,0.62,2.52\n"
            "0.22,0.12,0.42,1.82\n"
            "0.52,0.42,0.72,2.92\n"
            "0.12,0.32,0.22,1.22\n"
            "0.62,0.52,0.82,3.12\n"
            "0.22,0.22,0.32,1.62\n"
            "0.72,0.62,0.92,3.52\n"
            "0.32,0.12,0.22,1.32\n"
            "0.82,0.72,1.02,3.82\n"
        )
        file_like = io.BytesIO(csv_data.encode('utf-8'))
        res_upload = self.client.post(
            f"/api/projects/{proj_id}/upload",
            files={"file": ("regression.csv", file_like, "text/csv")},
            headers=headers
        )
        self.assertEqual(res_upload.status_code, 200)

        # Trigger model training with MLP & 2-Fold Cross-Validation (no tuning for speed)
        train_payload = {
            "model_name": "MLP",
            "target_col": "target",
            "use_hyperparameter_tuning": False,
            "validation_split": 0.2,
            "k_folds": 2
        }
        res_train = self.client.post(f"/api/projects/{proj_id}/training/train", json=train_payload, headers=headers)
        self.assertEqual(res_train.status_code, 200)
        model_id = res_train.json()["id"]

        # Poll training task
        completed = False
        for _ in range(30):
            res_status = self.client.get(f"/api/projects/{proj_id}/training/status", headers=headers)
            status_val = res_status.json()["status"]
            if status_val == "completed":
                completed = True
                break
            elif status_val == "failed":
                self.fail(f"Background training job failed: {res_status.json()}")
            time.sleep(1)
            
        self.assertTrue(completed, "Training task timed out")

        # Fetch evaluation metrics
        res_eval = self.client.get(f"/api/projects/{proj_id}/models/{model_id}/evaluation", headers=headers)
        self.assertEqual(res_eval.status_code, 200)
        eval_json = res_eval.json()
        
        self.assertEqual(eval_json["task_type"], "regression")
        self.assertIn("cv_metrics", eval_json["metrics"])
        cv_metrics = eval_json["metrics"]["cv_metrics"]
        self.assertIn("r2", cv_metrics)
        self.assertEqual(len(cv_metrics["r2"]["scores"]), 2)

if __name__ == '__main__':
    unittest.main()
