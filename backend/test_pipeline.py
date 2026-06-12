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
db_path = Path(__file__).resolve().parent / "test_pipeline.db"
os.environ["TURSO_DATABASE_URL"] = f"file:{db_path.resolve().as_posix()}"
os.environ["TRAINER_SERVICE_URL"] = "http://localhost:8011"
os.environ["PREDICTOR_SERVICE_URL"] = "http://localhost:8012"

from backend.app.main import app
from backend.app.db.session import run_migrations_sync

class TestInstaMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        db_path = (Path(__file__).resolve().parent / "test_pipeline.db").resolve()
        os.environ["TURSO_DATABASE_URL"] = f"file:{db_path.as_posix()}"
        # Make sure they communicate with localhost endpoints for testing
        os.environ["TRAINER_SERVICE_URL"] = "http://localhost:8011"
        os.environ["PREDICTOR_SERVICE_URL"] = "http://localhost:8012"
        
        # Clean slate: delete existing pipeline db file
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
            "uvicorn", "backend.app.main_trainer:app", "--host", "127.0.0.1", "--port", "8011"
        ], shell=True, env=env)
        
        cls.predictor_proc = subprocess.Popen([
            "uvicorn", "backend.app.main_predictor:app", "--host", "127.0.0.1", "--port", "8012"
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
            
        db_path = (Path(__file__).resolve().parent / "test_pipeline.db").resolve()
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

    def test_full_ml_pipeline_execution(self):
        # 1. Register & Login
        self.client.post("/api/auth/register", json={"username": "mldev", "email": "dev@instaml.com", "password": "securepass"})
        token = self.client.post("/api/auth/login", json={"username": "mldev", "password": "securepass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Create Project
        res_proj = self.client.post(
            "/api/projects", 
            json={"name": "Titanic Predictor", "description": "Survival prediction", "data_type": "tabular"},
            headers=headers
        )
        self.assertEqual(res_proj.status_code, 200)
        proj_id = res_proj.json()["id"]

        # 3. Create dummy dataset in memory & upload
        # 6 rows, columns: age (numeric with NaN), fare (numeric), embarked (categorical), survived (target)
        csv_data = (
            "age,fare,embarked,survived\n"
            "22,7.25,S,0\n"
            ",71.28,C,1\n"
            "26,7.92,S,1\n"
            "35,53.10,S,1\n"
            "54,51.86,S,0\n"
            ",8.05,Q,0\n"
        )
        file_like = io.BytesIO(csv_data.encode('utf-8'))
        
        res_upload = self.client.post(
            f"/api/projects/{proj_id}/upload",
            files={"file": ("titanic.csv", file_like, "text/csv")},
            headers=headers
        )
        self.assertEqual(res_upload.status_code, 200)
        self.assertIn("version", res_upload.json())

        # Verify current data info
        res_curr = self.client.get(f"/api/projects/{proj_id}/data/current", headers=headers)
        self.assertEqual(res_curr.status_code, 200)
        self.assertEqual(res_curr.json()["shape"], [6, 4])
        self.assertEqual(res_curr.json()["missing_counts"]["age"], 2)

        # 4. Apply preprocessing: Impute 'age' with mean, one-hot encode 'embarked'
        preprocess_payload = {
            "operations": [
                {"op": "fill_missing", "strategy": "mean", "columns": ["age"]},
                {"op": "encode", "method": "onehot", "columns": ["embarked"]}
            ]
        }
        res_prep = self.client.post(f"/api/projects/{proj_id}/preprocess", json=preprocess_payload, headers=headers)
        self.assertEqual(res_prep.status_code, 200)

        # Verify preprocessing shape changed (one-hot encoding added columns, imputation filled NaNs)
        res_curr_prep = self.client.get(f"/api/projects/{proj_id}/data/current", headers=headers)
        self.assertEqual(res_curr_prep.json()["missing_counts"]["age"], 0)
        self.assertIn("embarked_S", res_curr_prep.json()["columns"])

        # 5. Trigger model training (Random Forest)
        train_payload = {
            "model_name": "Random Forest",
            "target_col": "survived",
            "use_hyperparameter_tuning": False,
            "validation_split": 0.2
        }
        res_train = self.client.post(f"/api/projects/{proj_id}/training/train", json=train_payload, headers=headers)
        self.assertEqual(res_train.status_code, 200, msg=f"Train trigger failed: {res_train.text}")
        model_id = res_train.json()["id"]

        # Poll training task (allow up to 15 seconds)
        completed = False
        for _ in range(15):
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

        # 6. Deploy model
        res_deploy = self.client.post(f"/api/projects/{proj_id}/models/{model_id}/deploy", headers=headers)
        self.assertEqual(res_deploy.status_code, 200)

        # 7. Execute real-time prediction
        # Provide features matching preprocessed dataset format
        pred_payload = {
            "features": {
                "age": 30.0,
                "fare": 25.0,
                "embarked_C": 0,
                "embarked_Q": 0,
                "embarked_S": 1
            }
        }
        res_pred = self.client.post(f"/api/projects/{proj_id}/deploy/predict", json=pred_payload, headers=headers)
        self.assertEqual(res_pred.status_code, 200)
        self.assertIn("prediction", res_pred.json())
        self.assertIn(res_pred.json()["prediction"], [0, 1]) # should return binary survival prediction
        print(f"\n[Pipeline Test Output]: Predict Result = {res_pred.json()}")

if __name__ == '__main__':
    unittest.main()
