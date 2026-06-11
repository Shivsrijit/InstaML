import unittest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path

# Setup pathing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set test database URL before importing any database modules
os.environ["TURSO_DATABASE_URL"] = f"file:{Path(__file__).resolve().parent}/test_instaml.db"

from backend.app.main import app
from backend.app.db.session import query_sync, execute_sync, run_migrations_sync

class TestInstaMLAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Delete existing test DB to ensure a clean slate
        test_db_path = Path(__file__).resolve().parent / "test_instaml.db"
        if test_db_path.exists():
            try:
                test_db_path.unlink()
            except:
                pass
        # Re-create tables via migrations
        run_migrations_sync()
        cls.client = TestClient(app)
        
    @classmethod
    def tearDownClass(cls):
        # Cleanup test DB files
        test_db_path = Path(__file__).resolve().parent / "test_instaml.db"
        if test_db_path.exists():
            try:
                test_db_path.unlink()
            except:
                pass

    def setUp(self):
        # Clean all tables for fresh test
        execute_sync("DELETE FROM models")
        execute_sync("DELETE FROM dataset_versions")
        execute_sync("DELETE FROM projects")
        execute_sync("DELETE FROM users")

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_user_registration_and_login(self):
        # 1. Register user
        reg_payload = {
            "username": "testuser",
            "email": "testuser@instaml.com",
            "password": "securepassword123"
        }
        res_reg = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(res_reg.status_code, 200)
        self.assertEqual(res_reg.json()["username"], "testuser")
        
        # 2. Login
        login_payload = {
            "username": "testuser",
            "password": "securepassword123"
        }
        res_login = self.client.post("/api/auth/login", json=login_payload)
        self.assertEqual(res_login.status_code, 200)
        self.assertIn("access_token", res_login.json())
        token = res_login.json()["access_token"]
        
        # 2b. Case-insensitive username login
        res_login_ci = self.client.post("/api/auth/login", json={
            "username": "TESTuser",
            "password": "securepassword123"
        })
        self.assertEqual(res_login_ci.status_code, 200)
        
        # 2c. Email login
        res_login_email = self.client.post("/api/auth/login", json={
            "username": "testuser@instaml.com",
            "password": "securepassword123"
        })
        self.assertEqual(res_login_email.status_code, 200)
        
        # 2d. Case-insensitive email login
        res_login_email_ci = self.client.post("/api/auth/login", json={
            "username": "TESTuser@INSTAML.com",
            "password": "securepassword123"
        })
        self.assertEqual(res_login_email_ci.status_code, 200)
        
        # 3. Fetch user info
        headers = {"Authorization": f"Bearer {token}"}
        res_me = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.json()["username"], "testuser")

    def test_project_isolation(self):
        # Create user A & token
        self.client.post("/api/auth/register", json={"username": "userA", "email": "a@instaml.com", "password": "pass"})
        tokenA = self.client.post("/api/auth/login", json={"username": "userA", "password": "pass"}).json()["access_token"]
        
        # Create user B & token
        self.client.post("/api/auth/register", json={"username": "userB", "email": "b@instaml.com", "password": "pass"})
        tokenB = self.client.post("/api/auth/login", json={"username": "userB", "password": "pass"}).json()["access_token"]

        # User A creates a project
        proj_payload = {"name": "A's Project", "description": "Tabular model", "data_type": "tabular"}
        headersA = {"Authorization": f"Bearer {tokenA}"}
        res_proj = self.client.post("/api/projects", json=proj_payload, headers=headersA)
        self.assertEqual(res_proj.status_code, 200)
        proj_id = res_proj.json()["id"]

        # User B lists projects - should be empty
        headersB = {"Authorization": f"Bearer {tokenB}"}
        res_list_B = self.client.get("/api/projects", headers=headersB)
        self.assertEqual(res_list_B.status_code, 200)
        self.assertEqual(len(res_list_B.json()), 0)

        # User B attempts to access User A's project - should be 404/401 unauthorized
        res_access_B = self.client.get(f"/api/projects/{proj_id}", headers=headersB)
        self.assertEqual(res_access_B.status_code, 404)

    def test_dim_reduction_endpoint(self):
        # 1. Register & login user
        self.client.post("/api/auth/register", json={"username": "user", "email": "user@instaml.com", "password": "pass"})
        token = self.client.post("/api/auth/login", json={"username": "user", "password": "pass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Create project
        proj_res = self.client.post("/api/projects", json={"name": "Test Proj", "description": "Tabular", "data_type": "tabular"}, headers=headers)
        proj_id = proj_res.json()["id"]
        
        # 3. Create a dummy tabular file (parquet) and dataset version in database
        import pandas as pd
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0] * 5,
            "col2": [5.0, 4.0, 3.0, 2.0, 1.0] * 5,
            "target": ["A", "B", "A", "B", "A"] * 5
        })
        
        # Save temp file
        temp_dir = Path(__file__).resolve().parent / "test_data"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / "test_dataset.parquet"
        df.to_parquet(file_path)
        
        try:
            # Update target_col in DB for project
            execute_sync("UPDATE projects SET target_col = 'target' WHERE id = ?", [proj_id])
            
            # Insert dataset version
            import json
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    "v1", "Data Upload", "Test Upload", 25, 3,
                    json.dumps(["col1", "col2", "target"]),
                    json.dumps({"col1": "float64", "col2": "float64", "target": "object"}),
                    json.dumps({}),
                    str(file_path),
                    proj_id
                ]
            )
            
            # 4. Test dim-reduction endpoint
            # PCA
            res = self.client.get(f"/api/projects/{proj_id}/eda/dim-reduction?method=PCA", headers=headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["method"], "PCA")
            self.assertIn("points", data)
            self.assertEqual(len(data["points"]), 25)
            
            # t-SNE
            res = self.client.get(f"/api/projects/{proj_id}/eda/dim-reduction?method=TSNE", headers=headers)
            self.assertEqual(res.status_code, 200)
            
            # LDA
            res = self.client.get(f"/api/projects/{proj_id}/eda/dim-reduction?method=LDA", headers=headers)
            self.assertEqual(res.status_code, 200)
            
            # UMAP (falls back to Isomap/SpectralEmbedding)
            res = self.client.get(f"/api/projects/{proj_id}/eda/dim-reduction?method=UMAP", headers=headers)
            self.assertEqual(res.status_code, 200)
            
        finally:
            if file_path.exists():
                file_path.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except:
                    pass

    def test_boxplot_endpoint(self):
        # 1. Register & login user
        self.client.post("/api/auth/register", json={"username": "boxplotuser", "email": "boxplot@instaml.com", "password": "pass"})
        token = self.client.post("/api/auth/login", json={"username": "boxplotuser", "password": "pass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Create project
        proj_res = self.client.post("/api/projects", json={"name": "Boxplot Proj", "description": "Tabular", "data_type": "tabular"}, headers=headers)
        proj_id = proj_res.json()["id"]
        
        # 3. Create a dummy tabular file (parquet) and dataset version in database
        import pandas as pd
        df = pd.DataFrame({
            "col1": [10.0, 12.0, 11.0, 15.0, 9.0, 8.0, 50.0, -10.0]  # Includes outliers: 50.0 and -10.0
        })
        
        # Save temp file
        temp_dir = Path(__file__).resolve().parent / "test_data_boxplot"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / "test_boxplot.parquet"
        df.to_parquet(file_path)
        
        try:
            # Insert dataset version
            import json
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    "v2", "Data Upload", "Test Boxplot Upload", 8, 1,
                    json.dumps(["col1"]),
                    json.dumps({"col1": "float64"}),
                    json.dumps({}),
                    str(file_path),
                    proj_id
                ]
            )
            
            # 4. Test boxplot endpoint
            res = self.client.get(f"/api/projects/{proj_id}/eda/boxplot/col1", headers=headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            
            self.assertEqual(data["column"], "col1")
            self.assertIn("q1", data)
            self.assertIn("median", data)
            self.assertIn("q3", data)
            self.assertIn("lower_whisker", data)
            self.assertIn("upper_whisker", data)
            self.assertIn("min", data)
            self.assertIn("max", data)
            self.assertIn("outliers", data)
            self.assertEqual(data["total_rows"], 8)
            
            # col1 has min = -10, max = 50.
            # q1 (25th percentile) and q3 (75th percentile) of:
            # sorted: -10, 8, 9, 10, 11, 12, 15, 50
            # median is (10 + 11)/2 = 10.5
            self.assertEqual(data["median"], 10.5)
            self.assertEqual(data["min"], -10.0)
            self.assertEqual(data["max"], 50.0)
            
        finally:
            if file_path.exists():
                file_path.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except:
                    pass

    def test_pca_preprocessing(self):
        # 1. Register & login user
        self.client.post("/api/auth/register", json={"username": "pcauser", "email": "pca@instaml.com", "password": "pass"})
        token = self.client.post("/api/auth/login", json={"username": "pcauser", "password": "pass"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Create project
        proj_res = self.client.post("/api/projects", json={"name": "PCA Proj", "description": "Tabular", "data_type": "tabular"}, headers=headers)
        proj_id = proj_res.json()["id"]
        
        # 3. Create a dummy tabular file (parquet) and dataset version
        import pandas as pd
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": [2.0, 3.0, 4.0, 5.0, 6.0],
            "target": ["A", "B", "A", "B", "A"]
        })
        
        # Save temp file
        temp_dir = Path(__file__).resolve().parent / "test_data_pca"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / "test_pca.parquet"
        df.to_parquet(file_path)
        
        try:
            # Insert dataset version
            import json
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    "v3", "Data Upload", "Test PCA Upload", 5, 4,
                    json.dumps(["col1", "col2", "col3", "target"]),
                    json.dumps({"col1": "float64", "col2": "float64", "col3": "float64", "target": "object"}),
                    json.dumps({}),
                    str(file_path),
                    proj_id
                ]
            )
            
            # 4. Test preprocess endpoint with PCA operation
            preprocess_payload = {
                "operations": [
                    {"op": "pca", "columns": ["col1", "col2", "col3"], "n_components": 2}
                ]
            }
            res = self.client.post(f"/api/projects/{proj_id}/preprocess", json=preprocess_payload, headers=headers)
            self.assertEqual(res.status_code, 200)
            
            # 5. Verify preprocessed dataset version shape & columns
            res_curr = self.client.get(f"/api/projects/{proj_id}/data/current", headers=headers)
            self.assertEqual(res_curr.status_code, 200)
            data = res_curr.json()
            self.assertEqual(data["shape"], [5, 3]) # target + PC1 + PC2
            self.assertIn("PC1", data["columns"])
            self.assertIn("PC2", data["columns"])
            self.assertIn("target", data["columns"])
            self.assertNotIn("col1", data["columns"])
            
        finally:
            if file_path.exists():
                file_path.unlink()
            if temp_dir.exists():
                try:
                    temp_dir.rmdir()
                except:
                    pass

if __name__ == '__main__':
    unittest.main()
