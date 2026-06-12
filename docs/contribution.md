# InstaML Contribution Guide

This guide details the codebase architecture, design patterns, testing files, and commands for contributors.

## 1. Project Directory Layout

The workspace is organized into two main folders:

```
instaml/                   # Root Workspace
├── backend/               # Python Microservices & Unified Trainers
│   ├── app/               # FastAPI Web Applications
│   │   ├── api/           # API Routers (auth, data, deployment, eda, nlp, projects, training)
│   │   ├── core/          # App config, workers, and Supabase client caching helper
│   │   └── db/            # Pydantic schemas, database connections, and tables setup
│   ├── core/              # Core UnifiedModelTrainer package library code
│   ├── storage/           # Local folder directory caches (Git-ignored)
│   └── *.py               # Test suites (test_supabase_mock.py, test_cv_and_mlp.py, etc.)
├── frontend/              # React Client SPA
│   ├── src/               # React codebase
│   │   ├── components/    # Background canvas layouts, sidebars, page navigations
│   │   ├── context/       # AuthContext session states
│   │   ├── pages/         # Views (Dashboard, Upload, Preprocess, EDA, Train, Test, Deploy)
│   │   └── services/      # Axios request handlers
│   └── package.json       # Node dependency manifest
├── docs/                  # Platform documentation files (architecture, setup, deployment)
└── docker-compose.yml     # Docker microservices orchestrator
```

## 2. Database Schema Details

InstaML utilizes SQLite locally (`backend/instaml.db`) and Turso database clusters (LibSQL) in cloud environments. The tables are configured as follows:

### users Table
*   `id`: INTEGER PRIMARY KEY AUTOINCREMENT
*   `username`: TEXT UNIQUE NOT NULL (regex constrained to letters, numbers, underscores, and hyphens)
*   `email`: TEXT UNIQUE NOT NULL (validated via EmailStr)
*   `hashed_password`: TEXT NOT NULL (encrypted via bcrypt)
*   `created_at`: TEXT DEFAULT (datetime('now'))

### projects Table
*   `id`: INTEGER PRIMARY KEY AUTOINCREMENT
*   `name`: TEXT NOT NULL
*   `description`: TEXT
*   `data_type`: TEXT NOT NULL (tabular, text, image, audio, multi_dimensional)
*   `task`: TEXT DEFAULT 'Classification'
*   `target_col`: TEXT
*   `created_at`: TEXT DEFAULT (datetime('now'))
*   `user_id`: INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE

### dataset_versions Table
*   `id`: INTEGER PRIMARY KEY AUTOINCREMENT
*   `version_id`: TEXT UNIQUE NOT NULL
*   `step_name`: TEXT NOT NULL (e.g. Data Upload, Preprocessing)
*   `description`: TEXT
*   `shape_rows`: INTEGER
*   `shape_cols`: INTEGER
*   `columns_json`: TEXT
*   `dtypes_json`: TEXT
*   `metadata_json`: TEXT (stores preprocessing operations logs)
*   `file_path`: TEXT NOT NULL
*   `created_at`: TEXT DEFAULT (datetime('now'))
*   `project_id`: INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE

### models Table
*   `id`: INTEGER PRIMARY KEY AUTOINCREMENT
*   `name`: TEXT NOT NULL
*   `model_type`: TEXT NOT NULL (e.g. Random Forest, MLP)
*   `target_col`: TEXT
*   `metrics_json`: TEXT (stores accuracy, precision, recall, confusion matrix)
*   `file_path`: TEXT NOT NULL (path to pickled pipeline)
*   `is_deployed`: INTEGER DEFAULT 0
*   `created_at`: TEXT DEFAULT (datetime('now'))
*   `project_id`: INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE
*   `dataset_version_id`: INTEGER REFERENCES dataset_versions(id)

## 3. Core Coding Standards and Protocols

*   **SQL Injection Protection**: Never build raw query strings using string formatting or variable concatenation. Always pass arguments separately using parameter arrays `[arg1, arg2]` inside the session wrappers:
    ```python
    query_sync("SELECT * FROM projects WHERE id = ? AND user_id = ?", [project_id, user_id])
    ```
*   **Access Control Validation**: Prior to handling dataset processing, model evaluations, downloads, or configurations, always assert project ownership by verifying the project belongs to the authenticated user ID:
    ```python
    project = get_user_project(project_id, current_user.id)
    ```
*   **Password Safety**: Enforce `bcrypt` hashing on all user accounts. Passwords must validate against the Pydantic min-length constraints before execution.
*   **Caching & Storage Caching**: The platform operates statefully locally and statelessly in the cloud. Do not block logic if cloud parameters (`SUPABASE_URL`, `SUPABASE_KEY`) are missing; fall back to local disk read/write storage under the directory `backend/storage/`.

## 4. Running Verification Tests

Run the backend test suites after modifying routes, schemas, database tables, or machine learning algorithms. Ensure you have activated your Python virtual environment inside the `backend` folder first.

### A. Isolated Supabase Mock Tests
Verify that path resolution, environment parsing, REST headers, and fallback mechanisms work without breaking actual bucket data:
```bash
python -m unittest backend/test_supabase_mock.py
```
This test file asserts:
*   `is_supabase_configured()` logic with/without env variables.
*   Convertion of absolute windows/unix paths into standardized cloud keys under the `user_X` folder.
*   Mocks `requests.get` / `requests.post` API calls to check authentication headers.

### B. Machine Learning & K-Fold Tests
Test custom algorithms, cross-validation scoring, and Optuna tuning:
```bash
python -m unittest backend/test_cv_and_mlp.py
```
This test file asserts:
*   Training of Multi-Layer Perceptron (MLP) Classifier and Regressor models.
*   Verification of K-Fold cross-validation metrics.
*   Optuna hyperparameter tuning runs.

### C. End-to-End Modality Tests
Run full E2E runs covering sentiment, CV face detection, and audio noise reduction:
```bash
python backend/test_modalities_e2e.py
```
This test file asserts:
*   Project creation, dataset uploads, and mock WAV/TXT/PNG dataset setups.
*   Triggering specialized trainers and polling statuses.
*   Model deployments and prediction calls.

### D. User Authentication Tests
Verify registration casing rules, user lookups, and hashed login matches:
```bash
python backend/test_running_register.py
python backend/test_running_login.py
```
These test files assert:
*   Validations for duplicate emails or mixed-case usernames.
*   Unique database constraint throws.
*   Hashed password verification with bcrypt.
