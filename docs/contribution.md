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

## 2. Coding Standards and Protocols

*   **Database Query Formatting**: Always parameterize all database queries. Avoid raw string injections or concatenation. Pass arguments separately as list parameters:
    ```python
    query_sync("SELECT * FROM projects WHERE id = ? AND user_id = ?", [project_id, user_id])
    ```
*   **Access Constraints Check**: Before any operations read or write datasets, versions, models, or evaluations, verify the project's ownership:
    ```python
    project = get_user_project(project_id, current_user.id)
    ```
*   **Password Encryption**: User passwords must be hashed using `bcrypt` (using dynamic salts) and matched using safe validation routines during logins.
*   **Caching & Syncing Fallback**: The platform operates statefully locally and statelessly in the cloud. Do not block logic if cloud parameters (`SUPABASE_URL`, `SUPABASE_KEY`) are missing; fall back to local disk read/write storage under the directory `backend/storage/`.

## 3. Running Verification Tests

Run the backend test suites after modifying routes, schemas, database tables, or machine learning algorithms. Ensure you have activated your Python virtual environment inside the `backend` folder first.

### Isolated Supabase Mock Tests
Verify that path resolution and fallback mechanisms work without breaking actual bucket data:
```bash
python -m unittest backend/test_supabase_mock.py
```

### Machine Learning & K-Fold Tests
Test custom algorithms, cross-validation scoring, and Optuna tuning:
```bash
python -m unittest backend/test_cv_and_mlp.py
```

### End-to-End Modality Tests
Run full E2E runs covering sentiment, CV face detection, and audio noise reduction:
```bash
python backend/test_modalities_e2e.py
```

### User Authentication Tests
Verify registration casing rules, user lookups, and hashed login matches:
```bash
python backend/test_running_register.py
python backend/test_running_login.py
```
