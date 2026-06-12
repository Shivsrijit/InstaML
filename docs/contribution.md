# InstaML Contribution Guide

Thank you for your interest in contributing to InstaML! Follow these steps and guidelines to write code, configure components, and run verification tests.

---

## 1. Project Directory Structure

Ensure code edits are confined to the following structural directories:

```
instaml/                   # Root Workspace
├── backend/               # Python Microservices & Models
│   ├── app/               # FastAPI Application
│   │   ├── api/           # API Routers (auth, data, eda, training, deployment, etc.)
│   │   ├── core/          # Core Business Logic (caching, configurations, training workers)
│   │   └── db/            # Database Sessions, Schema Definitions, Migrations
│   ├── core/              # Unified Machine Learning Trainers (UnifiedModelTrainer)
│   ├── storage/           # Ephemeral Local Cache Files (Git-ignored)
│   └── *.py               # Test Suites (test_supabase_mock.py, test_cv_and_mlp.py, etc.)
├── frontend/              # React SPA
│   ├── src/               # React Codebase
│   │   ├── components/    # Reusable UI Elements (Sidebar, Charts, Modals)
│   │   ├── context/       # Auth state providers
│   │   ├── pages/         # Page Views (Dashboard, EDA, Preprocessing, Train, Deployment)
│   │   └── services/      # Axios API connection endpoints
│   └── package.json       # Node package manager configurations
├── docs/                  # Platform Documentation
└── docker-compose.yml     # Container Orchestration
```

---

## 2. Core Development Standards

- **Parameterization**: Never build raw query strings using string formatting or variable concatenation. Always pass arguments separately using parameter arrays `[arg1, arg2]` inside the session wrappers:
  ```python
  query_sync("SELECT * FROM projects WHERE id = ? AND user_id = ?", [project_id, user_id])
  ```
- **Password Safety**: Enforce `bcrypt` hashing on all user accounts. Passwords must validate against the Pydantic min-length constraints before execution.
- **Access Control Check**: Prior to handling dataset processing, model evaluations, or downloads, always assert project ownership:
  ```python
  project = get_user_project(project_id, current_user.id)
  ```
- **Local Fallback**: Maintain compatibility for both local offline installations (using local SQLite + local folders) and production setups (using Turso + Supabase). Do not hardcode cloud credentials.

---

## 3. Running Verification Tests

Run the backend test suites after modifying routes, schemas, database tables, or machine learning algorithms. Ensure you have activated your Python virtual environment inside the `backend` folder first.

### A. Isolated Supabase Mock Tests
Verify that path resolution and fallback mechanisms work without breaking actual bucket data:
```bash
python -m unittest backend/test_supabase_mock.py
```

### B. Machine Learning & K-Fold Tests
Test custom algorithms, cross-validation scoring, and Optuna tuning:
```bash
python -m unittest backend/test_cv_and_mlp.py
```

### C. End-to-End Modality Tests
Run full E2E runs covering sentiment, CV face detection, and audio noise reduction:
```bash
python backend/test_modalities_e2e.py
```

### D. User Authentication Tests
Verify registration casing rules, user lookups, and hashed login matches:
```bash
python backend/test_running_register.py
python backend/test_running_login.py
```
