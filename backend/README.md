# InstaML Backend Services

This directory houses the backend codebase for the InstaML platform, structured as a set of FastAPI microservices designed for dataset management, interactive preprocessing, CPU/GPU model training, and real-time/batch prediction.

---

## 1. Services Architecture

The backend consists of three distinct FastAPI applications:

1.  **Gateway API (`backend/app/main.py`)**: Runs on port `8000`. Handles user logins, project creation, EDA stats calculation, preprocessing execution, and forwards training/inference calls to the internal services.
2.  **Trainer Service (`backend/app/main_trainer.py`)**: Runs on port `8001` (isolated to `127.0.0.1` loopback in container builds). Runs background trainer worker threads to optimize hyperparameters (via Optuna) and train Scikit-Learn algorithms (Decision Trees, Random Forests, XGBoost, MLP) and CV/NLP/Audio pipelines.
3.  **Predictor Service (`backend/app/main_predictor.py`)**: Runs on port `8002` (isolated to `127.0.0.1` loopback in container builds). Standardizes raw user input matching the project's historical preprocessing operations and runs low-latency model predictions.

---

## 2. Directory Map

```
backend/
├── app/
│   ├── api/                   # Router Endpoints
│   │   ├── auth.py            # User registration & JWT generation
│   │   ├── data.py            # Dataset uploads, preprocessing, versions
│   │   ├── deployment.py      # Model deploy hooks & streaming downloads
│   │   ├── eda.py             # Descriptive stats, correlations, dimensions
│   │   ├── nlp.py             # Stateless text sentiment & keywords analyzer
│   │   ├── projects.py        # Project creations & deletions
│   │   └── training.py        # Active training job logs & options
│   ├── core/                  # Core Systems
│   │   ├── config.py          # App config & CORS permissions
│   │   ├── data_handler.py    # Parquet/pickle utilities & preprocessing pipelines
│   │   ├── security.py        # Bcrypt utilities & JWT signing
│   │   ├── specialized_pipelines.py # CV/Audio/NLP pre-trained networks
│   │   ├── supabase_storage.py # Cloud storage caching wrapper
│   │   └── training_worker.py # Background thread training runner
│   └── db/                    # Data Access
│       ├── schemas.py         # Pydantic validation schemas
│       └── session.py         # libSQL connection manager & migrations
├── core/                      # Shared Trainer Package
│   └── ML_models/             # Classifiers, Regressors, & CV/Audio modals
├── requirements.txt           # Python packages manifest
└── *.py                       # Backend Unit & Integration Tests
```

---

## 3. Database & Migrations

InstaML utilizes `libsql-client` to support SQLite locally and Turso LibSQL database clusters in production.

### Migrations
Migrations run automatically on service startup (via `run_migrations_sync()` inside `session.py`). The schema consists of four tables:

- **`users`**: Manages credentials, storing hashed passwords using `bcrypt`.
- **`projects`**: Tracks user projects, categorized by modality (`tabular`, `text`, `image`, `audio`, `multi_dimensional`) and tasks (`Classification`, `Regression`, specialized pre-trained tasks).
- **`dataset_versions`**: Records dataset check-points, column json lists, stats, and files paths.
- **`models`**: Tracks trained models, validation metrics, active deployments (`is_deployed`), and pickle output locations.

---

## 4. Caching & File Sync

Datasets and model files are saved using the `supabase_storage` wrapper. 

- **Local Cache**: Written locally to `backend/storage/user_<id>/project_<id>/...` first.
- **Supabase Cloud Sync**: If environment variables (`SUPABASE_URL`, `SUPABASE_KEY`) are present, files are mirrored in the cloud bucket. If a container needs a file that isn't cached locally (e.g., in stateless Render environments), it automatically downloads the file from Supabase Storage prior to loading it into memory.

---

## 5. Verification Commands

Run backend verification suites inside this directory:
```bash
# Run Supabase mock client tests
python -m unittest test_supabase_mock.py

# Run modeling pipelines & cross-validation tests
python -m unittest test_cv_and_mlp.py

# Run E2E text, image, and audio pipeline runs
python test_modalities_e2e.py
```
