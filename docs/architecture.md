# InstaML Architecture

This document provides a detailed breakdown of the InstaML system architecture, services interaction, data storage configurations, and communication flows.

---

## 1. System Overview

InstaML is built on a **decoupled microservices architecture** designed to separate API orchestration, model training execution, and real-time inference. This separation allows training tasks (which are highly CPU/memory intensive) and prediction tasks (which require high availability and low latency) to scale independently.

```mermaid
graph TD
    subgraph Client Layer
        ReactClient["React SPA (Vite)"]
    end

    subgraph Service Layer (Gateway & Microservices)
        Gateway["Gateway API (FastAPI, Port 8000)"]
        Trainer["Trainer Service (FastAPI, Port 8001)"]
        Predictor["Predictor Service (FastAPI, Port 8002)"]
    end

    subgraph Database Layer
        SQLite["Local SQLite (instaml.db)"]
        Turso["Turso Database (LibSQL, Cloud)"]
    end

    subgraph Storage Layer
        LocalStorage["Local Directory (backend/storage)"]
        Supabase["Supabase Storage Bucket"]
    end

    ReactClient <-->|HTTP / JSON / JWT| Gateway
    Gateway <-->|HTTP / JSON| Trainer
    Gateway <-->|HTTP / JSON / Streams| Predictor

    Gateway <-->|SQL Queries| SQLite
    Gateway <-->|SQL Queries| Turso
    Trainer <-->|SQL Queries| SQLite
    Trainer <-->|SQL Queries| Turso
    Predictor <-->|SQL Queries| SQLite
    Predictor <-->|SQL Queries| Turso

    Gateway <-->|Read / Write| LocalStorage
    Trainer <-->|Read / Write| LocalStorage
    Predictor <-->|Read / Write| LocalStorage

    LocalStorage <-->|Sync / Caching Fallback| Supabase
```

---

## 2. Component Descriptions

### A. Frontend Client (React)
- **Tech Stack**: React 18, Vite, Vanilla CSS, Recharts (for EDA and model evaluation plotting), React Router DOM, React Hot Toast.
- **Role**: Provides the no-code workspace. It guides the user step-by-step through:
  - Dataset Upload and modality selections.
  - Interactive Preprocessing (data cleaning, scaling, encoding).
  - Exploratory Data Analysis (univariate distribution plots, bivariate correlations, scatter/box charts).
  - Model Training specifications and real-time execution logs.
  - Model evaluation comparison and deployed test interfaces.
- **Authentication**: JWT tokens saved in memory/localStorage, attached to all requests in the `Authorization: Bearer <token>` header.

### B. Gateway API (Port 8000)
- **Tech Stack**: FastAPI, Uvicorn, SQLite/LibSQL client, Jose (JWT), Bcrypt.
- **Role**: The single entrypoint for the frontend client. It manages:
  - Authentication (user registration, login, token issue/refresh).
  - Projects and dataset versions metadata.
  - Preprocessing task triggers (delegated to `data_handler.py`).
  - Access control and ownership isolation (verifies `user_id` on all project-related requests).
  - Proxying training triggers to the Trainer Service and inference payloads to the Predictor Service.

### C. Trainer Service (Port 8001)
- **Tech Stack**: FastAPI, Uvicorn, Optuna, Scikit-Learn.
- **Role**: A background service dedicated to running model training jobs.
  - Receives asynchronous training requests from the Gateway.
  - Spawns background training worker threads using `UnifiedModelTrainer`.
  - Performs automated hyperparameter optimization (using Optuna trials) and K-Fold cross-validation.
  - Serializes the final trained pipeline as a `.pkl` file.
  - Captures and stores execution logs in memory.

### D. Predictor Service (Port 8002)
- **Tech Stack**: FastAPI, Uvicorn, Joblib, Pandas.
- **Role**: An inference service optimized for low latency.
  - Loads trained pipeline `.pkl` files dynamically into memory.
  - Preprocesses raw prediction features matching the historical preprocessing operations of the target dataset version.
  - Executes predictions and returns confidence scores / probabilities.
  - Reconstructs predicted values back to the original target column space if inverse transforms (like log, exp, square root) were applied.

---

## 3. Database Layer & Multi-User Isolation

InstaML handles database orchestration using `libsql-client`, supporting local file-based SQLite databases and remote Turso cloud clusters.

### A. Schema Diagram
```
  ┌──────────────┐          ┌────────────────┐
  │    users     │          │    projects    │
  ├──────────────┤          ├────────────────┤
  │ id (PK)      │◄──┐      │ id (PK)        │◄──┐
  │ username     │   └──────│ user_id (FK)   │   │
  │ email        │          │ name           │   │
  │ password     │          │ description    │   │
  │ created_at   │          │ data_type      │   │
  └──────────────┘          │ task           │   │
                            │ target_col     │   │
                            └────────────────┘   │
                                     ▲           │
                                     │           │
           ┌─────────────────────────┴─┐         │
           │                           │         │
  ┌────────┴─────────┐        ┌────────┴────────┐│
  │ dataset_versions │        │     models      ││
  ├──────────────────┤        ├─────────────────┤│
  │ id (PK)          │◄──┐    │ id (PK)         ││
  │ project_id (FK)  │   └────│ version_id (FK) ││
  │ version_id       │        │ project_id (FK) │┘
  │ step_name        │        │ name            │
  │ description      │        │ model_type      │
  │ file_path        │        │ target_col      │
  │ shape_rows       │        │ metrics_json    │
  │ shape_cols       │        │ file_path       │
  │ columns_json     │        │ is_deployed     │
  │ dtypes_json      │        └─────────────────┘
  │ metadata_json    │
  └──────────────────┘
```

### B. Security and Isolation Control
1. **Ownership Validation**: Every endpoint verifying user resource requests queries:
   ```sql
   SELECT * FROM projects WHERE id = ? AND user_id = ?
   ```
   If no rows are found, a `404 Not Found` response is raised immediately, preventing users from accessing or modifying other accounts' projects.
2. **Parameterized Queries**: All queries are parameterized (`?` place-holders) inside the database session manager to eliminate SQL injection vulnerabilities.
3. **Password Hashing**: User passwords are encrypted using `bcrypt` (gensalt rounds) before inserting into the database, with constant-time matching implemented during authentication.

---

## 4. File Storage Layer & Supabase Sync

The storage architecture implements an automatic caching sync wrapper to support both stateful local environments and stateless cloud environments.

```
                  ┌───────────────────────────────┐
                  │   Client requests file read   │
                  └───────────────┬───────────────┘
                                  │
                       [File exists locally?]
                                 / \
                                /   \
                              YES    NO
                              /       \
                             /         \
              ┌─────────────▼───┐     [Supabase configured?]
              │ Read local file │             / \
              └─────────────────┘            /   \
                                           YES    NO
                                           /       \
             ┌────────────────────────────▼──┐    ┌─▼────────────────────────┐
             │ Pull file from Supabase       │    │ Error: file not found    │
             │ Save copy to local directory  │    │ (Local fallback fallback)│
             │ Return local path for read    │    └──────────────────────────┘
             └───────────────────────────────┘
```

- **Local Path Formatting**: All paths under the `storage/` directory follow the naming structure:
  `user_<user_id>/project_<project_id>/[data|models]/<version_id_or_model_id>.[parquet|pkl]`
- **REST Syncing**: When `SUPABASE_URL` and `SUPABASE_KEY` are provided, the backend uploads newly created files to the cloud bucket under their relative storage path (e.g. `user_1/project_2/data/...`). When a service needs a dataset or model that isn't cached in its container filesystem, it pulls the file dynamically from Supabase.
