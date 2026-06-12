# InstaML Frontend Client

This directory houses the React Single Page Application (SPA) for the InstaML platform. It provides a visual, drag-and-drop workspace that interfaces with the backend Gateway API.

---

## 1. Technology Stack

-   **Core Framework**: React 18 with Vite (for fast module reloading and bundling).
-   **Routing**: React Router DOM (v6) for multi-stage pipelines and authentication guards.
-   **State & API**: Context API (`AuthContext`) for credentials persistent sessions, Axios for backend HTTP communication.
-   **Charting & Visuals**: Recharts (for drawing advanced EDA summaries, correlation heatmaps, line plots, and training metrics), Vanilla CSS for glassmorphism layout, and custom SVGs for indicators.

---

## 2. Directory Structure

```
frontend/
├── public/                 # Static public assets
├── src/                    # Source code
│   ├── assets/             # Images & visual styles
│   ├── components/         # Reusable UI Components
│   │   ├── InteractiveBackground.jsx # Smooth animated canvas backdrop
│   │   ├── Sidebar.jsx     # Side navigation menu
│   │   └── sidebarHelper.js # Mobile menu trigger helpers
│   ├── context/            # Authentication Context Providers
│   │   └── AuthContext.jsx # Logs users in, parses JWTs, and handles logging out
│   ├── pages/              # Primary View Pages
│   │   ├── Dashboard.jsx   # List user projects and modalities
│   │   ├── DataUpload.jsx  # Dataset upload drag-drop & modalities configurations
│   │   ├── Deployment.jsx  # Realtime & batch predict test client & deployed APIs
│   │   ├── EDA.jsx         # Statistical summaries, univariate and bivariate plots
│   │   ├── FeatureEngineering.jsx # Custom target lineage transformations & filters
│   │   ├── Guidelines.jsx  # Machine Learning tutorials and platform guides
│   │   ├── Landing.jsx     # Marketing home screen and welcome panel
│   │   ├── Login.jsx       # User credentials form & token store
│   │   ├── Preprocessing.jsx # Multi-stage preprocessing step configurator
│   │   ├── Register.jsx    # New account form & constraints checks
│   │   ├── Settings.jsx    # User email updates & password alterations
│   │   ├── TestModel.jsx   # Training results, evaluations, and confusion matrices
│   │   ├── TrainModel.jsx  # Model selectors, tuning options, and training console
│   │   └── Versions.jsx    # Checkpoints list & restore modal triggers
│   ├── services/           # Axios connection managers
│   │   └── api.js          # API client with token insertion interceptors
│   ├── App.jsx             # Routes map & pipeline routing guards
│   ├── index.css           # Global theme variables, colors, and design system
│   └── main.jsx            # DOM entrypoint
├── package.json            # Node configurations
└── vite.config.js          # Vite configuration options
```

---

## 3. Pipeline Views & Router Mapping

The platform is designed around a multi-stage **PipelineLayout** nested router (`/projects/:project_id/`):

1.  **`/upload`**: DataUpload page (upload CSV/Parquet/Excel/ZIP and detect shapes).
2.  **`/preprocess`**: Preprocessing page (interactively clean data and check shapes).
3.  **`/eda`**: EDA page (univariate histogram/boxplot/KDE/CDF, correlation heatmap).
4.  **`/feature-eng`**: FeatureEngineering page (apply math functions to target columns).
5.  **`/train`**: TrainModel page (configure algorithm types, Optuna trials, and start training).
6.  **`/test`**: TestModel page (explore validation scores, confusion matrix, and feature importances).
7.  **`/deploy`**: Deployment page (deploy models, test realtime JSON prediction, upload files for predict, or predict batch CSV).
8.  **`/versions`**: Versions page (view versions log and restore state checkpoints).
9.  **`/playground`**: NLPPlayground page (test sentiment and summarization utilities).

---

## 4. Run Scripts

Make sure you have installed Node.js 18+ and have run `npm install` inside this directory first.

### Run Development Server
Launches the React client with Hot Module Replacement (HMR) on port `5173`:
```bash
npm run dev
```

### Build Client
Bundles the React client into static production assets under the `dist/` directory:
```bash
npm run build
```
