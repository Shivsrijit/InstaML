# InstaML — No-Code Machine Learning Platform

## Overview

InstaML is a comprehensive no-code machine learning platform designed to democratize AI development. It provides an intuitive web-based interface that enables users to build, train, evaluate, and deploy machine learning models without writing code. The platform serves students learning ML concepts, researchers prototyping models, and business professionals solving real-world problems.

---

## Core Features

*   **Intelligent Data Upload**: CSV, Excel, Parquet, and ZIP files (containing structured images/audio) with auto-modality detection.
*   **Interactive Preprocessing**: Sequential multi-stage pipeline configuration (handling missing values, dropping columns, scaling features, outlier removals, and category encoders).
*   **Exploratory Data Analysis (EDA)**: Interactive statistical description cards, Univariate visuals (Box plot, Histogram, KDE, CDF, Pie chart), Bivariate correlations, and K-Means/PCA projections.
*   **Model Training**: Select custom classifiers or regressors (XGBoost, Random Forest, Multi-Layer Perceptrons) with automated hyperparameter optimization (using Optuna trials) and K-Fold cross-validation.
*   **Specialized Modalities**: Out-of-the-box pretrained networks for NLP (Named Entity Recognition, Sentiment, Text Summarization), Computer Vision (Face Detection, OCR, Image Super Resolution/Denoising), and Audio (Speech Recognition, Noise Reduction).
*   **Deployment**: One-click REST API deployments, real-time prediction widgets, bulk CSV predictions, and version restore capabilities.

---

## Documentation

For deep dives into configuration, architecture, and coding standards, refer to:

- 📘 **[System Architecture](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/architecture.md)** – Detailed microservices mapping, databases schemas, and storage sync caching.
- ⚙️ **[Installation & Setup Guide](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/setup.md)** – Launch the project locally with or without Docker.
- 🌐 **[Production Deployment Guide](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/deployment.md)** – Detailed instructions for hosting the platform on Render, Vercel, Turso, and Supabase free-tiers.
- 🚀 **[Contributing Guide](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/contribution.md)** – Repository structure, coding protocols, and test suites executions.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | React 18, Vite, Recharts, Vanilla CSS | Interactive single page client |
| **Gateway** | FastAPI, Uvicorn, Python-Jose, Bcrypt | Orchestrator, authorization, and route security |
| **Microservices** | FastAPI, Scikit-Learn, Joblib, Optuna | Background trainers and inference predictors |
| **Database** | SQLite (local) / Turso LibSQL (cloud) | Schema migrations and user/project indexing |
| **Storage** | Local Disk / Supabase Storage REST API | Caching sync for dataset check-points & pickled models |

---

## Quick Start (Docker Compose)

The fastest way to boot the complete environment is via Docker:

1.  Make sure you have Docker Desktop running.
2.  Launch the services:
    ```bash
    docker-compose up --build
    ```
3.  Access the React client at: **[http://localhost:5173](http://localhost:5173)** (Gateway runs on `8000`).

*(Refer to [docs/setup.md](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/setup.md) for running native python/npm processes directly on your host loops).*

---

## Contributing

We welcome contributions to enhance InstaML's capabilities! Please review our **[Contributing Guide](file:///c:/Users/SSN/OneDrive%20-%20Shiv%20Nadar%20University%20-%20Chennai/Desktop/instaml/docs/contribution.md)** and feel free to submit issues or pull requests.

---

## License

This project is licensed under the **MIT License**.

---

✨ **Built for the AI community** | ⭐ **Star this repository if you find it helpful**
