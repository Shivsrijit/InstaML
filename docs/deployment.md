# InstaML Production Deployment Guide

This guide details how to deploy the full-stack InstaML platform on **cloud free-tiers** (Vercel, Render, Turso, and Supabase) without leaking credentials or exposing sensitive resources.

---

## 1. Cloud Database Setup (Turso)

Turso provides a serverless LibSQL database built on SQLite, perfect for low-latency queries and stateless containers.

1.  **Register/Login**: Go to [turso.tech](https://turso.tech) and create an account.
2.  **Create Database**: Create a new database named `instaml`.
3.  **Retrieve Connection Details**:
    *   **Database URL**: Copy the connection URL. It looks like: `libsql://instaml-username.turso.io`.
    *   **Auth Token**: Create a new access token for the database.
4.  Keep these two credentials safe. They will be added as environment variables in the backend services.

---

## 2. Cloud Object Storage Setup (Supabase)

Supabase Storage hosts raw uploaded datasets and final serialized model `.pkl` pipelines.

1.  **Register/Login**: Go to [supabase.com](https://supabase.com) and create a new project.
2.  **Create Storage Bucket**:
    *   Navigate to **Storage** in the left sidebar.
    *   Click **New Bucket** and name it `instaml`.
    *   Make the bucket **Public** or **Private** (authenticated is recommended since our backend uses the service key).
3.  **Retrieve Keys**:
    *   Go to **Project Settings** → **API**.
    *   Copy the **Project URL** (e.g. `https://xyz.supabase.co`).
    *   Copy the **service_role** key (this has bypass row-level-security permissions, allowing the backend services to securely read/write objects).
4.  Keep these credentials safe for the backend configurations.

---

## 3. Backend Services Deployment (Render)

Render free tier hosts standard web services. Since InstaML uses a microservices layout, you will deploy **three separate Web Services** from your repository.

### Service 1: Gateway API (`gateway-api`)
- **Repository Folder Root**: `./`
- **Language**: `Python`
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  ```env
  SECRET_KEY=generate-a-secure-random-jwt-string
  TURSO_DATABASE_URL=libsql://instaml-username.turso.io
  TURSO_AUTH_TOKEN=your-turso-token
  SUPABASE_URL=https://your-supabase-id.supabase.co
  SUPABASE_KEY=your-supabase-service-role-key
  SUPABASE_BUCKET=instaml
  TRAINER_SERVICE_URL=https://your-trainer-service.onrender.com
  PREDICTOR_SERVICE_URL=https://your-predictor-service.onrender.com
  ```

### Service 2: Trainer Service (`trainer-service`)
- **Repository Folder Root**: `./`
- **Language**: `Python`
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.app.main_trainer:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  ```env
  TURSO_DATABASE_URL=libsql://instaml-username.turso.io
  TURSO_AUTH_TOKEN=your-turso-token
  SUPABASE_URL=https://your-supabase-id.supabase.co
  SUPABASE_KEY=your-supabase-service-role-key
  SUPABASE_BUCKET=instaml
  ```

### Service 3: Predictor Service (`predictor-service`)
- **Repository Folder Root**: `./`
- **Language**: `Python`
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.app.main_predictor:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  ```env
  TURSO_DATABASE_URL=libsql://instaml-username.turso.io
  TURSO_AUTH_TOKEN=your-turso-token
  SUPABASE_URL=https://your-supabase-id.supabase.co
  SUPABASE_KEY=your-supabase-service-role-key
  SUPABASE_BUCKET=instaml
  ```

---

## 4. Frontend Client Deployment (Vercel)

Vercel hosts the React SPA statically and serves it via high-speed CDNs.

1.  **Register/Login**: Go to [vercel.com](https://vercel.com) and import your GitHub repository.
2.  **Configuration Settings**:
    *   **Root Directory**: Set this to `frontend`.
    *   **Framework Preset**: Select `Vite`.
    *   **Build Command**: `npm run build`
    *   **Output Directory**: `dist`
3.  **Environment Variables**:
    *   Add a single environment variable to point Vite to the deployed Gateway API:
        *   **Name**: `VITE_API_URL`
        *   **Value**: `https://your-gateway-api.onrender.com/api`
4.  Click **Deploy**. Once complete, Vercel will provide a public HTTPS URL for your application dashboard!
