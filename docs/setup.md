# InstaML Setup Guide

This guide details how to set up and run the InstaML platform locally, either natively on your host machine or using Docker containers.

## 1. Environment Configurations

Create a file named `.env` in the `backend/` directory. Copy the variables below and adjust them according to your environment.

```env
# JWT Token Signing Security
SECRET_KEY=instaml-super-secret-key-change-in-prod

# Database Configurations
# (Leave empty to use local sqlite file backend/instaml.db automatically)
TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=

# Cloud File Storage (Supabase) Configurations
# (Leave empty to store files on local disk under backend/storage/ automatically)
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_BUCKET=instaml
```

> [!NOTE]
> If you choose to configure Supabase Storage, make sure to log into your Supabase Dashboard, create a bucket named `instaml` (or your custom bucket name), and set storage policies to allow authenticated or public insert/read access.

## 2. Option A: Local Run (Without Docker)

This setup runs the microservices directly on your loopback interface.

### Prerequisites
*   **Python**: Version 3.10 or higher.
*   **Node.js**: Version 18 or higher (with `npm`).

### Step 1: Set Up Backend Microservices
1.  Open a terminal, navigate to the `backend` folder, and set up a virtual environment:
    ```bash
    cd backend
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the three backend services:
    *   **Gateway API** (Port 8000):
        ```bash
        uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
        ```
    *   **Trainer Service** (Port 8001):
        ```bash
        uvicorn app.main_trainer:app --host 127.0.0.1 --port 8001 --reload
        ```
    *   **Predictor Service** (Port 8002):
        ```bash
        uvicorn app.main_predictor:app --host 127.0.0.1 --port 8002 --reload
        ```

### Step 2: Set Up Frontend Client
1.  Open a separate terminal, navigate to the `frontend` folder:
    ```bash
    cd frontend
    ```
2.  Install npm packages:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
4.  Access the web interface in your browser at: **[http://localhost:5173](http://localhost:5173)**

## 3. Option B: Containerized Run (With Docker)

This setup launches all microservices and the React client inside container sandboxes using a single command.

### Prerequisites
*   **Docker Desktop** installed and running on your system.

### Step 1: Configure Environment Variables
*   Ensure `backend/.env` is set up. The environment variables are forwarded to the containers automatically by Docker Compose.

### Step 2: Launch Containers
1.  Navigate to the root directory containing the `docker-compose.yml` file.
2.  Build and start the containers:
    ```bash
    docker-compose up --build
    ```
3.  Docker Compose will launch:
    *   `gateway-api` on port `8000`
    *   `trainer-service` on port `8001` (bound to `127.0.0.1` loopback for security)
    *   `predictor-service` on port `8002` (bound to `127.0.0.1` loopback for security)
    *   `frontend` on port `5173`
4.  Access the web dashboard at: **[http://localhost:5173](http://localhost:5173)**

To stop the services, run:
```bash
docker-compose down
```
