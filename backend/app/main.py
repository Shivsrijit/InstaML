import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Inject root paths for dependencies
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.config import ORIGINS
from backend.app.db.session import run_migrations_sync
from backend.app.api import auth, projects, data, eda, training, deployment, nlp

# Boot database tables
run_migrations_sync()

app = FastAPI(
    title="InstaML Backend API",
    description="REST API server supporting multi-user authentication, datasets version control, and ML pipelines execution.",
    version="1.0.0"
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(eda.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(deployment.router, prefix="/api")
app.include_router(nlp.router, prefix="/api")

@app.get("/health")
def health_check():
    """Simple verification route."""
    return {"status": "healthy", "service": "instaml-backend"}
