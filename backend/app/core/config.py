import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# Load environment variables from backend/.env
load_dotenv(BASE_DIR / ".env")

# Database config
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
    # Clean up standard URL prefixes if present
    clean_url = TURSO_DATABASE_URL
    if clean_url.startswith("libsql://"):
        clean_url = clean_url.replace("libsql://", "", 1)
    elif clean_url.startswith("https://"):
        clean_url = clean_url.replace("https://", "", 1)
    DATABASE_URL = f"sqlite+libsql://{clean_url}/?authToken={TURSO_AUTH_TOKEN}&secure=true"
else:
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/instaml.db")

# Security
import secrets
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Use dynamic cryptographically secure key in production to avoid static fallback vulnerabilities,
    # but use static key in local development to preserve login sessions across hot-reloads.
    if os.getenv("RENDER") or os.getenv("NODE_ENV") == "production":
        SECRET_KEY = secrets.token_hex(32)
    else:
        SECRET_KEY = "instaml-super-secret-key-change-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Allowed CORS Origins
ORIGINS = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://instamlx.vercel.app",  # Production Vercel client
]

cors_env = os.getenv("CORS_ORIGINS")
if cors_env:
    for origin in cors_env.split(","):
        origin = origin.strip()
        if origin and origin not in ORIGINS:
            ORIGINS.append(origin)
