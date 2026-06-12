import os
from pathlib import Path
import libsql_client
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# Determine URL and auth token
TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

if TURSO_URL:
    # Use specified remote/local database URL
    url = TURSO_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://", 1)
    token = TURSO_TOKEN
else:
    # Use local file default fallback
    url = f"file:{BASE_DIR}/instaml.db"
    token = None

print(f"DATABASE URL RESOLVED TO: {url}")

class Row(dict):
    """A dictionary subclass that supports dot-attribute access for columns."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Row' object has no attribute '{name}'")
            
    def __setattr__(self, name, value):
        self[name] = value

# Async query wrappers (for async operations/future-proofing)
async def query(sql: str, args: list = []) -> list:
    """Run a SELECT query asynchronously and return list of Row objects."""
    async with libsql_client.create_client(url=url, auth_token=token) as client:
        result = await client.execute(libsql_client.Statement(sql, args))
        return [Row(zip(result.columns, row)) for row in result.rows]

async def execute(sql: str, args: list = []) -> None:
    """Run INSERT / UPDATE / DELETE asynchronously."""
    async with libsql_client.create_client(url=url, auth_token=token) as client:
        await client.execute(libsql_client.Statement(sql, args))

async def execute_many(statements: list) -> None:
    """Run multiple statements in a batch asynchronously. Each item: (sql, args)."""
    async with libsql_client.create_client(url=url, auth_token=token) as client:
        batch = [libsql_client.Statement(sql, args) for sql, args in statements]
        await client.batch(batch)

# Sync query wrappers (for synchronous FastAPI endpoints and background tasks)
def query_sync(sql: str, args: list = []) -> list:
    """Run a SELECT query synchronously and return list of Row objects."""
    with libsql_client.create_client_sync(url=url, auth_token=token) as client:
        result = client.execute(libsql_client.Statement(sql, args))
        return [Row(zip(result.columns, row)) for row in result.rows]

def execute_sync(sql: str, args: list = []) -> None:
    """Run INSERT / UPDATE / DELETE synchronously."""
    with libsql_client.create_client_sync(url=url, auth_token=token) as client:
        client.execute(libsql_client.Statement(sql, args))

def execute_many_sync(statements: list) -> None:
    """Run multiple statements in a batch synchronously. Each item: (sql, args)."""
    with libsql_client.create_client_sync(url=url, auth_token=token) as client:
        batch = [libsql_client.Statement(sql, args) for sql, args in statements]
        client.batch(batch)

# Dummy helpers for migration compatibility
def get_db():
    """Dummy dependency to prevent import errors during transition."""
    yield None

class DummySession:
    """Dummy Session object to prevent syntax errors during transition."""
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass
    def refresh(self, obj): pass

def SessionLocal():
    """Dummy SessionLocal generator for background training worker."""
    return DummySession()

def run_migrations_sync():
    """Initialize database tables synchronously if they do not exist."""
    print("Running database migrations via libsql-client...")
    execute_sync("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """)
    execute_sync("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        data_type TEXT NOT NULL,
        task TEXT DEFAULT 'Classification',
        target_col TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    execute_sync("""
    CREATE TABLE IF NOT EXISTS dataset_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id TEXT UNIQUE NOT NULL,
        step_name TEXT NOT NULL,
        description TEXT,
        shape_rows INTEGER,
        shape_cols INTEGER,
        columns_json TEXT,
        dtypes_json TEXT,
        metadata_json TEXT,
        file_path TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE
    );
    """)
    execute_sync("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        target_col TEXT,
        metrics_json TEXT,
        file_path TEXT NOT NULL,
        is_deployed INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        dataset_version_id INTEGER REFERENCES dataset_versions(id)
    );
    """)
    
    # Run migrations for existing installations
    try:
        execute_sync("ALTER TABLE projects ADD COLUMN task TEXT DEFAULT 'Classification'")
        print("Migration: added task column to projects table successfully.")
    except Exception as e:
        # Column likely already exists
        pass
        
    try:
        execute_sync("ALTER TABLE projects ADD COLUMN target_col TEXT")
        print("Migration: added target_col column to projects table successfully.")
    except Exception as e:
        # Column likely already exists
        pass
        
    print("Database migrations complete.")
