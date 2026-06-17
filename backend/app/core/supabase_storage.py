import os
import requests
from pathlib import Path
from backend.app.core.config import STORAGE_DIR

# Supabase API Environment Configurations
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "instaml")

def is_supabase_configured() -> bool:
    """Check if both Supabase URL and Key are defined."""
    return bool(SUPABASE_URL and SUPABASE_KEY)

def is_supabase_required() -> bool:
    """Check if cloud storage (Supabase) is strictly required (e.g., in production/deployment)."""
    return bool(os.getenv("RENDER") or os.getenv("STRICT_CLOUD_STORAGE") or os.getenv("NODE_ENV") == "production")

def get_supabase_headers():
    """Build authorization and key headers for Supabase API requests."""
    return {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apiKey": SUPABASE_KEY,
    }

def get_relative_storage_path(local_path: Path) -> str:
    """Convert any local storage path to a standard relative path starting with user_X."""
    try:
        # Resolve to handle symlinks or relative paths cleanly
        resolved_local = Path(local_path).resolve()
        resolved_storage = Path(STORAGE_DIR).resolve()
        return str(resolved_local.relative_to(resolved_storage)).replace("\\", "/")
    except ValueError:
        parts = Path(local_path).parts
        for i, part in enumerate(parts):
            if part.startswith("user_"):
                return "/".join(parts[i:]).replace("\\", "/")
        return str(local_path).replace("\\", "/")

def upload_file_to_supabase(local_file_path: Path) -> bool:
    """Upload a local file to the Supabase Storage bucket."""
    if not is_supabase_configured():
        return False
    local_path = Path(local_file_path)
    if not local_path.exists() or not local_path.is_file():
        print(f"Supabase Upload: Local file not found: {local_file_path}")
        return False
    
    remote_path = get_relative_storage_path(local_path)
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{SUPABASE_BUCKET}/{remote_path}"
    
    headers = get_supabase_headers()
    headers["x-upsert"] = "true"  # Enable overwrite/upsert
    headers["Content-Type"] = "application/octet-stream"
    
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        res = requests.post(url, headers=headers, data=data)
        if res.status_code in [200, 201]:
            print(f"Supabase Upload Success: {remote_path}")
            return True
        else:
            err_msg = f"Supabase Upload Failed ({res.status_code}): {res.text}"
            print(err_msg)
            if is_supabase_required():
                raise RuntimeError(
                    f"Cloud storage upload failed: {err_msg}. "
                    f"Please ensure you have created a public/private storage bucket named '{SUPABASE_BUCKET}' "
                    f"in your Supabase project, and that your SUPABASE_KEY is a valid 'service_role' key (not the anon/publishable key) that bypasses Row-Level Security (RLS)."
                )
            return False
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        err_msg = f"Supabase Upload Exception: {e}"
        print(err_msg)
        if is_supabase_required():
            raise RuntimeError(f"Cloud storage upload failed due to network/unexpected error: {err_msg}")
        return False

def upload_dir_to_supabase(local_dir_path: Path) -> bool:
    """Recursively upload all files within a local directory to Supabase Storage."""
    if not is_supabase_configured():
        return False
    local_path = Path(local_dir_path)
    if not local_path.exists() or not local_path.is_dir():
        print(f"Supabase Upload: Local directory not found: {local_dir_path}")
        return False
    
    success = True
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            # Skip hidden files
            if file_path.name.startswith("."):
                continue
            if not upload_file_to_supabase(file_path):
                success = False
    return success

def download_file_from_supabase(remote_path: str, local_file_path: Path) -> bool:
    """Download a specific object from Supabase Storage to a local file path."""
    if not is_supabase_configured():
        return False
    
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/authenticated/{SUPABASE_BUCKET}/{remote_path}"
    headers = get_supabase_headers()
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            local_path = Path(local_file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(res.content)
            print(f"Supabase Download Success: {remote_path} -> {local_file_path}")
            return True
        else:
            print(f"Supabase Download Failed ({res.status_code}): {res.text}")
            return False
    except Exception as e:
        print(f"Supabase Download Exception: {e}")
        return False

def list_supabase_files(remote_prefix: str) -> list:
    """List objects located under a remote folder prefix."""
    if not is_supabase_configured():
        return []
    
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/list/{SUPABASE_BUCKET}"
    headers = get_supabase_headers()
    headers["Content-Type"] = "application/json"
    
    body = {
        "prefix": remote_prefix,
        "limit": 1000,
        "offset": 0,
    }
    
    try:
        res = requests.post(url, headers=headers, json=body)
        if res.status_code == 200:
            return res.json()
        else:
            print(f"Supabase List Failed ({res.status_code}): {res.text}")
            return []
    except Exception as e:
        print(f"Supabase List Exception: {e}")
        return []

def download_dir_from_supabase(remote_prefix: str, local_dir_path: Path) -> bool:
    """Download all objects under a remote directory prefix recursively to a local path."""
    if not is_supabase_configured():
        return False
    
    remote_prefix = remote_prefix.rstrip("/")
    files = list_supabase_files(remote_prefix)
    
    if not files:
        # Try listing with a trailing slash to hit subdirectory contents specifically
        files = list_supabase_files(remote_prefix + "/")
    
    if not files:
        print(f"Supabase Download: No files found under prefix: {remote_prefix}")
        return False
    
    success = True
    local_path = Path(local_dir_path)
    
    for item in files:
        name = item.get("name")
        if not name:
            continue
        # In Supabase Storage API, subdirectories returned by list have metadata = None or id = None
        if item.get("id") is None:
            # Recursively download subdirectories
            sub_prefix = f"{remote_prefix}/{name}"
            sub_dir = local_path / name
            if not download_dir_from_supabase(sub_prefix, sub_dir):
                success = False
        else:
            # Download file
            remote_file_path = f"{remote_prefix}/{name}"
            local_file_path = local_path / name
            if not download_file_from_supabase(remote_file_path, local_file_path):
                success = False
                
    return success

def ensure_local_file_exists(local_path_str: str) -> str:
    """Check if the path exists locally. If missing and Supabase is set up, download it."""
    if not local_path_str:
        return local_path_str
        
    local_path = Path(local_path_str)
    if local_path.exists():
        return local_path_str
        
    if not is_supabase_configured():
        return local_path_str
        
    remote_path = get_relative_storage_path(local_path)
    print(f"Local resource not found: {local_path}. Fetching from Supabase Storage: {remote_path}")
    
    # Try file download first
    if download_file_from_supabase(remote_path, local_path):
        return local_path_str
        
    # If file download fails, try folder download
    if download_dir_from_supabase(remote_path, local_path):
        return local_path_str
        
    if is_supabase_required():
        raise FileNotFoundError(
            f"File not found locally: {local_path}. Attempted to download from cloud storage (Supabase) using path '{remote_path}' but it failed. "
            f"Please verify that you have created a public/private storage bucket named '{SUPABASE_BUCKET}' in your Supabase project, "
            f"and that your SUPABASE_KEY is a valid 'service_role' key (not the anon/publishable key) that bypasses Row-Level Security (RLS)."
        )
        
    return local_path_str
