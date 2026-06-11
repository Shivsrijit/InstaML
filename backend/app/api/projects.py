import shutil
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from backend.app.db.session import query_sync, execute_sync
from backend.app.db.schemas import ProjectCreate, ProjectResponse, ProjectUpdate
from backend.app.api.deps import get_current_user
from backend.app.core.config import STORAGE_DIR

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("", response_model=List[ProjectResponse])
def list_projects(current_user: dict = Depends(get_current_user)):
    """List all projects for the authenticated user."""
    return query_sync("SELECT * FROM projects WHERE user_id = ?", [current_user.id])

@router.post("", response_model=ProjectResponse)
def create_project(project_in: ProjectCreate, current_user: dict = Depends(get_current_user)):
    """Create a new project for the authenticated user."""
    # Validate data_type
    allowed_types = ["tabular", "text", "image", "audio", "multi_dimensional"]
    if project_in.data_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_type. Must be one of {allowed_types}"
        )
    
    execute_sync(
        "INSERT INTO projects (name, description, data_type, task, user_id) VALUES (?, ?, ?, ?, ?)",
        [project_in.name, project_in.description, project_in.data_type, project_in.task or 'Classification', current_user.id]
    )
    
    # Retrieve the newly created project
    projects = query_sync(
        "SELECT * FROM projects WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        [current_user.id]
    )
    project = projects[0]
    
    # Create project storage directory
    project_dir = STORAGE_DIR / f"user_{current_user.id}" / f"project_{project.id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    
    return project

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, current_user: dict = Depends(get_current_user)):
    """Retrieve details of a single project."""
    projects = query_sync(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        [project_id, current_user.id]
    )
    if not projects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )
    return projects[0]

@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int,
    project_in: ProjectUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update details of a project, including its data_type modality."""
    projects = query_sync(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        [project_id, current_user.id]
    )
    if not projects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )
    
    if project_in.name is not None:
        execute_sync("UPDATE projects SET name = ? WHERE id = ?", [project_in.name, project_id])
    if project_in.description is not None:
        execute_sync("UPDATE projects SET description = ? WHERE id = ?", [project_in.description, project_id])
    if project_in.task is not None:
        execute_sync("UPDATE projects SET task = ? WHERE id = ?", [project_in.task, project_id])
    if project_in.target_col is not None:
        execute_sync("UPDATE projects SET target_col = ? WHERE id = ?", [project_in.target_col, project_id])
    if project_in.data_type is not None:
        allowed_types = ["tabular", "text", "image", "audio", "multi_dimensional"]
        if project_in.data_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_type. Must be one of {allowed_types}"
            )
        execute_sync("UPDATE projects SET data_type = ? WHERE id = ?", [project_in.data_type, project_id])
        
    updated_projects = query_sync("SELECT * FROM projects WHERE id = ?", [project_id])
    return updated_projects[0]

@router.delete("/{project_id}")
def delete_project(project_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a project and all its associated database entries and files."""
    projects = query_sync(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        [project_id, current_user.id]
    )
    if not projects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have access to it"
        )
    
    # Delete database entries
    execute_sync("DELETE FROM projects WHERE id = ?", [project_id])
    
    # Clean up physical storage directory
    project_dir = STORAGE_DIR / f"user_{current_user.id}" / f"project_{project_id}"
    if project_dir.exists():
        try:
            shutil.rmtree(project_dir)
        except Exception as e:
            # log or pass
            pass
            
    return {"detail": "Project and all associated data deleted successfully"}
