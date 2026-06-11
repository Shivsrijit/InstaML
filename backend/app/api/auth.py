from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from backend.app.db.session import query_sync, execute_sync
from backend.app.db.schemas import UserCreate, UserLogin, UserResponse, Token, UserUpdate, TokenData
from backend.app.core.security import verify_password, get_password_hash, create_access_token
from backend.app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate):
    """Register a new user."""
    # Check if username exists
    users_by_username = query_sync("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", [user_in.username])
    if users_by_username:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Check if email exists
    users_by_email = query_sync("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", [user_in.email])
    if users_by_email:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Hash password and save user
    hashed_password = get_password_hash(user_in.password)
    execute_sync(
        "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
        [user_in.username, user_in.email, hashed_password]
    )
    
    # Fetch created user to return
    created_users = query_sync("SELECT * FROM users WHERE username = ?", [user_in.username])
    return created_users[0]

@router.post("/login", response_model=Token)
def login_json(user_in: UserLogin):
    """Authenticate user and return access token via JSON."""
    users = query_sync("SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)", [user_in.username, user_in.username])
    user = users[0] if users else None
    
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(subject=user.username)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login-form", response_model=Token)
def login_form(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token via OAuth2 Form (for OpenAPI Docs)."""
    users = query_sync("SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)", [form_data.username, form_data.username])
    user = users[0] if users else None
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(subject=user.username)
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: dict = Depends(get_current_user)):
    """Retrieve details of the active logged-in user."""
    return current_user

@router.put("/me", response_model=UserResponse)
def update_current_user(user_update: UserUpdate, current_user: dict = Depends(get_current_user)):
    """Update current user email or password."""
    updates = []
    params = []
    
    if user_update.email is not None and user_update.email.strip() != "":
        # Check if email is already used by another user
        existing = query_sync("SELECT * FROM users WHERE LOWER(email) = LOWER(?) AND id != ?", [user_update.email, current_user.id])
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        updates.append("email = ?")
        params.append(user_update.email)
        
    if user_update.password is not None and user_update.password.strip() != "":
        hashed_pw = get_password_hash(user_update.password)
        updates.append("hashed_password = ?")
        params.append(hashed_pw)
        
    if not updates:
        return current_user
        
    params.append(current_user.id)
    execute_sync(
        f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
        params
    )
    
    # Return updated user
    updated_users = query_sync("SELECT * FROM users WHERE id = ?", [current_user.id])
    return updated_users[0]
