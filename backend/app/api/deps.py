from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from backend.app.core.config import SECRET_KEY, ALGORITHM
from backend.app.db.session import query_sync
from backend.app.db.schemas import TokenData

# OAuth2 scheme config
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login-form")

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency to retrieve and verify current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    users = query_sync("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", [token_data.username])
    if not users:
        raise credentials_exception
    return users[0]
