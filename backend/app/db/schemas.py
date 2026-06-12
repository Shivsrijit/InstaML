from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- User Schemas ---
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Project Schemas ---
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str  # tabular, image, audio, multi_dimensional
    task: Optional[str] = 'Classification'

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    task: Optional[str] = None
    target_col: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    data_type: str
    task: Optional[str] = None
    target_col: Optional[str] = None
    created_at: datetime
    user_id: int

    class Config:
        from_attributes = True

# --- Dataset Version Schemas ---
class DatasetVersionResponse(BaseModel):
    id: int
    version_id: str
    step_name: str
    description: Optional[str] = None
    shape_rows: Optional[int] = None
    shape_cols: Optional[int] = None
    columns_json: Optional[str] = None
    dtypes_json: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# --- Model Schemas ---
class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    target_col: Optional[str] = None
    metrics_json: Optional[str] = None
    is_deployed: bool
    created_at: datetime
    project_id: int
    dataset_version_id: Optional[int] = None

    class Config:
        from_attributes = True

# --- Prediction & Preprocessing Schemas ---
class PreprocessRequest(BaseModel):
    operations: List[Dict[str, Any]]
    # Example operations format:
    # [{"op": "drop_cols", "columns": ["id", "name"]},
    #  {"op": "fill_missing", "strategy": "mean", "columns": ["age"]},
    #  {"op": "scale", "method": "standard", "columns": ["age", "fare"]},
    #  {"op": "encode", "method": "onehot", "columns": ["sex", "embarked"]},
    #  {"op": "remove_outliers", "column": "fare"}]

class TrainRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    target_col: str = Field(..., min_length=1)
    text_col: Optional[str] = None
    use_hyperparameter_tuning: bool = False
    trials: int = Field(10, ge=1, le=100)
    validation_split: float = Field(0.2, ge=0.05, le=0.5)
    k_folds: Optional[int] = Field(None, ge=2, le=10)

class PredictRequest(BaseModel):
    features: Dict[str, Any]

class BatchPredictRequest(BaseModel):
    file_content: str  # base64 or csv string
