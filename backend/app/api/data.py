import datetime
import json
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
import pandas as pd
import shutil

from backend.app.db.session import query_sync, execute_sync
from backend.app.db.schemas import DatasetVersionResponse, PreprocessRequest
from backend.app.api.deps import get_current_user
from backend.app.core.config import STORAGE_DIR
from backend.app.core.data_handler import load_dataframe, save_dataframe, get_dataframe_preview, apply_preprocessing_operations

router = APIRouter(prefix="/projects/{project_id}", tags=["data"])

def get_user_project(project_id: int, user_id: int) -> dict:
    """Helper to check project ownership."""
    projects = query_sync("SELECT * FROM projects WHERE id = ? AND user_id = ?", [project_id, user_id])
    if not projects:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have permission to access it"
        )
    return projects[0]

def compute_dataframe_stats(df) -> dict:
    """Helper to calculate missing values and duplicate row counts in a DataFrame."""
    missing_counts = df.isnull().sum().to_dict()
    missing_total = int(df.isnull().sum().sum())
    # Convert keys to string and values to int for JSON compatibility
    missing_counts_str = {str(k): int(v) for k, v in missing_counts.items()}
    
    # Calculate duplicate rows
    try:
        duplicate_count = int(df.duplicated().sum())
    except:
        duplicate_count = 0
        
    return {
        "missing_counts": missing_counts_str,
        "missing_total": missing_total,
        "duplicate_count": duplicate_count
    }

@router.post("/upload")
def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a dataset file (CSV, Parquet, Excel, or ZIP for image/audio) and save as version 0."""
    project = get_user_project(project_id, current_user.id)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"upload_{timestamp}"
    
    project_dir = STORAGE_DIR / f"user_{current_user.id}" / f"project_{project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = Path(file.filename).suffix.lower()
    
    if project.data_type in ["image", "audio"]:
        valid_image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
        valid_audio_exts = [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"]
        
        is_zip = file_ext == ".zip"
        is_raw = (project.data_type == "image" and file_ext in valid_image_exts) or \
                 (project.data_type == "audio" and file_ext in valid_audio_exts)
                 
        if not is_zip and not is_raw:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Please upload a .zip archive or a raw file with extensions: "
                       f"{', '.join(valid_image_exts if project.data_type == 'image' else valid_audio_exts)}"
            )
            
        if not is_zip:
            # Process single raw file upload
            final_dir_path = project_dir / "data" / version_id
            class_dir_path = final_dir_path / "default"
            class_dir_path.mkdir(parents=True, exist_ok=True)
            
            dest_file = class_dir_path / file.filename
            with open(dest_file, "wb") as f:
                shutil.copyfileobj(file.file, f)
                
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    version_id,
                    "Data Upload",
                    f"Uploaded raw {project.data_type} file: {file.filename}",
                    1,
                    1,
                    json.dumps(["Class Name", "File Count"]),
                    json.dumps({"Class Name": "str", "File Count": "int"}),
                    json.dumps({"filename": file.filename, "raw_single_file": True}),
                    str(final_dir_path),
                    project_id
                ]
            )
            
            db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
            db_version = db_versions[0]
            
            return {
                "detail": f"Raw {project.data_type} file uploaded and registered successfully",
                "version": {
                    "id": db_version.id,
                    "version_id": db_version.version_id,
                    "step_name": db_version.step_name,
                    "shape": [1, 1]
                }
            }
            
        final_dir_path = project_dir / "data" / version_id
        final_dir_path.mkdir(parents=True, exist_ok=True)
        
        temp_zip = project_dir / f"temp_{version_id}.zip"
        with open(temp_zip, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            import zipfile
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(final_dir_path)
                
            # Count extracted files
            all_files = [p for p in final_dir_path.rglob('*') if p.is_file()]
            subdirs = [p.name for p in final_dir_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
            
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    version_id,
                    "Data Upload",
                    f"Uploaded zip dataset containing {len(all_files)} files in {len(subdirs)} classes.",
                    len(all_files),
                    len(subdirs),
                    json.dumps(["Class Name", "File Count"]),
                    json.dumps({"Class Name": "str", "File Count": "int"}),
                    json.dumps({"filename": file.filename}),
                    str(final_dir_path),
                    project_id
                ]
            )
            
            db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
            db_version = db_versions[0]
            
            return {
                "detail": "ZIP dataset uploaded and extracted successfully",
                "version": {
                    "id": db_version.id,
                    "version_id": db_version.version_id,
                    "step_name": db_version.step_name,
                    "shape": [len(all_files), len(subdirs)]
                }
            }
        except Exception as e:
            if final_dir_path.exists():
                shutil.rmtree(final_dir_path)
            raise HTTPException(status_code=500, detail=f"Failed to extract zip dataset: {str(e)}")
        finally:
            if temp_zip.exists():
                temp_zip.unlink()
                
    elif project.data_type == "text" and file_ext == ".zip":
        final_dir_path = project_dir / "data" / f"temp_extract_{version_id}"
        final_dir_path.mkdir(parents=True, exist_ok=True)
        
        temp_zip = project_dir / f"temp_{version_id}.zip"
        with open(temp_zip, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            import zipfile
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(final_dir_path)
                
            # Scan for .txt files
            records = []
            for txt_path in final_dir_path.rglob('*'):
                if txt_path.is_file() and txt_path.suffix.lower() == '.txt':
                    if txt_path.name.startswith('.') or '__MACOSX' in txt_path.parts:
                        continue
                    try:
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        if content:
                            label = txt_path.parent.name
                            records.append({"text": content, "label": label})
                    except Exception:
                        continue
            
            shutil.rmtree(final_dir_path)
            
            if not records:
                raise HTTPException(
                    status_code=400,
                    detail="No valid text files (.txt) found in the zip archive."
                )
                
            df = pd.DataFrame(records)
            final_file_path = project_dir / "data" / f"{version_id}.parquet"
            save_dataframe(df, str(final_file_path))
            
            shape_rows, shape_cols = df.shape
            columns_list = list(df.columns)
            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            
            metadata = {"filename": file.filename, "extracted_zip": True}
            try:
                stats = compute_dataframe_stats(df)
                metadata.update(stats)
            except Exception as stats_err:
                print(f"Warning computing stats: {stats_err}")

            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    version_id,
                    "Data Upload",
                    f"Uploaded ZIP dataset: extracted {shape_rows} text documents across unique categories.",
                    shape_rows,
                    shape_cols,
                    json.dumps(columns_list),
                    json.dumps(dtypes_dict),
                    json.dumps(metadata),
                    str(final_file_path),
                    project_id
                ]
            )
            
            db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
            db_version = db_versions[0]
            
            return {
                "detail": "ZIP text dataset uploaded and processed successfully",
                "version": {
                    "id": db_version.id,
                    "version_id": db_version.version_id,
                    "step_name": db_version.step_name,
                    "shape": [shape_rows, shape_cols]
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process zip text dataset: {str(e)}")
        finally:
            if temp_zip.exists():
                temp_zip.unlink()
            if final_dir_path.exists():
                try:
                    shutil.rmtree(final_dir_path)
                except Exception:
                    pass
                    
    elif project.data_type == "text" and file_ext == ".txt":
        temp_file_path = project_dir / f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            if not lines:
                raise HTTPException(status_code=400, detail="The uploaded text file is empty.")
                
            df = pd.DataFrame({"text": lines})
            final_file_path = project_dir / "data" / f"{version_id}.parquet"
            save_dataframe(df, str(final_file_path))
            
            shape_rows, shape_cols = df.shape
            columns_list = list(df.columns)
            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            
            metadata = {"filename": file.filename}
            try:
                stats = compute_dataframe_stats(df)
                metadata.update(stats)
            except Exception as stats_err:
                print(f"Warning computing stats: {stats_err}")

            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    version_id,
                    "Data Upload",
                    f"Uploaded raw text file: {file.filename} ({shape_rows} lines)",
                    shape_rows,
                    shape_cols,
                    json.dumps(columns_list),
                    json.dumps(dtypes_dict),
                    json.dumps(metadata),
                    str(final_file_path),
                    project_id
                ]
            )
            
            db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
            db_version = db_versions[0]
            
            return {
                "detail": "Text file processed and registered successfully",
                "version": {
                    "id": db_version.id,
                    "version_id": db_version.version_id,
                    "step_name": db_version.step_name,
                    "shape": [shape_rows, shape_cols]
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process text file: {str(e)}")
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()
    else:
        # Tabular / multi-dimensional datasets
        temp_file_path = project_dir / f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            df = None
            metadata = {"filename": file.filename}
            
            if file_ext == ".zip":
                temp_extract_dir = project_dir / f"temp_extract_{version_id}"
                temp_extract_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    import zipfile
                    with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                        
                    # Find all tabular files
                    tabular_files = []
                    for path in temp_extract_dir.rglob('*'):
                        if path.is_file() and path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.parquet']:
                            if path.name.startswith('.') or '__MACOSX' in path.parts:
                                continue
                            tabular_files.append(path)
                            
                    if not tabular_files:
                        raise HTTPException(status_code=400, detail="No valid CSV, Excel, or Parquet files found in the zip archive.")
                        
                    # Find train and test files
                    train_file = None
                    test_file = None
                    for f_path in tabular_files:
                        name_lower = f_path.name.lower()
                        if "train" in name_lower:
                            train_file = f_path
                        elif "test" in name_lower:
                            test_file = f_path
                            
                    if not train_file:
                        train_file = tabular_files[0]
                        if len(tabular_files) > 1 and not test_file:
                            test_file = tabular_files[1]
                            
                    # Load train
                    if train_file.suffix.lower() == ".csv":
                        df = pd.read_csv(train_file)
                    elif train_file.suffix.lower() in [".xlsx", ".xls"]:
                        df = pd.read_excel(train_file)
                    else:
                        df = pd.read_parquet(train_file)
                        
                    # Load test if present
                    if test_file:
                        try:
                            if test_file.suffix.lower() == ".csv":
                                df_test = pd.read_csv(test_file)
                            elif test_file.suffix.lower() in [".xlsx", ".xls"]:
                                df_test = pd.read_excel(test_file)
                            else:
                                df_test = pd.read_parquet(test_file)
                                
                            test_save_path = project_dir / "data" / f"{version_id}_test.parquet"
                            save_dataframe(df_test, str(test_save_path))
                            metadata["has_test"] = True
                            metadata["test_file_path"] = str(test_save_path)
                            metadata["test_filename"] = test_file.name
                        except Exception as test_err:
                            print(f"Warning loading test file: {test_err}")
                finally:
                    if temp_extract_dir.exists():
                        try:
                            shutil.rmtree(temp_extract_dir)
                        except Exception:
                            pass
            else:
                # Single file upload
                if file_ext == ".csv":
                    df = pd.read_csv(temp_file_path)
                elif file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(temp_file_path)
                elif file_ext == ".parquet":
                    df = pd.read_parquet(temp_file_path)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported format. Upload CSV, Excel, or Parquet.")
            
            if df is None:
                raise ValueError("Could not parse dataset DataFrame.")
                
            final_file_path = project_dir / "data" / f"{version_id}.parquet"
            save_dataframe(df, str(final_file_path))
            
            shape_rows, shape_cols = df.shape
            columns_list = list(df.columns)
            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            
            # Predict Target Column
            predicted_target = None
            possible_targets = ["target", "label", "class", "output", "y", "prediction", "diagnoses", "survived", "status", "price", "revenue"]
            for col in columns_list:
                if col.lower() in possible_targets:
                    predicted_target = col
                    break
            if not predicted_target and columns_list:
                predicted_target = columns_list[-1]
                
            # Predict Task (Classification vs Regression)
            predicted_task = "Classification"
            if predicted_target and predicted_target in df.columns:
                target_series = df[predicted_target].dropna()
                num_unique = target_series.nunique()
                target_dtype = str(target_series.dtype)
                
                if "float" in target_dtype:
                    if num_unique <= 5:
                        predicted_task = "Classification"
                    else:
                        predicted_task = "Regression"
                elif "int" in target_dtype:
                    if num_unique <= 10 or num_unique < 0.05 * len(target_series):
                        predicted_task = "Classification"
                    else:
                        predicted_task = "Regression"
                elif "bool" in target_dtype or "object" in target_dtype or "category" in target_dtype:
                    predicted_task = "Classification"
                    
            # Save predicted target and task in DB projects table
            execute_sync(
                "UPDATE projects SET target_col = ?, task = ? WHERE id = ?",
                [predicted_target, predicted_task, project_id]
            )
            
            try:
                stats = compute_dataframe_stats(df)
                metadata.update(stats)
            except Exception as stats_err:
                print(f"Warning computing stats: {stats_err}")

            # Save version
            execute_sync(
                """INSERT INTO dataset_versions (
                    version_id, step_name, description, shape_rows, shape_cols,
                    columns_json, dtypes_json, metadata_json, file_path, project_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    version_id,
                    "Data Upload",
                    f"Uploaded dataset: {file.filename}",
                    shape_rows,
                    shape_cols,
                    json.dumps(columns_list),
                    json.dumps(dtypes_dict),
                    json.dumps(metadata),
                    str(final_file_path),
                    project_id
                ]
            )
            
            db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
            db_version = db_versions[0]
            
            return {
                "detail": "File uploaded and registered successfully",
                "version": {
                    "id": db_version.id,
                    "version_id": db_version.version_id,
                    "step_name": db_version.step_name,
                    "shape": [shape_rows, shape_cols]
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

@router.get("/data/current")
def get_current_dataset(
    project_id: int,
    preview_rows: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve current (active) dataset details, schema, and preview records."""
    project = get_user_project(project_id, current_user.id)
    
    latest_versions = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        [project_id]
    )
        
    if not latest_versions:
        return {"data_loaded": False, "detail": "No dataset uploaded yet"}
    
    latest_version = latest_versions[0]
        
    try:
        if project.data_type in ["image", "audio"]:
            path = Path(latest_version.file_path)
            all_files = [p for p in path.rglob('*') if p.is_file()]
            subdirs = [p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith('.')]
            
            preview = []
            for sd in subdirs[:15]:
                class_files = [f for f in (path / sd).rglob('*') if f.is_file()]
                preview.append({
                    "Class Name": sd,
                    "File Count": len(class_files)
                })
                
            return {
                "data_loaded": True,
                "version_id": latest_version.version_id,
                "step_name": latest_version.step_name,
                "description": latest_version.description,
                "shape": [len(all_files), len(subdirs)],
                "columns": ["Class Name", "File Count"],
                "dtypes": {"Class Name": "str", "File Count": "int"},
                "missing_counts": {},
                "missing_total": 0,
                "duplicate_count": 0,
                "preview": preview,
                "created_at": latest_version.created_at
            }
        else:
            df = load_dataframe(latest_version.file_path)
            preview = get_dataframe_preview(df, preview_rows)
            
            # Retrieve cached stats if present in metadata_json
            metadata = {}
            if latest_version.metadata_json:
                try:
                    metadata = json.loads(latest_version.metadata_json)
                except:
                    pass
            
            missing_counts = metadata.get("missing_counts")
            missing_total = metadata.get("missing_total")
            duplicate_count = metadata.get("duplicate_count")
            
            if missing_counts is None or missing_total is None or duplicate_count is None:
                missing_counts = df.isnull().sum().to_dict()
                missing_counts = {str(k): int(v) for k, v in missing_counts.items()}
                missing_total = int(df.isnull().sum().sum())
                try:
                    duplicate_count = int(df.duplicated().sum())
                except:
                    duplicate_count = 0
            
            columns = json.loads(latest_version.columns_json)
            dtypes = json.loads(latest_version.dtypes_json)
            
            # Fetch first version (upload version) to get initial columns/dtypes
            first_versions = query_sync(
                "SELECT columns_json, dtypes_json FROM dataset_versions WHERE project_id = ? ORDER BY id ASC LIMIT 1",
                [project_id]
            )
            initial_columns = columns
            initial_dtypes = dtypes
            if first_versions:
                try:
                    initial_columns = json.loads(first_versions[0].columns_json)
                    initial_dtypes = json.loads(first_versions[0].dtypes_json)
                except:
                    pass
            
            return {
                "data_loaded": True,
                "version_id": latest_version.version_id,
                "step_name": latest_version.step_name,
                "description": latest_version.description,
                "shape": [latest_version.shape_rows, latest_version.shape_cols],
                "columns": columns,
                "dtypes": dtypes,
                "initial_columns": initial_columns,
                "initial_dtypes": initial_dtypes,
                "missing_counts": missing_counts,
                "missing_total": missing_total,
                "duplicate_count": duplicate_count,
                "preview": preview,
                "created_at": latest_version.created_at
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@router.post("/preprocess")
def preprocess_dataset(
    project_id: int,
    req: PreprocessRequest,
    current_user: dict = Depends(get_current_user)
):
    """Apply sequential cleaning, scaling, encoding, or outlier removals, generating a new version."""
    project = get_user_project(project_id, current_user.id)
    
    latest_versions = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        [project_id]
    )
        
    if not latest_versions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No dataset version available to preprocess.")
        
    latest_version = latest_versions[0]
        
    try:
        # Load current dataframe
        df = load_dataframe(latest_version.file_path)
        
        # Apply preprocessing operations
        df_processed = apply_preprocessing_operations(df, req.operations)
        
        # Save preprocessed version
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"preprocess_{timestamp}"
        
        project_dir = STORAGE_DIR / f"user_{current_user.id}" / f"project_{project_id}"
        final_file_path = project_dir / "data" / f"{version_id}.parquet"
        save_dataframe(df_processed, str(final_file_path))
        
        # Calculate new shapes/dtypes
        shape_rows, shape_cols = df_processed.shape
        columns_list = list(df_processed.columns)
        dtypes_dict = {col: str(dtype) for col, dtype in df_processed.dtypes.to_dict().items()}
        
        metadata = {"operations": req.operations}
        try:
            stats = compute_dataframe_stats(df_processed)
            metadata.update(stats)
        except Exception as stats_err:
            print(f"Warning computing stats: {stats_err}")

        execute_sync(
            """INSERT INTO dataset_versions (
                version_id, step_name, description, shape_rows, shape_cols,
                columns_json, dtypes_json, metadata_json, file_path, project_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                version_id,
                "Preprocessing",
                f"Applied operations: {', '.join([op['op'] for op in req.operations])}",
                shape_rows,
                shape_cols,
                json.dumps(columns_list),
                json.dumps(dtypes_dict),
                json.dumps(metadata),
                str(final_file_path),
                project_id
            ]
        )
        
        db_versions = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC LIMIT 1", [project_id])
        db_version = db_versions[0]
        
        return {
            "detail": "Dataset preprocessed successfully",
            "version": {
                "id": db_version.id,
                "version_id": db_version.version_id,
                "step_name": db_version.step_name,
                "shape": [shape_rows, shape_cols]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.get("/data/versions", response_model=List[DatasetVersionResponse])
def get_dataset_versions(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve the version log list for the project."""
    get_user_project(project_id, current_user.id)
    return query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY id DESC",
        [project_id]
    )

@router.post("/data/versions/{version_id}/restore")
def restore_dataset_version(
    project_id: int,
    version_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Restore dataset state to an older checkpoint version by creating a copy as the latest."""
    project = get_user_project(project_id, current_user.id)
    
    # Query source version
    source_vers = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? AND version_id = ?",
        [project_id, version_id]
    )
        
    if not source_vers:
        raise HTTPException(status_code=404, detail="Dataset version not found")
        
    source_ver = source_vers[0]
        
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_version_id = f"restore_{timestamp}"
        
        execute_sync(
            """INSERT INTO dataset_versions (
                version_id, step_name, description, shape_rows, shape_cols,
                columns_json, dtypes_json, metadata_json, file_path, project_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                new_version_id,
                "Restore Checkpoint",
                f"Restored to previous state: {source_ver.version_id} ({source_ver.step_name})",
                source_ver.shape_rows,
                source_ver.shape_cols,
                source_ver.columns_json,
                source_ver.dtypes_json,
                json.dumps({"restored_from": source_ver.version_id}),
                source_ver.file_path,
                project_id
            ]
        )
        
        return {"detail": f"Successfully restored project dataset to {version_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore: {str(e)}")

@router.get("/data/versions/compare")
def compare_dataset_versions(
    project_id: int,
    v1: str = Query(..., description="First version_id (older)"),
    v2: str = Query(..., description="Second version_id (newer)"),
    current_user: dict = Depends(get_current_user)
):
    """Compare shapes, columns, and data counts between two dataset versions."""
    project = get_user_project(project_id, current_user.id)
    
    ver1s = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? AND version_id = ?", [project_id, v1])
    ver2s = query_sync("SELECT * FROM dataset_versions WHERE project_id = ? AND version_id = ?", [project_id, v2])
    
    if not ver1s or not ver2s:
        raise HTTPException(status_code=404, detail="One or both dataset versions not found")
        
    ver1 = ver1s[0]
    ver2 = ver2s[0]
        
    try:
        df1 = load_dataframe(ver1.file_path)
        df2 = load_dataframe(ver2.file_path)
        
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        return {
            "v1": {
                "version_id": ver1.version_id,
                "step_name": ver1.step_name,
                "shape": [ver1.shape_rows, ver1.shape_cols],
                "missing_total": int(df1.isnull().sum().sum()),
                "duplicate_count": int(df1.duplicated().sum())
            },
            "v2": {
                "version_id": ver2.version_id,
                "step_name": ver2.step_name,
                "shape": [ver2.shape_rows, ver2.shape_cols],
                "missing_total": int(df2.isnull().sum().sum()),
                "duplicate_count": int(df2.duplicated().sum())
            },
            "changes": {
                "shape_changed": df1.shape != df2.shape,
                "row_diff": df2.shape[0] - df1.shape[0],
                "col_diff": df2.shape[1] - df1.shape[1],
                "columns_added": list(cols2 - cols1),
                "columns_removed": list(cols1 - cols2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
