from fastapi import APIRouter, Depends, HTTPException, status, Query
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

from backend.app.db.session import query_sync
from backend.app.api.deps import get_current_user
from backend.app.api.data import get_user_project
from backend.app.core.data_handler import load_dataframe

router = APIRouter(prefix="/projects/{project_id}/eda", tags=["eda"])

def get_latest_dataframe(project_id: int, user_id: int) -> pd.DataFrame:
    """Helper to get project's active dataset."""
    get_user_project(project_id, user_id)
    latest_version = query_sync(
        "SELECT * FROM dataset_versions WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        [project_id]
    )
    if not latest_version:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet")
    return load_dataframe(latest_version[0].file_path)

@router.get("/summary")
def get_eda_summary(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve statistical summaries (describe) for numerical and categorical columns."""
    df = get_latest_dataframe(project_id, current_user.id)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    summary = {}
    
    # Numerical description
    if numeric_cols:
        desc = df[numeric_cols].describe().replace({np.nan: None})
        summary["numeric"] = desc.to_dict()
        
    # Categorical description
    if categorical_cols:
        desc = df[categorical_cols].astype(str).describe().replace({np.nan: None})
        summary["categorical"] = desc.to_dict()
        
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "summary": summary
    }

@router.get("/histogram/{column}")
def get_histogram(
    project_id: int,
    column: str,
    bins: int = Query(30, ge=5, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Calculate histogram bins and frequencies for a numeric column."""
    df = get_latest_dataframe(project_id, current_user.id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
        
    col_data = df[column].dropna()
    if not np.issubdtype(col_data.dtype, np.number):
        raise HTTPException(status_code=400, detail="Histogram is only available for numerical columns")
        
    counts, bin_edges = np.histogram(col_data, bins=bins)
    
    # Map edges to bin centers or range labels
    bin_labels = []
    for i in range(len(counts)):
        bin_labels.append(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}")
        
    return {
        "labels": bin_labels,
        "counts": counts.tolist(),
        "mean": float(col_data.mean()),
        "median": float(col_data.median()),
        "std": float(col_data.std())
    }

@router.get("/categories/{column}")
def get_category_counts(
    project_id: int,
    column: str,
    top_n: int = Query(10, ge=3, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve frequency counts for categorical values."""
    df = get_latest_dataframe(project_id, current_user.id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
        
    counts = df[column].astype(str).value_counts()
    top_counts = counts.head(top_n)
    
    return {
        "labels": top_counts.index.tolist(),
        "values": top_counts.values.tolist(),
        "total_unique": len(counts)
    }

@router.get("/correlation")
def get_correlation_matrix(
    project_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Calculate the correlation matrix for all numeric columns."""
    df = get_latest_dataframe(project_id, current_user.id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="At least 2 numerical columns are required for correlation")
        
    corr = df[numeric_cols].corr().replace({np.nan: 0})
    return {
        "columns": numeric_cols,
        "matrix": corr.values.tolist()
    }

@router.get("/scatter")
def get_scatter_data(
    project_id: int,
    x: str = Query(...),
    y: str = Query(...),
    color_by: str = Query(None),
    max_points: int = Query(1000, le=5000),
    current_user: dict = Depends(get_current_user)
):
    """Get X-Y coordinates for scatter plot mapping (optionally sampled and grouped)."""
    df = get_latest_dataframe(project_id, current_user.id)
    
    for col in [x, y]:
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found")
            
    # Clean and sample data
    cols = [x, y]
    if color_by and color_by != "None" and color_by in df.columns:
        cols.append(color_by)
        
    sample_df = df[cols].dropna()
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(n=max_points, random_state=42)
        
    points = []
    for _, row in sample_df.iterrows():
        point = {
            "x": float(row[x]) if isinstance(row[x], (int, float, np.number)) else str(row[x]),
            "y": float(row[y]) if isinstance(row[y], (int, float, np.number)) else str(row[y])
        }
        if color_by and color_by != "None":
            point["color"] = str(row[color_by])
        points.append(point)
        
    return points

@router.get("/kmeans")
def get_kmeans_clusters(
    project_id: int,
    n_clusters: int = Query(3, ge=2, le=8),
    current_user: dict = Depends(get_current_user)
):
    """Run K-Means clustering and return coordinates map."""
    df = get_latest_dataframe(project_id, current_user.id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numerical columns for clustering")
        
    sample_df = df[numeric_cols].dropna()
    if len(sample_df) > 1000:
        sample_df = sample_df.sample(n=1000, random_state=42)
        
    scaled = StandardScaler().fit_transform(sample_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    
    points = []
    # Use first two numerical columns as X and Y
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        points.append({
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "cluster": int(labels[i])
        })
        
    return {
        "x_label": x_col,
        "y_label": y_col,
        "points": points
    }

@router.get("/pca")
def get_pca_projections(
    project_id: int,
    n_components: int = Query(2, ge=2, le=5),
    current_user: dict = Depends(get_current_user)
):
    """Calculate PCA principal components projection coordinates."""
    df = get_latest_dataframe(project_id, current_user.id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numerical columns for PCA")
        
    sample_df = df[numeric_cols].dropna()
    if len(sample_df) > 1000:
        sample_df = sample_df.sample(n=1000, random_state=42)
        
    scaled = StandardScaler().fit_transform(sample_df)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)
    
    points = []
    for i in range(len(reduced)):
        pt = {}
        for c in range(n_components):
            pt[f"PC{c+1}"] = float(reduced[i, c])
        points.append(pt)
        
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "points": points
    }

@router.get("/dim-reduction")
def get_dimensionality_reduction(
    project_id: int,
    method: str = Query("PCA"),  # PCA, t-SNE, LDA, UMAP
    color_by: str = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Compute 2D projection using PCA, t-SNE, LDA, or UMAP with custom grouping color."""
    df = get_latest_dataframe(project_id, current_user.id)
    
    # Load project metadata for target column
    project_rows = query_sync("SELECT * FROM projects WHERE id = ?", [project_id])
    if not project_rows:
        raise HTTPException(status_code=404, detail="Project not found")
    project = project_rows[0]
    target_col = project.target_col
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="At least 2 numerical columns are required for dimensionality reduction")
    
    # Resolve color column
    color_col = color_by if (color_by and color_by != "None" and color_by in df.columns) else target_col
    
    # Sample up to 1000 rows to ensure responsiveness
    cols_to_use = numeric_cols.copy()
    if color_col and color_col in df.columns and color_col not in cols_to_use:
        cols_to_use.append(color_col)
    if method == "LDA" and target_col and target_col in df.columns and target_col not in cols_to_use:
        cols_to_use.append(target_col)
        
    sample_df = df[cols_to_use].dropna(subset=numeric_cols)
    if len(sample_df) > 1000:
        sample_df = sample_df.sample(n=1000, random_state=42)
        
    X_num = sample_df[numeric_cols]
    if X_num.isnull().any().any():
        X_num = X_num.fillna(X_num.median())
        
    scaled = StandardScaler().fit_transform(X_num)
    reduced = None
    
    # Apply method
    method_upper = method.upper()
    if method_upper == "TSNE" or method_upper == "T-SNE":
        from sklearn.manifold import TSNE
        n_samples = scaled.shape[0]
        # Perplexity must be less than n_samples
        perplexity = min(30.0, max(2.0, float(n_samples - 1.0) / 3.0))
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=perplexity)
        reduced = tsne.fit_transform(scaled)
    elif method_upper == "LDA":
        y_lda = sample_df[target_col] if (target_col and target_col in sample_df.columns) else None
        if y_lda is not None and y_lda.nunique() >= 2:
            # For LDA, target must be discrete classes. If continuous, bin it.
            if y_lda.dtype in ['float64', 'float32'] or y_lda.nunique() > 50:
                try:
                    y_lda = pd.qcut(y_lda, q=3, labels=["Low", "Medium", "High"], duplicates='drop')
                except:
                    pass
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(n_components=min(2, y_lda.nunique() - 1))
            reduced = lda.fit_transform(scaled, y_lda)
            if reduced is not None and reduced.shape[1] == 1:
                # Pad to 2D
                reduced = np.hstack([reduced, np.zeros((reduced.shape[0], 1))])
            elif reduced is None or reduced.shape[1] == 0:
                # Fallback to PCA if 0 components (due to singular matrix or collinearity)
                pca = PCA(n_components=2, random_state=42)
                reduced = pca.fit_transform(scaled)
        else:
            # Fallback to PCA if LDA not applicable
            pca = PCA(n_components=2, random_state=42)
            reduced = pca.fit_transform(scaled)
    elif method_upper == "UMAP":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(scaled)
        except ImportError:
            # Fallback to Isomap or SpectralEmbedding
            try:
                from sklearn.manifold import Isomap
                reducer = Isomap(n_components=2, n_neighbors=5)
                reduced = reducer.fit_transform(scaled)
            except:
                from sklearn.manifold import SpectralEmbedding
                reducer = SpectralEmbedding(n_components=2, random_state=42)
                reduced = reducer.fit_transform(scaled)
    else:  # PCA
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(scaled)
        
    # Build points list
    points = []
    for i in range(len(reduced)):
        pt = {
            "x": float(reduced[i, 0]),
            "y": float(reduced[i, 1])
        }
        if color_col and color_col in sample_df.columns:
            val = sample_df.iloc[i][color_col]
            if pd.isnull(val):
                pt["color"] = "Unknown"
            else:
                pt["color"] = str(val)
        else:
            pt["color"] = "All"
        points.append(pt)
        
    # Candidates for color grouping (categorical columns + target column)
    color_candidates = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # Add integer columns with low cardinality as grouping options as well
    for col in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if df[col].nunique() <= 20 and col not in color_candidates:
            color_candidates.append(col)
    if target_col and target_col not in color_candidates and target_col in df.columns:
        color_candidates.append(target_col)
        
    return {
        "method": method,
        "color_by": color_col,
        "color_candidates": color_candidates,
        "points": points
    }
