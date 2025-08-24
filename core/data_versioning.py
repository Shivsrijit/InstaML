import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import hashlib
import pickle

class DataVersionManager:
    """Manages data versions and tracks changes throughout the ML pipeline."""
    
    def __init__(self, base_dir="data_versions"):
        """
        Initialize the data version manager.
        
        Args:
            base_dir: Directory to store data versions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()
        
    def _load_versions(self):
        """Load existing versions from file."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except:
                return {"versions": [], "current_version": None}
        return {"versions": [], "current_version": None}
    
    def _save_versions(self):
        """Save versions to file."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _generate_hash(self, df):
        """Generate a hash for the dataframe."""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def create_version(self, df, step_name, description, metadata=None):
        """
        Create a new version of the data.
        
        Args:
            df: DataFrame to save
            step_name: Name of the processing step
            description: Description of what was changed
            metadata: Additional metadata about the changes
            
        Returns:
            version_id: Unique identifier for this version
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{step_name}_{timestamp}"
        
        # Create version directory
        version_dir = self.base_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save data
        data_file = version_dir / "data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(df, f)
        
        # Create version info
        version_info = {
            "version_id": version_id,
            "step_name": step_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "data_hash": self._generate_hash(df),
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},  # Convert dtypes to strings
            "metadata": metadata or {},
            "data_file": str(data_file)
        }
        
        # Add to versions list
        self.versions["versions"].append(version_info)
        self.versions["current_version"] = version_id
        
        # Save versions
        self._save_versions()
        
        return version_id
    
    def get_version(self, version_id):
        """
        Load a specific version of the data.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            DataFrame: The data at that version
        """
        # Find version info
        version_info = None
        for v in self.versions["versions"]:
            if v["version_id"] == version_id:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version_id} not found")
        
        # Load data
        data_file = Path(version_info["data_file"])
        if not data_file.exists():
            raise FileNotFoundError(f"Data file for version {version_id} not found")
        
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
        
        return df
    
    def get_current_version(self):
        """Get the current version of the data."""
        if not self.versions["current_version"]:
            return None
        
        return self.get_version(self.versions["current_version"])
    
    def list_versions(self):
        """List all available versions."""
        return self.versions["versions"]
    
    def get_version_info(self, version_id):
        """Get information about a specific version."""
        for v in self.versions["versions"]:
            if v["version_id"] == version_id:
                return v
        return None
    
    def compare_versions(self, version_id1, version_id2):
        """
        Compare two versions of the data.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            dict: Comparison results
        """
        df1 = self.get_version(version_id1)
        df2 = self.get_version(version_id2)
        
        comparison = {
            "shape_changed": df1.shape != df2.shape,
            "shape_before": df1.shape,
            "shape_after": df2.shape,
            "columns_changed": list(df1.columns) != list(df2.columns),
            "columns_added": list(set(df2.columns) - set(df1.columns)),
            "columns_removed": list(set(df1.columns) - set(df2.columns)),
            "dtypes_changed": df1.dtypes.to_dict() != df2.dtypes.to_dict(),
            "missing_values_before": df1.isnull().sum().sum(),
            "missing_values_after": df2.isnull().sum().sum(),
            "duplicates_before": df1.duplicated().sum(),
            "duplicates_after": df2.duplicated().sum()
        }
        
        return comparison
    
    def restore_version(self, version_id):
        """
        Restore to a specific version.
        
        Args:
            version_id: Version ID to restore to
            
        Returns:
            DataFrame: The restored data
        """
        df = self.get_version(version_id)
        self.versions["current_version"] = version_id
        self._save_versions()
        return df
    
    def delete_version(self, version_id):
        """Delete a specific version."""
        # Find and remove version info
        self.versions["versions"] = [v for v in self.versions["versions"] if v["version_id"] != version_id]
        
        # Update current version if needed
        if self.versions["current_version"] == version_id:
            if self.versions["versions"]:
                self.versions["current_version"] = self.versions["versions"][-1]["version_id"]
            else:
                self.versions["current_version"] = None
        
        # Delete version directory
        version_dir = self.base_dir / version_id
        if version_dir.exists():
            import shutil
            shutil.rmtree(version_dir)
        
        # Save versions
        self._save_versions()
    
    def export_version(self, version_id, filepath, format="csv"):
        """
        Export a version to a file.
        
        Args:
            version_id: Version ID to export
            filepath: Path to save the file
            format: Export format (csv, excel, parquet, etc.)
        """
        df = self.get_version(version_id)
        
        if format.lower() == "csv":
            df.to_csv(filepath, index=False)
        elif format.lower() == "excel":
            df.to_excel(filepath, index=False)
        elif format.lower() == "parquet":
            df.to_parquet(filepath, index=False)
        elif format.lower() == "pickle":
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_change_log(self):
        """Get a summary of all changes made."""
        change_log = []
        
        for i, version in enumerate(self.versions["versions"]):
            change_info = {
                "version_id": version["version_id"],
                "step_name": version["step_name"],
                "description": version["description"],
                "timestamp": version["timestamp"],
                "shape": version["shape"],
                "is_current": version["version_id"] == self.versions["current_version"]
            }
            
            # Add comparison with previous version if available
            if i > 0:
                prev_version = self.versions["versions"][i-1]
                comparison = self.compare_versions(prev_version["version_id"], version["version_id"])
                change_info["changes"] = comparison
            
            change_log.append(change_info)
        
        return change_log

# Global instance for easy access
version_manager = DataVersionManager()
