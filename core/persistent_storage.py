import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
from pathlib import Path
from datetime import datetime
import hashlib

class PersistentStorage:
    """Persistent storage system for maintaining user progress across page refreshes and navigation."""
    
    def __init__(self, storage_dir="persistent_data"):
        """
        Initialize persistent storage.
        
        Args:
            storage_dir: Directory to store persistent data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # File paths for different types of data
        self.session_file = self.storage_dir / "session_data.json"
        self.data_file = self.storage_dir / "current_data.pkl"
        self.preprocessed_file = self.storage_dir / "preprocessed_data.pkl"
        self.model_file = self.storage_dir / "trained_model.pkl"
        self.metrics_file = self.storage_dir / "model_metrics.json"
        
    def save_session_state(self):
        """Save current session state to persistent storage."""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "data_loaded": "df" in st.session_state and st.session_state.df is not None,
            "data_type": st.session_state.get("data_type", None),
            "data_shape": st.session_state.df.shape if "df" in st.session_state and st.session_state.df is not None else None,
            "preprocessed": st.session_state.get("df_preprocessed") is not None,
            "model_trained": st.session_state.get("model_trained", False),
            "model_deployed": st.session_state.get("model_deployed", False),
            "target_col": st.session_state.get("target_col", None),
            "model_type": st.session_state.get("model_type", None),
            "current_page": st.session_state.get("current_page", "home"),
            "preprocessing_steps": st.session_state.get("preprocessing_steps", []),
            "data_hash": self._get_data_hash() if "df" in st.session_state and st.session_state.df is not None else None
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def load_session_state(self):
        """Load session state from persistent storage."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Restore session state flags
                st.session_state.data_loaded = session_data.get("data_loaded", False)
                st.session_state.data_type = session_data.get("data_type", None)
                st.session_state.preprocessed = session_data.get("preprocessed", False)
                st.session_state.model_trained = session_data.get("model_trained", False)
                st.session_state.model_deployed = session_data.get("model_deployed", False)
                st.session_state.target_col = session_data.get("target_col", None)
                st.session_state.model_type = session_data.get("model_type", None)
                st.session_state.current_page = session_data.get("current_page", "home")
                st.session_state.preprocessing_steps = session_data.get("preprocessing_steps", [])
                
                return session_data
            except Exception as e:
                print(f"Error loading session state: {e}")
                return None
        return None
    
    def save_data(self, df, data_type="tabular"):
        """Save current dataset to persistent storage."""
        if df is not None:
            try:
                # Save the dataframe
                with open(self.data_file, 'wb') as f:
                    pickle.dump(df, f)
                
                # Update session state
                st.session_state.df = df
                st.session_state.data_type = data_type
                st.session_state.data_loaded = True
                
                # Save session state
                self.save_session_state()
                
                return True
            except Exception as e:
                print(f"Error saving data: {e}")
                return False
        return False
    
    def load_data(self):
        """Load dataset from persistent storage."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    df = pickle.load(f)
                
                # Restore to session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                return df
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        return None
    
    def save_preprocessed_data(self, df):
        """Save preprocessed data to persistent storage."""
        if df is not None:
            try:
                with open(self.preprocessed_file, 'wb') as f:
                    pickle.dump(df, f)
                
                st.session_state.df_preprocessed = df
                st.session_state.preprocessed = True
                
                # Save session state
                self.save_session_state()
                
                return True
            except Exception as e:
                print(f"Error saving preprocessed data: {e}")
                return False
        return False
    
    def load_preprocessed_data(self):
        """Load preprocessed data from persistent storage."""
        if self.preprocessed_file.exists():
            try:
                with open(self.preprocessed_file, 'rb') as f:
                    df = pickle.load(f)
                
                st.session_state.df_preprocessed = df
                st.session_state.preprocessed = True
                
                return df
            except Exception as e:
                print(f"Error loading preprocessed data: {e}")
                return None
        return None
    
    def save_model(self, model, metrics, model_type, target_col):
        """Save trained model to persistent storage."""
        try:
            # Save model
            with open(self.model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Update session state
            st.session_state.model = model
            st.session_state.metrics = metrics
            st.session_state.model_type = model_type
            st.session_state.target_col = target_col
            st.session_state.model_trained = True
            
            # Save session state
            self.save_session_state()
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load trained model from persistent storage."""
        if self.model_file.exists() and self.metrics_file.exists():
            try:
                # Load model
                with open(self.model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Load metrics
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Restore to session state
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_trained = True
                
                return model, metrics
            except Exception as e:
                print(f"Error loading model: {e}")
                return None, None
        return None, None
    
    def add_preprocessing_step(self, step_name, description, metadata=None):
        """Add a preprocessing step to the history."""
        if "preprocessing_steps" not in st.session_state:
            st.session_state.preprocessing_steps = []
        
        step = {
            "step_name": step_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        st.session_state.preprocessing_steps.append(step)
        self.save_session_state()
    
    def get_preprocessing_history(self):
        """Get the history of preprocessing steps."""
        return st.session_state.get("preprocessing_steps", [])
    
    def clear_all_data(self):
        """Clear all persistent data."""
        try:
            # Remove all files
            for file_path in [self.session_file, self.data_file, self.preprocessed_file, 
                            self.model_file, self.metrics_file]:
                if file_path.exists():
                    file_path.unlink()
            
            # Clear session state - more comprehensive clearing
            keys_to_clear = [
                "df", "df_preprocessed", "model", "metrics", "data_type", 
                "target_col", "model_type", "preprocessing_steps", "data_loaded",
                "preprocessed", "model_trained", "model_deployed", "current_page",
                "image_file", "audio_file", "array_data", "image_directory", 
                "audio_directory", "fitted_scaler", "fitted_encoders"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Reset to initial state
            st.session_state.df = None
            st.session_state.df_preprocessed = None
            st.session_state.model_trained = None
            st.session_state.model_deployed = None
            
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
    
    def get_storage_info(self):
        """Get information about stored data."""
        info = {
            "session_exists": self.session_file.exists(),
            "data_exists": self.data_file.exists(),
            "preprocessed_exists": self.preprocessed_file.exists(),
            "model_exists": self.model_file.exists(),
            "metrics_exists": self.metrics_file.exists()
        }
        
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                info.update({
                    "last_saved": session_data.get("timestamp"),
                    "data_loaded": session_data.get("data_loaded"),
                    "preprocessed": session_data.get("preprocessed"),
                    "model_trained": session_data.get("model_trained")
                })
            except:
                pass
        
        return info
    
    def _get_data_hash(self):
        """Generate a hash for the current data."""
        if "df" in st.session_state and st.session_state.df is not None:
            return hashlib.md5(pd.util.hash_pandas_object(st.session_state.df).values).hexdigest()
        return None
    
    def is_data_changed(self):
        """Check if data has changed since last save."""
        current_hash = self._get_data_hash()
        if "df" in st.session_state and st.session_state.df is not None:
            stored_hash = st.session_state.get("data_hash")
            return current_hash != stored_hash
        return False

# Global instance for easy access
persistent_storage = PersistentStorage()

# Convenience functions for easy access
def save_progress():
    """Save current progress to persistent storage."""
    persistent_storage.save_session_state()

def load_progress():
    """Load progress from persistent storage."""
    return persistent_storage.load_session_state()

def auto_save_data(df, data_type="tabular"):
    """Automatically save data with progress tracking."""
    success = persistent_storage.save_data(df, data_type)
    if success:
        persistent_storage.add_preprocessing_step(
            "data_upload", 
            f"Data uploaded - {df.shape[0]} rows × {df.shape[1]} columns",
            {"data_type": data_type, "shape": df.shape}
        )
    return success

def auto_save_preprocessed(df):
    """Automatically save preprocessed data with progress tracking."""
    success = persistent_storage.save_preprocessed_data(df)
    if success:
        persistent_storage.add_preprocessing_step(
            "preprocessing_complete", 
            f"Preprocessing completed - {df.shape[0]} rows × {df.shape[1]} columns",
            {"final_shape": df.shape}
        )
    return success

def auto_save_model(model, metrics, model_type, target_col):
    """Automatically save model with progress tracking."""
    success = persistent_storage.save_model(model, metrics, model_type, target_col)
    if success:
        persistent_storage.add_preprocessing_step(
            "model_training", 
            f"Model trained - {model_type} for {target_col}",
            {"model_type": model_type, "target_col": target_col, "metrics": metrics}
        )
    return success
