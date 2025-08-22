import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the training modules using absolute imports
from core.ML_models.tabular_data import TabularModelTrainer
from core.ML_models.image_data import ImageModelTrainer
from core.ML_models.audio_data import AudioModelTrainer
from core.ML_models.multi_dimensional_data import MultiDimensionalTrainer

class UnifiedModelTrainer:
    """Unified interface for training models on different data types."""
    
    def __init__(self, data_source, target_col=None, data_type="auto", **kwargs):
        """
        Initialize the unified trainer.
        
        Args:
            data_source: Path to data or DataFrame/array
            target_col: Target column name (for supervised learning)
            data_type: "auto", "tabular", "image", "audio", or "multi_dimensional"
            **kwargs: Additional parameters for specific trainers
        """
        self.data_source = data_source
        self.target_col = target_col
        self.data_type = data_type
        self.kwargs = kwargs
        
        # Auto-detect data type if not specified
        if data_type == "auto":
            self.data_type = self._detect_data_type()
        
        # Initialize the appropriate trainer
        self._initialize_trainer()
    
    def _detect_data_type(self):
        """Auto-detect the type of data."""
        if isinstance(self.data_source, str) or isinstance(self.data_source, Path):
            file_path = Path(self.data_source)
            
            # Check file extension
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return "image"
            elif file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.aac']:
                return "audio"
            elif file_path.is_dir():
                # Check directory contents
                if self._is_image_directory(file_path):
                    return "image"
                elif self._is_audio_directory(file_path):
                    return "audio"
                else:
                    return "multi_dimensional"
            else:
                return "tabular"
        
        elif isinstance(self.data_source, pd.DataFrame):
            return "tabular"
        elif isinstance(self.data_source, np.ndarray):
            if len(self.data_source.shape) <= 2:
                return "tabular"
            else:
                return "multi_dimensional"
        else:
            return "tabular"
    
    def _is_image_directory(self, directory):
        """Check if directory contains image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                return True
        return False
    
    def _is_audio_directory(self, directory):
        """Check if directory contains audio files."""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                return True
        return False
    
    def _initialize_trainer(self):
        """Initialize the appropriate trainer based on data type."""
        if self.data_type == "tabular":
            if isinstance(self.data_source, pd.DataFrame):
                self.trainer = TabularModelTrainer(
                    self.data_source, 
                    self.target_col, 
                    scaling=self.kwargs.get('scaling', 'standard'),
                    test_size=self.kwargs.get('test_size', 0.2),
                    random_state=self.kwargs.get('random_state', 42)
                )
            else:
                # Load CSV file
                df = pd.read_csv(self.data_source)
                self.trainer = TabularModelTrainer(
                    df, 
                    self.target_col, 
                    scaling=self.kwargs.get('scaling', 'standard'),
                    test_size=self.kwargs.get('test_size', 0.2),
                    random_state=self.kwargs.get('random_state', 42)
                )
        
        elif self.data_type == "image":
            self.trainer = ImageModelTrainer(
                self.data_source,
                **self.kwargs
            )
        
        elif self.data_type == "audio":
            self.trainer = AudioModelTrainer(
                self.data_source,
                **self.kwargs
            )
        
        elif self.data_type == "multi_dimensional":
            self.trainer = MultiDimensionalTrainer(
                self.data_source,
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def get_available_models(self):
        """Get list of available models for the current data type."""
        if self.data_type == "tabular":
            return self.trainer.get_available_models()
        elif self.data_type == "image":
            return {
                "classification": ["resnet18", "resnet50", "vgg16", "mobilenet"],
                "detection": ["yolo"]
            }
        elif self.data_type == "audio":
            return ["cnn", "lstm"]
        elif self.data_type == "multi_dimensional":
            return self.trainer.get_available_models()
    
    def train_model(self, model_name, **kwargs):
        """
        Train a model using the appropriate trainer.
        
        Args:
            model_name: Name of the model to train
            **kwargs: Additional parameters for training
            
        Returns:
            Trained model and metrics
        """
        if self.data_type == "tabular":
            # For tabular data, only pass model_name and use_hyperparameter_tuning
            training_kwargs = {
                'use_hyperparameter_tuning': kwargs.get('use_hyperparameter_tuning', True)
            }
            return self.trainer.train_model(model_name, **training_kwargs)
        
        elif self.data_type == "image":
            if model_name == "yolo":
                return self.trainer.train_yolo_model(**kwargs)
            else:
                return self.trainer.train_classification_model(model_name, **kwargs)
        
        elif self.data_type == "audio":
            return self.trainer.train_classification_model(model_name, **kwargs)
        
        elif self.data_type == "multi_dimensional":
            return self.trainer.train_model(model_name, **kwargs)
    
    def get_data_info(self):
        """Get information about the data."""
        info = {
            "data_type": self.data_type,
            "data_source": str(self.data_source)
        }
        
        if self.data_type == "tabular":
            if hasattr(self.trainer, 'df'):
                info.update({
                    "shape": self.trainer.df.shape,
                    "columns": list(self.trainer.df.columns),
                    "dtypes": self.trainer.df.dtypes.to_dict()
                })
        elif self.data_type == "image":
            if hasattr(self.trainer, 'class_names'):
                info.update({
                    "num_classes": len(self.trainer.class_names),
                    "class_names": self.trainer.class_names
                })
        elif self.data_type == "audio":
            if hasattr(self.trainer, 'class_names'):
                info.update({
                    "num_classes": len(self.trainer.class_names),
                    "class_names": self.trainer.class_names
                })
        elif self.data_type == "multi_dimensional":
            if hasattr(self.trainer, 'data_shape'):
                info.update({
                    "data_shape": self.trainer.data_shape,
                    "num_dimensions": self.trainer.num_dimensions
                })
        
        return info
    
    def save_model(self, filepath):
        """Save the trained model."""
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'save_model'):
            self.trainer.save_model(filepath)
        else:
            raise ValueError("No trained model available to save.")
    
    def get_training_history(self):
        """Get training history if available."""
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'training_history'):
            return self.trainer.training_history
        else:
            return None

# Convenience functions for easy access
def train_model(data_source, target_col=None, data_type="auto", model_name="auto", **kwargs):
    """
    Convenience function for training models.
    
    Args:
        data_source: Path to data or DataFrame/array
        target_col: Target column name (for supervised learning)
        data_type: "auto", "tabular", "image", "audio", or "multi_dimensional"
        model_name: "auto" or specific model name
        **kwargs: Additional parameters
        
    Returns:
        Trained model and metrics
    """
    # Filter out parameters that should only be used during initialization
    init_kwargs = {
        'scaling': kwargs.get('scaling', 'standard'),
        'test_size': kwargs.get('test_size', 0.2),
        'random_state': kwargs.get('random_state', 42)
    }
    
    # Remove these from kwargs to avoid passing them to train_model
    training_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['scaling', 'test_size', 'random_state']}
    
    trainer = UnifiedModelTrainer(data_source, target_col, data_type, **init_kwargs)
    
    # Auto-select model if not specified
    if model_name == "auto":
        available_models = trainer.get_available_models()
        if data_type == "tabular":
            # For tabular data, default to Random Forest
            model_name = "Random Forest"
        elif data_type == "image":
            # For image data, default to ResNet18
            model_name = "resnet18"
        elif data_type == "audio":
            # For audio data, default to CNN
            model_name = "cnn"
        elif data_type == "multi_dimensional":
            # For multi-dimensional data, default to MLP
            model_name = "MLP"
    
    return trainer.train_model(model_name, **training_kwargs)

def get_data_info(data_source, data_type="auto"):
    """Get information about the data."""
    trainer = UnifiedModelTrainer(data_source, data_type=data_type)
    return trainer.get_data_info()

def get_available_models(data_source, data_type="auto"):
    """Get available models for the data."""
    trainer = UnifiedModelTrainer(data_source, data_type=data_type)
    return trainer.get_available_models()

# Example usage functions
def train_tabular_model_from_csv(csv_path, target_col, model_name="Random Forest", **kwargs):
    """Train a tabular model from a CSV file."""
    return train_model(csv_path, target_col, "tabular", model_name, **kwargs)

def train_image_classification_model(image_dir, model_name="resnet18", **kwargs):
    """Train an image classification model."""
    return train_model(image_dir, data_type="image", model_name=model_name, **kwargs)

def train_audio_classification_model(audio_dir, model_name="cnn", **kwargs):
    """Train an audio classification model."""
    return train_model(audio_dir, data_type="audio", model_name=model_name, **kwargs)

def train_multi_dimensional_model(data_source, task_type="regression", model_name="MLP", **kwargs):
    """Train a multi-dimensional model."""
    return train_model(data_source, data_type="multi_dimensional", model_name=model_name, 
                      task_type=task_type, **kwargs)
