import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow import keras
    from keras._tf_keras.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MultiDimensionalDataset(Dataset):
    """Custom PyTorch dataset for multi-dimensional data."""
    
    def __init__(self, data, labels=None, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            return sample, self.labels[idx]
        else:
            return sample

class MultiDimensionalTrainer:
    """Comprehensive trainer for multi-dimensional data models."""
    
    def __init__(self, data, task_type="regression", framework="pytorch", 
                 test_size=0.2, random_state=42):
        """
        Initialize the multi-dimensional trainer.
        
        Args:
            data: Input data (numpy array, pandas DataFrame, or file path)
            task_type: "regression", "classification", or "clustering"
            framework: "pytorch" or "tensorflow"
            test_size: Test set size
            random_state: Random seed
        """
        self.task_type = task_type
        self.framework = framework
        self.test_size = test_size
        self.random_state = random_state
        
        # Validate framework availability
        if framework == "pytorch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it first.")
        if framework == "tensorflow" and not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install it first.")
        
        # Load and prepare data
        self._load_data(data)
        self._prepare_data()
        
        # Initialize models
        self._initialize_models()
        
    def _load_data(self, data):
        """Load data from various sources."""
        if isinstance(data, str) or isinstance(data, Path):
            # Load from file
            file_path = Path(data)
            if file_path.suffix == '.csv':
                self.raw_data = pd.read_csv(file_path)
            elif file_path.suffix == '.npy':
                self.raw_data = np.load(file_path)
            elif file_path.suffix == '.npz':
                self.raw_data = np.load(file_path)
                # Handle .npz files (compressed numpy arrays)
                if isinstance(self.raw_data, np.lib.npyio.NpzFile):
                    # Get the first array
                    first_key = list(self.raw_data.keys())[0]
                    self.raw_data = self.raw_data[first_key]
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data
        elif isinstance(data, np.ndarray):
            self.raw_data = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Convert to numpy array if needed
        if isinstance(self.raw_data, pd.DataFrame):
            self.data_array = self.raw_data.values
        else:
            self.data_array = self.raw_data
        
        # Get data shape and dimensions
        self.data_shape = self.data_array.shape
        self.num_dimensions = len(self.data_shape)
        
        print(f"Data loaded with shape: {self.data_shape}")
        print(f"Number of dimensions: {self.num_dimensions}")
    
    def _prepare_data(self):
        """Prepare data for training."""
        # Reshape data if needed
        if self.num_dimensions == 1:
            # 1D data - reshape to (samples, features)
            self.data_array = self.data_array.reshape(-1, 1)
        elif self.num_dimensions == 2:
            # 2D data - assume (samples, features)
            pass
        elif self.num_dimensions == 3:
            # 3D data - reshape to (samples, channels, features)
            if self.data_array.shape[0] < self.data_array.shape[1]:
                # Assume (features, channels, samples) -> (samples, channels, features)
                self.data_array = np.transpose(self.data_array, (2, 1, 0))
        elif self.num_dimensions > 3:
            # Higher dimensional data - flatten to 2D
            original_shape = self.data_array.shape
            self.data_array = self.data_array.reshape(original_shape[0], -1)
            print(f"Reshaped {original_shape} to {self.data_array.shape}")
        
        # Split data
        if self.task_type == "clustering":
            # For clustering, no train/test split needed
            self.X_train = self.data_array
            self.X_test = self.data_array
            self.y_train = None
            self.y_test = None
        else:
            # For supervised learning, need labels
            if self.num_dimensions == 2:
                # Assume last column is target
                X = self.data_array[:, :-1]
                y = self.data_array[:, -1]
            else:
                # For higher dimensional data, need to specify target
                raise ValueError("For multi-dimensional data, please specify target column or use 2D data with target in last column")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train_scaled.shape}")
        if self.y_train is not None:
            print(f"Target shape: {self.y_train.shape}")
    
    def _initialize_models(self):
        """Initialize model dictionaries."""
        if self.task_type == "regression":
            self.models = {
                "MLP": "mlp",
                "CNN": "cnn",
                "LSTM": "lstm",
                "Transformer": "transformer"
            }
        elif self.task_type == "classification":
            self.models = {
                "MLP": "mlp",
                "CNN": "cnn",
                "LSTM": "lstm",
                "Transformer": "transformer"
            }
        elif self.task_type == "clustering":
            self.models = {
                "KMeans": "kmeans",
                "DBSCAN": "dbscan",
                "Hierarchical": "hierarchical"
            }
    
    def train_model(self, model_name, **kwargs):
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            **kwargs: Additional parameters for the model
            
        Returns:
            Trained model and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if self.task_type == "clustering":
            return self._train_clustering_model(model_name, **kwargs)
        elif self.framework == "pytorch":
            return self._train_pytorch_model(model_name, **kwargs)
        else:
            return self._train_tensorflow_model(model_name, **kwargs)
    
    def _train_clustering_model(self, model_name, **kwargs):
        """Train clustering models."""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        
        if model_name == "KMeans":
            n_clusters = kwargs.get('n_clusters', 3)
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        elif model_name == "DBSCAN":
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif model_name == "Hierarchical":
            n_clusters = kwargs.get('n_clusters', 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering model: {model_name}")
        
        # Fit model
        if model_name == "DBSCAN":
            # DBSCAN doesn't have predict method, use fit_predict
            labels = model.fit_predict(self.X_train_scaled)
        else:
            model.fit(self.X_train_scaled)
            labels = model.predict(self.X_train_scaled)
        
        # Calculate metrics
        metrics = self._evaluate_clustering(labels)
        
        self.trained_model = model
        self.cluster_labels = labels
        self.metrics = metrics
        
        return model, metrics
    
    def _evaluate_clustering(self, labels):
        """Evaluate clustering results."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        metrics = {}
        
        # Silhouette score (higher is better)
        try:
            metrics["silhouette_score"] = silhouette_score(self.X_train_scaled, labels)
        except:
            metrics["silhouette_score"] = None
        
        # Calinski-Harabasz score (higher is better)
        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(self.X_train_scaled, labels)
        except:
            metrics["calinski_harabasz_score"] = None
        
        # Number of clusters
        metrics["n_clusters"] = len(np.unique(labels))
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics["cluster_sizes"] = dict(zip(unique_labels, counts))
        
        return metrics
    
    def _train_pytorch_model(self, model_name, **kwargs):
        """Train PyTorch model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        if model_name == "MLP":
            model = self._create_pytorch_mlp(**kwargs)
        elif model_name == "CNN":
            model = self._create_pytorch_cnn(**kwargs)
        elif model_name == "LSTM":
            model = self._create_pytorch_lstm(**kwargs)
        elif model_name == "Transformer":
            model = self._create_pytorch_transformer(**kwargs)
        else:
            raise ValueError(f"Unsupported PyTorch model: {model_name}")
        
        model = model.to(device)
        
        # Create datasets and dataloaders
        train_dataset = MultiDimensionalDataset(self.X_train_scaled, self.y_train)
        val_dataset = MultiDimensionalDataset(self.X_test_scaled, self.y_test)
        
        batch_size = kwargs.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        epochs = kwargs.get('epochs', 100)
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Evaluate model
        metrics = self._evaluate_pytorch_model(model, val_loader, device)
        
        self.trained_model = model
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        self.metrics = metrics
        
        return model, metrics
    
    def _create_pytorch_mlp(self, **kwargs):
        """Create PyTorch MLP model."""
        input_size = self.X_train_scaled.shape[1]
        
        if self.task_type == "classification":
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64, 32])
        
        layers_list = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers_list.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers_list.append(nn.Linear(prev_size, output_size))
        
        if self.task_type == "classification":
            layers_list.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*layers_list)
    
    def _create_pytorch_cnn(self, **kwargs):
        """Create PyTorch CNN model for 1D data."""
        input_size = self.X_train_scaled.shape[1]
        
        if self.task_type == "classification":
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        class CNN1D(nn.Module):
            def __init__(self, input_size, output_size):
                super(CNN1D, self).__init__()
                
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(0.3)
                
                # Calculate output size after convolutions
                conv_output_size = input_size // 8 * 128
                
                self.fc1 = nn.Linear(conv_output_size, 256)
                self.fc2 = nn.Linear(256, output_size)
                
                self.relu = nn.ReLU()
                self.batch_norm1 = nn.BatchNorm1d(32)
                self.batch_norm2 = nn.BatchNorm1d(64)
                self.batch_norm3 = nn.BatchNorm1d(128)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                
                x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
                x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
                x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
                
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                
                return x
        
        return CNN1D(input_size, output_size)
    
    def _create_pytorch_lstm(self, **kwargs):
        """Create PyTorch LSTM model."""
        input_size = self.X_train_scaled.shape[1]
        
        if self.task_type == "classification":
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 2)
        
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTM, self).__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.3 if num_layers > 1 else 0
                )
                
                self.fc1 = nn.Linear(hidden_size, 256)
                self.fc2 = nn.Linear(256, output_size)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)  # Add feature dimension
                
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]  # Take last output
                
                x = self.dropout(self.relu(self.fc1(lstm_out)))
                x = self.fc2(x)
                
                return x
        
        return LSTM(input_size, hidden_size, num_layers, output_size)
    
    def _create_pytorch_transformer(self, **kwargs):
        """Create PyTorch Transformer model."""
        input_size = self.X_train_scaled.shape[1]
        
        if self.task_type == "classification":
            output_size = len(np.unique(self.y_train))
        else:
            output_size = 1
        
        d_model = kwargs.get('d_model', 128)
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 6)
        
        class Transformer(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, output_size):
                super(Transformer, self).__init__()
                
                self.input_projection = nn.Linear(1, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=512,
                    dropout=0.1
                )
                
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                self.fc1 = nn.Linear(d_model, 256)
                self.fc2 = nn.Linear(256, output_size)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)  # Add feature dimension
                
                # Project to d_model dimensions
                x = self.input_projection(x)  # (batch, seq_len, d_model)
                
                # Add positional encoding
                seq_len = x.size(1)
                x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
                
                # Transformer expects (seq_len, batch, d_model)
                x = x.transpose(0, 1)
                
                # Apply transformer
                x = self.transformer(x)
                
                # Take mean across sequence length
                x = x.mean(dim=0)
                
                # Final layers
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                
                return x
        
        return Transformer(input_size, d_model, nhead, num_layers, output_size)
    
    def _evaluate_pytorch_model(self, model, val_loader, device):
        """Evaluate PyTorch model."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                if self.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = outputs.squeeze()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        if self.task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {
                "accuracy": accuracy_score(all_labels, all_predictions),
                "f1_score": f1_score(all_labels, all_predictions, average='weighted')
            }
        else:
            metrics = {
                "mse": mean_squared_error(all_labels, all_predictions),
                "rmse": np.sqrt(mean_squared_error(all_labels, all_predictions)),
                "mae": mean_absolute_error(all_labels, all_predictions),
                "r2": r2_score(all_labels, all_predictions)
            }
        
        return metrics
    
    def _train_tensorflow_model(self, model_name, **kwargs):
        """Train TensorFlow model."""
        # This would implement TensorFlow training similar to PyTorch
        # For brevity, returning a placeholder
        raise NotImplementedError("TensorFlow training not implemented yet. Use PyTorch framework.")
    
    def get_available_models(self):
        """Get list of available models for the current task type."""
        if self.task_type == "regression":
            return {
                "regression": ["MLP", "CNN", "LSTM", "Transformer"]
            }
        elif self.task_type == "classification":
            return {
                "classification": ["MLP", "CNN", "LSTM", "Transformer"]
            }
        elif self.task_type == "clustering":
            return {
                "clustering": ["KMeans", "DBSCAN", "Hierarchical"]
            }
        else:
            return {
                "regression": ["MLP", "CNN", "LSTM", "Transformer"],
                "classification": ["MLP", "CNN", "LSTM", "Transformer"],
                "clustering": ["KMeans", "DBSCAN", "Hierarchical"]
            }
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        
        if self.framework == "pytorch":
            torch.save(self.trained_model.state_dict(), filepath)
        else:
            self.trained_model.save(filepath)
    
    def load_model(self, filepath):
        """Load a saved model."""
        if self.framework == "pytorch":
            # This would need to recreate the model architecture first
            raise NotImplementedError("PyTorch model loading not implemented yet.")
        else:
            self.trained_model = keras.models.load_model(filepath)
            return self.trained_model

# Convenience functions
def train_multi_dimensional_model(data, model_name="MLP", task_type="regression", 
                                 framework="pytorch", **kwargs):
    """Convenience function for training multi-dimensional models."""
    trainer = MultiDimensionalTrainer(data, task_type=task_type, framework=framework)
    return trainer.train_model(model_name, **kwargs)

def get_available_models():
    """Get list of available models."""
    return {
        "regression": ["MLP", "CNN", "LSTM", "Transformer"],
        "classification": ["MLP", "CNN", "LSTM", "Transformer"],
        "clustering": ["KMeans", "DBSCAN", "Hierarchical"]
    }
