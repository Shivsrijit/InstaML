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
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not available. Install with: pip install librosa")

class AudioDataset(Dataset):
    """Custom PyTorch dataset for audio data."""
    
    def __init__(self, audio_paths, labels, sample_rate=22050, duration=3):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa is required for audio processing.")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Pad or truncate to fixed length
        target_length = self.sample_rate * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        return audio_tensor, label

class AudioModelTrainer:
    """Comprehensive trainer for audio-based models."""
    
    def __init__(self, data_path, task_type="classification", framework="pytorch", 
                 sample_rate=22050, duration=3, batch_size=32):
        self.data_path = Path(data_path)
        self.task_type = task_type
        self.framework = framework
        self.sample_rate = sample_rate
        self.duration = duration
        self.batch_size = batch_size
        
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa is required for audio processing.")
        
        if framework == "pytorch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        if framework == "tensorflow" and not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available.")
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare audio data paths and labels."""
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist.")
        
        if (self.data_path / "train").exists() and (self.data_path / "val").exists():
            self.train_path = self.data_path / "train"
            self.val_path = self.data_path / "val"
        else:
            self.train_path = self.data_path
            self.val_path = self.data_path
        
        if self.task_type == "classification":
            self.class_names = self._get_class_names()
            self.num_classes = len(self.class_names)
        
        self._prepare_data_lists()
    
    def _get_class_names(self):
        """Get class names from directory structure."""
        if (self.data_path / "train").exists():
            class_dirs = [d for d in (self.data_path / "train").iterdir() if d.is_dir()]
            return [d.name for d in class_dirs]
        else:
            return ["class_0", "class_1"]
    
    def _prepare_data_lists(self):
        """Prepare lists of audio paths and labels."""
        if self.task_type == "classification":
            self.train_audio, self.train_labels = self._get_classification_data(self.train_path)
            self.val_audio, self.val_labels = self._get_classification_data(self.val_path)
        else:
            self.train_audio = self._get_audio_paths(self.train_path)
            self.val_audio = self._get_audio_paths(self.val_path)
    
    def _get_classification_data(self, data_path):
        """Get audio paths and labels for classification."""
        audio_files = []
        labels = []
        
        if data_path.exists():
            for class_idx, class_name in enumerate(self.class_names):
                class_path = data_path / class_name
                if class_path.exists():
                    for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
                        for audio_file in class_path.glob(ext):
                            audio_files.append(str(audio_file))
                            labels.append(class_idx)
        
        return audio_files, labels
    
    def _get_audio_paths(self, data_path):
        """Get all audio paths from directory."""
        audio_files = []
        if data_path.exists():
            for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
                audio_files.extend([str(p) for p in data_path.glob(ext)])
        return audio_files
    
    def extract_mel_spectrogram_features(self, audio_path):
        """Extract mel spectrogram features from audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        target_length = self.sample_rate * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def train_classification_model(self, model_name="cnn", epochs=10, learning_rate=0.001):
        """Train an audio classification model."""
        if self.task_type != "classification":
            raise ValueError("This method is only for classification tasks.")
        
        if self.framework == "pytorch":
            return self._train_pytorch_classification(model_name, epochs, learning_rate)
        else:
            return self._train_tensorflow_classification(model_name, epochs, learning_rate)
    
    def get_available_models(self):
        """Get list of available models for the current task type."""
        if self.task_type == "classification":
            return {
                "classification": ["cnn", "lstm"]
            }
        else:
            return {
                "classification": ["cnn", "lstm"]
            }
    
    def _train_pytorch_classification(self, model_name, epochs, learning_rate):
        """Train PyTorch audio classification model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == "cnn":
            model = self._create_audio_cnn()
        elif model_name == "lstm":
            model = self._create_audio_lstm()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(device)
        
        train_dataset = AudioDataset(self.train_audio, self.train_labels, 
                                   self.sample_rate, self.duration)
        val_dataset = AudioDataset(self.val_audio, self.val_labels, 
                                 self.sample_rate, self.duration)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for audio, labels in train_loader:
                audio, labels = audio.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(audio)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for audio, labels in val_loader:
                    audio, labels = audio.to(device), labels.to(device)
                    outputs = model(audio)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_acc = val_correct / val_total
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(epoch_val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        self.trained_model = model
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        }
        
        return model, self.training_history
    
    def _create_audio_cnn(self):
        """Create a CNN model for audio classification."""
        class AudioCNN(nn.Module):
            def __init__(self, num_classes, sample_rate, duration):
                super(AudioCNN, self).__init__()
                
                self.sample_rate = sample_rate
                self.duration = duration
                
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
                
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(0.5)
                
                input_size = sample_rate * duration
                conv_output_size = input_size // 8
                
                self.fc1 = nn.Linear(128 * conv_output_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, num_classes)
                
                self.relu = nn.ReLU()
                self.batch_norm1 = nn.BatchNorm1d(32)
                self.batch_norm2 = nn.BatchNorm1d(64)
                self.batch_norm3 = nn.BatchNorm1d(128)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
                
                x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
                x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
                x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
                
                x = x.view(x.size(0), -1)
                
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                
                return x
        
        return AudioCNN(self.num_classes, self.sample_rate, self.duration)
    
    def _create_audio_lstm(self):
        """Create an LSTM model for audio classification."""
        class AudioLSTM(nn.Module):
            def __init__(self, num_classes, hidden_size=128, num_layers=2):
                super(AudioLSTM, self).__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.5 if num_layers > 1 else 0
                )
                
                self.fc1 = nn.Linear(hidden_size, 256)
                self.fc2 = nn.Linear(256, num_classes)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)
                
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]
                
                x = self.dropout(self.relu(self.fc1(lstm_out)))
                x = self.fc2(x)
                
                return x
        
        return AudioLSTM(self.num_classes)
    
    def _train_tensorflow_classification(self, model_name, epochs, learning_rate):
        """Train TensorFlow audio classification model."""
        if model_name == "cnn":
            model = self._create_tf_audio_cnn()
        elif model_name == "lstm":
            model = self._create_tf_audio_lstm()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        train_generator = self._create_tf_data_generator(self.train_audio, self.train_labels)
        val_generator = self._create_tf_data_generator(self.val_audio, self.val_labels)
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        self.trained_model = model
        self.training_history = history.history
        
        return model, self.training_history
    
    def _create_tf_audio_cnn(self):
        """Create a TensorFlow CNN model for audio classification."""
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=(self.sample_rate * self.duration, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_tf_audio_lstm(self):
        """Create a TensorFlow LSTM model for audio classification."""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sample_rate * self.duration, 1)),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_tf_data_generator(self, audio_paths, labels):
        """Create TensorFlow data generator for audio data."""
        def generator():
            for audio_path, label in zip(audio_paths, labels):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
                target_length = self.sample_rate * self.duration
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                else:
                    audio = audio[:target_length]
                
                features = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
                yield features, label
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        return dataset.batch(self.batch_size)
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        
        if self.framework == "pytorch":
            torch.save(self.trained_model.state_dict(), filepath)
        else:
            self.trained_model.save(filepath)

# Convenience functions
def train_audio_classification(data_path, model_name="cnn", framework="pytorch", 
                              epochs=10, sample_rate=22050, duration=3, batch_size=32):
    """Convenience function for training audio classification models."""
    trainer = AudioModelTrainer(data_path, task_type="classification", framework=framework,
                               sample_rate=sample_rate, duration=duration, batch_size=batch_size)
    return trainer.train_classification_model(model_name, epochs)
