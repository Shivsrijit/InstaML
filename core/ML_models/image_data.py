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
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

class ImageDataset(Dataset):
    """Custom PyTorch dataset for image data."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageModelTrainer:
    """Comprehensive trainer for image-based models."""
    
    def __init__(self, data_path, task_type="classification", framework="pytorch", 
                 img_size=(224, 224), batch_size=32, random_state=42):
        """
        Initialize the image trainer.
        
        Args:
            data_path: Path to image data directory
            task_type: "classification", "detection", or "segmentation"
            framework: "pytorch" or "tensorflow"
            img_size: Image dimensions (width, height)
            batch_size: Batch size for training
            random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.task_type = task_type
        self.framework = framework
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Validate framework availability
        if framework == "pytorch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it first.")
        if framework == "tensorflow" and not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install it first.")
        
        # Prepare data
        self._prepare_data()
        
        # Initialize transforms
        self._initialize_transforms()
        
    def _prepare_data(self):
        """Prepare image data paths and labels."""
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist.")
        
        # Look for common directory structures
        if (self.data_path / "train").exists() and (self.data_path / "val").exists():
            # Standard train/val split
            self.train_path = self.data_path / "train"
            self.val_path = self.data_path / "val"
        elif (self.data_path / "images").exists() and (self.data_path / "labels").exists():
            # YOLO-style structure
            self.train_path = self.data_path / "images"
            self.val_path = self.data_path / "images"  # Will split later
        else:
            # Single directory - will split randomly
            self.train_path = self.data_path
            self.val_path = self.data_path
        
        # Get class names
        if self.task_type == "classification":
            self.class_names = self._get_class_names()
            self.num_classes = len(self.class_names)
        
        # Prepare data lists
        self._prepare_data_lists()
    
    def _get_class_names(self):
        """Get class names from directory structure."""
        if (self.data_path / "train").exists():
            # Classes are subdirectories in train folder
            class_dirs = [d for d in (self.data_path / "train").iterdir() if d.is_dir()]
            return [d.name for d in class_dirs]
        else:
            # Try to infer from data structure
            return ["class_0", "class_1"]  # Default fallback
    
    def _prepare_data_lists(self):
        """Prepare lists of image paths and labels."""
        if self.task_type == "classification":
            self.train_images, self.train_labels = self._get_classification_data(self.train_path)
            self.val_images, self.val_labels = self._get_classification_data(self.val_path)
        else:
            # For detection/segmentation, we'll need label files
            self.train_images = self._get_image_paths(self.train_path)
            self.val_images = self._get_image_paths(self.val_path)
    
    def _get_classification_data(self, data_path):
        """Get image paths and labels for classification."""
        images = []
        labels = []
        
        if data_path.exists():
            for class_idx, class_name in enumerate(self.class_names):
                class_path = data_path / class_name
                if class_path.exists():
                    for img_file in class_path.glob("*.jpg"):
                        images.append(str(img_file))
                        labels.append(class_idx)
                    for img_file in class_path.glob("*.png"):
                        images.append(str(img_file))
                        labels.append(class_idx)
                    for img_file in class_path.glob("*.jpeg"):
                        images.append(str(img_file))
                        labels.append(class_idx)
        
        return images, labels
    
    def _get_image_paths(self, data_path):
        """Get all image paths from directory."""
        images = []
        if data_path.exists():
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                images.extend([str(p) for p in data_path.glob(ext)])
        return images
    
    def _initialize_transforms(self):
        """Initialize image transforms."""
        if self.framework == "pytorch":
            self.train_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.val_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # TensorFlow
            self.train_transform = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            self.val_transform = ImageDataGenerator(rescale=1./255)
    
    def train_classification_model(self, model_name="resnet18", epochs=10, learning_rate=0.001):
        """Train an image classification model."""
        if self.task_type != "classification":
            raise ValueError("This method is only for classification tasks.")
        
        if self.framework == "pytorch":
            return self._train_pytorch_classification(model_name, epochs, learning_rate)
        else:
            return self._train_tensorflow_classification(model_name, epochs, learning_rate)
    
    def _train_pytorch_classification(self, model_name, epochs, learning_rate):
        """Train PyTorch classification model."""
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier[-1] = nn.Linear(4096, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(device)
        
        # Create datasets and dataloaders
        train_dataset = ImageDataset(self.train_images, self.train_labels, self.train_transform)
        val_dataset = ImageDataset(self.val_images, self.val_labels, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
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
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
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
        
        # Store results
        self.trained_model = model
        self.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        }
        
        return model, self.training_history
    
    def _train_tensorflow_classification(self, model_name, epochs, learning_rate):
        """Train TensorFlow classification model."""
        # Create model
        if model_name == "resnet50":
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
            base_model.trainable = False
            
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        elif model_name == "vgg16":
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
            base_model.trainable = False
            
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        elif model_name == "mobilenet":
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
            base_model.trainable = False
            
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create data generators
        train_generator = self.train_transform.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse'
        )
        
        val_generator = self.val_transform.flow_from_directory(
            self.val_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse'
        )
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        # Store results
        self.trained_model = model
        self.training_history = history.history
        
        return model, self.training_history
    
    def train_yolo_model(self, data_yaml_path, model_size="n", epochs=100, img_size=640):
        """Train YOLO model using Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        # Load YOLO model
        model = YOLO(f"yolo{model_size}.pt")
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=self.batch_size,
            project="yolo_training",
            name=f"yolo_{model_size}_custom"
        )
        
        self.trained_model = model
        self.training_history = results
        
        return model, results
    
    def evaluate_model(self, test_images=None, test_labels=None):
        """Evaluate the trained model."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        
        if test_images is None:
            test_images = self.val_images
            test_labels = self.val_labels
        
        if self.framework == "pytorch":
            return self._evaluate_pytorch_model(test_images, test_labels)
        else:
            return self._evaluate_tensorflow_model(test_images, test_labels)
    
    def _evaluate_pytorch_model(self, test_images, test_labels):
        """Evaluate PyTorch model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model.eval()
        
        test_dataset = ImageDataset(test_images, test_labels, self.val_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_labels = []
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.trained_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = test_correct / test_total
        avg_loss = test_loss / len(test_loader)
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "predictions": all_predictions,
            "true_labels": all_labels
        }
    
    def _evaluate_tensorflow_model(self, test_images, test_labels):
        """Evaluate TensorFlow model."""
        # This would need to be implemented based on the specific test data format
        # For now, return basic metrics
        return {
            "accuracy": 0.0,
            "loss": 0.0,
            "predictions": [],
            "true_labels": []
        }
    
    def get_available_models(self):
        """Get list of available models for the current task type."""
        if self.task_type == "classification":
            return {
                "classification": ["resnet18", "resnet50", "vgg16", "mobilenet"],
                "detection": ["yolo"]
            }
        elif self.task_type == "detection":
            return {
                "detection": ["yolo"]
            }
        else:
            return {
                "classification": ["resnet18", "resnet50", "vgg16", "mobilenet"]
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
def train_image_classification(data_path, model_name="resnet18", framework="pytorch", 
                              epochs=10, img_size=(224, 224), batch_size=32):
    """Convenience function for training image classification models."""
    trainer = ImageModelTrainer(data_path, task_type="classification", framework=framework,
                               img_size=img_size, batch_size=batch_size)
    return trainer.train_classification_model(model_name, epochs)

def train_yolo_detection(data_yaml_path, model_size="n", epochs=100, img_size=640):
    """Convenience function for training YOLO models."""
    trainer = ImageModelTrainer(data_yaml_path, task_type="detection", framework="pytorch")
    return trainer.train_yolo_model(data_yaml_path, model_size, epochs, img_size)
