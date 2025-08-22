# ML Training Modules Guide

This guide explains how to use the comprehensive ML training modules in InstaML for different types of data.

## 🚀 Quick Start

The training modules automatically detect your data type and provide appropriate models. Here's the simplest way to train a model:

```python
from core.unified_trainer import train_model

# Train a model (auto-detects data type and selects best model)
model, metrics = train_model(
    data_source="your_data.csv",  # or DataFrame, numpy array, directory
    target_col="target_column",    # for supervised learning
    data_type="auto"              # or specify: "tabular", "image", "audio", "multi_dimensional"
)
```

## 📊 Supported Data Types

### 1. Tabular Data (CSV, DataFrame, NumPy arrays)
- **Classification**: Random Forest, XGBoost, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes
- **Regression**: Random Forest, XGBoost, Linear Regression, Ridge, Lasso, SVR, KNN, Decision Tree
- **Features**: Automatic preprocessing, feature scaling, hyperparameter tuning

```python
from core.ML_models.tabular_data import TabularModelTrainer

# Initialize trainer
trainer = TabularModelTrainer(df, target_col='target', task_type='auto')

# Train with hyperparameter tuning
model, metrics, best_params = trainer.train_model("Random Forest", use_hyperparameter_tuning=True)

# Get feature importance
importance_df = trainer.get_feature_importance()
```

### 2. Image Data (Directories with train/val folders)
- **Classification**: ResNet18, ResNet50, VGG16, MobileNet
- **Detection**: YOLOv8
- **Features**: Data augmentation, transfer learning, automatic class detection

```python
from core.ML_models.image_data import ImageModelTrainer

# Initialize trainer
trainer = ImageModelTrainer("path/to/image/directory", task_type="classification")

# Train classification model
model, history = trainer.train_classification_model("resnet18", epochs=10)

# Train YOLO detection model
model, results = trainer.train_yolo_model("data.yaml", model_size="n", epochs=100)
```

### 3. Audio Data (Directories with train/val folders)
- **Classification**: CNN, LSTM
- **Features**: Mel spectrograms, MFCC, automatic preprocessing

```python
from core.ML_models.audio_data import AudioModelTrainer

# Initialize trainer
trainer = AudioModelTrainer("path/to/audio/directory", task_type="classification")

# Train model
model, history = trainer.train_classification_model("cnn", epochs=20)

# Extract features
mel_spec = trainer.extract_mel_spectrogram_features("audio_file.wav")
```

### 4. Multi-Dimensional Data (3D+ arrays, time series)
- **Classification/Regression**: MLP, CNN, LSTM, Transformer
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Features**: Automatic reshaping, normalization, advanced architectures

```python
from core.ML_models.multi_dimensional_data import MultiDimensionalTrainer

# Initialize trainer
trainer = MultiDimensionalTrainer(data_3d, task_type="classification")

# Train deep learning model
model, metrics = trainer.train_model("Transformer", epochs=100, learning_rate=0.001)

# Train clustering model
model, metrics = trainer.train_model("KMeans", n_clusters=5)
```

## 🔧 Advanced Usage

### Custom Model Parameters

```python
# Tabular data with custom parameters
model, metrics = train_model(
    df, 
    target_col="target",
    model_name="XGBoost",
    n_estimators=500,
    max_depth=10,
    learning_rate=0.1
)

# Image data with custom parameters
model, history = train_model(
    "image_directory",
    data_type="image",
    model_name="resnet50",
    epochs=50,
    batch_size=64,
    learning_rate=0.0001
)
```

### Cross-Validation

```python
from core.ML_models.tabular_data import TabularModelTrainer

trainer = TabularModelTrainer(df, target_col="target")
cv_results = trainer.cross_validate("Random Forest", cv=5)

print(f"CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
```

### Model Persistence

```python
# Save model
trainer.save_model("model.pkl")

# Load model (for some frameworks)
trainer.load_model("model.pkl")
```

## 📁 Directory Structures

### Image Data
```
image_directory/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_1/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class_0/
    └── class_1/
```

### Audio Data
```
audio_directory/
├── train/
│   ├── class_0/
│   │   ├── audio1.wav
│   │   └── audio2.mp3
│   └── class_1/
│       ├── audio3.flac
│       └── audio4.m4a
└── val/
    ├── class_0/
    └── class_1/
```

### YOLO Detection Data
```
yolo_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## 🎯 Model Selection Guide

### For Tabular Data
- **Small datasets (< 1000 samples)**: Logistic Regression, SVM
- **Medium datasets (1000-10000 samples)**: Random Forest, XGBoost
- **Large datasets (> 10000 samples)**: XGBoost, Deep Learning
- **High-dimensional data**: Lasso, Ridge, Feature Selection + Random Forest

### For Image Data
- **Small datasets**: ResNet18, MobileNet (with transfer learning)
- **Medium datasets**: ResNet50, VGG16
- **Large datasets**: ResNet101, EfficientNet
- **Real-time applications**: MobileNet, EfficientNet

### For Audio Data
- **Short audio clips**: CNN
- **Long audio sequences**: LSTM, Transformer
- **Real-time processing**: CNN with small receptive field

### For Multi-Dimensional Data
- **Time series**: LSTM, Transformer
- **Spatial data**: CNN
- **Mixed data types**: MLP with custom architecture

## 🚨 Common Issues and Solutions

### Memory Issues
```python
# Reduce batch size
trainer = ImageModelTrainer("data_path", batch_size=16)

# Use smaller models
model, history = trainer.train_classification_model("resnet18")  # instead of resnet50
```

### Overfitting
```python
# Increase regularization
model, metrics = trainer.train_model("Random Forest", max_depth=5, min_samples_split=10)

# Use early stopping
model, history = trainer.train_classification_model("resnet18", epochs=100, patience=10)
```

### Data Loading Issues
```python
# Check data structure
trainer = UnifiedModelTrainer("data_path")
info = trainer.get_data_info()
print(f"Data type: {info['data_type']}")
print(f"Available models: {trainer.get_available_models()}")
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
cd InstaML
python test_training_modules.py
```

## 📚 Examples

### Complete Tabular Training Example
```python
import pandas as pd
from core.unified_trainer import train_model

# Load data
df = pd.read_csv("data.csv")

# Train model
model, metrics = train_model(
    df,
    target_col="target",
    data_type="tabular",
    model_name="Random Forest",
    test_size=0.2,
    random_state=42
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Complete Image Training Example
```python
from core.unified_trainer import train_model

# Train image classification
model, history = train_model(
    "image_directory",
    data_type="image",
    model_name="resnet18",
    epochs=20,
    batch_size=32
)

print(f"Training completed with {len(history['train_losses'])} epochs")
```

## 🔗 Integration with Streamlit

The training modules are fully integrated with the Streamlit interface. Users can:

1. Upload data through the Data Upload page
2. Preprocess data in the Data Preprocessing page
3. Explore data in the EDA page
4. Train models in the Train Model page
5. Test models in the Test Model page
6. Deploy models in the Deploy Model page

The system automatically detects data types and suggests appropriate models and parameters.

## 📈 Performance Tips

1. **Use GPU acceleration** when available (PyTorch/TensorFlow automatically detect)
2. **Start with simple models** and gradually increase complexity
3. **Use cross-validation** for reliable performance estimates
4. **Monitor training progress** with the provided history objects
5. **Save intermediate checkpoints** for long training runs
6. **Use data augmentation** for image and audio data to prevent overfitting

## 🤝 Contributing

To add new models or data types:

1. Create a new trainer class in the appropriate module
2. Implement the required methods (`train_model`, `save_model`, etc.)
3. Add the new trainer to the `UnifiedModelTrainer` class
4. Update the test suite
5. Document the new functionality

## 📞 Support

For issues or questions:
1. Check the test suite output
2. Review the error messages for specific issues
3. Verify your data format matches the expected structure
4. Check that all dependencies are installed correctly
