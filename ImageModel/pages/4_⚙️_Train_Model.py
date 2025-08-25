# Enhanced Streamlit Integration for Machine Learning Training and Object Detection
# This file handles both tabular ML training and computer vision tasks

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import tempfile
import json

def ensure_image_format(image):
    """Ensure image is in the correct format for OpenCV processing"""
    try:
        # Handle different image formats and ensure proper conversion
        if image is None:
            return None
            
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            image = np.array(image)
        
        # Handle different data types
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle different channel configurations
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB image - convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:
                # RGBA image - convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 1:
                # Single channel image - convert to 3-channel
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2:
            # Grayscale image - convert to 3-channel BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    except Exception as e:
        st.error(f"Error in image format conversion: {str(e)}")
        return None

def opencv_object_detection_interface():
    """OpenCV-based Object Detection Interface for Streamlit"""

    st.header("🎯 OpenCV Object Detection")

    # Initialize session state for object detection
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Detection Configuration")

        # Detection method selection
        detection_method = st.selectbox(
            "Select Detection Method",
            ["Face Detection (Haar Cascade)", "Object Detection (HOG)", "Edge Detection", "Contour Detection"],
            help="Choose the OpenCV-based detection method",
            key="detection_method_select"
        )

        # Detection parameters
        st.subheader("Detection Parameters")
        
        if detection_method == "Face Detection (Haar Cascade)":
            scale_factor = st.slider("Scale Factor", min_value=1.01, max_value=1.5, value=1.1, step=0.01, key="face_scale_factor")
            min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, key="face_min_neighbors")
            min_size = st.slider("Min Face Size", min_value=20, max_value=100, value=30, key="face_min_size")
            
        elif detection_method == "Object Detection (HOG)":
            win_stride = st.slider("Win Stride", min_value=4, max_value=16, value=8, step=4, key="hog_win_stride")
            padding = st.slider("Padding", min_value=4, max_value=16, value=8, step=4, key="hog_padding")
            
        elif detection_method == "Edge Detection":
            low_threshold = st.slider("Low Threshold", min_value=50, max_value=200, value=100, key="edge_low_threshold")
            high_threshold = st.slider("High Threshold", min_value=100, max_value=300, value=200, key="edge_high_threshold")
            
        elif detection_method == "Contour Detection":
            blur_kernel = st.slider("Blur Kernel Size", min_value=3, max_value=15, value=5, step=2, key="contour_blur_kernel")
            canny_low = st.slider("Canny Low", min_value=50, max_value=200, value=100, key="contour_canny_low")
            canny_high = st.slider("Canny High", min_value=100, max_value=300, value=200, key="contour_canny_high")

    with col2:
        st.subheader("Image Selection and Detection")
        
        # Check if image data exists in session state
        if 'image_data' in st.session_state and st.session_state.image_data is not None and len(st.session_state.image_data) > 0:
            st.success("✅ Image data found from previous workflow!")
            
            # Select image from existing data
            image_data = st.session_state.image_data
            if len(image_data) > 1:
                image_idx = st.selectbox("Select image for detection:", 
                                       range(len(image_data)), 
                                       format_func=lambda x: image_data[x]['name'],
                                       key="image_selection_select")
            else:
                image_idx = 0
                st.info("Using the uploaded image")
            
            # Display selected image
            selected_image = image_data[image_idx]['data']
            st.image(selected_image, caption=f"Selected: {image_data[image_idx]['name']}", use_column_width=True)
            
        else:
            # Add direct image upload option
            st.warning("⚠️ No image data found. Upload an image below:")
            uploaded_file = st.file_uploader("Upload an image for detection", type=["png", "jpg", "jpeg"], key="detection_image_uploader")
            
            if uploaded_file is not None:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Store in session state for consistency
                st.session_state.image_data = [{
                    'name': uploaded_file.name,
                    'data': img_array
                }]
                
                # Display the uploaded image
                st.image(img_array, caption="Uploaded Image", use_column_width=True)
                st.experimental_rerun()
            else:
                # Show sample image option
                st.info("Or try with a sample image:")
                if st.button("🖼️ Use Sample Image"):
                    # Load sample image
                    sample_img = cv2.imread('sample.jpg')
                    if sample_img is None:
                        # Create a simple sample image if file doesn't exist
                        sample_img = np.zeros((300, 400, 3), dtype=np.uint8)
                        cv2.rectangle(sample_img, (100, 100), (300, 200), (0, 255, 0), -1)
                        cv2.putText(sample_img, "Sample Image", (120, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    else:
                        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
                    
                    st.session_state.image_data = [{
                        'name': 'sample.jpg',
                        'data': sample_img
                    }]
                    st.experimental_rerun()
                
                return

        # Run detection button
        if st.button("🔍 Run Object Detection", type="primary"):
            try:
                with st.spinner("Running object detection..."):
                    # Get the current image data
                    image_data = st.session_state.image_data
                    image_idx = 0  # Default to first image
                    if len(image_data) > 1:
                        image_idx = st.session_state.get('image_idx', 0)
                    
                    # Get selected image
                    selected_image = image_data[image_idx]['data']
                    
                    # Validate image data
                    if selected_image is None:
                        st.error("No image data available")
                        return
                    
                    # Convert image to proper format for OpenCV using helper function
                    cv_image = ensure_image_format(selected_image)
                    if cv_image is None:
                        st.error("Failed to convert image to proper format")
                        return
                    
                    # Run detection based on selected method
                    params = {
                        'scale_factor': scale_factor if detection_method == "Face Detection (Haar Cascade)" else None,
                        'min_neighbors': min_neighbors if detection_method == "Face Detection (Haar Cascade)" else None,
                        'min_size': min_size if detection_method == "Face Detection (Haar Cascade)" else None,
                        'win_stride': win_stride if detection_method == "Object Detection (HOG)" else None,
                        'padding': padding if detection_method == "Object Detection (HOG)" else None,
                        'low_threshold': low_threshold if detection_method == "Edge Detection" else None,
                        'high_threshold': high_threshold if detection_method == "Edge Detection" else None,
                        'blur_kernel': blur_kernel if detection_method == "Contour Detection" else None,
                        'canny_low': canny_low if detection_method == "Contour Detection" else None,
                        'canny_high': canny_high if detection_method == "Contour Detection" else None
                    }
                    
                    # Add additional validation
                    st.info(f"Processing image with shape: {cv_image.shape}, dtype: {cv_image.dtype}")
                    
                    try:
                        results = run_opencv_detection(cv_image, detection_method, params)
                    except Exception as e:
                        st.error(f"Detection failed with error: {str(e)}")
                        st.exception(e)
                        return
                    
                    if results is not None and len(results) > 0:
                        st.session_state.detection_results = results
                        st.session_state.detection_method = detection_method
                        
                        # Display results
                        st.subheader("🎯 Detection Results")
                        st.success(f"✅ Detection completed! Found {len(results)} objects/features")
                        
                        # Show result image
                        result_image = draw_detection_results(cv_image, results, detection_method)
                        # Convert back to RGB for display
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image_rgb, caption="Detection Results", use_column_width=True)
                    else:
                        st.warning("No objects detected with current parameters")
                        
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
                st.exception(e)

def display_detection_results():
    """Display detailed detection results"""
    if st.session_state.detection_results is None or st.session_state.detection_method is None:
        return
        
    st.subheader("📊 Detection Details")
    
    # Display detection method used
    st.info(f"**Detection Method:** {st.session_state.detection_method}")
    
    results = st.session_state.detection_results
    
    # Handle different result types
    if st.session_state.detection_method == "Face Detection (Haar Cascade)":
        st.success(f"**Total Faces Detected:** {len(results)}")
        st.write("**Face Detection Results:**")
        for i, (x, y, w, h) in enumerate(results):
            st.write(f"Face {i+1}: Position ({x}, {y}), Size {w}×{h}")
            
    elif st.session_state.detection_method == "Object Detection (HOG)":
        st.success(f"**Total Objects Detected:** {len(results)}")
        st.write("**HOG Detection Results:**")
        for i, (x, y, w, h) in enumerate(results):
            st.write(f"Object {i+1}: Position ({x}, {y}), Size {w}×{h}")
            
    elif st.session_state.detection_method == "Edge Detection":
        st.success(f"**Total Edge Points:** {len(results)}")
        st.write("**Edge Detection Results:**")
        st.write(f"Detected {len(results)} edge points")
        
    elif st.session_state.detection_method == "Contour Detection":
        st.success(f"**Total Contours:** {len(results)}")
        st.write("**Contour Detection Results:**")
        for i, contour in enumerate(results):
            area = cv2.contourArea(contour)
            st.write(f"Contour {i+1}: Area = {area:.2f}")
    
    # Download results as JSON
    try:
        results_data = {
            "detection_method": st.session_state.detection_method,
            "total_detections": len(results),
            "results": results.tolist() if hasattr(results, 'tolist') else str(results)
        }
        
        st.download_button(
            label="💾 Download Results as JSON",
            data=json.dumps(results_data, indent=2),
            file_name="opencv_detection_results.json",
            mime="application/json",
            key="download_detection_results"
        )
    except Exception as e:
        st.error(f"Error preparing results for download: {str(e)}")

def tabular_ml_training_interface():
    """Tabular Machine Learning Training Interface"""

    st.header("🏋️‍♂️ Tabular ML Model Training")

    # Check if tabular data exists
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No tabular data found.")
        
        # Add direct CSV upload option
        uploaded_file = st.file_uploader("Upload CSV for training", type=["csv"], key="training_csv_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success("✅ Data loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
        
        # Show sample dataset option
        st.info("Or try with a sample dataset:")
        if st.button("📊 Use Sample Dataset (Iris)"):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target
            st.session_state.df = df
            st.success("✅ Sample Iris dataset loaded!")
            st.experimental_rerun()
            
        return

    df = st.session_state.df
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Configuration")
        
        # Show data preview
        st.write("Preview of first 5 rows:")
        st.dataframe(df.head(), key="data_preview_dataframe")
        
        # Target variable selection
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(), key="target_variable_select")
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Features", 
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col][:min(5, len(df.columns)-1)],
            key="feature_selection_multiselect"
        )
        
        # Problem type detection
        if target_col:
            target_dtype = df[target_col].dtype
            if target_dtype in ['object', 'category'] or df[target_col].nunique() < 10:
                problem_type = "Classification"
            else:
                problem_type = "Regression"
            
            st.info(f"**Problem Type:** {problem_type}")

    with col2:
        st.subheader("Model Configuration")
        
        if problem_type == "Classification":
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Logistic Regression"],
                key="classification_model_select"
            )
        else:
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Linear Regression"],
                key="regression_model_select"
            )
        
        # Training parameters
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider")
        random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, key="random_state_input")
        
        if model_type == "Random Forest":
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100, key="n_estimators_input")
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10, key="max_depth_input")

    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not feature_cols:
            st.error("Please select features for training")
        else:
            try:
                with st.spinner("Training model..."):
                    # Prepare data
                    X = df[feature_cols].copy()
                    # Handle categorical features
                    X = pd.get_dummies(X)
                    y = df[target_col].copy()
                    
                    # Handle categorical target for classification
                    if problem_type == "Classification" and y.dtype == 'object':
                        y, y_labels = pd.factorize(y)
                        st.session_state.y_labels = y_labels
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    if model_type == "Random Forest":
                        if problem_type == "Classification":
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                        else:
                            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    elif model_type == "Logistic Regression" and problem_type == "Classification":
                        model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif model_type == "Linear Regression" and problem_type == "Regression":
                        model = LinearRegression()
                    else:
                        st.error("Invalid model type for the problem type")
                        return
                    
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Evaluate model
                    if problem_type == "Classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        st.success(f"✅ Training completed! Accuracy: {accuracy:.3f}")
                        
                        # Classification report
                        st.subheader("📊 Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.json(report)
                        
                        # Confusion matrix
                        from sklearn.metrics import confusion_matrix
                        import seaborn as sns
                        
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig, key="confusion_matrix_plot")
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.success(f"✅ Training completed! MSE: {mse:.3f}, R²: {r2:.3f}")
                        
                        # Plot predictions vs actual
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred)
                        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title('Predicted vs Actual')
                        st.pyplot(fig, key="predictions_vs_actual_plot")
                    
                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = model_type
                    st.session_state.problem_type = problem_type
                    st.session_state.feature_cols = feature_cols
                    st.session_state.target_col = target_col
                    st.session_state.X_columns = X.columns.tolist()
                    
                    # Download model
                    model_bytes = joblib.dump(model, None)
                    st.download_button(
                        label="💾 Download Trained Model",
                        data=model_bytes,
                        file_name=f"{model_type.lower().replace(' ', '_')}_model.pkl",
                        mime="application/octet-stream",
                        key="download_trained_model"
                    )
                    
                    # Show feature importance for tree-based models
                    if "Random Forest" in model_type:
                        st.subheader("🔍 Feature Importance")
                        if problem_type == "Classification":
                            importances = model.feature_importances_
                        else:
                            importances = model.feature_importances_
                            
                        # Create a DataFrame for plotting
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot
                        fig, ax = plt.subplots()
                        ax.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Feature Importances')
                        st.pyplot(fig, key="feature_importance_plot")
                    
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.exception(e)

# OpenCV Detection Functions
def run_opencv_detection(image, method, params):
    """Run OpenCV-based object detection"""
    try:
        # Add debugging information
        if image is not None:
            st.info(f"Image shape: {image.shape}, dtype: {image.dtype}, channels: {image.shape[2] if len(image.shape) > 2 else 1}")
        
        if method == "Face Detection (Haar Cascade)":
            return run_face_detection(image, params)
        elif method == "Object Detection (HOG)":
            return run_hog_detection(image, params)
        elif method == "Edge Detection":
            return run_edge_detection(image, params)
        elif method == "Contour Detection":
            return run_contour_detection(image, params)
        else:
            return None
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
        return None

def run_face_detection(image, params):
    """Run face detection using Haar Cascade"""
    try:
        # Load pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            st.error("Failed to load Haar Cascade classifier")
            return np.array([])
        
        # Ensure image has the correct format and channels
        if len(image.shape) == 3:
            # Image has 3 channels (RGB/BGR)
            if image.shape[2] == 3:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # Image has 4 channels (RGBA), convert to BGR first
                bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                st.error(f"Unexpected number of channels: {image.shape[2]}")
                return np.array([])
        elif len(image.shape) == 2:
            # Image is already grayscale
            gray = image
        else:
            st.error(f"Unexpected image shape: {image.shape}")
            return np.array([])
        
        # Ensure grayscale image is uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Add debugging information
        st.info(f"Grayscale image shape: {gray.shape}, dtype: {gray.dtype}, min: {gray.min()}, max: {gray.max()}")
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=params.get('scale_factor', 1.1),
            minNeighbors=params.get('min_neighbors', 5),
            minSize=(params.get('min_size', 30), params.get('min_size', 30))
        )
        
        return faces
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        return np.array([])

def run_hog_detection(image, params):
    """Run HOG-based pedestrian detection"""
    try:
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people
        boxes, _ = hog.detectMultiScale(
            image,
            winStride=(params.get('win_stride', 8), params.get('win_stride', 8)),
            padding=(params.get('padding', 8), params.get('padding', 8))
        )
        
        return boxes
    except Exception as e:
        st.error(f"HOG detection error: {str(e)}")
        return np.array([])

def run_edge_detection(image, params):
    """Run Canny edge detection"""
    try:
        # Ensure image has the correct format and channels
        if len(image.shape) == 3:
            # Image has 3 channels (RGB/BGR)
            if image.shape[2] == 3:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # Image has 4 channels (RGBA), convert to BGR first
                bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                st.error(f"Unexpected number of channels: {image.shape[2]}")
                return np.array([])
        elif len(image.shape) == 2:
            # Image is already grayscale
            gray = image
        else:
            st.error(f"Unexpected image shape: {image.shape}")
            return np.array([])
        
        # Ensure grayscale image is uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, params.get('low_threshold', 100), params.get('high_threshold', 200))
        
        return edges
    except Exception as e:
        st.error(f"Edge detection error: {str(e)}")
        return np.array([])

def run_contour_detection(image, params):
    """Run contour detection"""
    try:
        # Ensure image has the correct format and channels
        if len(image.shape) == 3:
            # Image has 3 channels (RGB/BGR)
            if image.shape[2] == 3:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                # Image has 4 channels (RGBA), convert to BGR first
                bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                st.error(f"Unexpected number of channels: {image.shape[2]}")
                return []
        elif len(image.shape) == 2:
            # Image is already grayscale
            gray = image
        else:
            st.error(f"Unexpected image shape: {image.shape}")
            return []
        
        # Ensure grayscale image is uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Apply Gaussian blur
        kernel_size = params.get('blur_kernel', 5)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, params.get('canny_low', 100), params.get('canny_high', 200))
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours
    except Exception as e:
        st.error(f"Contour detection error: {str(e)}")
        return []

def draw_detection_results(image, results, method):
    """Draw detection results on image"""
    result_image = image.copy()
    
    if method == "Face Detection (Haar Cascade)":
        for (x, y, w, h) in results:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    elif method == "Object Detection (HOG)":
        for (x, y, w, h) in results:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_image, 'Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
    elif method == "Edge Detection":
        # For edge detection, results is the edge image itself
        return results
        
    elif method == "Contour Detection":
        # Draw contours
        cv2.drawContours(result_image, results, -1, (0, 255, 255), 2)
    
    return result_image

# Main integration function
def add_ml_functionality():
    """Add Machine Learning functionality to existing Streamlit app"""

    st.markdown("---")
    st.title("🤖 Machine Learning & Computer Vision")

    # Tab selection
    tab1, tab2, tab3 = st.tabs(["🎯 Object Detection", "🏋️‍♂️ ML Training", "📊 Results Analysis"])

    with tab1:
        opencv_object_detection_interface()
        display_detection_results()

    with tab2:
        tabular_ml_training_interface()

    with tab3:
        st.header("📊 Model Performance Analysis")
        
        # Show detection results if available
        if 'detection_results' in st.session_state and st.session_state.detection_results is not None and st.session_state.detection_method is not None:
            st.success("✅ Object detection results available!")
            display_detection_results()
        
        # Show ML training results if available
        if 'trained_model' in st.session_state:
            st.success("✅ Trained ML model available!")
            st.info(f"**Model Type:** {st.session_state.model_type}")
            st.info(f"**Problem Type:** {st.session_state.problem_type}")
            st.info(f"**Features:** {len(st.session_state.feature_cols)} selected")
            st.info(f"**Target Variable:** {st.session_state.target_col}")
            
            # Show feature list
            with st.expander("View Selected Features", key="features_expander"):
                st.write(st.session_state.feature_cols)
            
            # Model download
            model_bytes = joblib.dump(st.session_state.trained_model, None)
            st.download_button(
                label="💾 Download Trained Model",
                data=model_bytes,
                file_name=f"{st.session_state.model_type.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                key="download_saved_model"
            )
            
            # Add prediction interface
            st.subheader("🔮 Make Predictions")
            st.write("Enter values for prediction:")
            
            input_data = {}
            for feature in st.session_state.feature_cols:
                # Determine input type based on data type
                if st.session_state.df[feature].dtype in ['int64', 'float64']:
                    min_val = float(st.session_state.df[feature].min())
                    max_val = float(st.session_state.df[feature].max())
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        min_value=min_val,
                        max_value=max_val,
                        value=float(st.session_state.df[feature].mean()),
                        key=f"prediction_{feature}_number"
                    )
                else:
                    unique_vals = st.session_state.df[feature].unique()
                    input_data[feature] = st.selectbox(
                        f"{feature}", 
                        options=unique_vals,
                        key=f"prediction_{feature}_select"
                    )
            
            if st.button("🔮 Predict", key="make_prediction_button"):
                try:
                    # Create input dataframe
                    input_df = pd.DataFrame([input_data])
                    # One-hot encode if needed
                    input_df = pd.get_dummies(input_df)
                    
                    # Ensure columns match training
                    for col in st.session_state.X_columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    # Only keep columns used in training
                    input_df = input_df[st.session_state.X_columns]
                    
                    # Make prediction
                    prediction = st.session_state.trained_model.predict(input_df)
                    
                    # Display result
                    st.success("Prediction Result:")
                    if st.session_state.problem_type == "Classification":
                        if 'y_labels' in st.session_state:
                            result = st.session_state.y_labels[prediction[0]]
                        else:
                            result = prediction[0]
                        st.metric("Predicted Class", result, key="predicted_class_metric")
                    else:
                        st.metric("Predicted Value", f"{prediction[0]:.4f}", key="predicted_value_metric")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.info("No trained models available. Train a model in the ML Training tab.")

# Example CSS styling for better UI
def add_custom_css():
    st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    .detection-results {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .st-bb {
        background-color: transparent;
    }
    
    .st-at {
        background-color: #4CAF50;
    }
    
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function to integrate with existing app
def main():
    """Main function - integrate this with your existing Streamlit app"""
    add_custom_css()
    add_ml_functionality()

if __name__ == "__main__":
    main()