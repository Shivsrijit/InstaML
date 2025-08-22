# app/pages/4_⚙️_Train_Model.py
import sys
import os
import pandas as pd # Import pandas as it's used for df.select_dtypes
import numpy as np # Import numpy as it's used for np.number

# Adjust the path to import from the parent directory if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from core.unified_trainer import train_model, get_data_info, get_available_models
from core.model_registry import save_model
from app.utilss.navigation import safe_switch_page

# Page configuration
st.set_page_config(page_title="Train Model", layout="wide")

# Initialize session state variables if they don't exist
# This is crucial to prevent AttributeError when accessing these variables before they are set
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_preprocessed' not in st.session_state:
    # This flag should be set by the preprocessing page (pages/2_🔧_Data_Preprocessing.py)
    # to indicate if data has been preprocessed. Default to False.
    st.session_state.df_preprocessed = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'model_saved' not in st.session_state:
    st.session_state.model_saved = False # Boolean flag to track if model was saved

# Custom CSS for better styling
st.markdown("""
<style>
    .help-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Targeting Streamlit's native st.metric component */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important; /* !important to override Streamlit defaults */
        font-weight: bold !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666 !important;
        font-size: 0.9rem !important;
    }
    
    .config-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .config-card h4 {
        color: #333;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    .model-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .training-progress {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("⚙️ Train Your Machine Learning Model")
st.markdown("Transform your clean data into a powerful predictive model")

# Collapsible help section
with st.expander("ℹ️ **What is Model Training and Why is it Important?**", expanded=False):
    st.markdown("""
    **Model Training** is like teaching a student to solve problems by showing them many examples. Your computer learns patterns from your data to make predictions on new, unseen data.
    
    **🎯 What happens during training:**
    - **Learning**: The algorithm finds patterns in your data
    - **Optimization**: It adjusts its parameters to minimize errors
    - **Validation**: It tests its performance on unseen data
    - **Generalization**: It learns to work well on new data
    
    **⚡ Why proper training matters:**
    - **Better predictions**: Well-trained models are more accurate
    - **Avoiding overfitting**: Models that generalize well to new data
    - **Business value**: Accurate predictions lead to better decisions
    """)

# Check if data is loaded
if 'df' not in st.session_state and 'data_type' not in st.session_state:
    st.error("❌ **No data loaded!** Please go to the Data Upload page first.")
    st.button("📂 Go to Data Upload", on_click=lambda: safe_switch_page("pages/1_📂_Data_Upload.py"))
    st.stop()

# Get data and data type
df = st.session_state.get('df', None)
data_type = st.session_state.get('data_type', 'auto')

# Data type information
st.header("🔍 **Data Type Detection**")
if data_type == 'tabular' and df is not None:
    st.success(f"📊 **Data Type:** Tabular Data (CSV, Excel, etc.)")
    st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
elif data_type == 'image':
    st.success(f"🖼️ **Data Type:** Image Data")
    st.info("**Format:** Image directory with train/val structure")
elif data_type == 'audio':
    st.success(f"🎵 **Data Type:** Audio Data")
    st.info("**Format:** Audio directory with train/val structure")
elif data_type == 'multi_dimensional':
    st.success(f"🔢 **Data Type:** Multi-dimensional Data")
    if hasattr(st.session_state, 'array_data'):
        st.info(f"**Shape:** {st.session_state.array_data.shape}")
    else:
        st.info("**Format:** Multi-dimensional array or time series")
else:
    st.warning("⚠️ **Data Type:** Auto-detecting...")
    if df is not None:
        if hasattr(df, 'shape') and len(df.shape) <= 2:
            st.session_state.data_type = 'tabular'
            data_type = 'tabular'
            st.success("📊 **Data Type Detected:** Tabular Data")
        else:
            st.session_state.data_type = 'multi_dimensional'
            data_type = 'multi_dimensional'
            st.success("🔢 **Data Type Detected:** Multi-dimensional Data")
    else:
        st.error("❌ **No data available for training!**")
        st.stop()

# Only show tabular data interface for tabular data
if data_type != 'tabular':
    st.info(f"""
    🎯 **{data_type.title()} Training Mode**
    
    For {data_type} data, the system will automatically:
    - Use appropriate model architectures
    - Apply suitable preprocessing
    - Handle data loading and batching
    """)
    
    # For non-tabular data, show simplified interface
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner(f"Training {data_type} model..."):
            try:
                if data_type == 'image':
                    model, metrics = train_model(
                        st.session_state.image_directory,
                        data_type="image",
                        model_name="resnet18"
                    )
                elif data_type == 'audio':
                    model, metrics = train_model(
                        st.session_state.audio_directory,
                        data_type="audio",
                        model_name="cnn"
                    )
                elif data_type == 'multi_dimensional':
                    data = st.session_state.get('array_data', df)
                    model, metrics = train_model(
                        data,
                        data_type="multi_dimensional",
                        model_name="MLP"
                    )
                
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_trained = True
                st.success("🎉 **Training completed successfully!**")
                
                # Show results
                st.header("📊 **Training Results**")
                st.json(metrics)
                
            except Exception as e:
                st.error(f"❌ **Training failed:** {str(e)}")
                st.exception(e)
    
    st.stop()

# Tabular data interface continues below...

# Check preprocessing status with enhanced guidance
st.header("🔍 **Data Readiness Check**")

# Fix the DataFrame truth value error by properly checking the boolean flag
df_preprocessed = st.session_state.df_preprocessed
if isinstance(df_preprocessed, bool) and df_preprocessed:
    st.success("""
    ✅ **Excellent! Your data has been preprocessed and is ready for training!**
    
    **What this means:**
    - Your data is clean and well-structured
    - Missing values have been handled
    - Data types are properly formatted
    - You're ready for optimal model performance
    """)
else:
    st.warning("""
    ⚠️ **Data Preprocessing Recommended**
    
    **Why preprocessing first is better:**
    - Clean data leads to better model performance
    - Proper formatting prevents training errors
    - Better data quality = better predictions
    - You'll save time and get better results
    
    **🔄 Quick Navigation:**
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Go to Preprocessing", type="primary"):
            safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
    with col2:
        if st.button("⚙️ Continue with Training", type="secondary"):
            st.info("ℹ️ You can continue, but results may be affected by data quality issues.")

# Enhanced Data Summary using st.metric
st.header("📊 **Training Dataset Overview**")

# Detect data type and show appropriate information
data_info = None
available_models = None

try:
    data_info = get_data_info(df)
    available_models = get_available_models(df)
    
    st.info(f"""
    **Data Type Detected:** {data_info['data_type'].title()}
    
    **Understanding your dataset helps you choose the right model and parameters:**
    - **Rows**: More data generally means better model performance
    - **Features**: More features can capture complex patterns but may cause overfitting
    - **Memory**: Affects training speed and resource requirements
    """)
except Exception as e:
    st.info("""
    **Understanding your dataset helps you choose the right model and parameters:**
    - **Rows**: More data generally means better model performance
    - **Features**: More features can capture complex patterns but may cause overfitting
    - **Memory**: Affects training speed and resource requirements
    """)
    st.warning(f"⚠️ Data type detection failed: {str(e)}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows (samples)", f"{df.shape[0]:,}")
    
with col2:
    st.metric("Features", f"{df.shape[1]:,}")
    
with col3:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.metric("Memory (MB)", f"{memory_mb:.2f}")

with col4:
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    st.metric("Numeric/Categorical", f"{numeric_cols}/{categorical_cols}")

# Model Configuration Section
st.header("⚙️ **Model Configuration**")
st.info("""
**Choose the right settings for your model:**
- **Target Column**: The variable you want to predict
- **Model Type**: Different algorithms work better for different problems
- **Scaling**: Normalizing data can improve performance
- **Test Size**: How much data to reserve for testing
""")

# Target column selection with enhanced guidance
st.subheader("🎯 **Select Your Target Variable**")

# Provide a default value for selectbox to prevent errors if df is empty or columns change
# Use st.session_state.target_col for persistence if already set
default_target_index = 0
if st.session_state.target_col and st.session_state.target_col in df.columns:
    default_target_index = df.columns.get_loc(st.session_state.target_col)

target_col = st.selectbox(
    "Choose the column you want to predict:",
    df.columns.tolist(), # Convert to list for selectbox
    index=default_target_index,
    help="This is the variable your model will learn to predict"
)

if target_col:
    # Enhanced target column analysis
    st.markdown(f"""
    <div class="model-info-card">
        <h3>🎯 Target: {target_col}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df[target_col].dtype in ['object', 'category']:
            unique_classes = df[target_col].nunique()
            st.markdown(f"""
            <div class="config-card">
                <h4>📊 Classification Task</h4>
                <p style="color: black;"><strong>Number of classes:</strong> {unique_classes}</p>
                <p style="color: black;"><strong>Task type:</strong> {'Binary' if unique_classes == 2 else 'Multi-class'} Classification</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Class distribution
            st.subheader("📊 Class Distribution")
            class_counts = df[target_col].value_counts()
            st.bar_chart(class_counts)
            
        else:
            target_range = f"{df[target_col].min():.2f} to {df[target_col].max():.2f}"
            st.markdown(f"""
            <div class="config-card">
                <h4>📊 Regression Task</h4>
                <p style="color: black;"><strong>Target range:</strong> {target_range}</p>
                <p style="color: black;"><strong>Mean value:</strong> {df[target_col].mean():.2f}</p>
                <p style="color: black;"><strong>Standard deviation:</strong> {df[target_col].std():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Data quality check for target
        missing_target = df[target_col].isnull().sum()
        if missing_target > 0:
            st.warning(f"""
            ⚠️ **Target Column Issue**
            
            **Missing values:** {missing_target} ({missing_target/len(df)*100:.1f}%)
            
            **Recommendation:** Handle missing values in preprocessing
            """)
        else:
            st.success(f"""
            ✅ **Target Column Ready**
            
            **No missing values** in target column
            **Ready for training**
            """)

# Model configuration options
st.subheader("🤖 **Choose Your Model Settings**")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="config-card">
        <h4>🔧 Model Algorithm</h4>
    </div>
    """, unsafe_allow_html=True) # Changed h4 to be more specific
    
    # Get available models based on data type
    try:
        if data_info and available_models and data_info.get("data_type") == "tabular":
            if df[target_col].dtype in ['object', 'category']:
                model_options = available_models.get("classification", ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"])
            else:
                model_options = available_models.get("regression", ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"])
        else:
            # Fallback to basic models based on target column type
            if df[target_col].dtype in ['object', 'category']:
                model_options = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"]
            else:
                model_options = ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"]
    except Exception as e:
        # Fallback to basic models
        if df[target_col].dtype in ['object', 'category']:
            model_options = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"]
        else:
            model_options = ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"]
        st.warning(f"⚠️ Model detection failed, using fallback options: {str(e)}")
    
    model_type = st.selectbox(
        "Model Algorithm:",
        model_options,
        index=0,  # Always start with first option to avoid index errors
        help="Random Forest and XGBoost are good for most problems. Linear models are faster but may miss complex patterns."
    )
    
    scaling = st.selectbox(
        "Data Scaling:",
        ["standard", "minmax", "robust", "none"],
        help="Standard scaling works well for most cases. MinMax for bounded data. Robust for data with outliers."
    )

with col2:
    st.markdown("""
    <div class="config-card">
        <h4>📊 Training Parameters</h4>
    </div>
    """, unsafe_allow_html=True) # Changed h4 to be more specific
    
    test_size = st.slider(
        "Test Set Size (%):",
        min_value=10,
        max_value=50,
        value=20,
        help="20% is a good default. More test data = more reliable evaluation but less training data."
    )
    
    random_state = st.number_input(
        "Random Seed:",
        min_value=1,
        max_value=1000,
        value=42,
        help="Fixed seed ensures reproducible results. Change for different random splits."
    )

# Training section
st.header("🚀 **Start Training**")
st.info("""
**Ready to train your model? Here's what will happen:**
1. Data will be split into training and test sets
2. Features will be scaled according to your selection
3. The model will learn patterns from your training data
4. Performance will be evaluated on the test set
5. Results will be displayed with detailed metrics
""")

# Training button with enhanced feedback
if st.button("🚀 Start Training", type="primary", use_container_width=True):
    if not target_col:
        st.error("❌ Please select a target column first!")
    else:
        # Training progress section
        st.markdown("""
        <div class="training-progress">
            <h4 style="color: black;">🔄 Training in Progress...</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Training your model... This may take a few minutes."):
            try:
                # Prepare progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training progress
                for i in range(5):
                    progress_bar.progress((i + 1) * 20)
                    if i == 0:
                        status_text.text("🔄 Loading and preprocessing data...")
                    elif i == 1:
                        status_text.text("🔧 Splitting data into train/test sets...")
                    elif i == 2:
                        status_text.text("⚙️ Training model...")
                    elif i == 3:
                        status_text.text("📊 Evaluating performance...")
                    elif i == 4:
                        status_text.text("✅ Finalizing results...")
                
                # Actual training using unified trainer
                # Determine the correct data type
                training_data_type = st.session_state.get('data_type', 'auto')
                if training_data_type == 'auto':
                    # Auto-detect based on data structure
                    if hasattr(df, 'shape') and len(df.shape) <= 2:
                        training_data_type = 'tabular'
                    else:
                        training_data_type = 'multi_dimensional'
                
                # Handle different data types
                training_result = None
                
                if training_data_type == 'tabular':
                    # Tabular data training
                    training_result = train_model(
                        df, 
                        target_col=target_col, 
                        data_type="tabular", 
                        model_name=model_type,
                        test_size=test_size/100, 
                        random_state=random_state,
                        scaling=scaling
                    )
                elif training_data_type == 'image':
                    # Image data training
                    if hasattr(st.session_state, 'image_directory'):
                        training_result = train_model(
                            st.session_state.image_directory,
                            data_type="image",
                            model_name="resnet18",  # Default image model
                            task_type="classification"
                        )
                    else:
                        st.error("❌ **Image directory not found!** Please upload or specify an image directory first.")
                        # Clear progress indicators before continuing
                        progress_bar.empty()
                        status_text.empty()
                elif training_data_type == 'audio':
                    # Audio data training
                    if hasattr(st.session_state, 'audio_directory'):
                        training_result = train_model(
                            st.session_state.audio_directory,
                            data_type="audio",
                            model_name="cnn",  # Default audio model
                            task_type="classification"
                        )
                    else:
                        st.error("❌ **Audio directory not found!** Please upload or specify an audio directory first.")
                        # Clear progress indicators before continuing
                        progress_bar.empty()
                        status_text.empty()
                elif training_data_type == 'multi_dimensional':
                    # Multi-dimensional data training
                    if hasattr(st.session_state, 'array_data'):
                        data = st.session_state.array_data
                    else:
                        data = df
                    
                    training_result = train_model(
                        data,
                        target_col=target_col if target_col else None,
                        data_type="multi_dimensional",
                        model_name="MLP",  # Default multi-dimensional model
                        framework="pytorch"
                    )
                else:
                    st.error(f"❌ **Unsupported data type: {training_data_type}**")
                    # Clear progress indicators before continuing
                    progress_bar.empty()
                    status_text.empty()
                
                # Handle different return formats from train_model
                if training_result is not None:
                    if isinstance(training_result, tuple):
                        if len(training_result) == 2:
                            model, metrics = training_result
                        elif len(training_result) == 3:
                            model, metrics, additional_info = training_result
                        elif len(training_result) == 4:
                            model, metrics, additional_info, extra = training_result
                        else:
                            # Handle unexpected number of return values
                            st.error(f"❌ **Unexpected return format from train_model:** {len(training_result)} values returned")
                            model = training_result[0] if len(training_result) > 0 else None
                            metrics = training_result[1] if len(training_result) > 1 else {}
                    else:
                        # Single return value (probably just the model)
                        model = training_result
                        metrics = {"status": "training_completed"}
                else:
                    model = None
                    metrics = None
                
                # Only proceed with success display if we have valid results
                if model is not None and metrics is not None:
                    # Store results in session state
                    st.session_state.model = model
                    st.session_state.metrics = metrics
                    st.session_state.model_type = model_type
                    st.session_state.target_col = target_col
                    # It's good practice to set a flag that training is complete
                    st.session_state.model_trained = True 
                    
                    # Success message
                    st.success("🎉 **Model Training Completed Successfully!**")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.header("📊 **Training Results**")
                    
                    # Model performance metrics
                    if isinstance(metrics, dict):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="config-card">
                                <h4>📈 Model Performance Metrics</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display key metrics using st.metric
                            if "accuracy" in metrics:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            if "precision" in metrics:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                            if "recall" in metrics:
                                st.metric("Recall", f"{metrics['recall']:.4f}")
                            if "f1_score" in metrics:
                                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                            if "r2_score" in metrics:
                                st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                            if "mean_squared_error" in metrics:
                                st.metric("MSE", f"{metrics['mean_squared_error']:.4f}")
                            
                        with col2:
                            st.markdown("""
                            <div class="config-card">
                                <h4>🔍 Detailed Results</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show all metrics in expandable section
                            with st.expander("📋 View All Metrics", expanded=True):
                                st.json(metrics)
                    else:
                        st.write("**Training Results:**", metrics)
                    
                    # Model saving section
                    st.header("💾 **Save Your Model**")
                    st.info("""
                    **Save your trained model to use it later for:**
                    - Making predictions on new data
                    - Deploying to production
                    - Sharing with team members
                    - Comparing with other models
                    """)
                    
                    if st.button("💾 Save Model", type="secondary", use_container_width=True):
                        try:
                            # IMPORTANT: Ensure save_model accepts 'scaling' and 'target_col'
                            # for complete metadata saving.
                            model_path, meta_path = save_model(
                                model, metrics, model_type, scaling, target_col
                            )
                            st.success(f"✅ **Model saved successfully!**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**Model Path:** {model_path}")
                            with col2:
                                st.info(f"**Metadata Path:** {meta_path}")
                                
                            st.session_state.model_saved = True
                            
                        except Exception as e:
                            st.error(f"❌ **Failed to save model:** {str(e)}")
                
            except Exception as e:
                st.error(f"""
                ❌ **Training Failed**
                
                **Error:** {str(e)}
                
                **🔧 Common solutions:**
                - Check if your data is properly preprocessed
                - Ensure the target column has no missing values
                - Try a different model type or scaling method
                - Verify your data types are appropriate for the selected model
                """)
                
                st.info("💡 **Need help?** Go back to the preprocessing page to clean your data.")

# Navigation section - Only show if a model has been successfully trained
if st.session_state.model is not None:
    st.success("🎉 **Congratulations! You now have a trained model ready for testing and deployment!**")
    
    st.header("🚀 **What's Next?**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🧪 Test Your Model**")
        st.write("Evaluate your model on new data to ensure it generalizes well")
        if st.button("🧪 Go to Testing", type="primary", use_container_width=True):
            safe_switch_page("pages/5_🧪_Test_Model.py")
    
    with col2:
        st.info("**🚀 Deploy Your Model**")
        st.write("Make predictions on new data using your trained model")
        if st.button("🚀 Go to Deployment", type="secondary", use_container_width=True):
            safe_switch_page("pages/6_🚀_Deploy_Model.py")

else:
    # Getting started guide (only shown if no model is trained yet)
    st.info("""
    📋 **Getting Started Guide**
    
    **Step 1:** Ensure your data is loaded and preferably preprocessed
    **Step 2:** Select the column you want to predict (target variable)
    **Step 3:** Choose your model type and configuration settings
    **Step 4:** Click "Start Training" to begin the process
    **Step 5:** Review results and save your model
    **Step 6:** Move to testing or deployment
    """)