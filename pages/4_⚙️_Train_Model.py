# app/pages/4_⚙️_Train_Model.py
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.utilss.navigation import safe_switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header

import streamlit as st
from core.unified_trainer import train_model, get_data_info, get_available_models
from core.model_registry import save_model
from core.persistent_storage import auto_save_model, persistent_storage

# Set page config
st.set_page_config(layout="wide", page_title="Train Model", page_icon="⚙️")

# === CUSTOM CSS (Same as EDA) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    .nav-icon {
        font-size: 1.5rem; margin-right: 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-value {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #718096; font-size: 0.9rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 12px;
        padding: 0.8rem 2rem; font-weight: 600; transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    .stSuccess, .stInfo, .stWarning { border-radius: 12px; border: none; }
    
    .config-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_preprocessed' not in st.session_state:
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
    st.session_state.model_saved = False

# === Main Container ===
with stylable_container("main_container", css_styles="""
    { background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 2rem;
      backdrop-filter: blur(10px); box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
"""):
    # === Header ===
    with stylable_container("page_header", css_styles="""
        { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
          color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; }
    """):
        st.markdown("""
        <div style="font-size: 3rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            <i class="fas fa-cogs nav-icon"></i>Train Your Machine Learning Model
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">Transform your clean data into a powerful predictive model</div>
        """, unsafe_allow_html=True)

    # === Progress Indicator ===
    if "df" in st.session_state and st.session_state.df is not None:
        with stylable_container("progress_card", css_styles="""
            { background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
              border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }
        """):
            st.markdown("""
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
                <i class="fas fa-chart-line nav-icon"></i>Current Progress
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = "✅" if st.session_state.df is not None else "❌"
                st.markdown(f"**Data Loaded:** {status}")
            with col2:
                # Fixed the DataFrame boolean evaluation issue
                df_preprocessed = st.session_state.get('df_preprocessed', False)
                if isinstance(df_preprocessed, pd.DataFrame):
                    # If it's a DataFrame, check if it's not empty
                    status = "✅" if not df_preprocessed.empty else "❌"
                elif isinstance(df_preprocessed, bool):
                    status = "✅" if df_preprocessed else "❌"
                else:
                    # For other types, convert to boolean safely
                    status = "✅" if bool(df_preprocessed) else "❌"
                st.markdown(f"**Preprocessed:** {status}")
            with col3:
                status = "✅" if st.session_state.get('model_trained') else "❌"
                st.markdown(f"**Model Trained:** {status}")
            with col4:
                target = st.session_state.get('target_col', 'Not selected')
                st.markdown(f"**Target:** {target}")

    # === Help Section ===
    with st.expander("What is Model Training and Why is it Important?"):
        st.markdown("""
        **Model Training** is like teaching a student to solve problems by showing them many examples. 
        Your computer learns patterns from your data to make predictions on new, unseen data.
        
        **What happens during training:**
        - **Learning**: The algorithm finds patterns in your data
        - **Optimization**: It adjusts its parameters to minimize errors
        - **Validation**: It tests its performance on unseen data
        - **Generalization**: It learns to work well on new data
        
        **Why proper training matters:**
        - **Better predictions**: Well-trained models are more accurate
        - **Avoiding overfitting**: Models that generalize well to new data
        - **Business value**: Accurate predictions lead to better decisions
        """)

    # === Data Check ===
    if 'df' not in st.session_state and 'data_type' not in st.session_state:
        st.error("No data loaded! Please go to the Data Upload page first.")
        if st.button("Go to Data Upload", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")
        st.stop()

    # Get data and data type
    df = st.session_state.get('df', None)
    data_type = st.session_state.get('data_type', 'auto')

    # === Data Type Detection ===
    colored_header("Data Type Detection", "Analyzing your data structure", "blue-70")
    
    if data_type == 'tabular' and df is not None:
        st.success("Data Type: Tabular Data (CSV, Excel, etc.)")
        st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    elif data_type == 'image':
        st.success("Data Type: Image Data")
        st.info("Format: Image directory with train/val structure")
    elif data_type == 'audio':
        st.success("Data Type: Audio Data")
        st.info("Format: Audio directory with train/val structure")
    elif data_type == 'multi_dimensional':
        st.success("Data Type: Multi-dimensional Data")
        if hasattr(st.session_state, 'array_data'):
            st.info(f"Shape: {st.session_state.array_data.shape}")
        else:
            st.info("Format: Multi-dimensional array or time series")
    else:
        st.warning("Data Type: Auto-detecting...")
        if df is not None:
            if hasattr(df, 'shape') and len(df.shape) <= 2:
                st.session_state.data_type = 'tabular'
                data_type = 'tabular'
                st.success("Data Type Detected: Tabular Data")
            else:
                st.session_state.data_type = 'multi_dimensional'
                data_type = 'multi_dimensional'
                st.success("Data Type Detected: Multi-dimensional Data")
        else:
            st.error("No data available for training!")
            st.stop()

    # Handle non-tabular data
    if data_type != 'tabular':
        st.info(f"""
        **{data_type.title()} Training Mode**
        
        For {data_type} data, the system will automatically:
        - Use appropriate model architectures
        - Apply suitable preprocessing
        - Handle data loading and batching
        """)
        
        if st.button("Start Training", type="primary", use_container_width=True):
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
                    st.success("Training completed successfully!")
                    
                    colored_header("Training Results", "Model performance metrics", "green-70")
                    st.json(metrics)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
        st.stop()

    # === Data Readiness Check ===
    colored_header("Data Readiness Check", "Ensuring optimal training conditions", "green-70")
    
    # Fixed the DataFrame boolean evaluation issue here as well
    df_preprocessed = st.session_state.get('df_preprocessed', False)
    is_preprocessed = False
    
    if isinstance(df_preprocessed, pd.DataFrame):
        # If it's a DataFrame, check if it's not empty
        is_preprocessed = not df_preprocessed.empty
    elif isinstance(df_preprocessed, bool):
        is_preprocessed = df_preprocessed
    else:
        # For other types, try to convert to boolean safely
        try:
            is_preprocessed = bool(df_preprocessed)
        except:
            is_preprocessed = False
    
    if is_preprocessed:
        st.success("""
        **Excellent! Your data has been preprocessed and is ready for training!**
        
        **What this means:**
        - Your data is clean and well-structured
        - Missing values have been handled
        - Data types are properly formatted
        - You're ready for optimal model performance
        """)
    else:
        st.warning("""
        **Data Preprocessing Recommended**
        
        **Why preprocessing first is better:**
        - Clean data leads to better model performance
        - Proper formatting prevents training errors
        - Better data quality = better predictions
        - You'll save time and get better results
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Preprocessing", type="primary"):
                safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
        with col2:
            if st.button("Continue with Training"):
                st.info("You can continue, but results may be affected by data quality issues.")

    # === Dataset Overview ===
    colored_header("Training Dataset Overview", "Understanding your data characteristics", "violet-70")
    
    # Get data info
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
        st.warning(f"Data type detection failed: {str(e)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container("metric_rows", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[0]:,}</div><div class='metric-label'>Rows (samples)</div>", unsafe_allow_html=True)
    with col2:
        with stylable_container("metric_features", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[1]:,}</div><div class='metric-label'>Features</div>", unsafe_allow_html=True)
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        with stylable_container("metric_memory", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{memory_mb:.2f}</div><div class='metric-label'>Memory (MB)</div>", unsafe_allow_html=True)
    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
        with stylable_container("metric_types", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{numeric_cols}/{categorical_cols}</div><div class='metric-label'>Numeric/Categorical</div>", unsafe_allow_html=True)

    # === Model Configuration ===
    colored_header("Model Configuration", "Choose the right settings for your model", "orange-70")
    
    # Target column selection
    st.subheader("Select Your Target Variable")
    
    default_target_index = 0
    if st.session_state.target_col and st.session_state.target_col in df.columns:
        default_target_index = df.columns.get_loc(st.session_state.target_col)

    target_col = st.selectbox(
        "Choose the column you want to predict:",
        df.columns.tolist(),
        index=default_target_index,
        help="This is the variable your model will learn to predict"
    )

    if target_col:
        col1, col2 = st.columns(2)
        
        with col1:
            # Task detection
            from core.ML_models.tabular_data import TabularModelTrainer
            temp_trainer = TabularModelTrainer(df, target_col, task_type="auto")
            detected_task = temp_trainer.task_type
            
            with stylable_container("task_info", css_styles="""
                { background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                  border-radius: 12px; padding: 1.5rem; }
            """):
                if detected_task == "classification":
                    unique_classes = df[target_col].nunique()
                    st.markdown(f"""
                    **<i class="fas fa-chart-pie"></i> Classification Task (Auto-detected)**
                    
                    - **Number of classes:** {unique_classes}
                    - **Task type:** {'Binary' if unique_classes == 2 else 'Multi-class'} Classification
                    - **Data type:** {df[target_col].dtype}
                    - **Unique values:** {unique_classes}
                    """)
                    
                    # Class distribution
                    st.subheader("Class Distribution")
                    class_counts = df[target_col].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Class balance check
                    if unique_classes > 1:
                        balance_ratio = class_counts.min() / class_counts.max()
                        if balance_ratio < 0.1:
                            st.warning("Imbalanced classes detected! Consider using techniques like SMOTE or class weights.")
                        elif balance_ratio < 0.3:
                            st.info("Moderately imbalanced classes. Consider using balanced accuracy metrics.")
                        else:
                            st.success("Well-balanced classes.")
                
                else:  # Regression
                    target_range = f"{df[target_col].min():.2f} to {df[target_col].max():.2f}"
                    st.markdown(f"""
                    ** Regression Task (Auto-detected)**
                    
                    - **Target range:** {target_range}
                    - **Mean value:** {df[target_col].mean():.2f}
                    - **Standard deviation:** {df[target_col].std():.2f}
                    - **Data type:** {df[target_col].dtype}
                    - **Unique values:** {df[target_col].nunique()}
                    """)
                    
                    st.subheader("Target Distribution")
                    hist_data = df[target_col].value_counts().sort_index()
                    st.bar_chart(hist_data)
        
        with col2:
            # Data quality check
            missing_target = df[target_col].isnull().sum()
            with stylable_container("quality_check", css_styles="""
                { background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
                  border-radius: 12px; padding: 1.5rem; }
            """):
                if missing_target > 0:
                    st.markdown(f"""
                    **Target Column Issue**
                    
                    **Missing values:** {missing_target} ({missing_target/len(df)*100:.1f}%)
                    
                    **Recommendation:** Handle missing values in preprocessing
                    """)
                else:
                    st.markdown("""
                    **Target Column Ready**
                    
                    **No missing values** in target column
                    **Ready for training**
                    """)

    # === Model Settings ===
    st.subheader("Choose Your Model Settings")
    col1, col2 = st.columns(2)

    with col1:
        with stylable_container("model_config", css_styles="""
            { background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
              border-radius: 12px; padding: 1.5rem; }
        """):
            st.markdown("** Model Algorithm**")
            
            # Get available models
            try:
                if data_info and available_models and data_info.get("data_type") == "tabular":
                    if detected_task == "classification":
                        model_options = available_models.get("classification", ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"])
                    else:
                        model_options = available_models.get("regression", ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"])
                else:
                    if detected_task == "classification":
                        model_options = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"]
                    else:
                        model_options = ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"]
            except Exception as e:
                if detected_task == "classification":
                    model_options = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN"]
                else:
                    model_options = ["Random Forest", "XGBoost", "Linear Regression", "Ridge", "Lasso"]
                st.warning(f"Model detection failed, using fallback options: {str(e)}")
            
            model_type = st.selectbox(
                "Model Algorithm:",
                model_options,
                index=0,
                help="Random Forest and XGBoost are good for most problems. Linear models are faster but may miss complex patterns."
            )
            
            scaling = st.selectbox(
                "Data Scaling:",
                ["standard", "minmax", "robust", "none"],
                help="Standard scaling works well for most cases. MinMax for bounded data. Robust for data with outliers."
            )

    with col2:
        with stylable_container("training_params", css_styles="""
            { background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
              border-radius: 12px; padding: 1.5rem; }
        """):
            st.markdown("**Training Parameters**")
            
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

    # === Training Section ===
    colored_header("Start Training", "Ready to train your model", "red-70")
    
    st.info("""
    **Ready to train your model? Here's what will happen:**
    1. Data will be split into training and test sets
    2. Features will be scaled according to your selection
    3. The model will learn patterns from your training data
    4. Performance will be evaluated on the test set
    5. Results will be displayed with detailed metrics
    """)

    # Training button
    if st.button("Start Training", type="primary", use_container_width=True):
        if not target_col:
            st.error("Please select a target column first!")
        else:
            with st.spinner("Training your model... This may take a few minutes."):
                try:
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Training progress simulation
                    for i in range(5):
                        progress_bar.progress((i + 1) * 20)
                        if i == 0:
                            status_text.text("Loading and preprocessing data...")
                        elif i == 1:
                            status_text.text("Splitting data into train/test sets...")
                        elif i == 2:
                            status_text.text("Training model...")
                        elif i == 3:
                            status_text.text("Evaluating performance...")
                        elif i == 4:
                            status_text.text("Finalizing results...")
                    
                    # Actual training
                    training_data_type = st.session_state.get('data_type', 'auto')
                    if training_data_type == 'auto':
                        if hasattr(df, 'shape') and len(df.shape) <= 2:
                            training_data_type = 'tabular'
                        else:
                            training_data_type = 'multi_dimensional'
                    
                    training_result = None
                    
                    if training_data_type == 'tabular':
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
                        if hasattr(st.session_state, 'image_directory'):
                            training_result = train_model(
                                st.session_state.image_directory,
                                data_type="image",
                                model_name="resnet18",
                                task_type="classification"
                            )
                        else:
                            st.error("Image directory not found! Please upload or specify an image directory first.")
                            progress_bar.empty()
                            status_text.empty()
                    elif training_data_type == 'audio':
                        if hasattr(st.session_state, 'audio_directory'):
                            training_result = train_model(
                                st.session_state.audio_directory,
                                data_type="audio",
                                model_name="cnn",
                                task_type="classification"
                            )
                        else:
                            st.error("Audio directory not found! Please upload or specify an audio directory first.")
                            progress_bar.empty()
                            status_text.empty()
                    elif training_data_type == 'multi_dimensional':
                        data = st.session_state.get('array_data', df)
                        training_result = train_model(
                            data,
                            target_col=target_col if target_col else None,
                            data_type="multi_dimensional",
                            model_name="MLP",
                            framework="pytorch"
                        )
                    else:
                        st.error(f"Unsupported data type: {training_data_type}")
                        progress_bar.empty()
                        status_text.empty()
                    
                    # Handle results
                    if training_result is not None:
                        if isinstance(training_result, tuple):
                            if len(training_result) == 2:
                                model, metrics = training_result
                            elif len(training_result) == 3:
                                model, metrics, additional_info = training_result
                            elif len(training_result) == 4:
                                model, metrics, additional_info, extra = training_result
                            else:
                                st.error(f"Unexpected return format from train_model: {len(training_result)} values returned")
                                model = training_result[0] if len(training_result) > 0 else None
                                metrics = training_result[1] if len(training_result) > 1 else {}
                        else:
                            model = training_result
                            metrics = {"status": "training_completed"}
                    else:
                        model = None
                        metrics = None
                    
                    # Success handling
                    if model is not None and metrics is not None:
                        # Store results
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        st.session_state.model_type = model_type
                        st.session_state.target_col = target_col
                        st.session_state.model_trained = True
                        
                        # Save to persistent storage
                        auto_save_model(model, metrics, model_type, target_col) 
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success("Model Training Completed Successfully!")
                        
                        # === Training Results ===
                        colored_header("Training Results", "Model performance metrics", "green-70")
                        
                        if isinstance(metrics, dict):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Model Performance Metrics")
                                
                                if "accuracy" in metrics:
                                    with stylable_container("accuracy_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['accuracy']:.4f}</div><div class='metric-label'>Accuracy</div>", unsafe_allow_html=True)
                                if "precision" in metrics:
                                    with stylable_container("precision_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['precision']:.4f}</div><div class='metric-label'>Precision</div>", unsafe_allow_html=True)
                                if "recall" in metrics:
                                    with stylable_container("recall_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['recall']:.4f}</div><div class='metric-label'>Recall</div>", unsafe_allow_html=True)
                                if "f1_score" in metrics:
                                    with stylable_container("f1_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['f1_score']:.4f}</div><div class='metric-label'>F1 Score</div>", unsafe_allow_html=True)
                                if "r2_score" in metrics:
                                    with stylable_container("r2_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['r2_score']:.4f}</div><div class='metric-label'>R² Score</div>", unsafe_allow_html=True)
                                if "mean_squared_error" in metrics:
                                    with stylable_container("mse_metric", css_styles="""{ text-align:center; padding:1rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{metrics['mean_squared_error']:.4f}</div><div class='metric-label'>MSE</div>", unsafe_allow_html=True)
                                
                            with col2:
                                st.subheader("Detailed Results")
                                with st.expander("View All Metrics", expanded=True):
                                    st.json(metrics)
                        else:
                            st.write("**Training Results:**", metrics)
                        
                        # === Model Saving Section ===
                        colored_header("Save Your Model", "Preserve your trained model for future use", "blue-70")
                        
                        st.info("""
                        **Save your trained model to use it later for:**
                        - Making predictions on new data
                        - Deploying to production
                        - Sharing with team members
                        - Comparing with other models
                        """)
                        
                        if st.button("Save Model", type="secondary", use_container_width=True):
                            try:
                                model_path, meta_path = save_model(
                                    model, metrics, model_type, scaling, target_col
                                )
                                st.success("Model saved successfully!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"**Model Path:** {model_path}")
                                with col2:
                                    st.info(f"**Metadata Path:** {meta_path}")
                                    
                                st.session_state.model_saved = True
                                
                            except Exception as e:
                                st.error(f"Failed to save model: {str(e)}")
                
                except Exception as e:
                    st.error(f"""
                    **Training Failed**
                    
                    **Error:** {str(e)}
                    
                    **Common solutions:**
                    - Check if your data is properly preprocessed
                    - Ensure the target column has no missing values
                    - Try a different model type or scaling method
                    - Verify your data types are appropriate for the selected model
                    """)
                    
                    st.info("Need help? Go back to the preprocessing page to clean your data.")

    # === Navigation Section ===
    if st.session_state.model is not None:
        st.success("Congratulations! You now have a trained model ready for testing and deployment!")
        
        colored_header("What's Next?", "Continue your machine learning journey", "violet-70")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with stylable_container("next_step_test", css_styles="""
                { background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
                  border-radius: 12px; padding: 1.5rem; text-align: center; }
            """):
                st.markdown("""
                **Test Your Model**
                
                Evaluate your model on new data to ensure it generalizes well
                """)
                if st.button("Go to Testing", type="primary", use_container_width=True):
                    safe_switch_page("pages/5_🧪_Test_Model.py")
        
        with col2:
            with stylable_container("next_step_deploy", css_styles="""
                { background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
                  border-radius: 12px; padding: 1.5rem; text-align: center; }
            """):
                st.markdown("""
                **Deploy Your Model**
                
                Make predictions on new data using your trained model
                """)
                if st.button("Go to Deployment", use_container_width=True):
                    safe_switch_page("pages/6_🚀_Deploy_Model.py")

    else:
        # === Getting Started Guide ===
        with stylable_container("getting_started", css_styles="""
            { background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
              border-radius: 12px; padding: 2rem; margin-top: 2rem; }
        """):
            st.markdown("""
            **Getting Started Guide**
            
            **Step 1:** Ensure your data is loaded and preferably preprocessed
            **Step 2:** Select the column you want to predict (target variable)
            **Step 3:** Choose your model type and configuration settings
            **Step 4:** Click "Start Training" to begin the process
            **Step 5:** Review results and save your model
            **Step 6:** Move to testing or deployment
            """)

    # === Final Navigation ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Data Upload", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")
    with col2:
        if st.button("Go to Preprocessing", use_container_width=True):
            safe_switch_page("pages/2_🔧_Data_Preprocessing.py")