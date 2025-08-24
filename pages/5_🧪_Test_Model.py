# app/pages/5_Test_Model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
from core.evaluator import evaluate_classification
import plotly.express as px
import joblib
from app.utilss.navigation import safe_switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header

# Page configuration
st.set_page_config(page_title="Test Model", layout="wide", page_icon="🧪")

# === CUSTOM CSS ===
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
    .stSuccess, .stInfo, .stWarning, .stError { border-radius: 12px; border: none; }
    
    /* Upload area styling */
    .upload-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #059669, #047857) !important;
    }
    
    /* Test card styling */
    .test-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            <i class="fas fa-flask nav-icon"></i>Test Your Model
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">Evaluate and validate your model performance</div>
        """, unsafe_allow_html=True)

    # === Check if model exists ===
    if "model" not in st.session_state:
        st.error("No trained model found. Please go to Training first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Training", use_container_width=True):
                safe_switch_page("pages/4_Train_Model.py")
        with col2:
            st.info("Train a model before testing it.")
        st.stop()

    # === Model Info Overview ===
    model_type = st.session_state.get("model_type", "Unknown")
    target_col = st.session_state.get("target_col", "Unknown")

    colored_header("Model Status", "Your trained model is ready for testing", "blue-70")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container("metric_model", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'><i class='fas fa-check-circle'></i></div><div class='metric-label'>Model Ready</div>", unsafe_allow_html=True)
    with col2:
        with stylable_container("metric_type", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{model_type}</div><div class='metric-label'>Model Type</div>", unsafe_allow_html=True)
    with col3:
        with stylable_container("metric_target", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{target_col}</div><div class='metric-label'>Target Variable</div>", unsafe_allow_html=True)
    with col4:
        status = "Ready"
        with stylable_container("metric_status", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{status}</div><div class='metric-label'>Status</div>", unsafe_allow_html=True)

    st.markdown("---")

    # === Upload Test Data ===
    colored_header("Upload Test Data", "Validate your model with new data", "green-70")
    
    st.markdown("""
    <div class="upload-info">
        <h4><i class="fas fa-upload"></i> Test Data Requirements</h4>
        <p><strong><i class="fas fa-check"></i> Same Format:</strong> Should match training data structure</p>
        <p><strong><i class="fas fa-target"></i> Include Target:</strong> For performance evaluation (optional)</p>
        <p><strong><i class="fas fa-info-circle"></i> No Target:</strong> For predictions only</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("**Data Requirements**", expanded=False):
        st.markdown("""
        **Requirements:**
        - Same columns as training data
        - Correct data types and formats
        - No structural changes
        
        **Avoid:**
        - Using the same training data
        - Missing required columns
        - Data type mismatches
        """, unsafe_allow_html=True)

    # File upload section
    with stylable_container("upload_section", css_styles="""
        { background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
          border-radius: 15px; padding: 1.5rem; margin: 1rem 0; }
    """):
        st.markdown("### <i class='fas fa-file-upload'></i> Upload Test Data", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a CSV file for testing",
            type=["csv"],
            help="Upload a CSV file containing test data for model validation"
        )

    if uploaded is not None:
        try:
            test_df = pd.read_csv(uploaded)
            st.success(f"Successfully uploaded: **{uploaded.name}**")

            # === Test Data Overview ===
            colored_header("Test Data Overview", "Quick stats about your test data", "purple-70")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with stylable_container("test_rows", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                    st.markdown(f"<div class='metric-value'>{test_df.shape[0]:,}</div><div class='metric-label'>Rows</div>", unsafe_allow_html=True)
            with col2:
                with stylable_container("test_cols", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                    st.markdown(f"<div class='metric-value'>{test_df.shape[1]}</div><div class='metric-label'>Columns</div>", unsafe_allow_html=True)
            with col3:
                size_mb = uploaded.size / (1024 * 1024)
                with stylable_container("test_size", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                    st.markdown(f"<div class='metric-value'>{size_mb:.1f}</div><div class='metric-label'>Size (MB)</div>", unsafe_allow_html=True)
            with col4:
                missing = test_df.isnull().sum().sum()
                with stylable_container("test_missing", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                    st.markdown(f"<div class='metric-value'>{missing:,}</div><div class='metric-label'>Missing Values</div>", unsafe_allow_html=True)

            # === Data Preview ===
            st.subheader("Data Preview")
            st.dataframe(test_df.head(10), use_container_width=True)

            has_target = target_col in test_df.columns

            if has_target:
                st.success(f"Target column `{target_col}` found! Full evaluation available.")
            else:
                st.info(f"Target column `{target_col}` not found. Predictions only.")

            # === Testing Options ===
            colored_header("Testing Options", "Choose how to test your model", "orange-70")

            col1, col2 = st.columns(2)

            with col1:
                with stylable_container("predict_card", css_styles="""
                    { background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
                      border-radius: 15px; padding: 2rem; margin: 1rem 0; text-align: center; }
                """):
                    st.markdown("### <i class='fas fa-magic'></i> Make Predictions", unsafe_allow_html=True)
                    st.markdown("Generate predictions without performance evaluation")
                    
                    if st.button("Start Predictions", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Making predictions..."):
                                X_test = test_df.drop(columns=[target_col]) if has_target else test_df
                                preds = st.session_state.model.predict(X_test)
                                results = test_df.copy()
                                results['Prediction'] = preds
                                
                                # Add confidence if available
                                if hasattr(st.session_state.model, "predict_proba"):
                                    proba = st.session_state.model.predict_proba(X_test)
                                    results["Confidence"] = np.max(proba, axis=1)

                            st.success("Predictions completed successfully!")
                            
                            # Results preview
                            colored_header("Prediction Results", "Your predictions are ready", "green-70")
                            st.dataframe(results.head(20), use_container_width=True)

                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                "Download Predictions (CSV)",
                                data=csv,
                                file_name="test_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")

            with col2:
                if has_target:
                    with stylable_container("evaluate_card", css_styles="""
                        { background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(124, 58, 237, 0.1));
                          border-radius: 15px; padding: 2rem; margin: 1rem 0; text-align: center; }
                    """):
                        st.markdown("### <i class='fas fa-chart-line'></i> Evaluate Model", unsafe_allow_html=True)
                        st.markdown("Measure accuracy and performance metrics")
                        
                        if st.button("Start Evaluation", type="primary", use_container_width=True):
                            try:
                                with st.spinner("Evaluating model performance..."):
                                    X_test = test_df.drop(columns=[target_col])
                                    y_test = test_df[target_col]
                                    y_pred = st.session_state.model.predict(X_test)

                                    # Calculate metrics
                                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                    acc = accuracy_score(y_test, y_pred)
                                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                                st.success("Model evaluation completed!")
                                
                                # Display metrics
                                colored_header("Performance Metrics", "Model evaluation results", "blue-70")
                                
                                m1, m2, m3, m4 = st.columns(4)
                                with m1:
                                    with stylable_container("acc_metric", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{acc:.3f}</div><div class='metric-label'>Accuracy</div>", unsafe_allow_html=True)
                                with m2:
                                    with stylable_container("prec_metric", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{prec:.3f}</div><div class='metric-label'>Precision</div>", unsafe_allow_html=True)
                                with m3:
                                    with stylable_container("rec_metric", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{rec:.3f}</div><div class='metric-label'>Recall</div>", unsafe_allow_html=True)
                                with m4:
                                    with stylable_container("f1_metric", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                                        st.markdown(f"<div class='metric-value'>{f1:.3f}</div><div class='metric-label'>F1 Score</div>", unsafe_allow_html=True)

                                # Store metrics in session state
                                st.session_state.test_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
                                
                                # Performance interpretation
                                with stylable_container("performance_info", css_styles="""
                                    { background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.1));
                                      border-radius: 15px; padding: 1.5rem; margin: 1rem 0; }
                                """):
                                    st.markdown("### <i class='fas fa-lightbulb'></i> Performance Insights", unsafe_allow_html=True)
                                    if acc >= 0.9:
                                        st.success("Excellent performance! Your model is highly accurate.")
                                    elif acc >= 0.8:
                                        st.info("Good performance. Consider fine-tuning for improvement.")
                                    elif acc >= 0.7:
                                        st.warning("Moderate performance. Model may need improvement.")
                                    else:
                                        st.error("Low performance. Consider retraining with different parameters.")

                            except Exception as e:
                                st.error(f"Evaluation failed: {str(e)}")
                else:
                    with stylable_container("no_target_card", css_styles="""
                        { background: rgba(156, 163, 175, 0.1); border-radius: 15px; padding: 2rem; 
                          margin: 1rem 0; text-align: center; }
                    """):
                        st.markdown("### <i class='fas fa-info-circle'></i> Evaluation Unavailable", unsafe_allow_html=True)
                        st.markdown("Target column required for model evaluation")
                        st.info("Upload data with target column to enable evaluation")

        except Exception as e:
            st.error(f"Failed to read uploaded file: {str(e)}")

    # === Navigation ===
    if "model" in st.session_state:
        st.markdown("---")
        colored_header("What's Next?", "Continue your ML journey", "blue-70")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Deploy Model", use_container_width=True):
                safe_switch_page("pages/6_🚀_Deploy_Model.py")
        with col2:
            if st.button("Improve Model", use_container_width=True):
                safe_switch_page("pages/4_⚙️_Train_Model.py")