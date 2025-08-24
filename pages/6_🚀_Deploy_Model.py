# app/pages/6_Deploy_Model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from app.utilss.navigation import safe_switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header

# Page config
st.set_page_config(page_title="Deploy Model", layout="wide", page_icon="🚀")

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
        font-size: 2rem; font-weight: 700;
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
    
    /* Upload area styling */
    .upload-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(45, 55, 72, 0.9) !important;
        border-radius: 8px !important;
        border: 1px solid #667eea !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
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
            <i class="fas fa-rocket nav-icon"></i>Deploy Model
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">Deploy your trained model for real-world use</div>
        """, unsafe_allow_html=True)

    # === Check if model exists ===
    if "model" not in st.session_state:
        st.warning("No trained model available. Please train a model first from the Training page.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Training", use_container_width=True):
                safe_switch_page("pages/4_Train_Model.py")
        with col2:
            st.info("You need a trained model to deploy it for predictions.")
        st.stop()

    # === Model Info Overview ===
    model_type = st.session_state.get("model_type", "Unknown")
    target_col = st.session_state.get("target_col", "Unknown")

    colored_header("Model Status", "Your trained model is ready for deployment", "blue-70")
    
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
        status = "Saved" if st.session_state.get("model_saved", False) else "Memory"
        with stylable_container("metric_status", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{status}</div><div class='metric-label'>Storage</div>", unsafe_allow_html=True)

    st.markdown("---")

    # === Deployment Method Selection ===
    colored_header("Deployment Methods", "Choose how you want to deploy your model", "violet-70")
    
    st.markdown("""
    <div class="upload-info">
        <h4><i class="fas fa-bullseye"></i> Available Deployment Options</h4>
        <p><strong><i class="fas fa-sync"></i> Batch Predictions:</strong> Process multiple records at once for reports and analysis</p>
        <p><strong><i class="fas fa-bolt"></i> Real-time API:</strong> Make instant predictions for web apps and live systems</p>
        <p><strong><i class="fas fa-download"></i> Download Model:</strong> Export your model for custom deployment environments</p>
    </div>
    """, unsafe_allow_html=True)

    deployment_method = st.radio(
        "Select your deployment method:",
        ["Batch Predictions", "Real-time API", "Download Model"],
        horizontal=True
    )

    st.markdown("---")

    # === Batch Predictions ===
    if deployment_method == "Batch Predictions":
        colored_header("Batch Prediction Service", "Process large datasets efficiently", "green-70")
        
        st.info("Perfect for: Large datasets, scheduled reports, offline analysis, and bulk processing")

        # File upload section
        with stylable_container("upload_section", css_styles="""
            { background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
              border-radius: 15px; padding: 1.5rem; margin: 1rem 0; }
        """):
            st.markdown("### <i class='fas fa-file-upload'></i> Upload Your Data", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Choose a CSV file for batch predictions",
                type=["csv"],
                help="Upload a CSV file containing the same features used during training"
            )

            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    st.success(f"Successfully uploaded: **{uploaded.name}**")

                    # Data overview
                    colored_header("Data Overview", "Quick stats about your uploaded data", "blue-70")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        with stylable_container("batch_records", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                            st.markdown(f"<div class='metric-value'>{df.shape[0]:,}</div><div class='metric-label'>Records</div>", unsafe_allow_html=True)
                    with col2:
                        with stylable_container("batch_features", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                            st.markdown(f"<div class='metric-value'>{df.shape[1]}</div><div class='metric-label'>Features</div>", unsafe_allow_html=True)
                    with col3:
                        size_mb = uploaded.size / (1024 * 1024)
                        with stylable_container("batch_size", css_styles="""{ text-align:center; padding:1.5rem; }"""):
                            st.markdown(f"<div class='metric-value'>{size_mb:.1f}</div><div class='metric-label'>Size (MB)</div>", unsafe_allow_html=True)

                    # Data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Process predictions
                    if st.button("Start Batch Predictions", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Processing predictions..."):
                                preds = st.session_state.model.predict(df)
                                results = df.copy()
                                results["Prediction"] = preds
                                
                                # Add confidence if available
                                if hasattr(st.session_state.model, "predict_proba"):
                                    proba = st.session_state.model.predict_proba(df)
                                    results["Confidence"] = np.max(proba, axis=1)

                            st.success("Batch predictions completed successfully!")
                            
                            # Results preview
                            colored_header("Prediction Results", "Your predictions are ready", "green-70")
                            st.dataframe(results.head(20), use_container_width=True)

                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                "Download Predictions (CSV)",
                                data=csv,
                                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")

                except Exception as e:
                    st.error(f"Failed to read file: {str(e)}")

    # === Real-time API ===
    elif deployment_method == "Real-time API":
        colored_header("Real-time Prediction API", "Instant predictions for live applications", "orange-70")
        
        st.info("Perfect for: Web applications, mobile apps, live decision systems, and instant predictions")

        # API Documentation
        colored_header("API Documentation", "How to integrate with your applications", "blue-70")
        
        with stylable_container("api_docs", css_styles="""
            { background: rgba(45, 55, 72, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
        """):
            st.markdown("### <i class='fas fa-plug'></i> API Endpoint", unsafe_allow_html=True)
            st.code("""
POST /api/predict
Content-Type: application/json

{
    "features": {
        "feature1": value1,
        "feature2": value2,
        "feature3": value3
    }
}
            """, language="json")

        # Live API Test
        colored_header("Test Your API", "Try real-time predictions", "green-70")
        
        if "df" in st.session_state:
            feature_cols = [col for col in st.session_state.df.columns if col != target_col]
            
            st.markdown("### <i class='fas fa-sliders-h'></i> Enter Feature Values", unsafe_allow_html=True)
            input_data = {}
            
            # Create input fields dynamically
            cols = st.columns(min(3, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                with cols[i % len(cols)]:
                    if st.session_state.df[col].dtype == "object":
                        unique_vals = st.session_state.df[col].unique()
                        input_data[col] = st.selectbox(f"Select {col}", unique_vals, key=f"api_{col}")
                    else:
                        mean_val = float(st.session_state.df[col].mean())
                        input_data[col] = st.number_input(f"Enter {col}", value=mean_val, key=f"api_{col}")

            if st.button("Make Real-time Prediction", type="primary", use_container_width=True):
                try:
                    with st.spinner("Making prediction..."):
                        input_df = pd.DataFrame([input_data])
                        pred = st.session_state.model.predict(input_df)[0]
                        conf = None
                        if hasattr(st.session_state.model, "predict_proba"):
                            conf = np.max(st.session_state.model.predict_proba(input_df))

                    # Display results
                    colored_header("Prediction Results", "Your real-time prediction", "green-70")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        with stylable_container("pred_result", css_styles="""{ text-align:center; padding:2rem; }"""):
                            st.markdown(f"<div class='metric-value'>{pred}</div><div class='metric-label'>Prediction</div>", unsafe_allow_html=True)
                    with c2:
                        if conf:
                            with stylable_container("pred_confidence", css_styles="""{ text-align:center; padding:2rem; }"""):
                                st.markdown(f"<div class='metric-value'>{conf:.3f}</div><div class='metric-label'>Confidence</div>", unsafe_allow_html=True)

                    st.markdown("### <i class='fas fa-clipboard-list'></i> Input Data Used", unsafe_allow_html=True)
                    st.json(input_data)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    # === Download Model ===
    elif deployment_method == "Download Model":
        colored_header("Download Your Model", "Export for custom deployment environments", "voilet")
        
        st.info("Perfect for: Custom environments, production servers, edge devices, and offline deployment")

        if st.session_state.get("model_saved", False):
            colored_header("Model Ready for Download", "Your model has been saved successfully", "green-70")
            
            # Model information
            with stylable_container("model_info", css_styles="""
                { background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
                  border-radius: 15px; padding: 2rem; margin: 1rem 0; }
            """):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**<i class='fas fa-robot'></i> Model Type:** `{model_type}`", unsafe_allow_html=True)
                    st.markdown(f"**<i class='fas fa-bullseye'></i> Target Variable:** `{target_col}`", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**<i class='fas fa-save'></i> Format:** `Pickle (.pkl)`", unsafe_allow_html=True)
                    st.markdown(f"**<i class='fas fa-info-circle'></i> Metadata:** `JSON included`", unsafe_allow_html=True)

            # Download instructions
            colored_header("Download Instructions", "How to use your downloaded model", "blue-70")
            
            st.markdown("""
            ### <i class='fas fa-wrench'></i> Usage Steps:
            1. **<i class='fas fa-folder-open'></i> Locate Files:** Navigate to the `models/` directory
            2. **<i class='fas fa-copy'></i> Copy Files:** 
               - `latest_model.pkl` (your trained model)
               - `latest_model_meta.json` (model metadata)
            3. **<i class='fas fa-code'></i> Load in Python:**
            """, unsafe_allow_html=True)
            
            st.code("""
import joblib
import json

# Load the model
model = joblib.load('latest_model.pkl')

# Load metadata (optional)
with open('latest_model_meta.json', 'r') as f:
    metadata = json.load(f)

# Make predictions
predictions = model.predict(your_data)
            """, language="python")

        else:
            st.warning("Your model has not been saved yet. Please go to the Training page to save it first.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go Save Model", type="primary", use_container_width=True):
                    safe_switch_page("pages/4_Train_Model.py")
            with col2:
                st.info("Saving your model creates the necessary files for download and deployment.")

    # === Production Deployment Guide ===
    st.markdown("---")
    colored_header("Production Deployment Guide", "Advanced deployment strategies", "red-70")
    
    with st.expander("**Advanced Deployment Options**", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Cloud", "Containers", "Serverless", "Edge"])
        
        with tab1:
            st.markdown("""
            ### <i class='fas fa-cloud'></i> **Cloud Platforms**
            - **AWS:** SageMaker, EC2, Batch
            - **Google Cloud:** AI Platform, Compute Engine  
            - **Azure:** Machine Learning, Container Instances
            - **Benefits:** Auto-scaling, managed infrastructure, high availability
            """, unsafe_allow_html=True)
            
        with tab2:
            st.markdown("""
            ### <i class='fab fa-docker'></i> **Container Deployment**
            - **Docker:** Containerize your model and dependencies
            - **Kubernetes:** Orchestrate containers at scale
            - **Benefits:** Consistency, portability, easy scaling
            """, unsafe_allow_html=True)
            
        with tab3:
            st.markdown("""
            ### <i class='fas fa-bolt'></i> **Serverless Functions**
            - **AWS Lambda:** Event-driven predictions
            - **Google Cloud Functions:** HTTP trigger predictions
            - **Benefits:** No server management, pay-per-use, auto-scaling
            """, unsafe_allow_html=True)
            
        with tab4:
            st.markdown("""
            ### <i class='fas fa-mobile-alt'></i> **Edge Deployment**
            - **Mobile Apps:** On-device inference
            - **IoT Devices:** Real-time edge computing
            - **Benefits:** Low latency, offline capability, privacy
            """, unsafe_allow_html=True)

    # === Navigation ===
    st.markdown("---")
    colored_header("What's Next?", "Continue your ML journey", "blue-70")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Training", use_container_width=True):
            safe_switch_page("pages/4_Train_Model.py")
    with col2:
        if st.button("Back to Home", use_container_width=True):
            safe_switch_page("app.py")