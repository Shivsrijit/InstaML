# app/pages/6_🚀_Deploy_Model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime
from app.utilss.navigation import safe_switch_page

# Page configuration
st.set_page_config(page_title="Deploy Model", layout="wide")

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
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    .deploy-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .deploy-card h4 {
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
    
    .upload-area {
        background: #f8f9fa;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: #f0f2f6;
    }
    
    .deployment-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .api-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .code-block {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🚀 Deploy Your Machine Learning Model")
st.markdown("Transform your trained model into a production-ready prediction service")

# Collapsible help section
with st.expander("ℹ️ **What is Model Deployment and Why is it Important?**", expanded=False):
    st.markdown("""
    **Model Deployment** is like opening a restaurant after perfecting your recipes. It's the final step that makes your model available to users and integrates it into real-world applications.
    
    **🎯 What deployment enables:**
    - **Real-time predictions**: Make predictions on new data instantly
    - **Integration**: Connect your model to web apps, mobile apps, or APIs
    - **Scalability**: Handle multiple prediction requests simultaneously
    - **Monitoring**: Track how your model performs in production
    - **Business value**: Turn your ML work into actionable insights
    
    **⚡ Why deployment matters:**
    - **Value realization**: Models only create value when they're used
    - **Feedback loop**: Production data helps improve future models
    - **User experience**: Seamless integration with existing workflows
    - **Competitive advantage**: Operational ML capabilities set you apart
    """)

# Check if model exists
if "model" not in st.session_state:
    st.error("""
    ❌ **No Trained Model Available**
    
    Please train a model first from the Train Model page.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️ Go to Training", type="primary"):
            safe_switch_page("pages/4_⚙️_Train_Model.py")
    with col2:
        st.info("💡 **Tip:** You need a trained model to deploy it for predictions.")
    st.stop()

# Model information display
st.header("🤖 **Your Model Ready for Deployment**")
model_type = st.session_state.get("model_type", "Unknown")
target_col = st.session_state.get("target_col", "Unknown")
model_metrics = st.session_state.get("metrics", {})

st.markdown(f"""
<div class="deployment-header">
    <h3>🚀 Model Ready for Production</h3>
    <p><strong>Model Type:</strong> {model_type}</p>
    <p><strong>Target Variable:</strong> {target_col}</p>
    <p><strong>Status:</strong> ✅ Trained and Ready for Deployment</p>
</div>
""", unsafe_allow_html=True)

# Deployment options
st.header("🎯 **Choose Your Deployment Method**")
st.info("""
**Select how you want to deploy your model:**
- **Batch Predictions**: Process multiple records at once (good for reports, analysis)
- **Real-time API**: Make instant predictions on individual records (good for user-facing apps)
- **Download Model**: Get your model file for custom deployment
""")

# Deployment method selection
deployment_method = st.radio(
    "Select deployment method:",
    ["Batch Predictions", "Real-time API", "Download Model"],
    horizontal=True
)

if deployment_method == "Batch Predictions":
    st.subheader("📊 **Batch Prediction Service**")
    st.info("""
    **Batch predictions are perfect for:**
    - Processing large datasets overnight
    - Generating reports and analytics
    - Bulk data analysis
    - Offline prediction workflows
    """)
    
    # Batch upload section
    st.markdown("""
    <div class="deploy-card">
        <h4>📤 Upload Data for Batch Predictions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-area">
        <h3>📁 Drop your CSV file here for batch predictions</h3>
        <p>or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    batch_uploaded = st.file_uploader(
        "Upload CSV for batch predictions", 
        type=["csv"],
        label_visibility="collapsed"
    )
    
    if batch_uploaded is not None:
        try:
            batch_df = pd.read_csv(batch_uploaded)
            st.success(f"✅ **Successfully uploaded: {batch_uploaded.name}**")
            
            # Show batch data overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", f"{batch_df.shape[0]:,}")
            with col2:
                st.metric("Features", batch_df.shape[1])
            with col3:
                file_size_mb = batch_uploaded.size / (1024 * 1024)
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            
            # Data preview
            st.subheader("🔍 **Data Preview**")
            st.dataframe(batch_df.head(10))
            
            # Batch prediction button
            if st.button("🚀 Start Batch Predictions", type="primary", use_container_width=True):
                try:
                    with st.spinner("Processing batch predictions..."):
                        # Make predictions
                        predictions = st.session_state.model.predict(batch_df)
                        
                        # Create results dataframe
                        results_df = batch_df.copy()
                        results_df['Prediction'] = predictions
                        
                        # Add confidence scores if available
                        if hasattr(st.session_state.model, 'predict_proba'):
                            proba = st.session_state.model.predict_proba(batch_df)
                            if proba.shape[1] == 2:  # Binary classification
                                results_df['Confidence'] = np.max(proba, axis=1)
                            else:  # Multi-class
                                results_df['Confidence'] = np.max(proba, axis=1)
                    
                    st.success("✅ **Batch predictions completed successfully!**")
                    
                    # Display results
                    st.subheader("📊 **Batch Prediction Results**")
                    st.dataframe(results_df.head(20))
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Batch Predictions (CSV)",
                        data=csv,
                        file_name=f"batch_predictions_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("📈 **Prediction Summary**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        if 'Confidence' in results_df.columns:
                            avg_confidence = results_df['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    with col3:
                        if hasattr(st.session_state.model, 'predict_proba'):
                            unique_predictions = len(np.unique(predictions))
                            st.metric("Unique Predictions", unique_predictions)
                    
                except Exception as e:
                    st.error(f"""
                    ❌ **Batch prediction failed: {str(e)}**
                    
                    **🔧 Common solutions:**
                    - Ensure data has the same columns as training data
                    - Check data types and formats
                    - Handle any missing values
                    - Verify data preprocessing requirements
                    """)
                    
        except Exception as e:
            st.error(f"❌ **Failed to read file: {str(e)}**")

elif deployment_method == "Real-time API":
    st.subheader("⚡ **Real-time Prediction API**")
    st.info("""
    **Real-time API is perfect for:**
    - Web applications and dashboards
    - Mobile apps
    - IoT devices
    - Live user interactions
    - Instant decision-making systems
    """)
    
    # API documentation
    st.markdown("""
    <div class="deploy-card">
        <h4>📚 API Documentation</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # API endpoint information
    st.markdown("""
    <div class="api-section">
        <h5>🌐 API Endpoint</h5>
        <div class="code-block">
POST /api/predict
Content-Type: application/json

{
    "features": {
        "feature1": value1,
        "feature2": value2,
        ...
    }
}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive API testing
    st.subheader("🧪 **Test Your API**")
    st.info("""
    **Test your model with sample data to see how the API works:**
    - Enter feature values manually
    - Upload a single record
    - See real-time predictions
    """)
    
    # Manual input testing
    st.markdown("""
    <div class="deploy-card">
        <h4>✏️ Manual Input Testing</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Get feature columns from training data
    if "df" in st.session_state:
        training_df = st.session_state.df
        feature_cols = [col for col in training_df.columns if col != target_col]
        
        # Create input form
        input_data = {}
        cols = st.columns(3)
        
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                if training_df[col].dtype in ['object', 'category']:
                    # Categorical input
                    unique_vals = training_df[col].unique()
                    input_data[col] = st.selectbox(f"{col}:", unique_vals)
                else:
                    # Numeric input
                    min_val = float(training_df[col].min())
                    max_val = float(training_df[col].max())
                    mean_val = float(training_df[col].mean())
                    input_data[col] = st.number_input(
                        f"{col}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
        
        # Make prediction button
        if st.button("🚀 Make Real-time Prediction", type="primary", use_container_width=True):
            try:
                # Create input dataframe
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = st.session_state.model.predict(input_df)[0]
                    
                    # Get confidence if available
                    confidence = None
                    if hasattr(st.session_state.model, 'predict_proba'):
                        proba = st.session_state.model.predict_proba(input_df)
                        confidence = np.max(proba)
                
                st.success("✅ **Prediction completed!**")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prediction}</div>
                        <div class="metric-label">Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if confidence is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{confidence:.3f}</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show input data
                st.subheader("📋 **Input Data Used**")
                st.json(input_data)
                
            except Exception as e:
                st.error(f"❌ **Prediction failed: {str(e)}**")
    
    # Single record upload testing
    st.subheader("📤 **Upload Single Record for Testing**")
    single_uploaded = st.file_uploader(
        "Upload single CSV record for testing",
        type=["csv"],
        help="Upload a CSV with one row of data to test the API"
    )
    
    if single_uploaded is not None:
        try:
            single_df = pd.read_csv(single_uploaded)
            if single_df.shape[0] == 1:
                st.success("✅ **Single record uploaded successfully!**")
                st.dataframe(single_df)
                
                if st.button("🚀 Test with This Record", type="secondary"):
                    try:
                        with st.spinner("Testing prediction..."):
                            prediction = st.session_state.model.predict(single_df)[0]
                        
                        st.success(f"✅ **Prediction: {prediction}**")
                        
                    except Exception as e:
                        st.error(f"❌ **Test failed: {str(e)}**")
            else:
                st.warning("⚠️ **Please upload a CSV with exactly one row for testing.**")
                
        except Exception as e:
            st.error(f"❌ **Failed to read file: {str(e)}**")

elif deployment_method == "Download Model":
    st.subheader("💾 **Download Your Model**")
    st.info("""
    **Download your model for custom deployment:**
    - Deploy to your own servers
    - Integrate with custom applications
    - Use in different environments
    - Share with team members
    """)
    
    # Model download section
    st.markdown("""
    <div class="deploy-card">
        <h4>📦 Model Files</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if model was saved
    if "model_saved" in st.session_state:
        st.success("✅ **Your model has been saved and is ready for download!**")
        
        # Model metadata
        st.subheader("📋 **Model Information**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model Type:** {model_type}")
            st.info(f"**Target Variable:** {target_col}")
        
        with col2:
            if model_metrics:
                if "accuracy" in model_metrics:
                    st.metric("Training Accuracy", f"{model_metrics['accuracy']:.4f}")
                elif "r2_score" in model_metrics:
                    st.metric("Training R²", f"{model_metrics['r2_score']:.4f}")
        
        # Download instructions
        st.subheader("📥 **Download Instructions**")
        st.markdown("""
        **To download your model:**
        1. Navigate to the `models/` directory in your project
        2. Find the following files:
           - `latest_model.pkl` - Your trained model
           - `latest_model_meta.json` - Model metadata and performance
        3. Copy these files to your deployment environment
        
        **💡 Tip:** The model file contains everything needed to make predictions!
        """)
        
        # Show model file paths
        st.subheader("📍 **Model File Locations**")
        st.code("models/latest_model.pkl", language="bash")
        st.code("models/latest_model_meta.json", language="bash")
        
    else:
        st.warning("""
        ⚠️ **Model not saved yet**
        
        Please save your model first on the Training page before downloading.
        """)
        
        if st.button("💾 Go Save Model", type="primary"):
            safe_switch_page("pages/4_⚙️_Train_Model.py")

# Production deployment guidance
st.header("🏭 **Production Deployment Guide**")
with st.expander("📚 **Advanced Deployment Options**", expanded=False):
    st.markdown("""
    **🚀 Production Deployment Strategies:**
    
    **1. Cloud Deployment:**
    - **AWS SageMaker**: Managed ML platform with auto-scaling
    - **Google Cloud AI Platform**: Enterprise ML deployment
    - **Azure Machine Learning**: Microsoft's ML service
    - **Heroku**: Simple deployment for small models
    
    **2. Container Deployment:**
    - **Docker**: Package your model in a container
    - **Kubernetes**: Orchestrate multiple model instances
    - **Docker Compose**: Simple multi-service setup
    
    **3. Serverless Deployment:**
    - **AWS Lambda**: Event-driven predictions
    - **Google Cloud Functions**: Serverless ML inference
    - **Azure Functions**: Microsoft's serverless platform
    
    **4. Edge Deployment:**
    - **Mobile apps**: On-device predictions
    - **IoT devices**: Local inference capabilities
    - **Edge servers**: Distributed prediction services
    
    **🔧 Deployment Checklist:**
    - [ ] Model performance validation
    - [ ] Data preprocessing pipeline
    - [ ] API endpoint design
    - [ ] Error handling and logging
    - [ ] Monitoring and alerting
    - [ ] Security and authentication
    - [ ] Scalability planning
    - [ ] Backup and recovery
    """)

# Navigation section
st.header("🚀 **What's Next?**")
col1, col2 = st.columns(2)

with col1:
    st.info("**🔄 Iterate and Improve**")
    st.write("Go back to training to improve your model based on deployment feedback")
    if st.button("⚙️ Back to Training", type="secondary", use_container_width=True):
        safe_switch_page("pages/4_⚙️_Train_Model.py")

with col2:
    st.info("**📊 Monitor Performance**")
    st.write("Track how your deployed model performs in production")
    st.info("💡 **Tip:** Monitor prediction accuracy, response times, and user feedback")

# Getting started guide
st.info("""
📋 **Deployment Success Guide**

**Step 1:** Choose your deployment method (batch, real-time, or download)
**Step 2:** Test your deployment with sample data
**Step 3:** Integrate with your applications or systems
**Step 4:** Monitor performance and gather feedback
**Step 5:** Iterate and improve based on real-world usage
**Step 6:** Scale up as demand grows
""") 