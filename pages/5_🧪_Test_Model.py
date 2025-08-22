# app/pages/5_🧪_Test_Model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
from core.evaluator import evaluate_classification
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from app.utilss.navigation import safe_switch_page

# Page configuration
st.set_page_config(page_title="Test Model", layout="wide")

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
    
    .test-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .test-card h4 {
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
    
    .results-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🧪 Test Your Machine Learning Model")
st.markdown("Evaluate your model's performance on new data to ensure it generalizes well")

# Collapsible help section
with st.expander("ℹ️ **What is Model Testing and Why is it Critical?**", expanded=False):
    st.markdown("""
    **Model Testing** is like giving your student a final exam to see how well they've learned. It's the crucial step that tells you whether your model will work well on new, unseen data.
    
    **🎯 What testing reveals:**
    - **Generalization**: How well your model works on new data
    - **Overfitting**: Whether your model memorized training data instead of learning patterns
    - **Real-world performance**: How your model will perform in production
    - **Bias and fairness**: Whether your model works equally well for different groups
    
    **⚡ Why testing is essential:**
    - **Avoid surprises**: Catch problems before deploying to production
    - **Build confidence**: Ensure your model works reliably
    - **Improve performance**: Identify areas for model enhancement
    - **Business validation**: Prove your model adds real value
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
        st.info("💡 **Tip:** You need a trained model to test it on new data.")
    st.stop()

# Model information display
st.header("🤖 **Your Trained Model**")
model_type = st.session_state.get("model_type", "Unknown")
target_col = st.session_state.get("target_col", "Unknown")

st.markdown(f"""
<div class="results-card">
    <h3>🎯 Model Ready for Testing</h3>
    <p><strong>Model Type:</strong> {model_type}</p>
    <p><strong>Target Variable:</strong> {target_col}</p>
    <p><strong>Status:</strong> ✅ Trained and Ready</p>
</div>
""", unsafe_allow_html=True)

# Test data upload section
st.header("📤 **Upload Test Data**")
st.info("""
**Upload a CSV file with new data to test your model:**
- **Same format**: Your test data should have the same structure as training data
- **Include target**: If you want to evaluate performance, include the target column
- **No target needed**: You can also just make predictions without evaluation
- **File size**: Keep under 100MB for reliable processing
""")

# Collapsible format guidance
with st.expander("📋 **Test Data Requirements & Tips**", expanded=False):
    st.markdown("""
    **✅ What your test data should have:**
    - **Same columns**: All feature columns used during training
    - **Same data types**: Numeric columns should be numeric, categorical should be categorical
    - **Same format**: CSV format with headers (column names)
    - **No missing values**: In columns that were used for training
    
    **💡 Best practices:**
    - Use data that represents real-world scenarios
    - Test on data collected at different times or from different sources
    - Include edge cases and unusual scenarios
    - Ensure data quality is similar to training data
    
    **⚠️ Common mistakes to avoid:**
    - Using training data as test data
    - Testing on data with different column names
    - Ignoring data type mismatches
    - Testing on data with different scales or distributions
    """)

# Upload area
st.markdown("""
<div class="upload-area">
    <h3>📁 Drop your test CSV file here</h3>
    <p>or click to browse</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload test CSV file", 
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        st.success(f"✅ **Successfully uploaded test data: {uploaded.name}**")
        
        # Test data overview
        st.subheader("📊 **Test Data Overview**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{test_df.shape[0]:,}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{test_df.shape[1]:,}</div>
                <div class="metric-label">Features</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            file_size_mb = uploaded.size / (1024 * 1024)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{file_size_mb:.1f}</div>
                <div class="metric-label">File Size (MB)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            missing_count = test_df.isnull().sum().sum()
            if missing_count > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">⚠️ {missing_count}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">✅ 0</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Data preview
        st.subheader("🔍 **Test Data Preview**")
        st.dataframe(test_df.head(10))
        
        # Check if target column exists
        has_target = target_col in test_df.columns
        
        if has_target:
            st.success(f"""
            ✅ **Target column '{target_col}' found in test data!**
            
            You can now evaluate your model's performance on this new data.
            """)
        else:
            st.info(f"""
            ℹ️ **Target column '{target_col}' not found in test data.**
            
            You can still make predictions, but won't be able to evaluate performance.
            """)
        
        # Testing options
        st.header("🧪 **Choose Testing Mode**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="test-card">
                <h4>🎯 Make Predictions</h4>
                <p>Get predictions for your test data without evaluation</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Make Predictions", type="primary", use_container_width=True):
                try:
                    # Prepare features (exclude target if present)
                    if has_target:
                        X_test = test_df.drop(columns=[target_col])
                    else:
                        X_test = test_df
                    
                    # Make predictions
                    with st.spinner("Making predictions..."):
                        predictions = st.session_state.model.predict(X_test)
                    
                    st.success("✅ **Predictions completed successfully!**")
                    
                    # Display predictions
                    st.subheader("📊 **Prediction Results**")
                    
                    # Create results dataframe
                    results_df = test_df.copy()
                    results_df['Prediction'] = predictions
                    
                    # Show first few predictions
                    st.write("**First 10 predictions:**")
                    st.dataframe(results_df[['Prediction'] + [col for col in test_df.columns if col != target_col]].head(10))
                    
                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name="model_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"""
                    ❌ **Prediction failed: {str(e)}**
                    
                    **🔧 Common solutions:**
                    - Ensure test data has the same columns as training data
                    - Check that data types match (numeric vs categorical)
                    - Verify no missing values in feature columns
                    - Make sure data is properly formatted
                    """)
        
        with col2:
            if has_target:
                st.markdown("""
                <div class="test-card">
                    <h4>📊 Evaluate Performance</h4>
                    <p>Test your model's accuracy on new data</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 Evaluate Model", type="secondary", use_container_width=True):
                    try:
                        with st.spinner("Evaluating model performance..."):
                            # Prepare test data
                            X_test = test_df.drop(columns=[target_col])
                            y_test = test_df[target_col]
                            
                            # Make predictions
                            y_pred = st.session_state.model.predict(X_test)
                            
                            # Evaluate performance
                            if hasattr(st.session_state.model, 'predict_proba'):
                                # Classification model
                                metrics = evaluate_classification(st.session_state.model, X_test, y_test)
                            else:
                                # Regression model
                                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                metrics = {
                                    'mse': mean_squared_error(y_test, y_pred),
                                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                    'mae': mean_absolute_error(y_test, y_pred),
                                    'r2': r2_score(y_test, y_pred)
                                }
                        
                        st.success("✅ **Model evaluation completed successfully!**")
                        
                        # Display evaluation results
                        st.subheader("📈 **Performance on Test Data**")
                        
                        # Show key metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="test-card">
                                <h4>📊 Key Metrics</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display appropriate metrics
                            if 'accuracy' in metrics:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            if 'precision' in metrics:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                            if 'recall' in metrics:
                                st.metric("Recall", f"{metrics['recall']:.4f}")
                            if 'f1_score' in metrics:
                                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                            if 'r2' in metrics:
                                st.metric("R² Score", f"{metrics['r2']:.4f}")
                            if 'rmse' in metrics:
                                st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        
                        with col2:
                            st.markdown("""
                            <div class="test-card">
                                <h4>🔍 Detailed Results</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show all metrics
                            with st.expander("📋 View All Metrics", expanded=True):
                                st.json(metrics)
                        
                        # Performance insights
                        st.subheader("💡 **Performance Insights**")
                        
                        if 'accuracy' in metrics:
                            accuracy = metrics['accuracy']
                            if accuracy >= 0.9:
                                st.success("🎉 **Excellent performance!** Your model is highly accurate on new data.")
                            elif accuracy >= 0.8:
                                st.info("👍 **Good performance!** Your model generalizes well to new data.")
                            elif accuracy >= 0.7:
                                st.warning("⚠️ **Moderate performance.** Consider improving your model or data.")
                            else:
                                st.error("❌ **Poor performance.** Your model may be overfitting or the data quality is low.")
                        
                        # Store results for comparison
                        st.session_state.test_metrics = metrics
                        st.session_state.test_predictions = y_pred
                        
                    except Exception as e:
                        st.error(f"""
                        ❌ **Evaluation failed: {str(e)}**
                        
                        **🔧 Common solutions:**
                        - Ensure test data has the same structure as training data
                        - Check that all required columns are present
                        - Verify data types and formats match
                        - Handle any missing values in the test data
                        """)
            else:
                st.info("""
                ℹ️ **Evaluation requires target column**
                
                To evaluate performance, your test data must include the target column '{target_col}'.
                """)
        
    except Exception as e:
        st.error(f"""
        ❌ **Failed to read test file: {str(e)}**
        
        **🔧 Common solutions:**
        - Make sure your file is a valid CSV
        - Check if the file isn't corrupted
        - Verify the file has proper headers
        - Ensure the file size isn't too large
        """)

# Navigation section
if "model" in st.session_state:
    st.header("🚀 **What's Next?**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🚀 Deploy Your Model**")
        st.write("Use your tested model to make predictions on new data")
        if st.button("🚀 Deploy Model", type="primary", use_container_width=True):
            safe_switch_page("pages/6_🚀_Deploy_Model.py")
    
    with col2:
        st.info("**⚙️ Improve Your Model**")
        st.write("Go back to training to try different parameters or algorithms")
        if st.button("⚙️ Back to Training", type="secondary", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")

else:
    # Getting started guide
    st.info("""
    📋 **Getting Started Guide**
    
    **Step 1:** Train a model on the Training page
    **Step 2:** Upload test data in the same format as training data
    **Step 3:** Choose whether to make predictions or evaluate performance
    **Step 4:** Review results and insights
    **Step 5:** Move to deployment or model improvement
    """) 