# app/app.py
import streamlit as st
import pandas as pd
# from streamlit_option_menu import option_menu
from app.utilss.navigation import safe_switch_page

# Initialize session state variables if they don't exist
if "df" not in st.session_state:
    st.session_state.df = None
if "df_preprocessed" not in st.session_state:
    st.session_state.df_preprocessed = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = None
if "model_deployed" not in st.session_state:
    st.session_state.model_deployed = None

# Page configuration
st.set_page_config(
    page_title="InstaML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .nav-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    .nav-card h3 {
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .nav-card p {
        color: #666;
        margin-bottom: 1rem;
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
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    .progress-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin: 0;">🚀 InstaML</h1>
    <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9;">No-Code Machine Learning Platform</p>
    <p style="font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.8;">Transform your data into insights with AI</p>
</div>
""", unsafe_allow_html=True)

# Navigation cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/1_📂_Data_Upload.py'">
        <h3>📂 Data Upload</h3>
        <p>Upload your datasets and get started with machine learning</p>
        <span class="status-badge status-info">Step 1</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/2_🔧_Data_Preprocessing.py'">
        <h3>🔧 Data Preprocessing</h3>
        <p>Clean, transform, and prepare your data for analysis</p>
        <span class="status-badge status-info">Step 2</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/3_📊_EDA.py'">
        <h3>📊 Exploratory Analysis</h3>
        <p>Discover patterns and insights in your data</p>
        <span class="status-badge status-info">Step 3</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/4_⚙️_Train_Model.py'">
        <h3>⚙️ Train Model</h3>
        <p>Build and train machine learning models</p>
        <span class="status-badge status-info">Step 4</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/5_🧪_Test_Model.py'">
        <h3>🧪 Test Model</h3>
        <p>Evaluate and validate your model performance</p>
        <span class="status-badge status-info">Step 5</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-card" onclick="window.location.href='pages/6_🚀_Deploy_Model.py'">
        <h3>🚀 Deploy Model</h3>
        <p>Deploy your model for real-world use</p>
        <span class="status-badge status-info">Step 6</span>
    </div>
    """, unsafe_allow_html=True)

# Progress section
if "df" in st.session_state and st.session_state.df is not None:
    st.markdown("""
    <div class="progress-section">
        <h3 >🎯 Your Progress</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">✅</div>
            <div class="metric-label">Data Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if "df_preprocessed" in st.session_state:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">✅</div>
                <div class="metric-label">Preprocessed</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">⏳</div>
                <div class="metric-label">Preprocessing</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if "model_trained" in st.session_state:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">✅</div>
                <div class="metric-label">Model Trained</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">⏳</div>
                <div class="metric-label">Training</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if "model_deployed" in st.session_state:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">✅</div>
                <div class="metric-label">Deployed</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">⏳</div>
                <div class="metric-label">Deployment</div>
            </div>
            """, unsafe_allow_html=True)

# Quick start section
st.markdown("""
<div class="progress-section">
    <h3 style="color: black;">🚀 Quick Start</h3>
    <p style="color: black;">Ready to begin? Click on any step above to get started, or use the quick navigation below:</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📂 Start with Data Upload", type="primary", use_container_width=True):
        safe_switch_page("pages/1_📂_Data_Upload.py")

with col2:
    if st.button("📊 Jump to EDA", use_container_width=True):
        safe_switch_page("pages/3_📊_EDA.py")

with col3:
    if st.button("⚙️ Train Model", use_container_width=True):
        safe_switch_page("pages/4_⚙️_Train_Model.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | InstaML Platform</p>
</div>
""", unsafe_allow_html=True)
