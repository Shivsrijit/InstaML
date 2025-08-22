# app/pages/2_🔧_Data_Preprocessing.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.utilss.navigation import safe_switch_page

st.set_page_config(page_title="Data Preprocessing", layout="wide")

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
    
    .step-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .step-card h4 {
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
</style>
""", unsafe_allow_html=True)

st.title("🔧 Data Preprocessing & Cleaning")

# Collapsible help section
with st.expander("ℹ️ **What is Data Preprocessing and Why is it Critical?**", expanded=False):
    st.markdown("""
    Think of preprocessing as **cleaning and organizing your kitchen** before cooking a gourmet meal. Just like you wouldn't cook with dirty utensils or spoiled ingredients, you shouldn't train a machine learning model with messy data!
    
    **🔍 What happens during preprocessing:**
    - **Data Cleaning**: Remove errors, duplicates, and inconsistencies
    - **Missing Values**: Fill in or remove incomplete data
    - **Data Types**: Convert text to numbers, categories to codes
    - **Scaling**: Make all numbers comparable (like converting inches to centimeters)
    - **Encoding**: Convert text categories to numbers the computer can understand
    
    **⚡ Why this matters for your model:**
    - **Better Performance**: Clean data = More accurate predictions
    - **Faster Training**: Well-structured data trains faster
    - **Fewer Errors**: Proper formatting prevents crashes and bugs
    - **Better Insights**: Clean data reveals true patterns, not noise
    """)

# Check if data is loaded
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload or load data first from the Data Upload page.")
    st.stop()

# Store original dataframe
if "df_original" not in st.session_state:
    st.session_state.df_original = st.session_state.df.copy()

df = st.session_state.df.copy()
df_original = st.session_state.df_original

# Sidebar for preprocessing options
st.sidebar.header("🔧 Preprocessing Options")

# Collapsible step-by-step guidance
with st.sidebar.expander("📋 Recommended Order", expanded=False):
    st.markdown("""
    **📋 Recommended Order:**
    1. **🧹 Data Cleaning** - Remove duplicates, select columns
    2. **🔢 Missing Values** - Handle incomplete data
    3. **📏 Scaling & Encoding** - Prepare for ML algorithms
    4. **📈 Outlier Detection** - Find and handle extreme values
    5. **💾 Save** - Store your cleaned data
    
    **💡 Pro Tips:**
    - Always start with cleaning
    - Handle missing values before scaling
    - Check outliers after scaling
    - Save your work frequently
    - Test small changes first
    """)

# Data Overview
st.header("📊 Data Overview")
st.info("**This section shows you the current state of your data. Use these metrics to understand your dataset.**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[0]:,}</div>
        <div class="metric-label">Total Rows</div>
    </div>
    """, unsafe_allow_html=True)
    if df.shape[0] < 100:
        st.warning("⚠️ Small dataset")
    elif df.shape[0] < 1000:
        st.info("📊 Medium dataset")
    else:
        st.success("🚀 Large dataset")
        
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]:,}</div>
        <div class="metric-label">Total Columns</div>
    </div>
    """, unsafe_allow_html=True)
    if df.shape[1] < 5:
        st.info("📊 Few features")
    elif df.shape[1] < 20:
        st.success("🚀 Good feature count")
    else:
        st.warning("⚠️ Many features - consider selection")
        
with col3:
    missing_count = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{missing_count:,}</div>
        <div class="metric-label">Missing Values</div>
    </div>
    """, unsafe_allow_html=True)
    if missing_count > 0:
        st.warning("⚠️ Missing data detected")
    else:
        st.success("✅ No missing data")
        
with col4:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{memory_mb:.2f}</div>
        <div class="metric-label">Memory (MB)</div>
    </div>
    """, unsafe_allow_html=True)

# Data Cleaning Section
st.header("🧹 Data Cleaning")
with st.expander("**Step 1: Clean your data**", expanded=True):
    st.markdown("""
    <div class="step-card">
        <h4>🧹 Basic Cleaning</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Remove duplicates
        if st.button("🔍 Check for Duplicates"):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"Found {duplicate_count} duplicate rows")
                if st.button("🗑️ Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.session_state.df = df
                    st.success(f"Removed {duplicate_count} duplicate rows")
                    st.rerun()
            else:
                st.success("✅ No duplicates found")
        
        # Column selection
        st.subheader("📋 Select Columns")
        selected_columns = st.multiselect(
            "Choose columns to keep:",
            df.columns.tolist(),
            default=df.columns.tolist(),
            help="Select which columns you want to keep for analysis"
        )
        
        if st.button("✂️ Apply Column Selection"):
            df = df[selected_columns]
            st.session_state.df = df
            st.success(f"Selected {len(selected_columns)} columns")
            st.rerun()
    
    with col2:
        # Data types info
        st.subheader("🔢 Data Types")
        dtype_info = df.dtypes.value_counts()
        st.write(dtype_info)
        
        # Memory optimization
        if st.button("💾 Optimize Memory"):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
            st.session_state.df = df
            st.success("Memory optimized!")
            st.rerun()

# Missing Values Section
st.header("🔢 Missing Values")
with st.expander("**Step 2: Handle missing data**", expanded=True):
    st.markdown("""
    <div class="step-card">
        <h4>🔢 Missing Value Analysis</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Missing values summary
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    
    if len(missing_summary) > 0:
        st.warning(f"Found missing values in {len(missing_summary)} columns")
        
        # Show missing values chart
        fig = px.bar(
            x=missing_summary.index, 
            y=missing_summary.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing value handling
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Missing Value Strategy")
            strategy = st.selectbox(
                "Choose strategy:",
                ["Drop rows with missing values", "Fill with mean/median", "Fill with mode", "Forward fill"]
            )
        
        with col2:
            if strategy == "Fill with mean/median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fill_method = st.selectbox("Fill method:", ["mean", "median"])
                    if st.button("🔧 Apply Strategy"):
                        for col in numeric_cols:
                            if df[col].isnull().sum() > 0:
                                if fill_method == "mean":
                                    df[col].fillna(df[col].mean(), inplace=True)
                                else:
                                    df[col].fillna(df[col].median(), inplace=True)
                        st.session_state.df = df
                        st.success("Missing values filled!")
                        st.rerun()
            
            elif strategy == "Drop rows with missing values":
                if st.button("🗑️ Drop Missing Rows"):
                    df = df.dropna()
                    st.session_state.df = df
                    st.success("Rows with missing values dropped!")
                    st.rerun()
    else:
        st.success("✅ No missing values found!")

# Scaling and Encoding Section
st.header("📏 Scaling & Encoding")
with st.expander("**Step 3: Prepare data for ML**", expanded=True):
    st.markdown("""
    <div class="step-card">
        <h4>📏 Feature Scaling</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Numeric Scaling")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            scale_method = st.selectbox(
                "Scaling method:",
                ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (robust to outliers)"]
            )
            
            if st.button("📏 Apply Scaling"):
                if scale_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                elif scale_method == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.session_state.df = df
                st.success(f"Applied {scale_method}!")
                st.rerun()
        else:
            st.info("No numeric columns found for scaling")
    
    with col2:
        st.subheader("🔤 Categorical Encoding")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            encode_method = st.selectbox(
                "Encoding method:",
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            if st.button("🔤 Apply Encoding"):
                if encode_method == "Label Encoding":
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                else:
                    df = pd.get_dummies(df, columns=categorical_cols)
                
                st.session_state.df = df
                st.success(f"Applied {encode_method}!")
                st.rerun()
        else:
            st.info("No categorical columns found for encoding")

# Outlier Detection Section
st.header("📈 Outlier Detection")
with st.expander("**Step 4: Find and handle outliers**", expanded=True):
    st.markdown("""
    <div class="step-card">
        <h4>📈 Outlier Analysis</h4>
    </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
        
        if selected_col:
            # Box plot for outlier detection
            fig = px.box(df, y=selected_col, title=f"Outlier Detection - {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier statistics
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lower Bound", f"{lower_bound:.2f}")
            with col2:
                st.metric("Upper Bound", f"{upper_bound:.2f}")
            with col3:
                st.metric("Outliers Found", len(outliers))
            
            if len(outliers) > 0:
                if st.button("🗑️ Remove Outliers"):
                    df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                    st.session_state.df = df
                    st.success(f"Removed {len(outliers)} outliers!")
                    st.rerun()
    else:
        st.info("No numeric columns found for outlier detection")

# Save and Reset Section
st.header("💾 Save & Reset")
with st.expander("**Step 5: Save your work**", expanded=True):
    st.markdown("""
    <div class="step-card">
        <h4>💾 Save Preprocessed Data</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Preprocessed Data", type="primary"):
            st.session_state.df_preprocessed = df.copy()
            st.success("✅ Preprocessed data saved!")
    
    with col2:
        if st.button("🔄 Reset to Original"):
            df = df_original.copy()
            st.session_state.df = df
            st.success("✅ Reset to original data!")

# Navigation
st.header("🚀 What's Next?")
col1, col2 = st.columns(2)

with col1:
    st.info("**📊 Next Step: Exploratory Data Analysis**")
    st.write("Explore patterns and insights in your cleaned data")
    if st.button("📊 Go to EDA", type="primary", use_container_width=True):
        safe_switch_page("pages/3_📊_EDA.py")

with col2:
    st.info("**⚙️ Alternative: Train Model**")
    st.write("Start training your machine learning model")
    if st.button("⚙️ Train Model", type="primary", use_container_width=True):
        safe_switch_page("pages/4_⚙️_Train_Model.py") 