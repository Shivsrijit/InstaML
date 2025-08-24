# app/pages/2_🔧_Data_Preprocessing.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.utilss.navigation import safe_switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(layout="wide", page_title="Data Preprocessing", page_icon="🔧")

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
    
    .step-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; padding: 1rem; border-radius: 10px;
        margin-bottom: 1rem; font-weight: 600;
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
            <i class="fas fa-cogs nav-icon"></i>Data Preprocessing & Cleaning
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">Clean, transform, and prepare your data for analysis</div>
        """, unsafe_allow_html=True)

    # === Help Section ===
    with st.expander("What is Data Preprocessing and Why is it Critical?"):
        st.markdown("""
        Think of preprocessing as **cleaning and organizing your kitchen** before cooking a gourmet meal. 
        Just like you wouldn't cook with dirty utensils or spoiled ingredients, you shouldn't train a 
        machine learning model with messy data!
        
        **What happens during preprocessing:**
        - **Data Cleaning**: Remove errors, duplicates, and inconsistencies
        - **Missing Values**: Fill in or remove incomplete data
        - **Data Types**: Convert text to numbers, categories to codes
        - **Scaling**: Make all numbers comparable (like converting inches to centimeters)
        - **Encoding**: Convert text categories to numbers the computer can understand
        
        **Why this matters for your model:**
        - **Better Performance**: Clean data = More accurate predictions
        - **Faster Training**: Well-structured data trains faster
        - **Fewer Errors**: Proper formatting prevents crashes and bugs
        - **Better Insights**: Clean data reveals true patterns, not noise
        """)

    # === Data Check ===
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No data loaded. Please go to Data Upload.")
        if st.button("Go to Data Upload", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")
        st.stop()

    # Store original dataframe
    if "df_original" not in st.session_state:
        st.session_state.df_original = st.session_state.df.copy()

    df = st.session_state.df.copy()
    df_original = st.session_state.df_original

    # === Dataset Overview ===
    colored_header("Dataset Overview", "Current state of your data", "blue-70")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container("metric_rows", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[0]:,}</div><div class='metric-label'>Total Rows</div>", unsafe_allow_html=True)
            if df.shape[0] < 100:
                st.warning("Small dataset")
            elif df.shape[0] < 1000:
                st.info("Medium dataset")
            else:
                st.success("Large dataset")

    with col2:
        with stylable_container("metric_cols", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[1]:,}</div><div class='metric-label'>Total Columns</div>", unsafe_allow_html=True)
            if df.shape[1] < 5:
                st.info("Few features")
            elif df.shape[1] < 20:
                st.success("Good feature count")
            else:
                st.warning("Many features - consider selection")

    with col3:
        missing_count = df.isnull().sum().sum()
        with stylable_container("metric_missing", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{missing_count:,}</div><div class='metric-label'>Missing Values</div>", unsafe_allow_html=True)
            if missing_count > 0:
                st.warning("Missing data detected")
            else:
                st.success("No missing data")

    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        with stylable_container("metric_memory", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{memory_mb:.2f}</div><div class='metric-label'>Memory (MB)</div>", unsafe_allow_html=True)

    st.markdown("---")

    # === Data Cleaning Section ===
    colored_header("Data Cleaning", "Remove duplicates and select columns", "green-70")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Duplicate Detection")
        if st.button("Check for Duplicates"):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"Found {duplicate_count} duplicate rows")
                if st.button("Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.session_state.df = df
                    st.success(f"Removed {duplicate_count} duplicate rows")
                    st.rerun()
            else:
                st.success("No duplicates found")
        
        # Column selection
        st.subheader("Column Selection")
        selected_columns = st.multiselect(
            "Choose columns to keep:",
            df.columns.tolist(),
            default=df.columns.tolist(),
            help="Select which columns you want to keep for analysis"
        )
        
        if st.button("Apply Column Selection"):
            df = df[selected_columns]
            st.session_state.df = df
            st.success(f"Selected {len(selected_columns)} columns")
            st.rerun()
    
    with col2:
        # Data types info
        st.subheader("Data Types")
        dtype_info = df.dtypes.value_counts()
        fig = px.bar(x=dtype_info.index.astype(str), y=dtype_info.values, 
                     title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory optimization
        if st.button("Optimize Memory"):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
            st.session_state.df = df
            st.success("Memory optimized!")
            st.rerun()

    # === Missing Values Section ===
    colored_header("Missing Values", "Handle incomplete data", "orange-70")
    
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    
    if len(missing_summary) > 0:
        st.warning(f"Found missing values in {len(missing_summary)} columns")
        
        # Show missing values chart
        fig = px.bar(
            x=missing_summary.index, 
            y=missing_summary.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing value handling
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Value Strategy")
            strategy = st.selectbox(
                "Choose strategy:",
                ["Drop rows with missing values", "Fill with mean/median", "Fill with mode", "Forward fill"]
            )
        
        with col2:
            if strategy == "Fill with mean/median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fill_method = st.selectbox("Fill method:", ["mean", "median"])
                    if st.button("Apply Strategy"):
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
                if st.button("Drop Missing Rows"):
                    df = df.dropna()
                    st.session_state.df = df
                    st.success("Rows with missing values dropped!")
                    st.rerun()
    else:
        st.success("No missing values found!")

    # === Scaling and Encoding Section ===
    colored_header("Scaling & Encoding", "Prepare data for machine learning", "violet-70")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Scaling")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            scale_method = st.selectbox(
                "Scaling method:",
                ["StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (robust to outliers)"]
            )
            
            if st.button("Apply Scaling"):
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
        st.subheader("Categorical Encoding")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            encode_method = st.selectbox(
                "Encoding method:",
                ["Label Encoding", "One-Hot Encoding"]
            )
            
            if st.button("Apply Encoding"):
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

    # === Outlier Detection Section ===
    colored_header("Outlier Detection", "Find and handle extreme values", "red-70")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
        
        if selected_col:
            # Box plot for outlier detection
            fig = px.box(df, y=selected_col, title=f"Outlier Detection - {selected_col}",
                        color_discrete_sequence=['#667eea'])
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
                with stylable_container("outlier_lower", css_styles="""{ text-align:center; padding:1rem; }"""):
                    st.markdown(f"<div class='metric-value'>{lower_bound:.2f}</div><div class='metric-label'>Lower Bound</div>", unsafe_allow_html=True)
            with col2:
                with stylable_container("outlier_upper", css_styles="""{ text-align:center; padding:1rem; }"""):
                    st.markdown(f"<div class='metric-value'>{upper_bound:.2f}</div><div class='metric-label'>Upper Bound</div>", unsafe_allow_html=True)
            with col3:
                with stylable_container("outlier_count", css_styles="""{ text-align:center; padding:1rem; }"""):
                    st.markdown(f"<div class='metric-value'>{len(outliers)}</div><div class='metric-label'>Outliers Found</div>", unsafe_allow_html=True)
            
            if len(outliers) > 0:
                if st.button("Remove Outliers"):
                    df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                    st.session_state.df = df
                    st.success(f"Removed {len(outliers)} outliers!")
                    st.rerun()
    else:
        st.info("No numeric columns found for outlier detection")

    # === Save and Reset Section ===
    colored_header("Save & Reset", "Save your preprocessed data", "blue-70")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Preprocessed Data", type="primary", use_container_width=True):
            st.session_state.df_preprocessed = df.copy()
            st.success("Preprocessed data saved!")
    
    with col2:
        if st.button("Reset to Original", use_container_width=True):
            df = df_original.copy()
            st.session_state.df = df
            st.success("Reset to original data!")
            st.rerun()

    # === Final Navigation ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to EDA", use_container_width=True):
            safe_switch_page("pages/3_📊_EDA.py")
    with col2:
        if st.button("Train Model", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")