# app/pages/3_📊_EDA.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.utilss import charts
from app.utilss.navigation import safe_switch_page

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration for a wider layout and a more professional look
st.set_page_config(layout="wide", page_title="EDA", page_icon="📊")

# Custom CSS for a cleaner, more modern look
st.markdown("""
<style>
/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}

/* Header and subheader styling */
h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #4B4B4B;
}
h2 {
    font-size: 2rem;
    font-weight: 600;
    color: #4F84C4;
    border-bottom: 2px solid #F0F2F6;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
}
h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #6C757D;
    margin-top: 1.5rem;
}

/* Info and Warning boxes */
.stAlert {
    border-left: 5px solid;
    border-radius: 5px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Metric styling */
[data-testid="stMetric"] {
    background-color: #F8F9FA;
    border: 1px solid #E9ECEF;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

[data-testid="stMetricLabel"] > div {
    font-size: 1.2rem;
    font-weight: 600;
    color: #495057;
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #343A40 !important;
}

/* Button styling */
.stButton>button {
    font-weight: bold;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
.stButton>button.primary {
    background-color: #4F84C4;
    color: white;
}
.stButton>button.secondary {
    background-color: #ADB5BD;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 Exploratory Data Analysis (EDA)")

# Comprehensive guidance in an expander
with st.expander("🎯 **What is EDA and Why Should You Care?**"):
    st.markdown("""
    **Exploratory Data Analysis (EDA)** is like being a detective investigating your data! It's the process of exploring, visualizing, and understanding your dataset before building machine learning models.

    **🔍 What EDA helps you discover:**
    - **Patterns and trends** hidden in your data
    - **Relationships** between different variables
    - **Data quality issues** that might affect your models
    - **Insights** that guide your modeling decisions
    - **Outliers and anomalies** that need attention

    **⚡ Why EDA is crucial for ML success:**
    - **Better model selection**: Understand your data to choose the right algorithms
    - **Feature engineering**: Discover which variables are most important
    - **Data validation**: Catch errors before they ruin your models
    - **Business insights**: Find valuable patterns for decision-making
    - **Model interpretation**: Understand why your models make certain predictions

    **🚨 What happens if you skip EDA:**
    - Choose wrong algorithms for your data
    - Miss important features or relationships
    - Build models on poor quality data
    - Get misleading results and insights
    - Waste time on ineffective approaches
    """)


if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload or load data first from the Data Upload page.")
    st.stop()

df = st.session_state.df

# Enhanced Data Status Check
st.markdown("## 📋 **Data Status Check**")
with st.container():
    if "df_preprocessed" in st.session_state:
        st.success("✅ **Your data has been preprocessed and is ready for analysis!**")
        with st.expander("See what this means"):
            st.markdown("""
            **What this means:**
            - Your data is clean and well-structured
            - Missing values have been handled
            - Data types are properly formatted
            - You're ready for comprehensive analysis
            """)
    else:
        st.warning("⚠️ **Data Preprocessing Recommended**")
        with st.expander("Why you should preprocess first"):
            st.markdown("""
            - Clean data reveals true patterns, not noise
            - Proper formatting prevents analysis errors
            - Better insights lead to better models
            - You'll save time in the long run
            """)
        st.markdown("💡 **What to do:** Go to the preprocessing page to clean and transform your data.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Go to Preprocessing", type="primary", use_container_width=True):
                safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
        with col2:
            if st.button("📊 Continue with EDA", type="secondary", use_container_width=True):
                st.info("ℹ️ You can continue, but results may be affected by data quality issues.")


# Enhanced Data Summary with explanations and metrics
st.markdown("## 📊 **Dataset Overview - Know Your Data**")
st.info("These metrics give you a quick understanding of your dataset. More details in the expander below.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", f"{df.shape[1]:,}")
with col3:
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    st.metric("Memory", f"{memory_mb:.2f} MB")
with col4:
    missing_count = df.isnull().sum().sum()
    missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
    st.metric("Missing Data", f"{missing_percentage:.1f}%")

with st.expander("More details on these metrics"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if df.shape[0] < 100: st.warning("⚠️ Small dataset")
        elif df.shape[0] < 1000: st.info("📊 Medium dataset")
        else: st.success("🚀 Large dataset")
        
    with col2:
        if df.shape[1] < 5: st.info("📊 Few features")
        elif df.shape[1] < 20: st.success("✅ Good feature count")
        else: st.warning("⚠️ Many features")
        
    with col3:
        if memory_mb < 10: st.success("✅ Small file")
        elif memory_mb < 100: st.info("📊 Medium file")
        else: st.warning("⚠️ Large file")
        
    with col4:
        if missing_percentage == 0: st.success("✅ No missing data")
        elif missing_percentage < 5: st.info("📊 Low missing data")
        else: st.warning("⚠️ High missing data")

# Separator
st.markdown("---")

# Enhanced Analysis Type Selection with guidance
st.markdown("## 🔍 **Choose Your Analysis Approach**")
st.info("Each analysis type is a different 'microscope' for your data. Choose one to get started.")

analysis_type = st.radio(
    "Select analysis type:",
    ["Univariate", "Bivariate", "Multivariate", "Dimensionality Reduction", "Time Series & Specialized"],
    horizontal=True,
    help="Choose based on what you want to discover about your data"
)

st.markdown("---")

# Detailed explanations for each analysis type inside a consistent layout
if analysis_type == "Univariate":
    st.subheader("📊 **Univariate Analysis**")
    with st.expander("What to look for in Univariate Analysis"):
        st.markdown("""
        **What this reveals:**
        - **Distribution patterns**: How values are spread out
        - **Central tendencies**: What's typical or average
        - **Variability**: How much values differ from each other
        
        **💡 Key insights to find:**
        - **Normal distribution**: Bell-shaped curves are good for many ML algorithms
        - **Skewed data**: Asymmetric distributions might need transformation
        - **Outliers**: Extreme values that might be errors or important signals
        """)
    charts.univariate_analysis(df)

elif analysis_type == "Bivariate":
    st.subheader("🔗 **Bivariate Analysis**")
    with st.expander("What to look for in Bivariate Analysis"):
        st.markdown("""
        **What this reveals:**
        - **Correlations**: How two variables move together
        - **Causal relationships**: Potential cause-and-effect connections
        - **Feature importance**: Which variables might predict your target
        
        **💡 Key insights to find:**
        - **Strong correlations**: Variables that move together (positive or negative)
        - **No correlation**: Independent variables that don't affect each other
        - **Non-linear relationships**: Curved patterns that correlation might miss
        """)
    charts.bivariate_analysis(df)

elif analysis_type == "Multivariate":
    st.subheader("🌐 **Multivariate Analysis**")
    with st.expander("What to look for in Multivariate Analysis"):
        st.markdown("""
        **What this reveals:**
        - **Interaction effects**: How multiple variables work together
        - **Hidden patterns**: Complex relationships not visible in simpler analyses
        - **Feature combinations**: Which groups of variables are most important
        
        **💡 Key insights to find:**
        - **Interaction effects**: Variables that work together differently than alone
        - **Feature groups**: Clusters of related variables
        - **Redundancy**: Variables that provide similar information
        """)
    charts.multivariate_analysis(df)

elif analysis_type == "Dimensionality Reduction":
    st.subheader("📉 **Dimensionality Reduction**")
    with st.expander("What to look for in Dimensionality Reduction"):
        st.markdown("""
        **What this reveals:**
        - **Hidden structure**: Underlying patterns in high-dimensional data
        - **Feature importance**: Which variables contribute most to patterns
        - **Data visualization**: Making complex data easier to understand
        
        **💡 Key insights to find:**
        - **Principal components**: New variables that capture most variation
        - **Explained variance**: How much information each component preserves
        - **Clustering patterns**: Natural groups in your data
        """)
    charts.dimensionality_reduction(df)

elif analysis_type == "Time Series & Specialized":
    st.subheader("⏰ **Time Series & Specialized Analysis**")
    with st.expander("What to look for in Time Series Analysis"):
        st.markdown("""
        **What this reveals:**
        - **Temporal patterns**: How data changes over time
        - **Seasonality**: Repeating patterns (daily, weekly, yearly)
        - **Trends**: Long-term changes or directions
        
        **💡 Key insights to find:**
        - **Trends**: Overall direction of change over time
        - **Seasonality**: Regular repeating patterns
        - **Anomalies**: Unusual time periods or events
        """)
    charts.time_series_analysis(df)

# Separator
st.markdown("---")

# Enhanced navigation and next steps with a clear, two-column layout
st.markdown("## 🚀 **What's Next After EDA?**")
st.info("You've explored your data. Now choose your next step.")

col1, col2 = st.columns(2)
with col1:
    with st.container():
        st.subheader("⚙️ **Train Model**")
        st.markdown("Ready to build a model? Your EDA insights will help you choose the right algorithm and features.")
        if st.button("⚙️ Train Model", type="primary", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")

with col2:
    with st.container():
        st.subheader("🔧 **Refine Preprocessing**")
        st.markdown("Did EDA reveal data quality issues? Go back to preprocessing to clean and transform your data.")
        if st.button("🔧 Go to Preprocessing", type="primary", use_container_width=True):
            safe_switch_page("pages/2_🔧_Data_Preprocessing.py")

# Separator
st.markdown("---")

# Add EDA best practices and tips in a collapsed section
with st.expander("💡 **EDA Best Practices & Tips**"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 Analysis Strategy:**")
        st.markdown("""
        • **Start simple**: Begin with basic statistics and distributions
        • **Ask questions**: What patterns do you expect to find?
        • **Document insights**: Keep notes of what you discover
        • **Iterate**: Go back and forth between different analysis types
        """)
    with col2:
        st.markdown("**🚨 Common Mistakes:**")
        st.markdown("""
        • **Rushing**: Take time to understand your data
        • **Ignoring outliers**: They might be important signals
        • **Missing context**: Understand what your variables mean
        • **Over-analyzing**: Don't get lost in endless exploration
        """)

# Final encouragement in a visually appealing box
st.success("""
🎉 **You're doing great! EDA is a crucial skill that will make you a better data scientist.**
""")