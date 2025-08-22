# app/utilss/ui_helpers.py
import streamlit as st
import pandas as pd
from .navigation import safe_switch_page

def centered_button(label, key=None):
    """Create a centered button with consistent styling"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        return st.button(label, key=key, use_container_width=True)

def metric_card(title, value, subtitle="", status="info"):
    """Create a styled metric card"""
    status_colors = {
        "success": "#d4edda",
        "warning": "#fff3cd", 
        "error": "#f8d7da",
        "info": "#d1ecf1"
    }
    
    status_borders = {
        "success": "#c3e6cb",
        "warning": "#ffeaa7",
        "error": "#f5c6cb", 
        "info": "#bee5eb"
    }
    
    status_text = {
        "success": "#155724",
        "warning": "#856404",
        "error": "#721c24",
        "info": "#0c5460"
    }
    
    st.markdown(f"""
    <div style="
        background: {status_colors.get(status, status_colors['info'])};
        border: 1px solid {status_borders.get(status, status_borders['info'])};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    ">
        <div style="
            font-size: 1.5rem;
            font-weight: bold;
            color: {status_text.get(status, status_text['info'])};
            margin-bottom: 0.5rem;
        ">{value}</div>
        <div style="
            font-size: 1rem;
            font-weight: 600;
            color: {status_text.get(status, status_text['info'])};
            margin-bottom: 0.25rem;
        ">{title}</div>
        {f'<div style="font-size: 0.8rem; color: #666;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def progress_bar(current_step, total_steps, labels=None):
    """Create a progress bar showing current step"""
    if labels is None:
        labels = [f"Step {i+1}" for i in range(total_steps)]
    
    progress = (current_step + 1) / total_steps
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="
            background: #e9ecef;
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                height: 100%;
                width: {progress * 100}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #666;
        ">
            {''.join([f'<span>{label}</span>' for label in labels])}
        </div>
    </div>
    """, unsafe_allow_html=True)

def info_card(title, content, icon="ℹ️", color="#d1ecf1"):
    """Create an info card with icon and content"""
    st.markdown(f"""
    <div style="
        background: {color};
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        ">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <strong>{title}</strong>
        </div>
        <div style="color: #0c5460;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def success_card(title, content, icon="✅"):
    """Create a success card"""
    return info_card(title, content, icon, "#d4edda")

def warning_card(title, content, icon="⚠️"):
    """Create a warning card"""
    return info_card(title, content, icon, "#fff3cd")

def error_card(title, content, icon="❌"):
    """Create an error card"""
    return info_card(title, content, icon, "#f8d7da")

def step_card(title, content, step_number=None, expanded=True):
    """Create a collapsible step card"""
    if step_number:
        title = f"**Step {step_number}:** {title}"
    
    with st.expander(title, expanded=expanded):
        st.markdown(content)

def data_quality_summary(df):
    """Display a summary of data quality metrics"""
    if df is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Total Rows", f"{df.shape[0]:,}", "Samples")
    
    with col2:
        metric_card("Total Columns", f"{df.shape[1]:,}", "Features")
    
    with col3:
        missing_count = df.isnull().sum().sum()
        status = "success" if missing_count == 0 else "warning"
        metric_card("Missing Values", f"{missing_count:,}", "Data points", status)
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        metric_card("Memory Usage", f"{memory_mb:.2f} MB", "Storage")

def navigation_buttons():
    """Create navigation buttons for moving between pages"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")
    
    with col2:
        if st.button("🏠 Home", use_container_width=True):
            safe_switch_page("app.py")
    
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            safe_switch_page("pages/3_📊_EDA.py")

def create_sidebar_nav():
    """Create a navigation sidebar"""
    st.sidebar.title("🧭 Navigation")
    
    pages = [
        ("🏠 Home", "app.py"),
        ("📂 Data Upload", "pages/1_📂_Data_Upload.py"),
        ("🔧 Data Preprocessing", "pages/2_🔧_Data_Preprocessing.py"),
        ("📊 EDA", "pages/3_📊_EDA.py"),
        ("⚙️ Train Model", "pages/4_⚙️_Train_Model.py"),
        ("🧪 Test Model", "pages/5_🧪_Test_Model.py"),
        ("🚀 Deploy Model", "pages/6_🚀_Deploy_Model.py")
    ]
    
    for page_name, page_path in pages:
        if st.sidebar.button(page_name, use_container_width=True):
            safe_switch_page(page_path)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Quick Tips**")
    st.sidebar.info("Use the expandable sections for detailed explanations")
    st.sidebar.info("Save your work frequently")
    st.sidebar.info("Check data quality before proceeding")

def status_indicator(status, text):
    """Create a status indicator with colored dot"""
    colors = {
        "success": "#28a745",
        "warning": "#ffc107", 
        "error": "#dc3545",
        "info": "#17a2b8"
    }
    
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    ">
        <div style="
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {colors.get(status, colors['info'])};
            margin-right: 0.5rem;
        "></div>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)
