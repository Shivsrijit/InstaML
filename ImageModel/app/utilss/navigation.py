# app/utilss/navigation.py
import streamlit as st
import os

def safe_switch_page(page_path):
    """
    Safely switch pages with backward compatibility for older Streamlit versions.
    
    Args:
        page_path (str): Path to the target page
        
    Returns:
        bool: True if navigation was successful, False otherwise
    """
    try:
        # Try the modern st.switch_page() method (Streamlit >= 1.27.0)
        if hasattr(st, 'switch_page'):
            st.switch_page(page_path)
            return True
        else:
            # Fallback for older Streamlit versions
            st.warning("⚠️ **Page Navigation Not Available**")
            st.info(f"""
            **Target Page:** {page_path}
            
            **Manual Navigation Required:**
            Please manually navigate to the target page using the sidebar or URL.
            
            **💡 Tip:** Update Streamlit to version 1.27.0+ for automatic page switching.
            """)
            
            # Show the target path for manual navigation
            st.code(f"Navigate to: {page_path}", language="bash")
            return False
            
    except Exception as e:
        st.error(f"❌ **Navigation Error:** {str(e)}")
        st.info(f"**Target Page:** {page_path}")
        return False

def create_navigation_buttons(prev_page=None, next_page=None, home_page="app.py"):
    """
    Create navigation buttons that work with all Streamlit versions.
    
    Args:
        prev_page (str, optional): Path to previous page
        next_page (str, optional): Path to next page  
        home_page (str): Path to home page
    """
    cols = []
    
    if prev_page:
        cols.append(1)
    cols.append(1)  # Home button
    if next_page:
        cols.append(1)
    
    if len(cols) == 1:
        cols = [1]
    elif len(cols) == 2:
        cols = [1, 1]
    else:
        cols = [1, 1, 1]
    
    col_list = st.columns(cols)
    col_idx = 0
    
    if prev_page:
        with col_list[col_idx]:
            if st.button("⬅️ Previous", use_container_width=True):
                safe_switch_page(prev_page)
        col_idx += 1
    
    with col_list[col_idx]:
        if st.button("🏠 Home", use_container_width=True):
            safe_switch_page(home_page)
    col_idx += 1
    
    if next_page:
        with col_list[col_idx]:
            if st.button("Next ➡️", use_container_width=True):
                safe_switch_page(next_page)

def create_sidebar_navigation():
    """
    Create a navigation sidebar that works with all Streamlit versions.
    """
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

def get_current_page_name():
    """
    Get the current page name from the URL or session state.
    """
    # Try to get from session state first
    if 'current_page' in st.session_state:
        return st.session_state.current_page
    
    # Try to infer from the current file path
    try:
        # This is a fallback method
        return "Unknown Page"
    except:
        return "Unknown Page"

def set_current_page(page_name):
    """
    Set the current page name in session state.
    """
    st.session_state.current_page = page_name 