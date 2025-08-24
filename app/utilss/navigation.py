# app/utilss/navigation.py
import streamlit as st
import os
from typing import Optional, List, Tuple

def safe_switch_page(page_path: str) -> bool:
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
            # Enhanced fallback for older Streamlit versions
            with st.container():
                st.warning("⚠️ **Page Navigation Not Available**")
                
                # Create an enhanced info box
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                    border-left: 5px solid #2196f3;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
                ">
                    <h4 style="color: #1565c0; margin: 0 0 0.5rem 0;">
                        📍 Manual Navigation Required
                    </h4>
                    <p style="color: #1976d2; margin: 0.5rem 0;">
                        <strong>Target Page:</strong> <code>{}</code>
                    </p>
                    <p style="color: #424242; margin: 0.5rem 0;">
                        Please manually navigate to the target page using the sidebar or URL bar.
                    </p>
                    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        💡 <strong>Tip:</strong> Update Streamlit to version 1.27.0+ for automatic page switching.
                    </p>
                </div>
                """.format(page_path), unsafe_allow_html=True)
                
            return False
            
    except Exception as e:
        # Enhanced error display
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            border-left: 5px solid #f44336;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(244, 67, 54, 0.15);
        ">
            <h4 style="color: #c62828; margin: 0 0 0.5rem 0;">
                ❌ Navigation Error
            </h4>
            <p style="color: #d32f2f; margin: 0.5rem 0;">
                <strong>Error:</strong> {}
            </p>
            <p style="color: #424242; margin: 0.5rem 0;">
                <strong>Target Page:</strong> <code>{}</code>
            </p>
        </div>
        """.format(str(e), page_path), unsafe_allow_html=True)
        return False


def create_navigation_buttons(
    prev_page: Optional[str] = None, 
    next_page: Optional[str] = None, 
    home_page: str = "app.py"
) -> None:
    """
    Create sleek, React-style navigation buttons with enhanced styling and animations.
    
    Args:
        prev_page (str, optional): Path to previous page
        next_page (str, optional): Path to next page
        home_page (str): Path to home page (default: "app.py")
    """
    # Enhanced CSS with modern design principles
    st.markdown("""
        <style>
        .nav-container {
            display: flex;
            gap: 1rem;
            justify-content: center;
            align-items: center;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .nav-button-wrapper {
            flex: 1;
            min-width: 120px;
            max-width: 200px;
        }
        
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #2d3748 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.5rem !important;
            text-transform: none !important;
            letter-spacing: 0.025em !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2) !important;
            border-color: #667eea !important;
            background: linear-gradient(145deg, #667eea, #764ba2) !important;
            color: white !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .nav-button-prev {
            background: linear-gradient(145deg, #fef7ed, #fed7aa) !important;
            border-color: #f59e0b !important;
            color: #92400e !important;
        }
        
        .nav-button-prev:hover {
            background: linear-gradient(145deg, #f59e0b, #d97706) !important;
            color: white !important;
        }
        
        .nav-button-home {
            background: linear-gradient(145deg, #ecfdf5, #d1fae5) !important;
            border-color: #10b981 !important;
            color: #047857 !important;
        }
        
        .nav-button-home:hover {
            background: linear-gradient(145deg, #10b981, #059669) !important;
            color: white !important;
        }
        
        .nav-button-next {
            background: linear-gradient(145deg, #eff6ff, #dbeafe) !important;
            border-color: #3b82f6 !important;
            color: #1d4ed8 !important;
        }
        
        .nav-button-next:hover {
            background: linear-gradient(145deg, #3b82f6, #2563eb) !important;
            color: white !important;
        }
        
        @media (max-width: 768px) {
            .nav-container {
                flex-direction: column;
            }
            .nav-button-wrapper {
                max-width: 100%;
                width: 100%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Calculate column layout
    cols = []
    if prev_page:
        cols.append(1)
    cols.append(1)  # Home button always present
    if next_page:
        cols.append(1)

    col_list = st.columns(cols, gap="large")
    col_idx = 0

    def render_enhanced_button(col, label: str, icon: str, page_path: str, button_type: str = "default"):
        """Render an enhanced navigation button with custom styling"""
        with col:
            button_key = f"nav_{button_type}_{label.lower().replace(' ', '_')}_{hash(page_path)}"
            
            # Add custom CSS class based on button type
            if button_type == "prev":
                st.markdown('<div class="nav-button-wrapper nav-button-prev-wrapper">', unsafe_allow_html=True)
            elif button_type == "home":
                st.markdown('<div class="nav-button-wrapper nav-button-home-wrapper">', unsafe_allow_html=True)
            elif button_type == "next":
                st.markdown('<div class="nav-button-wrapper nav-button-next-wrapper">', unsafe_allow_html=True)
            else:
                st.markdown('<div class="nav-button-wrapper">', unsafe_allow_html=True)
                
            if st.button(f"{icon} {label}", key=button_key, use_container_width=True):
                safe_switch_page(page_path)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # Render buttons with enhanced styling
    if prev_page:
        render_enhanced_button(col_list[col_idx], "Previous", "⬅️", prev_page, "prev")
        col_idx += 1

    render_enhanced_button(col_list[col_idx], "Home", "🏠", home_page, "home")
    col_idx += 1

    if next_page:
        render_enhanced_button(col_list[col_idx], "Next", "➡️", next_page, "next")


def create_sidebar_navigation() -> None:
    """
    Create a modern, React-inspired sidebar with enhanced styling and smooth interactions.
    """
    # Enhanced sidebar CSS with modern design
    st.sidebar.markdown("""
        <style>
        .sidebar-container {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 10px;
            text-align: center;
            justify-content: center;
        }
        
        .sidebar-nav-section {
            margin-bottom: 2rem;
        }
        
        .sidebar-nav-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #374151 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
            text-align: left !important;
            justify-content: flex-start !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }
        
        .stButton > button:hover {
            transform: translateX(4px) !important;
            background: linear-gradient(145deg, #667eea, #764ba2) !important;
            color: white !important;
            border-color: #667eea !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
        }
        
        .stButton > button:active {
            transform: translateX(2px) !important;
        }
        
        .sidebar-divider {
            margin: 2rem 0;
            border: none;
            border-top: 2px solid #e2e8f0;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
            height: 1px;
        }
        
        .sidebar-tips {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 12px;
            padding: 1.25rem;
            border-left: 4px solid #0ea5e9;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
        }
        
        .sidebar-tips h4 {
            color: #0c4a6e;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .sidebar-tips ul {
            margin: 0;
            padding-left: 1.25rem;
            color: #0369a1;
            line-height: 1.6;
        }
        
        .sidebar-tips li {
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .sidebar-tips li:last-child {
            margin-bottom: 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            background: #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .workflow-progress {
            background: linear-gradient(135deg, #fefce8, #fef3c7);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #f59e0b;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
        }
        
        .progress-title {
            color: #92400e;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #fed7aa;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #f59e0b, #d97706);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        </style>
    """, unsafe_allow_html=True)

    # Enhanced sidebar content
    st.sidebar.markdown("""
    <div class="sidebar-container">
        <div class="sidebar-title">
            <span class="status-indicator"></span>
            🧭 InstaML Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation pages with enhanced structure
    page_groups = [
        ("Data Workflow", [
            ("🏠 Dashboard", "app.py"),
            ("📂 Data Upload", "pages/1_📂_Data_Upload.py"),
            ("🔧 Preprocessing", "pages/2_🔧_Data_Preprocessing.py"),
            ("📊 EDA", "pages/3_📊_EDA.py"),
        ]),
        ("ML Pipeline", [
            ("⚙️ Train Model", "pages/4_⚙️_Train_Model.py"),
            ("🧪 Test Model", "pages/5_🧪_Test_Model.py"),
            ("🚀 Deploy Model", "pages/6_🚀_Deploy_Model.py"),
        ])
    ]

    for group_name, pages in page_groups:
        st.sidebar.markdown(f"""
        <div class="sidebar-nav-section">
            <div class="sidebar-nav-header">{group_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        for page_name, page_path in pages:
            button_key = f"nav_sidebar_{page_path.replace('/', '_').replace('.py', '')}"
            if st.sidebar.button(page_name, key=button_key, use_container_width=True):
                safe_switch_page(page_path)

    # Enhanced divider and tips section
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # Dynamic progress indicator (if data is available)
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        progress_steps = [
            ("Data Loaded", True),
            ("Preprocessed", hasattr(st.session_state, 'df_preprocessed') and st.session_state.df_preprocessed is not None),
            ("Model Trained", st.session_state.get("model_trained", False)),
            ("Deployed", st.session_state.get("model_deployed", False))
        ]
        
        completed_steps = sum(1 for _, completed in progress_steps if completed)
        progress_percentage = (completed_steps / len(progress_steps)) * 100
        
        st.sidebar.markdown(f"""
        <div class="workflow-progress">
            <div class="progress-title">Workflow Progress ({completed_steps}/{len(progress_steps)})</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_percentage}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced tips section
    st.sidebar.markdown("""
    <div class="sidebar-tips">
        <h4>💡 Pro Tips</h4>
        <ul>
            <li>Save your progress frequently using the built-in persistence</li>
            <li>Check data quality metrics before proceeding to training</li>
            <li>Use EDA to understand your data patterns and relationships</li>
            <li>Compare multiple models for best performance</li>
            <li>Test thoroughly before deploying to production</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def get_current_page_name() -> str:
    """
    Get the current page name from the URL or session state with enhanced detection.
    
    Returns:
        str: Current page name or "Unknown Page" if not detectable
    """
    try:
        # Try to get from session state first
        if 'current_page' in st.session_state:
            return st.session_state.current_page
            
        # Try to get from query params (newer Streamlit versions)
        if hasattr(st, 'query_params'):
            query_params = st.query_params
            if 'page' in query_params:
                return query_params['page']
        
        # Fallback to script runner context (if available)
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx and hasattr(ctx, 'page_script_hash'):
                # This is a rough approximation
                return "Current Page"
        except ImportError:
            pass
            
        return "Dashboard"
        
    except Exception:
        return "Unknown Page"


def set_current_page(page_name: str) -> None:
    """
    Set the current page name in session state with validation.
    
    Args:
        page_name (str): Name of the current page
    """
    try:
        if isinstance(page_name, str) and page_name.strip():
            st.session_state.current_page = page_name.strip()
        else:
            st.session_state.current_page = "Unknown Page"
    except Exception:
        # Fail silently to avoid breaking the app
        pass


def create_breadcrumb_navigation(current_page: str, page_hierarchy: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Create a breadcrumb navigation component.
    
    Args:
        current_page (str): Current page name
        page_hierarchy (List[Tuple[str, str]], optional): List of (page_name, page_path) tuples
    """
    if not page_hierarchy:
        page_hierarchy = [
            ("Home", "app.py"),
            (current_page, "")
        ]
    
    st.markdown("""
    <style>
    .breadcrumb-container {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .breadcrumb {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .breadcrumb-item {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    
    .breadcrumb-item:hover {
        color: #5a67d8;
    }
    
    .breadcrumb-separator {
        color: #94a3b8;
        font-size: 0.8rem;
    }
    
    .breadcrumb-current {
        color: #374151;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    breadcrumb_html = '<div class="breadcrumb-container"><nav class="breadcrumb">'
    
    for i, (page_name, page_path) in enumerate(page_hierarchy):
        if i > 0:
            breadcrumb_html += '<span class="breadcrumb-separator">›</span>'
            
        if i == len(page_hierarchy) - 1 or not page_path:
            # Current page (no link)
            breadcrumb_html += f'<span class="breadcrumb-current">{page_name}</span>'
        else:
            # Clickable page
            breadcrumb_html += f'<span class="breadcrumb-item" onclick="window.location.href=\'{page_path}\'">{page_name}</span>'
    
    breadcrumb_html += '</nav></div>'
    st.markdown(breadcrumb_html, unsafe_allow_html=True)