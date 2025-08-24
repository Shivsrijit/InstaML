import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards
import streamlit_extras as extras

def apply_modern_theme():
    """Apply modern React-like theme to the entire app"""
    st.markdown("""
    <style>
        /* Import Google Fonts for better typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Theme Variables */
        :root {
            --primary-color: #3b82f6;
            --primary-dark: #1d4ed8;
            --primary-light: #60a5fa;
            --secondary-color: #8b5cf6;
            --accent-color: #f59e0b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-bg: #0f172a;
            --sidebar-bg: #1e293b;
            --card-bg: #ffffff;
            --surface-bg: #f8fafc;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --border-light: #f1f5f9;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            --border-radius: 8px;
            --border-radius-lg: 12px;
            --border-radius-xl: 16px;
            --spacing-xs: 0.5rem;
            --spacing-sm: 0.75rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-2xl: 3rem;
        }
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .main {
            background: var(--surface-bg);
            padding: var(--spacing-lg);
        }
        
        .stApp {
            background: var(--surface-bg);
        }
        
        /* Sidebar Styling - Modern Dark Theme */
        .css-1d391kg {
            background: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color);
        }
        
        .css-1d391kg .css-1lcbmhc {
            background: transparent !important;
        }
        
        /* Main Content Area */
        .main .block-container {
            background: transparent;
            padding: var(--spacing-xl);
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Modern Header with Better Layout */
        .modern-header {
            background: var(--card-bg);
            border-radius: var(--border-radius-xl);
            padding: var(--spacing-2xl);
            margin-bottom: var(--spacing-2xl);
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--border-light);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .modern-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }
        
        .modern-header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin: 0 0 var(--spacing-sm) 0;
            color: var(--text-primary);
            letter-spacing: -0.025em;
        }
        
        .modern-header p {
            font-size: 1.125rem;
            margin: 0;
            color: var(--text-secondary);
            font-weight: 400;
            line-height: 1.6;
        }
        
        /* Modern Card System */
        .modern-card {
            background: var(--card-bg);
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-xl);
            margin: var(--spacing-lg) 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-light);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .modern-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            transform: scaleY(0);
            transition: transform 0.2s ease;
        }
        
        .modern-card:hover::before {
            transform: scaleY(1);
        }
        
        .modern-card:hover {
            box-shadow: var(--shadow-xl);
            transform: translateY(-2px);
        }
        
        .modern-card h3 {
            color: var(--text-primary);
            margin: 0 0 var(--spacing-sm) 0;
            font-weight: 600;
            font-size: 1.25rem;
            line-height: 1.4;
        }
        
        .modern-card p {
            color: var(--text-secondary);
            margin: 0;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        
        /* Dashboard Cards with Better Layout */
        .dashboard-card {
            background: var(--card-bg);
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-xl);
            margin: var(--spacing-md) 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-light);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .dashboard-card::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 0 24px 24px 0;
            border-color: transparent var(--primary-color) transparent transparent;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        
        .dashboard-card:hover::after {
            opacity: 1;
        }
        
        .dashboard-card:hover {
            box-shadow: var(--shadow-xl);
            transform: translateY(-2px);
        }
        
        /* Metric Cards with Better Typography */
        .metric-card {
            background: var(--card-bg);
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-xl);
            text-align: center;
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-md);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transform: scaleX(0);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover::before {
            transform: scaleX(1);
        }
        
        .metric-card:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: var(--spacing-xs);
            line-height: 1;
            letter-spacing: -0.025em;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Navigation Cards with Better Layout */
        .nav-card {
            background: var(--card-bg);
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-xl);
            margin: var(--spacing-md) 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-light);
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 160px;
        }
        
        .nav-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            transform: scaleY(0);
            transition: transform 0.2s ease;
        }
        
        .nav-card:hover::before {
            transform: scaleY(1);
        }
        
        .nav-card:hover {
            box-shadow: var(--shadow-xl);
            transform: translateY(-3px);
            border-color: var(--primary-color);
        }
        
        .nav-card h3 {
            color: var(--text-primary);
            margin: 0 0 var(--spacing-sm) 0;
            font-weight: 600;
            font-size: 1.25rem;
            line-height: 1.4;
        }
        
        .nav-card p {
            color: var(--text-secondary);
            margin: 0 0 var(--spacing-md) 0;
            line-height: 1.6;
            font-size: 0.95rem;
            flex-grow: 1;
        }
        
        /* Status Badges with Better Design */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            box-shadow: var(--shadow-sm);
            border: 1px solid transparent;
        }
        
        .status-success {
            background: #dcfce7;
            color: #166534;
            border-color: #bbf7d0;
        }
        
        .status-info {
            background: #dbeafe;
            color: #1e40af;
            border-color: #bfdbfe;
        }
        
        .status-warning {
            background: #fef3c7;
            color: #92400e;
            border-color: #fde68a;
        }
        
        .status-danger {
            background: #fee2e2;
            color: #991b1b;
            border-color: #fecaca;
        }
        
        /* Button Styles with Better Design */
        .stButton > button {
            border-radius: var(--border-radius);
            font-weight: 500;
            font-size: 0.875rem;
            text-transform: none;
            letter-spacing: 0;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            padding: var(--spacing-sm) var(--spacing-lg);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        /* Form Elements with Better Design */
        .stSelectbox, .stTextInput, .stNumberInput, .stTextArea {
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .stSelectbox:focus, .stTextInput:focus, .stNumberInput:focus, .stTextArea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1);
        }
        
        .stSelectbox > div > div {
            border-radius: var(--border-radius);
        }
        
        /* Progress Bars with Better Design */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            border-radius: var(--border-radius);
        }
        
        /* Dataframe Styling */
        .dataframe {
            border-radius: var(--border-radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-light);
        }
        
        /* Chart Containers */
        .chart-container {
            background: var(--card-bg);
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-lg);
            margin: var(--spacing-lg) 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-light);
        }
        
        /* Upload Area with Better Design */
        .upload-area {
            background: var(--surface-bg);
            border: 2px dashed var(--border-color);
            border-radius: var(--border-radius-xl);
            padding: var(--spacing-2xl);
            text-align: center;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgb(59 130 246 / 0.05) 50%, transparent 70%);
            transform: translateX(-100%);
            transition: transform 0.6s;
        }
        
        .upload-area:hover::before {
            transform: translateX(100%);
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
            background: var(--card-bg);
            transform: scale(1.01);
        }
        
        .upload-area h3 {
            color: var(--text-primary);
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0 0 var(--spacing-sm) 0;
        }
        
        .upload-area p {
            color: var(--text-secondary);
            margin: 0;
            font-size: 0.95rem;
        }
        
        /* Info Boxes with Better Design */
        .info-box {
            background: #dbeafe;
            border: 1px solid #bfdbfe;
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-lg);
            margin: var(--spacing-lg) 0;
            border-left: 4px solid var(--primary-color);
            position: relative;
            overflow: hidden;
        }
        
        .info-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--primary-color);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .success-box {
            background: #dcfce7;
            border: 1px solid #bbf7d0;
            border-left: 4px solid var(--success-color);
        }
        
        .warning-box {
            background: #fef3c7;
            border: 1px solid #fde68a;
            border-left: 4px solid var(--warning-color);
        }
        
        .error-box {
            background: #fee2e2;
            border: 1px solid #fecaca;
            border-left: 4px solid var(--danger-color);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .modern-header h1 {
                font-size: 2rem;
            }
            
            .metric-value {
                font-size: 2rem;
            }
            
            .modern-card, .dashboard-card, .nav-card {
                padding: var(--spacing-lg);
            }
            
            .main .block-container {
                padding: var(--spacing-lg);
            }
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--surface-bg);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        
        /* Loading Animation */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Floating Action Button */
        .fab {
            position: fixed;
            bottom: var(--spacing-xl);
            right: var(--spacing-xl);
            width: 56px;
            height: 56px;
            background: var(--primary-color);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            box-shadow: var(--shadow-xl);
            cursor: pointer;
            transition: all 0.2s ease;
            z-index: 1000;
            border: none;
        }
        
        .fab:hover {
            transform: scale(1.1);
            box-shadow: var(--shadow-xl);
            background: var(--primary-dark);
        }
        
        /* Section Headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: var(--spacing-xl) 0 var(--spacing-lg) 0;
            padding-bottom: var(--spacing-sm);
            border-bottom: 2px solid var(--border-light);
        }
        
        /* Grid Layout */
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: var(--spacing-lg);
            margin: var(--spacing-lg) 0;
        }
        
        /* Card Grid */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: var(--spacing-lg);
            margin: var(--spacing-lg) 0;
        }
    </style>
    """, unsafe_allow_html=True)

def modern_header(title, subtitle="", icon="🚀"):
    """Create a modern header with better layout"""
    st.markdown(f"""
    <div class="modern-header">
        <h1>{icon} {title}</h1>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def modern_card(title, content, icon="📋", status=None):
    """Create a modern card with better layout"""
    status_html = f'<span class="status-badge status-{status}">{status.upper()}</span>' if status else ""
    st.markdown(f"""
    <div class="modern-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
        {status_html}
    </div>
    """, unsafe_allow_html=True)

def modern_metric_card(value, label, icon="📊"):
    """Create a modern metric card with better typography"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def modern_nav_card(title, description, step, icon="📋"):
    """Create a modern navigation card with better layout"""
    st.markdown(f"""
    <div class="nav-card">
        <div>
            <h3>{icon} {title}</h3>
            <p>{description}</p>
        </div>
        <span class="status-badge status-info">Step {step}</span>
    </div>
    """, unsafe_allow_html=True)

def modern_upload_area():
    """Create a modern upload area with better design"""
    st.markdown("""
    <div class="upload-area">
        <h3>📁 Drop your file here or click to browse</h3>
        <p>Support for CSV, Excel, Parquet, and HDF5 files</p>
    </div>
    """, unsafe_allow_html=True)

def modern_info_box(message, type="info"):
    """Create a modern info box with better design"""
    st.markdown(f"""
    <div class="{type}-box">
        {message}
    </div>
    """, unsafe_allow_html=True)

def modern_table_container():
    """Create a modern table container"""
    return st.container()

def modern_chart_container():
    """Create a modern chart container"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    container = st.container()
    st.markdown('</div>', unsafe_allow_html=True)
    return container

def create_metric_row(metrics_data):
    """Create a row of metric cards with better layout"""
    cols = st.columns(len(metrics_data))
    for i, (value, label, icon) in enumerate(metrics_data):
        with cols[i]:
            modern_metric_card(value, label, icon)

def create_nav_grid(nav_items):
    """Create a grid of navigation cards with better layout"""
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (title, description, step, icon) in enumerate(nav_items):
        with cols[i % 2]:
            modern_nav_card(title, description, step, icon)
    st.markdown('</div>', unsafe_allow_html=True)

def modern_sidebar():
    """Create a modern sidebar style"""
    st.markdown("""
    <style>
        .css-1d391kg {
            background: var(--sidebar-bg);
        }
    </style>
    """, unsafe_allow_html=True)

def dashboard_card(title, value, trend=None, icon="📊"):
    """Create a dashboard-style metric card with better layout"""
    trend_html = ""
    if trend:
        trend_color = "var(--success-color)" if trend > 0 else "var(--danger-color)"
        trend_arrow = "↗" if trend > 0 else "↘"
        trend_html = f'<div style="color: {trend_color}; font-size: 0.875rem; margin-top: var(--spacing-xs); font-weight: 500;">{trend_arrow} {abs(trend):.2f}%</div>'
    
    st.markdown(f"""
    <div class="dashboard-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: var(--spacing-xs); font-weight: 500;">{title}</div>
                <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1;">{value}</div>
                {trend_html}
            </div>
            <div style="font-size: 2rem; opacity: 0.7; margin-left: var(--spacing-md);">{icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_dashboard_grid(dashboard_data):
    """Create a grid of dashboard cards with better layout"""
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    cols = st.columns(len(dashboard_data))
    for i, (title, value, trend, icon) in enumerate(dashboard_data):
        with cols[i]:
            dashboard_card(title, value, trend, icon)
    st.markdown('</div>', unsafe_allow_html=True)

def section_header(title):
    """Create a modern section header"""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
