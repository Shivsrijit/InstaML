# app/app.py
import streamlit as st
import pandas as pd
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from app.utilss.navigation import safe_switch_page
from core.persistent_storage import persistent_storage, load_progress, save_progress

# Initialize session state variables if they don't exist
if "df" not in st.session_state:
    st.session_state.df = None
if "df_preprocessed" not in st.session_state:
    st.session_state.df_preprocessed = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = None
if "model_deployed" not in st.session_state:
    st.session_state.model_deployed = None

# Load persistent data on app startup
@st.cache_resource
def load_persistent_data():
    """Load persistent data on app startup."""
    session_data = load_progress()
    
    df = persistent_storage.load_data()
    if df is not None:
        st.session_state.df = df
    
    preprocessed_df = persistent_storage.load_preprocessed_data()
    if preprocessed_df is not None:
        st.session_state.df_preprocessed = preprocessed_df
    
    model, metrics = persistent_storage.load_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.metrics = metrics
    
    return session_data

session_data = load_persistent_data()

# Page configuration with dark theme
st.set_page_config(
    page_title="InstaML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern React-like styling with FontAwesome
st.markdown("""
<style>
    /* Import Google Fonts and FontAwesome */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
        opacity: 0.3;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        margin: 0.5rem 0;
        opacity: 0.9;
        font-weight: 400;
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced Card styling */
    .workflow-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: none;
        border-radius: 20px;
        padding: 2rem;
        margin: 0.8rem 0;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.05),
            0 10px 25px rgba(0, 0, 0, 0.08),
            0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.08);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .workflow-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 0 3px 3px 0;
        transition: width 0.3s ease;
    }
    
    .workflow-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.03) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .workflow-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.15),
            0 20px 50px rgba(102, 126, 234, 0.1),
            0 4px 15px rgba(0, 0, 0, 0.05);
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    .workflow-card:hover::before {
        width: 8px;
    }
    
    .workflow-card:hover::after {
        opacity: 1;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }
    
    .card-icon {
        font-size: 2rem;
        margin-right: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        transition: transform 0.3s ease;
    }
    
    .workflow-card:hover .card-icon {
        transform: scale(1.1);
    }
    
    .card-title {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.5rem;
        margin: 0;
        flex: 1;
    }
    
    .card-step-badge {
        background: linear-gradient(135deg, #e2e8f0, #cbd5e0);
        color: #4a5568;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: auto;
        transition: all 0.3s ease;
    }
    
    .workflow-card:hover .card-step-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transform: scale(1.05);
    }
    
    .card-description {
        color: #718096;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        flex: 1;
        position: relative;
        z-index: 2;
    }
    
    .card-features {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
        position: relative;
        z-index: 2;
    }
    
    .card-features li {
        color: #718096;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .card-features li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #48bb78;
        font-weight: bold;
    }
    
    .card-button-container {
        margin-top: auto;
        position: relative;
        z-index: 2;
    }
    
    /* Icon styling */
    .nav-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .status-completed {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
    }
    
    .status-pending {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
        color: white;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.3);
    }
    
    .status-ready {
        background: linear-gradient(135deg, #4299e1, #3182ce);
        color: white;
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
    }
    
    /* Progress section */
    .progress-container {
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .progress-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #718096;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Quick actions section */
    .quick-actions {
        background: linear-gradient(135deg, #e6fffa, #f0fff4);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(72, 187, 120, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    
    /* Success/info/warning styling */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main container
with stylable_container(
    key="main_container",
    css_styles="""
    {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    """
):
    # Dashboard Header
    with stylable_container(
        key="dashboard_header",
        css_styles="""
        {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        """
    ):
        st.markdown("""
        <div class="header-title"><i class="fas fa-rocket nav-icon"></i>InstaML</div>
        <div class="header-subtitle">No-Code Machine Learning Platform</div>
        <p style="font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.8;">Transform your data into insights with AI</p>
        """, unsafe_allow_html=True)

    # Progress restoration section
    if session_data and session_data.get("data_loaded"):
        with stylable_container(
            key="progress_restoration",
            css_styles="""
            {
                background: linear-gradient(135deg, #e6fffa, #f0fff4);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                border: 1px solid rgba(72, 187, 120, 0.2);
            }
            """
        ):
            st.success("""
            **Progress Restored!** 
            
            Your previous session has been loaded. You can continue from where you left off.
            """)
            
            # Progress metrics
            my_grid = grid(4, vertical_align="center")
            
            if session_data.get("data_loaded"):
                my_grid.info(f"**Data Loaded**\n{session_data.get('data_shape', 'Unknown shape')}")
            
            if session_data.get("preprocessed"):
                my_grid.success("**Preprocessed**\nData ready for training")
            else:
                my_grid.warning("**Not Preprocessed**\nConsider preprocessing first")
            
            if session_data.get("model_trained"):
                my_grid.success("**Model Trained**\nReady for testing/deployment")
            else:
                my_grid.info("**No Model**\nReady for training")
            
            if session_data.get("model_deployed"):
                my_grid.success("**Deployed**\nModel is live")
            else:
                my_grid.info("**Not Deployed**\nReady for deployment")

        # Quick actions
        colored_header(
            label="Quick Actions",
            description="Jump to any step in your ML workflow",
            color_name="blue-70",
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Continue with EDA", use_container_width=True):
                safe_switch_page("pages/3_📊_EDA.py")
        
        with col2:
            if st.button("Continue Training", use_container_width=True):
                safe_switch_page("pages/4_⚙️_Train_Model.py")
        
        with col3:
            if st.button("Start Fresh", use_container_width=True):
                if persistent_storage.clear_all_data():
                    st.success("All data cleared! Starting fresh...")
                    st.rerun()  # ✅ Fixed: Changed from st.experimental_rerun() to st.rerun()
        
        st.markdown("---")

    # Navigation section
    colored_header(
        label="ML Workflow",
        description="Follow these steps to build your machine learning solution",
        color_name="violet-70",
    )

    # Navigation cards in 3x2 grid layout for better spacing
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        # Data Upload Card
        with stylable_container(
            key="nav_card_1",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-upload card-icon"></i>
                <h3 class="card-title">Data Upload</h3>
                <span class="card-step-badge">Step 1</span>
            </div>
            <div class="card-description">
                Upload your datasets and get started with machine learning
            </div>
            <ul class="card-features">
                <li>CSV, Excel, JSON support</li>
                <li>Auto data type detection</li>
                <li>Instant data preview</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start Upload", key="nav_upload", use_container_width=True):
                safe_switch_page("pages/1_📂_Data_Upload.py")
        
        # Model Training Card
        with stylable_container(
            key="nav_card_4",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-brain card-icon"></i>
                <h3 class="card-title">Train Model</h3>
                <span class="card-step-badge">Step 4</span>
            </div>
            <div class="card-description">
                Build and train machine learning models
            </div>
            <ul class="card-features">
                <li>Multiple algorithms</li>
                <li>Auto parameter tuning</li>
                <li>Performance metrics</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start Training", key="nav_train", use_container_width=True):
                safe_switch_page("pages/4_⚙️_Train_Model.py")

    with col2:
        # Data Preprocessing Card
        with stylable_container(
            key="nav_card_2",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-cogs card-icon"></i>
                <h3 class="card-title">Preprocessing</h3>
                <span class="card-step-badge">Step 2</span>
            </div>
            <div class="card-description">
                Clean, transform, and prepare your data for analysis
            </div>
            <ul class="card-features">
                <li>Handle missing values</li>
                <li>Feature scaling</li>
                <li>Data transformation</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start Preprocessing", key="nav_preprocess", use_container_width=True):
                safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
        
        # Model Testing Card
        with stylable_container(
            key="nav_card_5",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-flask card-icon"></i>
                <h3 class="card-title">Test Model</h3>
                <span class="card-step-badge">Step 5</span>
            </div>
            <div class="card-description">
                Evaluate and validate your model performance
            </div>
            <ul class="card-features">
                <li>Cross validation</li>
                <li>Performance analysis</li>
                <li>Model comparison</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start Testing", key="nav_test", use_container_width=True):
                safe_switch_page("pages/5_🧪_Test_Model.py")

    with col3:
        # EDA Card
        with stylable_container(
            key="nav_card_3",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-chart-bar card-icon"></i>
                <h3 class="card-title">EDA</h3>
                <span class="card-step-badge">Step 3</span>
            </div>
            <div class="card-description">
                Discover patterns and insights in your data
            </div>
            <ul class="card-features">
                <li>Statistical summaries</li>
                <li>Data visualization</li>
                <li>Correlation analysis</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start EDA", key="nav_eda", use_container_width=True):
                safe_switch_page("pages/3_📊_EDA.py")
        
        # Model Deployment Card
        with stylable_container(
            key="nav_card_6",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                margin: 0.8rem 0;
                box-shadow: 
                    0 4px 6px rgba(0, 0, 0, 0.05),
                    0 10px 25px rgba(0, 0, 0, 0.08);
                transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
                cursor: pointer;
                position: relative;
                border: 1px solid rgba(102, 126, 234, 0.08);
                height: 280px;
                display: flex;
                flex-direction: column;
            }
            """
        ):
            st.markdown("""
            <div class="card-header">
                <i class="fas fa-rocket card-icon"></i>
                <h3 class="card-title">Deploy</h3>
                <span class="card-step-badge">Step 6</span>
            </div>
            <div class="card-description">
                Deploy your model for real-world use
            </div>
            <ul class="card-features">
                <li>Real-time predictions</li>
                <li>API integration</li>
                <li>Model monitoring</li>
            </ul>
            """, unsafe_allow_html=True)
            
            if st.button("Start Deployment", key="nav_deploy", use_container_width=True):
                safe_switch_page("pages/6_🚀_Deploy_Model.py")

    # Progress section
    if "df" in st.session_state and st.session_state.df is not None:
        colored_header(
            label="Your Progress",
            description="Track your machine learning workflow progress",
            color_name="green-70",
        )
        
        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with stylable_container(
                key="metric_1",
                css_styles="""
                {
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid rgba(72, 187, 120, 0.2);
                }
                """
            ):
                st.markdown(
                    """
                    <div class="metric-value">
                        <i class="fas fa-check-circle" style="color: #48bb78;"></i>
                    </div>
                    <div class="metric-label">Data Loaded</div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            status_icon = "fas fa-check-circle" if ("df_preprocessed" in st.session_state and st.session_state.df_preprocessed is not None) else "fas fa-clock"
            status_color = "#48bb78" if ("df_preprocessed" in st.session_state and st.session_state.df_preprocessed is not None) else "#ed8936"
            border_color = "rgba(72, 187, 120, 0.2)" if ("df_preprocessed" in st.session_state and st.session_state.df_preprocessed is not None) else "rgba(237, 137, 54, 0.2)"
            
            with stylable_container(
                key="metric_2",
                css_styles=f"""
                {{
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid {border_color};
                }}
                """
            ):
                st.markdown(f"""
                <div class="metric-value"><i class="{status_icon}" style="color: {status_color};"></i></div>
                <div class="metric-label">Preprocessed</div>
                """, unsafe_allow_html=True)
        
        with col3:
            status_icon = "fas fa-check-circle" if st.session_state.get("model_trained") else "fas fa-clock"
            status_color = "#48bb78" if st.session_state.get("model_trained") else "#4299e1"
            border_color = "rgba(72, 187, 120, 0.2)" if st.session_state.get("model_trained") else "rgba(66, 153, 225, 0.2)"
            
            with stylable_container(
                key="metric_3",
                css_styles=f"""
                {{
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid {border_color};
                }}
                """
            ):
                st.markdown(f"""
                <div class="metric-value"><i class="{status_icon}" style="color: {status_color};"></i></div>
                <div class="metric-label">Model Trained</div>
                """, unsafe_allow_html=True)
        
        with col4:
            status_icon = "fas fa-check-circle" if st.session_state.get("model_deployed") else "fas fa-clock"
            status_color = "#48bb78" if st.session_state.get("model_deployed") else "#4299e1"
            border_color = "rgba(72, 187, 120, 0.2)" if st.session_state.get("model_deployed") else "rgba(66, 153, 225, 0.2)"
            
            with stylable_container(
                key="metric_4",
                css_styles=f"""
                {{
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid {border_color};
                }}
                """
            ):
                st.markdown(f"""
                <div class="metric-value"><i class="{status_icon}" style="color: {status_color};"></i></div>
                <div class="metric-label">Deployed</div>
                """, unsafe_allow_html=True)

    # Quick start section
    colored_header(
        label="Quick Start",
        description="Ready to begin? Jump to any step or start from the beginning",
        color_name="blue-green-70",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Start with Data Upload", type="primary", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")

    with col2:
        if st.button("Jump to EDA", use_container_width=True):
            safe_switch_page("pages/3_📊_EDA.py")

    with col3:
        if st.button("Train Model", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")

    # Footer
    st.markdown("---")
    with stylable_container(
        key="footer",
        css_styles="""
        {
            text-align: center;
            color: #718096;
            padding: 2rem;
            background: linear-gradient(135deg, #f7fafc, #edf2f7);
            border-radius: 15px;
            margin-top: 2rem;
        }
        """
    ):
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: 500;">
            <p><i class="fas fa-rocket"></i> InstaML Platform | Built for ML Lovers</p>
        </div>
        """, unsafe_allow_html=True)