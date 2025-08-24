# app/pages/1_📂_Data_Upload.py
import streamlit as st
import pandas as pd
import os
import numpy as np
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.grid import grid
from app.utilss.navigation import safe_switch_page
from core.persistent_storage import auto_save_data, persistent_storage

# Try to import optional dependencies
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    st.warning("Excel support not available. Install with: `pip install openpyxl`")

try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    st.warning("Parquet support not available. Install with: `pip install pyarrow`")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    st.warning("HDF5 support not available. Install with: `pip install h5py`")

# Page configuration
st.set_page_config(page_title="Data Upload", layout="wide")

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
    
    /* Icon styling */
    .nav-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
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

    /* Custom button styling for image/audio buttons */
    .custom-data-button {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        display: block;
        color: inherit;
        margin: 1rem 0;
    }
    
    .custom-data-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
        border-color: #764ba2;
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    }
    
    .custom-data-button h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    .custom-data-button p {
        margin: 0;
        color: #718096;
        font-size: 1rem;
    }
    
    .custom-data-button i {
        font-size: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
    # Page Header
    with stylable_container(
        key="page_header",
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
        <div style="font-size: 3rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);">
            <i class="fas fa-upload nav-icon"></i>Data Upload
        </div>
        <div style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.9; font-weight: 400;">
            Upload your datasets and get started with machine learning
        </div>
        <p style="font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.8;">Transform your data into insights with AI</p>
        """, unsafe_allow_html=True)

    # Progress indicator
    if "df" in st.session_state and st.session_state.df is not None:
        colored_header(
            label="Current Progress",
            description="Your current workflow status",
            color_name="green-70",
        )
        
        # Progress metrics
        my_grid = grid(4, vertical_align="center")
        
        my_grid.info(f"**Data Loaded**\n{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns")
        my_grid.info(f"**Data Type**\n{st.session_state.get('data_type', 'Unknown')}")
        my_grid.success("**Preprocessed**\nYes" if st.session_state.get('df_preprocessed') is not None else "**Preprocessed**\nNo")
        my_grid.info("**Model Trained**\nYes" if st.session_state.get('model_trained') else "**Model Trained**\nNo")

    # Help section
    with st.expander("What is this step and why is it important?", expanded=False):
        st.markdown("""
        This is the **first and most crucial step** in your machine learning journey! Think of it as preparing the ingredients before cooking a meal.
        
        **Why data quality matters:**
        - **Garbage in = Garbage out**: Poor quality data leads to poor model performance
        - **Foundation**: Everything else depends on this data
        - **Efficiency**: Good data means faster training and better results
        """)

    # Special Data Type Buttons Section
    colored_header(
        label="Special Data Types",
        description="Work with specialized data formats",
        color_name="red-70",
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <a href="http://localhost:8503/" target="_blank" class="custom-data-button">
            <div>
                <i class="fas fa-image"></i>
                <h3>Work with Image Data</h3>
                <p>Train models on images, photos, and visual data</p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown("""
    #     <a href="https://localhost:8504" target="_blank" class="custom-data-button">
    #         <div>
    #             <i class="fas fa-music"></i>
    #             <h3>Work with Audio Data</h3>
    #             <p>Process sound, music, and audio signals</p>
    #         </div>
    #     </a>
    #     """, unsafe_allow_html=True)

    # Mode selection
    colored_header(
        label="Choose Your Data Source",
        description="Select how you want to load your data",
        color_name="blue-70",
    )
    
    mode = st.radio(
        "Select data source:", 
        ["Upload file", "Load from local path"],
        horizontal=True,
        key="data_source_mode_selection"
    )

    # --- Upload mode ---
    if mode == "Upload file":
        colored_header(
            label="Upload Your Data File",
            description="Drag and drop or click to browse",
            color_name="violet-70",
        )
        
        # Collapsible format info
        with st.expander("Supported Formats & Tips", expanded=False):
            st.markdown("""
            **Supported Formats:**
            - **CSV files** (Comma Separated Values) - Most common and recommended
            
            **Tips for best results:**
            - Make sure your CSV has headers (column names in the first row)
            - Avoid special characters in column names
            - Keep file size under 100MB for reliable uploads
            - Ensure your data is clean and well-structured
            """)
        
        # Upload area
        with stylable_container(
            key="upload_area",
            css_styles="""
            {
                background: linear-gradient(145deg, #ffffff, #f8f9fa);
                border: 2px dashed #667eea;
                border-radius: 20px;
                padding: 3rem;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                margin: 2rem 0;
            }
            {
                &:hover {
                    border-color: #764ba2;
                    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                    transform: scale(1.02);
                }
            }
            """
        ):
            st.markdown("""
            <h3 style="color: #2d3748; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">
                <i class="fas fa-cloud-upload-alt" style="margin-right: 0.5rem;"></i>Drop your data file here
            </h3>
            <p style="color: #718096; font-size: 1rem; margin: 0;">Supports CSV, Excel, JSON, images, audio, and more!</p>
            """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Upload data file", 
            type=["csv", "xlsx", "xls", "json", "parquet", "feather", "pickle", "pkl", 
                  "jpg", "jpeg", "png", "bmp", "tiff", "gif",
                  "wav", "mp3", "flac", "m4a", "aac", "ogg",
                  "npy", "npz", "h5", "hdf5"],
            label_visibility="collapsed",
            key="main_file_uploader"
        )
        
        if uploaded is not None:
            uploaded_size_mb = uploaded.size / (1024 * 1024)
            
            # File size guidance
            if uploaded_size_mb > 100:
                st.warning(f"""
                **Large File Warning** 
                
                Your file is **{uploaded_size_mb:.1f} MB**. This might cause slower processing.
                Consider using "Load from local path" for large files.
                """)
            elif uploaded_size_mb > 50:
                st.info(f"File size: {uploaded_size_mb:.1f} MB - Good size for analysis!")
            else:
                st.success(f"File size: {uploaded_size_mb:.1f} MB - Perfect size!")

            try:
                # Process different file types
                file_extension = uploaded.name.lower().split('.')[-1]
                
                if file_extension in ['csv']:
                    df = pd.read_csv(uploaded)
                    data_type = "tabular"
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded)
                    data_type = "tabular"
                elif file_extension in ['json']:
                    df = pd.read_json(uploaded)
                    data_type = "tabular"
                elif file_extension in ['parquet']:
                    df = pd.read_parquet(uploaded)
                    data_type = "tabular"
                elif file_extension in ['feather']:
                    df = pd.read_feather(uploaded)
                    data_type = "tabular"
                elif file_extension in ['pickle', 'pkl']:
                    df = pd.read_pickle(uploaded)
                    data_type = "tabular"
                elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
                    # For image files, we'll store the file path and mark as image data
                    df = None
                    data_type = "image"
                    st.session_state.image_file = uploaded
                elif file_extension in ['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg']:
                    # For audio files, we'll store the file path and mark as audio data
                    df = None
                    data_type = "audio"
                    st.session_state.audio_file = uploaded
                elif file_extension in ['npy', 'npz', 'h5', 'hdf5']:
                    # For numpy/array files, we'll load them differently
                    if file_extension == 'npy':
                        data = np.load(uploaded)
                        if len(data.shape) <= 2:
                            df = pd.DataFrame(data)
                            data_type = "tabular"
                        else:
                            df = None
                            data_type = "multi_dimensional"
                            st.session_state.array_data = data
                    else:
                        st.error(f"File type {file_extension} not yet supported. Please convert to CSV or other supported format.")
                        df = None
                        data_type = None
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    df = None
                    data_type = None
                
                # Store data and type in session state (only if successfully processed)
                if df is not None:
                    # Save to persistent storage
                    if auto_save_data(df, data_type):
                        st.session_state.df = df
                        st.session_state.data_type = data_type
                    else:
                        st.error("Failed to save data to persistent storage!")
                elif data_type is not None:
                    st.session_state.data_type = data_type
                
                # Show success and data info only if we successfully processed the file
                if data_type is not None:
                    # Success message
                    st.success(f"Successfully uploaded {uploaded.name}!")
                    st.info(f"**Data Type Detected:** {data_type.title()}")
                    
                    # Show different content based on data type
                    if data_type == "tabular" and df is not None:
                        # Data metrics
                        colored_header(
                            label="Data Overview",
                            description="Key metrics about your uploaded data",
                            color_name="blue-70",
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            with stylable_container(
                                key="metric_rows",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                st.markdown(f"""
                                <div class="metric-value">{df.shape[0]:,}</div>
                                <div class="metric-label">Rows (samples)</div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            with stylable_container(
                                key="metric_cols",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                st.markdown(f"""
                                <div class="metric-value">{df.shape[1]:,}</div>
                                <div class="metric-label">Columns (features)</div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            with stylable_container(
                                key="metric_size",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                st.markdown(f"""
                                <div class="metric-value">{uploaded_size_mb:.1f}</div>
                                <div class="metric-label">File Size (MB)</div>
                                """, unsafe_allow_html=True)
                        
                        # Data preview
                        colored_header(
                            label="Data Preview",
                            description="First 10 rows of your data",
                            color_name="violet-70",
                        )
                        
                        with stylable_container(
                            key="data_preview",
                            css_styles="""
                            {
                                background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                border-radius: 15px;
                                padding: 2rem;
                                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                border: 1px solid rgba(0, 0, 0, 0.05);
                            }
                            """
                        ):
                            st.dataframe(df.head(10))
                        
                        # Quick quality check
                        colored_header(
                            label="Quick Data Quality Check",
                            description="Quality metrics for your data",
                            color_name="green-70",
                        )
                        
                        missing_count = df.isnull().sum().sum()
                        duplicate_count = df.duplicated().sum()
                        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            with stylable_container(
                                key="quality_missing",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                status = "warning" if missing_count > 0 else "success"
                                icon = "fas fa-exclamation-triangle" if missing_count > 0 else "fas fa-check-circle"
                                color = "#ed8936" if missing_count > 0 else "#48bb78"
                                st.markdown(f"""
                                <div class="metric-value"><i class="{icon}" style="color: {color};"></i> {missing_count}</div>
                                <div class="metric-label">Missing Values</div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            with stylable_container(
                                key="quality_duplicates",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                status = "warning" if duplicate_count > 0 else "success"
                                icon = "fas fa-exclamation-triangle" if duplicate_count > 0 else "fas fa-check-circle"
                                color = "#ed8936" if duplicate_count > 0 else "#48bb78"
                                st.markdown(f"""
                                <div class="metric-value"><i class="{icon}" style="color: {color};"></i> {duplicate_count}</div>
                                <div class="metric-label">Duplicate Rows</div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            with stylable_container(
                                key="quality_memory",
                                css_styles="""
                                {
                                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                    border: 1px solid rgba(0, 0, 0, 0.05);
                                }
                                """
                            ):
                                st.markdown(f"""
                                <div class="metric-value">{memory_mb:.2f}</div>
                                <div class="metric-label">Memory Usage (MB)</div>
                                """, unsafe_allow_html=True)
                            
                    elif data_type == "image":
                        st.info("**Image file uploaded successfully!** For image training, please organize your images in folders with the following structure:\n\n```\nimages/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
                        
                    elif data_type == "audio":
                        st.info("**Audio file uploaded successfully!** For audio training, please organize your audio files in folders with the following structure:\n\n```\naudio/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
                        
                    elif data_type == "multi_dimensional":
                        st.info("**Multi-dimensional data uploaded successfully!** This data can be used for advanced ML models like CNNs, LSTMs, and Transformers.")
                    
            except Exception as e:
                st.error(f"""
                **Failed to read your file!**
                
                **Error:** {e}
                
                **Common solutions:**
                - Make sure your file is in a supported format
                - Check if the file isn't corrupted
                - For large files, try using "Load from local path"
                - For images/audio, ensure proper directory structure
                """)

    # --- Local path mode ---
    elif mode == "Load from local path":
        colored_header(
            label="Load Data from Your Computer",
            description="Load data from your local file system",
            color_name="orange-70",
        )
        
        # Collapsible info
        with st.expander("When to use this option", expanded=False):
            st.markdown("""
            **When to use this option:**
            - You have large files (>100MB)
            - You're working with the same dataset repeatedly
            - You want to avoid uploading the same file multiple times
            - You're working in a local development environment
            - You have image/audio directories that need to be loaded
            """)
        
        # File type selection
        file_type = st.selectbox(
            "Select file type:",
            ["CSV", "Excel", "JSON", "Parquet", "Image Directory", "Audio Directory", "Multi-dimensional Data"],
            help="Choose the type of data you want to load",
            key="local_file_type_selection"
        )
        
        if file_type == "CSV":
            default_path = "datasets/tabular/airlines_flights_data.csv"
            file_path = st.text_input(
                "Enter CSV file path:", 
                default_path,
                help="Type the full path to your CSV file",
                key="csv_file_path_input"
            )
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    # Save to persistent storage
                    if auto_save_data(df, "tabular"):
                        st.session_state.df = df
                        st.session_state.data_type = "tabular"
                    else:
                        st.error("Failed to save data to persistent storage!")
                    
                    # Get file size
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    
                    st.success(f"Successfully loaded from {file_path}!")
                    
                    # Data metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        with stylable_container(
                            key="local_metric_1",
                            css_styles="""
                            {
                                background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                border-radius: 15px;
                                padding: 2rem;
                                text-align: center;
                                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                border: 1px solid rgba(0, 0, 0, 0.05);
                            }
                            """
                        ):
                            st.markdown(f"""
                            <div class="metric-value">{df.shape[0]:,}</div>
                            <div class="metric-label">Rows (samples)</div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        with stylable_container(
                            key="local_metric_2",
                            css_styles="""
                            {
                                background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                border-radius: 15px;
                                padding: 2rem;
                                text-align: center;
                                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                border: 1px solid rgba(0, 0, 0, 0.05);
                            }
                            """
                        ):
                            st.markdown(f"""
                            <div class="metric-value">{df.shape[1]:,}</div>
                            <div class="metric-label">Columns (features)</div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        with stylable_container(
                            key="local_metric_3",
                            css_styles="""
                            {
                                background: linear-gradient(145deg, #ffffff, #f8f9fa);
                                border-radius: 15px;
                                padding: 2rem;
                                text-align: center;
                                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                                border: 1px solid rgba(0, 0, 0, 0.05);
                            }
                            """
                        ):
                            st.markdown(f"""
                            <div class="metric-value">{file_size:.2f}</div>
                            <div class="metric-label">File size (MB)</div>
                            """, unsafe_allow_html=True)
                    
                    # Data preview
                    colored_header(
                        label="Data Preview",
                        description="First 10 rows of your data",
                        color_name="violet-70",
                    )
                    
                    with stylable_container(
                        key="local_data_preview",
                        css_styles="""
                        {
                            background: linear-gradient(145deg, #ffffff, #f8f9fa);
                            border-radius: 15px;
                            padding: 2rem;
                            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                            border: 1px solid rgba(0, 0, 0, 0.05);
                        }
                        """
                    ):
                        st.dataframe(df.head(10))
                    
                except Exception as e:
                    st.error(f"Failed to load CSV file: {e}")

    # Navigation section
    # Initialize df in session state if it doesn't exist
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Check if we have valid data loaded
    def has_valid_data():
        """Check if we have valid data loaded in session state."""
        return (st.session_state.df is not None and 
                not st.session_state.df.empty and
                len(st.session_state.df.columns) > 0)

    # Show navigation only if we have valid data
    if has_valid_data():
        st.success("Great job! Your dataset is loaded and ready for the next step.")
        
        colored_header(
            label="What's Next?",
            description="Choose your next step in the ML workflow",
            color_name="blue-green-70",
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            with stylable_container(
                key="next_preprocessing",
                css_styles="""
                {
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }
                """
            ):
                st.info("**Next Step: Data Preprocessing**")
                st.write("Clean and prepare your data for analysis")
                if st.button("Go to Data Preprocessing", type="primary", use_container_width=True, key="nav_to_preprocessing"):
                    safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
        
        with col2:
            with stylable_container(
                key="next_eda",
                css_styles="""
                {
                    background: linear-gradient(145deg, #ffffff, #f8f9fa);
                    border-radius: 15px;
                    padding: 2rem;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }
                """
            ):
                st.info("**Alternative: Skip to EDA**")
                st.write("Explore your data first without preprocessing")
                if st.button("Go to EDA", use_container_width=True, key="nav_to_eda"):
                    safe_switch_page("pages/3_📊_EDA.py")
        
        # Add Start Fresh option
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info("Want to start fresh? Clear all data and begin a new project.")
        with col2:
            if st.button("Start Fresh", type="secondary", use_container_width=True, key="start_fresh_upload"):
                if persistent_storage.clear_all_data():
                    st.success("All data cleared! Starting fresh...")
                    st.experimental_rerun()
        with col3:
            if st.button("Go Home", type="secondary", use_container_width=True, key="go_home_upload"):
                safe_switch_page("app.py")

    # Show getting started guide when no valid data is loaded
    elif not has_valid_data():
        # Getting started guide
        with stylable_container(
            key="getting_started",
            css_styles="""
            {
                background: linear-gradient(135deg, #e6fffa, #f0fff4);
                border-radius: 15px;
                padding: 2rem;
                border: 1px solid rgba(72, 187, 120, 0.2);
            }
            """
        ):
            st.info("""
            **Getting Started Guide**
            
            **Step 1:** Choose how you want to load your data (upload or local path)
            **Step 2:** Select your CSV file
            **Step 3:** Review the data preview to make sure it loaded correctly
            **Step 4:** Check the data quality summary
            **Step 5:** Move to the next step when you're satisfied
            """)