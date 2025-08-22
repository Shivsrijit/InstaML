# app/pages/1_📂_Data_Upload.py
import streamlit as st
import pandas as pd
import os
import numpy as np
from app.utilss.navigation import safe_switch_page

# Try to import optional dependencies
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    st.warning("⚠️ Excel support not available. Install with: `pip install openpyxl`")

try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    st.warning("⚠️ Parquet support not available. Install with: `pip install pyarrow`")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    st.warning("⚠️ HDF5 support not available. Install with: `pip install h5py`")

# Page configuration
st.set_page_config(page_title="Data Upload", layout="wide")

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
    
    .upload-card {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        color : "black";
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: #764ba2;
        background: #f8f9fa;
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

# Title and header
st.title("📂 Data Upload")
st.markdown("Upload your datasets and get started with machine learning")

# Collapsible help section
with st.expander("ℹ️ **What is this step and why is it important?**", expanded=False):
    st.markdown("""
    This is the **first and most crucial step** in your machine learning journey! Think of it as preparing the ingredients before cooking a meal.
    
    **Why data quality matters:**
    - 🎯 **Garbage in = Garbage out**: Poor quality data leads to poor model performance
    - 📊 **Foundation**: Everything else depends on this data
    - ⚡ **Efficiency**: Good data means faster training and better results
    """)

# Mode selection
st.header("🚀 Choose Your Data Source")
mode = st.radio(
    "Select data source:", 
    ["Upload file", "Load from local path"],
    horizontal=True
)

# --- Upload mode ---
if mode == "Upload file":
    st.subheader("📤 Upload Your Data File")
    
    # Collapsible format info
    with st.expander("📋 Supported Formats & Tips", expanded=False):
        st.markdown("""
        **✅ Supported Formats:**
        - **CSV files** (Comma Separated Values) - Most common and recommended
        
        **💡 Tips for best results:**
        - Make sure your CSV has headers (column names in the first row)
        - Avoid special characters in column names
        - Keep file size under 100MB for reliable uploads
        - Ensure your data is clean and well-structured
        """)
    
    # Upload area
    st.markdown("""
    <div class="upload-card">
        <h3 style="color: black;">📁 Drop your data file here</h3>
        <p style="color: black;">Supports CSV, Excel, JSON, images, audio, and more!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload data file", 
        type=["csv", "xlsx", "xls", "json", "parquet", "feather", "pickle", "pkl", 
              "jpg", "jpeg", "png", "bmp", "tiff", "gif",
              "wav", "mp3", "flac", "m4a", "aac", "ogg",
              "npy", "npz", "h5", "hdf5"],
        label_visibility="collapsed"
    )
    
    if uploaded is not None:
        uploaded_size_mb = uploaded.size / (1024 * 1024)
        
        # File size guidance
        if uploaded_size_mb > 100:
            st.warning(f"""
            ⚠️ **Large File Warning** 
            
            Your file is **{uploaded_size_mb:.1f} MB**. This might cause slower processing.
            Consider using "Load from local path" for large files.
            """)
        elif uploaded_size_mb > 50:
            st.info(f"📊 File size: {uploaded_size_mb:.1f} MB - Good size for analysis!")
        else:
            st.success(f"📊 File size: {uploaded_size_mb:.1f} MB - Perfect size!")

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
                st.session_state.df = df
                st.session_state.data_type = data_type
            elif data_type is not None:
                st.session_state.data_type = data_type
            
            # Show success and data info only if we successfully processed the file
            if data_type is not None:
                # Success message
                st.success(f"✅ **Successfully uploaded {uploaded.name}!**")
                st.info(f"📊 **Data Type Detected:** {data_type.title()}")
                
                # Show different content based on data type
                if data_type == "tabular" and df is not None:
                    # Data metrics for tabular data
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{df.shape[0]:,}</div>
                            <div class="metric-label">Rows (samples)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{df.shape[1]:,}</div>
                            <div class="metric-label">Columns (features)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{uploaded_size_mb:.1f} MB</div>
                            <div class="metric-label">File size</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Data preview
                    st.subheader("🔍 Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Quick quality check
                    st.subheader("🔍 Quick Data Quality Check")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        missing_count = df.isnull().sum().sum()
                        if missing_count > 0:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">⚠️ {missing_count}</div>
                                <div class="metric-label">Missing Values</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">✅ 0</div>
                                <div class="metric-label">Missing Values</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col2:
                        duplicate_count = df.duplicated().sum()
                        if duplicate_count > 0:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">⚠️ {duplicate_count}</div>
                                <div class="metric-label">Duplicate Rows</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">✅ 0</div>
                                <div class="metric-label">Duplicate Rows</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col3:
                        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{memory_mb:.2f}</div>
                            <div class="metric-label">Memory Usage (MB)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                elif data_type == "image":
                    st.info("🖼️ **Image file uploaded successfully!** For image training, please organize your images in folders with the following structure:\n\n```\nimages/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
                    
                elif data_type == "audio":
                    st.info("🎵 **Audio file uploaded successfully!** For audio training, please organize your audio files in folders with the following structure:\n\n```\naudio/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
                    
                elif data_type == "multi_dimensional":
                    st.info("🔢 **Multi-dimensional data uploaded successfully!** This data can be used for advanced ML models like CNNs, LSTMs, and Transformers.")
                
        except Exception as e:
            st.error(f"""
            ❌ **Failed to read your file!**
            
            **Error:** {e}
            
            **🔧 Common solutions:**
            - Make sure your file is in a supported format
            - Check if the file isn't corrupted
            - For large files, try using "Load from local path"
            - For images/audio, ensure proper directory structure
            """)

# --- Local path mode ---
elif mode == "Load from local path":
    st.subheader("📁 Load Data from Your Computer")
    
    # Collapsible info
    with st.expander("💡 When to use this option", expanded=False):
        st.markdown("""
        **💡 When to use this option:**
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
        help="Choose the type of data you want to load"
    )
    
    if file_type == "CSV":
        default_path = "datasets/tabular/airlines_flights_data.csv"
        file_path = st.text_input(
            "Enter CSV file path:", 
            default_path,
            help="Type the full path to your CSV file"
        )
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                st.session_state.df = df
                st.session_state.data_type = "tabular"
                
                # Get file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                
                st.success(f"✅ **Successfully loaded from {file_path}!**")
                
                # Data metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df.shape[0]:,}</div>
                        <div class="metric-label">Rows (samples)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{df.shape[1]:,}</div>
                        <div class="metric-label">Columns (features)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{file_size:.2f}</div>
                        <div class="metric-label">File size (MB)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"❌ **Failed to load CSV file:** {e}")
                
    elif file_type == "Excel":
        file_path = st.text_input(
            "Enter Excel file path:", 
            help="Type the full path to your Excel file (.xlsx or .xls)"
        )
        
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                st.session_state.df = df
                st.session_state.data_type = "tabular"
                st.success(f"✅ **Successfully loaded Excel file from {file_path}!**")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"❌ **Failed to load Excel file:** {e}")
                
    elif file_type == "Image Directory":
        directory_path = st.text_input(
            "Enter image directory path:", 
            help="Path to directory containing train/val subdirectories with class folders"
        )
        
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                st.session_state.data_type = "image"
                st.session_state.image_directory = directory_path
                st.success(f"✅ **Image directory loaded from {directory_path}!**")
                st.info("🖼️ **Image directory ready for training!** Make sure it has the structure:\n\n```\nimages/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
            else:
                st.error("❌ **Path exists but is not a directory!**")
                
    elif file_type == "Audio Directory":
        directory_path = st.text_input(
            "Enter audio directory path:", 
            help="Path to directory containing train/val subdirectories with class folders"
        )
        
        if os.path.exists(directory_path):
            if os.path.isdir(directory_path):
                st.session_state.data_type = "audio"
                st.session_state.audio_directory = directory_path
                st.success(f"✅ **Audio directory loaded from {directory_path}!**")
                st.info("🎵 **Audio directory ready for training!** Make sure it has the structure:\n\n```\naudio/\n├── train/\n│   ├── class_1/\n│   └── class_2/\n└── val/\n    ├── class_1/\n    └── class_2/\n```")
            else:
                st.error("❌ **Path exists but is not a directory!**")
                
    elif file_type == "Multi-dimensional Data":
        file_path = st.text_input(
            "Enter data file path:", 
            help="Path to .npy, .npz, .h5, or .hdf5 file"
        )
        
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.npy'):
                    data = np.load(file_path)
                elif file_path.endswith('.npz'):
                    data = np.load(file_path)
                    # For .npz files, we'll use the first array
                    data = data[data.files[0]]
                elif file_path.endswith(('.h5', '.hdf5')):
                    import h5py
                    with h5py.File(file_path, 'r') as f:
                        # Use the first dataset
                        key = list(f.keys())[0]
                        data = f[key][:]
                else:
                    st.error("❌ **Unsupported file format!** Please use .npy, .npz, .h5, or .hdf5 files.")
                    data = None
                
                if data is not None:
                    if len(data.shape) <= 2:
                        df = pd.DataFrame(data)
                        st.session_state.df = df
                        st.session_state.data_type = "tabular"
                        st.success(f"✅ **Successfully loaded tabular data from {file_path}!**")
                        st.dataframe(df.head(10))
                    else:
                        st.session_state.data_type = "multi_dimensional"
                        st.session_state.array_data = data
                        st.success(f"✅ **Successfully loaded multi-dimensional data from {file_path}!**")
                        st.info(f"🔢 **Data shape:** {data.shape}\n**Number of dimensions:** {len(data.shape)}")
                        
            except Exception as e:
                st.error(f"❌ **Failed to load data file:** {e}")

# Navigation section
# Initialize df in session state if it doesn't exist
if 'df' not in st.session_state:
    st.session_state.df = None

if st.session_state.df is not None:
    st.success("🎉 **Great job! Your dataset is loaded and ready for the next step.**")
    
    st.header("🚀 What's Next?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🔧 Next Step: Data Preprocessing**")
        st.write("Clean and prepare your data for analysis")
        if st.button("🚀 Go to Data Preprocessing", type="primary", use_container_width=True):
            safe_switch_page("pages/2_🔧_Data_Preprocessing.py")
    
    with col2:
        st.info("**📊 Alternative: Skip to EDA**")
        st.write("Explore your data first without preprocessing")
        if st.button("📊 Go to EDA", use_container_width=True):
            safe_switch_page("pages/3_📊_EDA.py")

else:
    # Getting started guide
    st.info("""
    📋 **Getting Started Guide**
    
    **Step 1:** Choose how you want to load your data (upload or local path)
    **Step 2:** Select your CSV file
    **Step 3:** Review the data preview to make sure it loaded correctly
    **Step 4:** Check the data quality summary
    **Step 5:** Move to the next step when you're satisfied
    """)