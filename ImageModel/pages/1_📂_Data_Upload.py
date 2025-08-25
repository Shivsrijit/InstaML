# app/pages/1_📂_Data_Upload.py
import streamlit as st
import pandas as pd
import os
import numpy as np
from app.utilss.navigation import safe_switch_page

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "data_type" not in st.session_state:
    st.session_state.data_type = None
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "image_metadata" not in st.session_state:
    st.session_state.image_metadata = None
if "image_pipeline_config" not in st.session_state:
    st.session_state.image_pipeline_config = {
        'model_input_size': (224, 224),
        'color_mode': 'RGB',
        'normalization_range': (0, 1),
        'augmentation_enabled': False,
        'denoising_enabled': False
    }

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
    
    .pipeline-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("📂 Data Upload")
st.markdown("Upload your datasets and get started with machine learning")

# Data type selector
st.header("🎯 Choose Your Data Type")
data_type_choice = st.radio(
    "Select the type of data you want to work with:",
    [ "🖼️ Image Data (JPG, PNG, etc.)"],
    
)

# Store the choice in session state
if data_type_choice == "📊 Tabular Data (CSV, Excel, etc.)":
    st.session_state.data_type = "tabular"
else:
    st.session_state.data_type = "image"

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
if st.session_state.data_type == "tabular":
    mode = st.radio(
        "Select data source:", 
        ["Upload file", "Load from local path"],
        horizontal=True
    )
else:  # image data
    mode = st.radio(
        "Select data source:", 
        ["🖼️ Single Image Upload", "📁 Batch Image Upload", "🔗 API/Dashboard Input"],
        horizontal=True
    )

# --- Tabular Data Upload (Unchanged) ---
if mode == "Upload file" and st.session_state.data_type == "tabular":
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
        <p style="color: black;">Supports CSV, Excel, JSON, and more!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload data file", 
        type=["csv", "xlsx", "xls", "json", "parquet", "feather", "pickle", "pkl"],
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
            else:
                st.error(f"Unsupported file type: {file_extension}")
                df = None
                data_type = None
            
            # Store data and type in session state (only if successfully processed)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_type = data_type
            
            # Show success and data info only if we successfully processed the file
            if data_type is not None:
                # Success message
                st.success(f"✅ **Successfully uploaded {uploaded.name}!**")
                st.info(f"📊 **Data Type Detected:** {data_type.title()}")
                
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
                
        except Exception as e:
            st.error(f"""
            ❌ **Failed to read your file!**
            
            **Error:** {e}
            
            **🔧 Common solutions:**
            - Make sure your file is in a supported format
            - Check if the file isn't corrupted
            - For large files, try using "Load from local path"
            """)

# --- Image Data Upload (New Comprehensive Workflow) ---
elif st.session_state.data_type == "image":
    if mode == "🖼️ Single Image Upload":
        st.subheader("🖼️ Single Image Upload")
        
        # Upload area
        st.markdown("""
        <div class="upload-card">
            <h3 style="color: black;">🖼️ Drop your image here</h3>
            <p style="color: black;">Supports JPG, PNG, BMP, TIFF, GIF</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Upload image file", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "gif"],
            label_visibility="collapsed"
        )
        
        if uploaded_image is not None:
            try:
                import cv2
                from PIL import Image
                import io
                
                # Read and process image
                image = Image.open(uploaded_image)
                image_array = np.array(image)
                
                # Store image data with metadata
                image_data = [{
                    'name': uploaded_image.name,
                    'data': image_array,
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format,
                    'original_path': None
                }]
                
                image_metadata = {
                    'total_images': 1,
                    'formats': {image.format or 'Unknown': 1},
                    'sizes': [image.size],
                    'total_size_mb': uploaded_image.size / (1024 * 1024)
                }
                
                # Store in session state
                st.session_state.image_data = image_data
                st.session_state.image_metadata = image_metadata
                
                # Success message and metrics
                st.success(f"✅ **Successfully uploaded {uploaded_image.name}!**")
                
                # Image metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{image.size[0]}×{image.size[1]}</div>
                        <div class="metric-label">Original Size</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{image.mode}</div>
                        <div class="metric-label">Color Mode</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{image_metadata['total_size_mb']:.2f}</div>
                        <div class="metric-label">File Size (MB)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Image preview
                st.subheader("🖼️ Image Preview")
                st.image(image_array, caption=uploaded_image.name, use_column_width=True)
                
                # Upload status
                st.subheader("✅ Upload Status")
                st.success("**Image successfully uploaded and ready for processing!**")
                st.info("💡 **Tip:** You can configure image preprocessing settings in the Data Preprocessing page.")
                
            except Exception as e:
                st.error(f"Failed to process image: {str(e)}")
    
    elif mode == "📁 Batch Image Upload":
        st.subheader("📁 Batch Image Upload")
        
        # Batch upload for multiple images
        uploaded_images = st.file_uploader(
            "Upload multiple image files", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "gif"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_images:
            try:
                import cv2
                from PIL import Image
                
                image_data = []
                image_metadata = {
                    'total_images': len(uploaded_images),
                    'formats': {},
                    'sizes': [],
                    'total_size_mb': 0
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_images):
                    status_text.text(f"Processing image {i+1}/{len(uploaded_images)}: {uploaded_file.name}")
                    
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    image_data.append({
                        'name': uploaded_file.name,
                        'data': image_array,
                        'size': image.size,
                        'mode': image.mode,
                        'format': image.format,
                        'original_path': None
                    })
                    
                    # Update metadata
                    format_type = image.format or 'Unknown'
                    image_metadata['formats'][format_type] = image_metadata['formats'].get(format_type, 0) + 1
                    image_metadata['sizes'].append(image.size)
                    image_metadata['total_size_mb'] += uploaded_file.size / (1024 * 1024)
                    
                    progress_bar.progress((i + 1) / len(uploaded_images))
                
                status_text.text("✅ Batch processing complete!")
                
                if image_data:
                    st.session_state.image_data = image_data
                    st.session_state.image_metadata = image_metadata
                    
                    st.success(f"✅ **Successfully uploaded {len(uploaded_images)} images!**")
                    
                    # Batch metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(uploaded_images)}</div>
                            <div class="metric-label">Total Images</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        avg_size = np.mean([img['size'][0] * img['size'][1] for img in image_data])
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_size:.0f}</div>
                            <div class="metric-label">Avg Pixels</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{image_metadata['total_size_mb']:.1f}</div>
                            <div class="metric-label">Total Size (MB)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Image preview grid
                    st.subheader("🖼️ Image Preview")
                    preview_cols = st.columns(4)
                    for i, img_info in enumerate(image_data[:8]):
                        with preview_cols[i % 4]:
                            st.image(img_info['data'], caption=img_info['name'], use_column_width=True)
                    
                    if len(image_data) > 8:
                        st.info(f"Showing first 8 images. Total: {len(image_data)} images uploaded.")
                
            except Exception as e:
                st.error(f"Failed to process batch images: {str(e)}")
    
    elif mode == "🔗 API/Dashboard Input":
        st.subheader("🔗 API/Dashboard Input")
        
        st.info("""
        **API/Dashboard Input Mode**
        
        This mode allows you to input image data from external sources:
        - **URL Input**: Provide image URLs for processing
        - **Base64 Input**: Paste base64 encoded images
        - **API Integration**: Connect to external image APIs
        """)
        
        input_method = st.selectbox(
            "Select input method:",
            ["URL Input", "Base64 Input", "API Integration"]
        )
        
        if input_method == "URL Input":
            image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
            if st.button("🔗 Load from URL"):
                if image_url:
                    try:
                        import requests
                        from PIL import Image
                        import io
                        
                        response = requests.get(image_url)
                        image = Image.open(io.BytesIO(response.content))
                        image_array = np.array(image)
                        
                        image_data = [{
                            'name': f"url_image_{len(image_url)}",
                            'data': image_array,
                            'size': image.size,
                            'mode': image.mode,
                            'format': image.format,
                            'original_path': image_url
                        }]
                        
                        st.session_state.image_data = image_data
                        st.session_state.image_metadata = {
                            'total_images': 1,
                            'formats': {image.format or 'Unknown': 1},
                            'sizes': [image.size],
                            'total_size_mb': len(response.content) / (1024 * 1024)
                        }
                        
                        st.success("✅ **Successfully loaded image from URL!**")
                        st.image(image_array, caption="Loaded from URL", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Failed to load image from URL: {str(e)}")
        
        elif input_method == "Base64 Input":
            base64_data = st.text_area("Paste base64 encoded image:", height=100)
            if st.button("🔗 Load from Base64"):
                if base64_data:
                    try:
                        import base64
                        from PIL import Image
                        import io
                        
                        # Remove data URL prefix if present
                        if base64_data.startswith('data:image'):
                            base64_data = base64_data.split(',')[1]
                        
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        image_array = np.array(image)
                        
                        image_data = [{
                            'name': "base64_image",
                            'data': image_array,
                            'size': image.size,
                            'mode': image.mode,
                            'format': image.format,
                            'original_path': "base64_input"
                        }]
                        
                        st.session_state.image_data = image_data
                        st.session_state.image_metadata = {
                            'total_images': 1,
                            'formats': {image.format or 'Unknown': 1},
                            'sizes': [image.size],
                            'total_size_mb': len(image_bytes) / (1024 * 1024)
                        }
                        
                        st.success("✅ **Successfully loaded image from Base64!**")
                        st.image(image_array, caption="Loaded from Base64", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Failed to load image from Base64: {str(e)}")

# --- Local path mode (Unchanged for tabular) ---
elif mode == "Load from local path" and st.session_state.data_type == "tabular":
    st.subheader("📁 Load Data from Your Computer")
    
    # Collapsible info
    with st.expander("💡 When to use this option", expanded=False):
        st.markdown("""
        **💡 When to use this option:**
        - You have large files (>100MB)
        - You're working with the same dataset repeatedly
        - You want to avoid uploading the same file multiple times
        - You're working in a local development environment
        """)
    
    file_type = st.selectbox(
        "Select file type:",
        ["CSV", "Excel", "JSON", "Parquet"],
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

# Navigation section
# Check if data is loaded (either tabular or image)
data_loaded = (st.session_state.df is not None or 
               st.session_state.image_data is not None or 
               st.session_state.data_type is not None)

if data_loaded:
    data_type = st.session_state.data_type or "unknown"
    st.success(f"🎉 **Great job! Your {data_type} dataset is loaded and ready for the next step.**")
    
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
    **Step 2:** Select your data file
    **Step 3:** Review the data preview to make sure it loaded correctly
    **Step 4:** Check the data quality summary
    **Step 5:** Move to the next step when you're satisfied
    """)