
# app/pages/2_🔧_Data_Preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
from app.utilss.navigation import safe_switch_page
try:
    from PIL import Image, ImageEnhance, ImageFilter
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    st.warning("⚠️ Image processing libraries not available. Install with: `pip install opencv-python Pillow matplotlib`")

# Page configuration
st.set_page_config(page_title="Data Preprocessing", layout="wide")

# Helper function for proper image display
def display_image_safely(image_data, caption="", use_column_width=True):
    """Safely display image data with proper type conversion"""
    if image_data is None:
        return
    
    # Convert to uint8 if needed for display
    if image_data.dtype != np.uint8:
        if image_data.max() <= 1.0:
            display_data = (image_data * 255).astype(np.uint8)
        else:
            display_data = image_data.astype(np.uint8)
    else:
        display_data = image_data
    
    st.image(display_data, caption=caption, use_column_width=use_column_width)

# Custom CSS for better styling
st.markdown("""
<style>
    .step-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    .pipeline-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .preview-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    
    .comparison-info {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if "df" not in st.session_state:
    st.session_state.df = None
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "data_type" not in st.session_state:
    st.session_state.data_type = None

# Title and header
st.title("🔧 Data Preprocessing")
st.markdown("Clean, transform, and prepare your data for machine learning")

# Check if data is loaded
if st.session_state.df is None and st.session_state.image_data is None:
    st.error("❌ **No data loaded!** Please go back to the Data Upload page and load your data first.")
    if st.button("📂 Go to Data Upload", type="primary"):
        safe_switch_page("pages/1_📂_Data_Upload.py")
    st.stop()

# Determine data type and load data
data_type = st.session_state.data_type
if data_type == "tabular":
    if "df_original" not in st.session_state:
        st.session_state.df_original = st.session_state.df.copy()
    df = st.session_state.df.copy()
    df_original = st.session_state.df_original
else:  # image data
    # Ensure original data is preserved before any processing
    if "image_data_original" not in st.session_state or st.session_state.image_data_original is None:
        if st.session_state.image_data is not None:
            # Deep copy the original data to preserve it
            import copy
            st.session_state.image_data_original = copy.deepcopy(st.session_state.image_data)
        else:
            st.session_state.image_data_original = None
    
    # Get current working copy and original reference
    image_data = st.session_state.image_data.copy() if st.session_state.image_data else None
    image_data_original = st.session_state.image_data_original

# Sidebar with recommended order
with st.sidebar.expander("📋 Recommended Order", expanded=False):
    if data_type == "tabular":
        st.markdown("""
        **📋 Recommended Order:**
        1. **🧹 Data Cleaning** - Remove duplicates, select columns
        2. **🔢 Missing Values** - Handle incomplete data
        3. **📏 Scaling & Encoding** - Prepare for ML algorithms
        4. **📈 Outlier Detection** - Find and handle extreme values
        5. **💾 Save** - Store your cleaned data
        """)
    else:  # image data
        st.markdown("""
        **📋 Recommended Order:**
        1. **📏 Image Resizing** - Resize to model input size
        2. **🎨 Color Conversion** - Convert to RGB/Grayscale
        3. **📊 Normalization** - Normalize pixel values
        4. **🔧 Denoising & Augmentation** - Enhance image quality
        5. **💾 Save** - Store your processed images
        """)

# Data Overview
st.header("📊 Data Overview")

# Add reset functionality for image data
if data_type == "image" and image_data_original is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **Tip:** Use the comparison sections to see the difference between original and processed images.")
    with col2:
        if st.button("🔄 Reset to Original", help="Restore original image data"):
            st.session_state.image_data = copy.deepcopy(image_data_original)
            st.success("✅ Reset to original data!")
            st.rerun()

if data_type == "tabular":
    # Tabular data overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
        if df.shape[0] < 100:
            st.warning("⚠️ Small dataset")
        elif df.shape[0] < 1000:
            st.info("📊 Medium dataset")
        else:
            st.success("🚀 Large dataset")
            
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]:,}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
        if df.shape[1] < 5:
            st.info("📊 Few features")
        elif df.shape[1] < 20:
            st.success("🚀 Good feature count")
        else:
            st.warning("⚠️ Many features - consider selection")
            
    with col3:
        missing_count = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing_count:,}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
        if missing_count > 0:
            st.warning("⚠️ Missing data detected")
        else:
            st.success("✅ No missing data")
            
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{memory_mb:.2f}</div>
            <div class="metric-label">Memory (MB)</div>
        </div>
        """, unsafe_allow_html=True)

else:  # image data
    # Image data overview
    if image_data is not None and len(image_data) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(image_data)}</div>
                <div class="metric-label">Total Images</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            avg_width = np.mean([img['size'][0] for img in image_data])
            avg_height = np.mean([img['size'][1] for img in image_data])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_width:.0f}×{avg_height:.0f}</div>
                <div class="metric-label">Avg Size</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            formats = set([img['format'] for img in image_data if img['format']])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(formats)}</div>
                <div class="metric-label">Formats</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            total_size = sum([img['size'][0] * img['size'][1] for img in image_data])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_size:,}</div>
                <div class="metric-label">Total Pixels</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No image data available")

# Preprocessing tabs
if data_type == "tabular":
    # Tabular preprocessing (unchanged)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧹 Data Cleaning", "🔢 Missing Values", "📏 Scaling & Encoding", "📈 Outlier Detection", "💾 Save Data"])
    
    with tab1:
        st.subheader("🧹 Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-card">
                <h4>1. Remove Duplicates</h4>
                <p>Remove duplicate rows from your dataset</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Check for Duplicates"):
                duplicate_count = df.duplicated().sum()
                st.info(f"Found {duplicate_count} duplicate rows")
                
                if duplicate_count > 0:
                    if st.button("🗑️ Remove Duplicates"):
                        df = df.drop_duplicates()
                        st.success(f"✅ Removed {duplicate_count} duplicate rows")
        
        with col2:
            st.markdown("""
            <div class="step-card">
                <h4>2. Column Selection</h4>
                <p>Select which columns to keep for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            all_columns = list(df.columns)
            selected_columns = st.multiselect(
                "Select columns to keep:",
                all_columns
            )
            
            if st.button("✅ Apply Column Selection"):
                df = df[selected_columns]
                st.success(f"✅ Kept {len(selected_columns)} columns")
    
    with tab2:
        st.subheader("🔢 Missing Values")
        
        # Missing value summary
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) > 0:
            st.write("**Columns with missing values:**")
            for col in missing_summary.index:
                st.write(f"**{col}** ({missing_summary[col]} missing values)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox(
                "Method for handling missing values:",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill"]
            )
            
            if st.button("🔧 Apply Missing Value Treatment"):
                if method == "Drop rows":
                    df = df.dropna()
                    st.success("✅ Dropped rows with missing values")
                elif method == "Fill with mean":
                    df = df.fillna(df.mean())
                    st.success("✅ Filled missing values with mean")
                elif method == "Fill with median":
                    df = df.fillna(df.median())
                    st.success("✅ Filled missing values with median")
                elif method == "Fill with mode":
                    df = df.fillna(df.mode().iloc[0])
                    st.success("✅ Filled missing values with mode")
                elif method == "Forward fill":
                    df = df.fillna(method='ffill')
                    st.success("✅ Filled missing values with forward fill")
            else:
                st.success("✅ No missing values found!")
    
    with tab3:
        st.subheader("📏 Scaling & Encoding")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="step-card">
                <h4>Numeric Scaling</h4>
                <p>Scale numeric features for better model performance</p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaling_method = st.selectbox(
                    "Scaling method:",
                    ["No scaling", "StandardScaler", "MinMaxScaler", "RobustScaler"]
                )
                
                if st.button("📏 Apply Scaling"):
                    if scaling_method != "No scaling":
                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                        
                        if scaling_method == "StandardScaler":
                            scaler = StandardScaler()
                        elif scaling_method == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        elif scaling_method == "RobustScaler":
                            scaler = RobustScaler()
                        
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        st.success(f"✅ Applied {scaling_method} to numeric columns")
            else:
                st.info("ℹ️ No numeric columns found")
        
        with col2:
            st.markdown("""
            <div class="step-card">
                <h4>Categorical Encoding</h4>
                <p>Encode categorical variables for ML algorithms</p>
            </div>
            """, unsafe_allow_html=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                encoding_method = st.selectbox(
                    "Encoding method:",
                    ["No encoding", "One-Hot Encoding", "Label Encoding"]
                )
                
                if st.button("🔤 Apply Encoding"):
                    if encoding_method == "One-Hot Encoding":
                        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                        st.success(f"✅ Applied One-Hot Encoding to categorical columns")
                    elif encoding_method == "Label Encoding":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        for col in categorical_cols:
                            df[col] = le.fit_transform(df[col])
                        st.success(f"✅ Applied Label Encoding to categorical columns")
            else:
                st.info("ℹ️ No categorical columns found")
    
    with tab4:
        st.subheader("📈 Outlier Detection")
        
        st.info("Outlier detection helps identify and handle extreme values that might affect model performance.")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_col = st.selectbox("Select column for outlier detection:", numeric_cols)
            
            if st.button("🔍 Detect Outliers"):
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[outlier_col] < Q1 - 1.5*IQR) | (df[outlier_col] > Q3 + 1.5*IQR)]
                
                st.write(f"**Found {len(outliers)} outliers in {outlier_col}**")
                st.write(f"**IQR Range:** {Q1:.2f} to {Q3:.2f}")
                st.write(f"**Outlier Range:** < {Q1 - 1.5*IQR:.2f} or > {Q3 + 1.5*IQR:.2f}")
                
                if len(outliers) > 0:
                    if st.button("🗑️ Remove Outliers"):
                        df = df[~((df[outlier_col] < Q1 - 1.5*IQR) | (df[outlier_col] > Q3 + 1.5*IQR))]
                        st.success(f"✅ Removed {len(outliers)} outliers")
        else:
            st.info("ℹ️ No numeric columns found for outlier detection")
    
    with tab5:
        st.subheader("💾 Save Data")
        
        st.markdown("""
        <div class="step-card">
            <h4>Save Preprocessed Data</h4>
            <p>Save your cleaned and preprocessed data for later use</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💾 Save Preprocessed Data"):
            st.session_state.df_preprocessed = df.copy()
            st.success("✅ **Preprocessed data saved successfully!**")
            st.info("Your data is now ready for the next step in the pipeline.")

else:  # image data
    # Image preprocessing pipeline
    if image_data is not None and len(image_data) > 0:
        st.markdown("""
        <div class="pipeline-step">
            <h3>🔄 Image Preprocessing Pipeline</h3>
            <p>Comprehensive image preprocessing for machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📏 Resize", "🎨 Color Conversion", "📊 Normalization", "🔧 Denoising & Augmentation", "💾 Save"])
        
        with tab1:
            st.subheader("📏 Image Resizing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="step-card">
                    <h4>Resize to Model Input Size</h4>
                    <p>Resize images to the target dimensions for your model</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get target size from pipeline config
                target_width, target_height = st.session_state.image_pipeline_config.get('model_input_size', (224, 224))
                
                resize_width = st.number_input("Target Width", min_value=32, max_value=1024, value=target_width, step=32)
                resize_height = st.number_input("Target Height", min_value=32, max_value=1024, value=target_height, step=32)
                
                resize_method = st.selectbox(
                    "Resize method:",
                    ["Resize to fixed size", "Resize maintaining aspect ratio", "Crop to square"]
                )
                
                if st.button("📏 Apply Resizing"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a new copy for processing to avoid modifying original
                    processed_image_data = []
                    
                    for i, img_info in enumerate(image_data):
                        status_text.text(f"Resizing image {i+1}/{len(image_data)}")
                        
                        img_array = img_info['data']
                        pil_img = Image.fromarray(img_array)
                        
                        if resize_method == "Resize to fixed size":
                            resized_img = pil_img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
                        elif resize_method == "Resize maintaining aspect ratio":
                            # Calculate aspect ratio
                            aspect_ratio = pil_img.width / pil_img.height
                            if resize_width / resize_height > aspect_ratio:
                                new_height = resize_height
                                new_width = int(resize_height * aspect_ratio)
                            else:
                                new_width = resize_width
                                new_height = int(resize_width / aspect_ratio)
                            resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        else:  # Crop to square
                            # Resize to make the shorter side equal to target
                            if pil_img.width < pil_img.height:
                                new_width = resize_width
                                new_height = int(resize_width * pil_img.height / pil_img.width)
                            else:
                                new_height = resize_height
                                new_width = int(resize_height * pil_img.width / pil_img.height)
                            resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # Crop to square
                            left = (new_width - resize_width) // 2
                            top = (new_height - resize_height) // 2
                            right = left + resize_width
                            bottom = top + resize_height
                            resized_img = resized_img.crop((left, top, right, bottom))
                        
                        # Create new image info with resized data
                        new_img_info = img_info.copy()
                        new_img_info['data'] = np.array(resized_img)
                        new_img_info['size'] = resized_img.size
                        processed_image_data.append(new_img_info)
                        
                        progress_bar.progress((i + 1) / len(image_data))
                    
                    # Update session state with processed data
                    st.session_state.image_data = processed_image_data
                    image_data = processed_image_data
                    
                    status_text.text("✅ Resizing complete!")
                    st.success(f"✅ **Resized {len(image_data)} images to {resize_width}×{resize_height}!**")
            
            with col2:
                if image_data is not None and len(image_data) > 0:
                    st.markdown("""
                    <div class="step-card">
                        <h4>Size Comparison</h4>
                        <p>Before vs After resizing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show before/after comparison
                    max_idx = max(0, len(image_data)-1)
                    if max_idx == 0:
                        sample_idx = 0
                        st.info("Only one image available")
                    else:
                        sample_idx = st.slider("Select sample image:", 0, max_idx, 0)
                    
                    # Add comparison info
                    st.markdown("""
                    <div class="comparison-info">
                        <strong>📊 Comparison Info:</strong> The left image shows the original data, the right shows the processed result.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**Original**")
                        if image_data_original and sample_idx < len(image_data_original):
                            orig_data = image_data_original[sample_idx]['data']
                            display_image_safely(orig_data, caption=f"Original: {image_data_original[sample_idx]['size']}")
                            # Show debug info
                            st.caption(f"Type: {orig_data.dtype}, Range: {orig_data.min():.1f}-{orig_data.max():.1f}")
                    
                    with col_b:
                        st.write("**Resized**")
                        if sample_idx < len(image_data):
                            proc_data = image_data[sample_idx]['data']
                            display_image_safely(proc_data, caption=f"Resized: {image_data[sample_idx]['size']}")
                            # Show debug info
                            st.caption(f"Type: {proc_data.dtype}, Range: {proc_data.min():.1f}-{proc_data.max():.1f}")
        
        with tab2:
            st.subheader("🎨 Color Conversion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="step-card">
                    <h4>Color Mode Conversion</h4>
                    <p>Convert images to RGB or Grayscale as needed</p>
                </div>
                """, unsafe_allow_html=True)
                
                target_color_mode = st.selectbox(
                    "Target color mode:",
                    ["RGB", "Grayscale"],
                    help="RGB for color images, Grayscale for monochrome"
                )
                
                if st.button("🎨 Apply Color Conversion"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a new copy for processing to avoid modifying original
                    processed_image_data = []
                    
                    for i, img_info in enumerate(image_data):
                        status_text.text(f"Converting image {i+1}/{len(image_data)}")
                        
                        img_array = img_info['data']
                        pil_img = Image.fromarray(img_array)
                        
                        if target_color_mode == "RGB":
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                        else:  # Grayscale
                            if pil_img.mode != "L":
                                pil_img = pil_img.convert("L")
                        
                        # Create new image info with converted data
                        new_img_info = img_info.copy()
                        new_img_info['data'] = np.array(pil_img)
                        new_img_info['mode'] = pil_img.mode
                        processed_image_data.append(new_img_info)
                        
                        progress_bar.progress((i + 1) / len(image_data))
                    
                    # Update session state with processed data
                    st.session_state.image_data = processed_image_data
                    image_data = processed_image_data
                    
                    status_text.text("✅ Color conversion complete!")
                    st.success(f"✅ **Converted {len(image_data)} images to {target_color_mode}!**")
            
            with col2:
                if image_data is not None and len(image_data) > 0:
                    st.markdown("""
                    <div class="step-card">
                        <h4>Color Mode Preview</h4>
                        <p>See the effect of color conversion</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if image_data is not None and len(image_data) > 0:
                        # Show before/after comparison
                        max_idx = max(0, len(image_data)-1)
                        if max_idx == 0:
                            sample_idx = 0
                            st.info("Only one image available")
                        else:
                            sample_idx = st.slider("Select sample image:", 0, max_idx, 0, key="color_preview")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Original**")
                            if image_data_original and sample_idx < len(image_data_original):
                                orig_data = image_data_original[sample_idx]['data']
                                display_image_safely(orig_data, caption="Original")
                                st.caption(f"Type: {orig_data.dtype}, Range: {orig_data.min():.1f}-{orig_data.max():.1f}")
                        
                        with col_b:
                            st.write("**Converted**")
                            if sample_idx < len(image_data):
                                proc_data = image_data[sample_idx]['data']
                                display_image_safely(proc_data, caption="Converted")
                                st.caption(f"Type: {proc_data.dtype}, Range: {proc_data.min():.1f}-{proc_data.max():.1f}")
        
        with tab3:
            st.subheader("📊 Normalization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="step-card">
                    <h4>Pixel Value Normalization</h4>
                    <p>Normalize pixel values to specified range</p>
                </div>
                """, unsafe_allow_html=True)
                
                norm_range = st.selectbox(
                    "Normalization range:",
                    ["0-1", "-1 to 1"],
                    help="Pixel value normalization range"
                )
                
                if st.button("📊 Apply Normalization"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a new copy for processing to avoid modifying original
                    processed_image_data = []
                    
                    for i, img_info in enumerate(image_data):
                        status_text.text(f"Normalizing image {i+1}/{len(image_data)}")
                        
                        img_array = img_info['data'].astype(np.float32)
                        
                        if norm_range == "0-1":
                            # Normalize to [0, 1]
                            img_array = img_array / 255.0
                        else:  # -1 to 1
                            # Normalize to [-1, 1]
                            img_array = (img_array / 255.0) * 2 - 1
                        
                        # Create new image info with normalized data
                        new_img_info = img_info.copy()
                        new_img_info['data'] = img_array
                        processed_image_data.append(new_img_info)
                        
                        progress_bar.progress((i + 1) / len(image_data))
                    
                    # Update session state with processed data
                    st.session_state.image_data = processed_image_data
                    image_data = processed_image_data
                    
                    status_text.text("✅ Normalization complete!")
                    st.success(f"✅ **Normalized {len(image_data)} images to range {norm_range}!**")
            
            with col2:
                if image_data is not None and len(image_data) > 0:
                    st.markdown("""
                    <div class="step-card">
                        <h4>Normalization Preview</h4>
                        <p>See the effect of normalization</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if image_data is not None and len(image_data) > 0:
                        # Show before/after comparison
                        max_idx = max(0, len(image_data)-1)
                        if max_idx == 0:
                            sample_idx = 0
                            st.info("Only one image available")
                        else:
                            sample_idx = st.slider("Select sample image:", 0, max_idx, 0, key="norm_preview")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Original**")
                            if image_data_original and sample_idx < len(image_data_original):
                                orig_data = image_data_original[sample_idx]['data']
                                display_image_safely(orig_data, caption="Original")
                                st.caption(f"Type: {orig_data.dtype}, Range: {orig_data.min():.1f}-{orig_data.max():.1f}")
                        
                        with col_b:
                            st.write("**Normalized**")
                            if sample_idx < len(image_data):
                                proc_data = image_data[sample_idx]['data']
                                display_image_safely(proc_data, caption="Normalized")
                                st.caption(f"Type: {proc_data.dtype}, Range: {proc_data.min():.1f}-{proc_data.max():.1f}")
        
        with tab4:
            st.subheader("🔧 Denoising & Augmentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="step-card">
                    <h4>Image Enhancement</h4>
                    <p>Apply denoising and augmentation techniques</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Denoising options
                denoise_enabled = st.checkbox("Apply denoising", value=False)
                if denoise_enabled:
                    denoise_method = st.selectbox("Denoising method:", ["Median Filter", "Gaussian Blur", "Bilateral Filter"])
                
                # Augmentation options
                augmentation_enabled = st.checkbox("Apply augmentation", value=False)
                if augmentation_enabled:
                    st.write("**Augmentation options:**")
                    flip_horizontal = st.checkbox("Horizontal flip", value=False)
                    flip_vertical = st.checkbox("Vertical flip", value=False)
                    rotate = st.checkbox("Random rotation", value=False)
                    if rotate:
                        rotation_angle = st.slider("Rotation angle (±degrees)", 0, 45, 15)
                
                if st.button("🔧 Apply Enhancements"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a new copy for processing to avoid modifying original
                    processed_image_data = []
                    
                    for i, img_info in enumerate(image_data):
                        status_text.text(f"Enhancing image {i+1}/{len(image_data)}")
                        
                        img_array = img_info['data']
                        pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
                        
                        # Apply denoising
                        if denoise_enabled:
                            if denoise_method == "Median Filter":
                                pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
                            elif denoise_method == "Gaussian Blur":
                                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
                            elif denoise_method == "Bilateral Filter":
                                # Convert to OpenCV for bilateral filtering
                                try:
                                    import cv2
                                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                    cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
                                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                                except ImportError:
                                    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
                        
                        # Apply augmentation
                        if augmentation_enabled:
                            if flip_horizontal:
                                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                            if flip_vertical:
                                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                            if rotate:
                                import random
                                angle = random.uniform(-rotation_angle, rotation_angle)
                                pil_img = pil_img.rotate(angle, expand=True)
                        
                        # Create new image info with enhanced data
                        new_img_info = img_info.copy()
                        new_img_info['data'] = np.array(pil_img).astype(np.float32) / 255.0
                        processed_image_data.append(new_img_info)
                        
                        progress_bar.progress((i + 1) / len(image_data))
                    
                    # Update session state with processed data
                    st.session_state.image_data = processed_image_data
                    image_data = processed_image_data
                    
                    status_text.text("✅ Enhancement complete!")
                    st.success(f"✅ **Applied enhancements to {len(image_data)} images!**")
            
            with col2:
                if image_data is not None and len(image_data) > 0:
                    st.markdown("""
                    <div class="step-card">
                        <h4>Enhancement Preview</h4>
                        <p>Compare original vs enhanced images</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if image_data is not None and len(image_data) > 0:
                        max_idx = max(0, len(image_data)-1)
                        if max_idx > 0:
                            sample_idx = st.slider("Select sample image:", 0, max_idx, 0, key="enhancement_preview")
                        else:
                            st.warning("No images available for preview.")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Before Enhancement**")
                            if image_data_original and sample_idx < len(image_data_original):
                                orig_data = image_data_original[sample_idx]['data']
                                display_image_safely(orig_data, caption="Before")
                                st.caption(f"Type: {orig_data.dtype}, Range: {orig_data.min():.1f}-{orig_data.max():.1f}")
                        
                        with col_b:
                            st.write("**After Enhancement**")
                            if sample_idx < len(image_data):
                                proc_data = image_data[sample_idx]['data']
                                display_image_safely(proc_data, caption="After")
                                st.caption(f"Type: {proc_data.dtype}, Range: {proc_data.min():.1f}-{proc_data.max():.1f}")
        
        with tab5:
            st.subheader("💾 Save Processed Images")
            
            st.markdown("""
            <div class="step-card">
                <h4>Save Preprocessed Images</h4>
                <p>Save your processed images for model training</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💾 Save Processed Images"):
                st.session_state.image_preprocessed = image_data.copy()
                st.session_state.image_data = image_data  # Update current data
                st.success("✅ **Processed images saved successfully!**")
                st.info("Your images are now ready for the next step in the pipeline.")
                
                # Show final statistics
                st.subheader("📊 Final Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Images", len(image_data))
                
                with col2:
                    avg_size = np.mean([img['size'][0] * img['size'][1] for img in image_data])
                    st.metric("Avg Pixels", f"{avg_size:.0f}")
                
                with col3:
                    formats = set([img['format'] for img in image_data if img['format']])
                    st.metric("Formats", len(formats))

# Navigation section
st.header("🚀 What's Next?")

col1, col2 = st.columns(2)

with col1:
    st.info("**📊 Next Step: Exploratory Data Analysis**")
    st.write("Explore and analyze your preprocessed data")
    if st.button("📊 Go to EDA", type="primary", use_container_width=True):
        safe_switch_page("pages/3_📊_EDA.py")

with col2:
    st.info("**⚙️ Alternative: Model Training**")
    st.write("Train machine learning models with your analyzed data")
    if st.button("⚙️ Go to Model Training", use_container_width=True):
        safe_switch_page("pages/4_⚙️_Train_Model.py") 