# app/pages/3_📊_EDA.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from app.utilss.navigation import safe_switch_page
from app.utilss.charts import *

# Image processing imports
try:
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    st.warning("⚠️ Image processing libraries not available. Install with: `pip install opencv-python pillow matplotlib`")

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

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
    .help-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
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
    
    .chart-card {
        background: white;
        border: 1px solid #e0e0e0;
    border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chart-card h4 {
        color: #333;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
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

st.title("📊 Exploratory Data Analysis")

# Collapsible help section
with st.expander("ℹ️ **What is EDA and Why is it Important?**", expanded=False):
    st.markdown("""
    **Exploratory Data Analysis (EDA)** is like being a detective with your data! It's the process of investigating your dataset to understand its patterns, characteristics, and potential insights.

    **🔍 What EDA helps you discover:**
    - **Data Distribution**: How your data is spread out
    - **Relationships**: Connections between different variables
    - **Outliers**: Unusual or extreme values
    - **Missing Patterns**: Where data is incomplete
    - **Quality Issues**: Problems that need fixing
    
    **⚡ Why this matters:**
    - **Better Models**: Understanding data leads to better ML models
    - **Feature Engineering**: Discover new useful features
    - **Data Quality**: Identify and fix data problems
    - **Business Insights**: Find valuable patterns for decision-making
    """)

# Check if data is loaded
if ("df" not in st.session_state or st.session_state.df is None) and \
   ("image_data" not in st.session_state or st.session_state.image_data is None):
    st.warning("⚠️ Please upload or load data first from the Data Upload page.")
    st.stop()

# Determine data type
data_type = st.session_state.get("data_type", "tabular")
if data_type == "image" and st.session_state.image_data is None:
    st.warning("⚠️ Please upload image data first from the Data Upload page.")
    st.stop()

# Load data
if data_type == "tabular":
    df = st.session_state.df.copy()
    if df is None:
        st.warning("⚠️ Please upload tabular data first from the Data Upload page.")
        st.stop()
else:  # image data
    image_data = st.session_state.image_data.copy()
    if image_data is None:
        st.warning("⚠️ Please upload image data first from the Data Upload page.")
        st.stop()

# Data Overview
st.header("📊 Data Overview")

# Add info about data type handling for image data
if data_type == "image":
    st.info("💡 **Image Data Handling:** This page automatically handles different image data types (0-255, 0-1 normalized) for proper analysis and visualization.")
    
st.info(f"**This section provides an overview of your {data_type} dataset.**")

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
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]:,}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            missing_count = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_count:,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        
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
        formats = list(set([img['format'] for img in image_data if img['format']]))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(formats)}</div>
            <div class="metric-label">Formats</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        total_pixels = sum([img['size'][0] * img['size'][1] for img in image_data])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_pixels:,}</div>
            <div class="metric-label">Total Pixels</div>
        </div>
        """, unsafe_allow_html=True)

# Analysis tabs
if data_type == "tabular":
    # Tabular EDA tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Univariate Analysis", 
        "🔗 Bivariate Analysis", 
        "🌐 Multivariate Analysis", 
        "📊 Dimensionality Reduction", 
        "📋 Summary Statistics"
    ])
    
    # Tab 1: Univariate Analysis
    with tab1:
        st.subheader("📈 Univariate Analysis")
        st.write("**Analyze individual variables to understand their distributions and characteristics.**")
        
        # Select column for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="chart-card">
                <h4>Numeric Variables</h4>
                <p>Analyze continuous and discrete numeric variables</p>
            </div>
            """, unsafe_allow_html=True)
            
            if numeric_cols:
                selected_numeric = st.selectbox("Select numeric column:", numeric_cols)
                
                if selected_numeric:
                    # Histogram
                    fig_hist = px.histogram(
                        df, 
                        x=selected_numeric, 
                        title=f"Distribution of {selected_numeric}",
                        nbins=30
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Box plot
                    fig_box = px.box(
                        df, 
                        y=selected_numeric, 
                        title=f"Box Plot of {selected_numeric}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Statistics
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    with col_stats1:
                        st.metric("Mean", f"{df[selected_numeric].mean():.2f}")
                    with col_stats2:
                        st.metric("Median", f"{df[selected_numeric].median():.2f}")
                    with col_stats3:
                        st.metric("Std Dev", f"{df[selected_numeric].std():.2f}")
                    with col_stats4:
                        st.metric("Range", f"{df[selected_numeric].max() - df[selected_numeric].min():.2f}")
            else:
                st.info("ℹ️ No numeric columns found")
        
        with col2:
            st.markdown("""
            <div class="chart-card">
                <h4>Categorical Variables</h4>
                <p>Analyze categorical and discrete variables</p>
            </div>
            """, unsafe_allow_html=True)
            
            if categorical_cols:
                selected_categorical = st.selectbox("Select categorical column:", categorical_cols)
                
                if selected_categorical:
                    # Value counts
                    value_counts = df[selected_categorical].value_counts()
                    
                    # Bar chart
                    fig_bar = px.bar(
                        x=value_counts.index, 
                        y=value_counts.values,
                        title=f"Distribution of {selected_categorical}",
                        labels={'x': selected_categorical, 'y': 'Count'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Pie chart
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart of {selected_categorical}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Statistics
                    st.write(f"**Unique values:** {df[selected_categorical].nunique()}")
                    st.write(f"**Most common:** {value_counts.index[0]} ({value_counts.iloc[0]} times)")
            else:
                st.info("ℹ️ No categorical columns found")
    
    # Tab 2: Bivariate Analysis
    with tab2:
        st.subheader("🔗 Bivariate Analysis")
        st.write("**Explore relationships between pairs of variables.**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="chart-card">
                <h4>Numeric vs Numeric</h4>
                <p>Scatter plots and correlations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X variable:", numeric_cols, key="bivariate_x")
                y_col = st.selectbox("Select Y variable:", [col for col in numeric_cols if col != x_col], key="bivariate_y")
                
                if x_col and y_col:
                    # Scatter plot
                    try:
                        fig_scatter = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{x_col} vs {y_col}",
                            trendline="ols"
                        )
                    except ModuleNotFoundError:
                        # Fallback without trendline if statsmodels is not available
                        fig_scatter = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{x_col} vs {y_col} (No trendline - statsmodels not available)"
                        )
                        st.info("💡 **Tip:** Install statsmodels for trendline analysis: `pip install statsmodels`")
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Correlation
                    correlation = df[x_col].corr(df[y_col])
                    st.metric("Correlation", f"{correlation:.3f}")
            else:
                st.info("ℹ️ Need at least 2 numeric columns for bivariate analysis")
        
        with col2:
            st.markdown("""
            <div class="chart-card">
                <h4>Categorical vs Numeric</h4>
                <p>Box plots and group comparisons</p>
            </div>
            """, unsafe_allow_html=True)
            
            if categorical_cols and numeric_cols:
                cat_col = st.selectbox("Select categorical variable:", categorical_cols, key="bivariate_cat")
                num_col = st.selectbox("Select numeric variable:", numeric_cols, key="bivariate_num")
                
                if cat_col and num_col:
                    # Box plot
                    fig_box = px.box(
                        df, 
                        x=cat_col, 
                        y=num_col,
                        title=f"{num_col} by {cat_col}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Group statistics
                    group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
                    st.write("**Group Statistics:**")
                    st.dataframe(group_stats)
            else:
                st.info("ℹ️ Need both categorical and numeric columns for this analysis")
    
    # Tab 3: Multivariate Analysis
    with tab3:
        st.subheader("🌐 Multivariate Analysis")
        st.write("**Analyze relationships between multiple variables simultaneously.**")
        
        if len(numeric_cols) >= 3:
            # Correlation matrix
            st.markdown("""
            <div class="chart-card">
                <h4>Correlation Matrix</h4>
                <p>Heatmap showing correlations between all numeric variables</p>
            </div>
            """, unsafe_allow_html=True)
            
            correlation_matrix = df[numeric_cols].corr()
            
            fig_heatmap = px.imshow(
                correlation_matrix,
                title="Correlation Matrix Heatmap",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 3D Scatter plot
            if len(numeric_cols) >= 3:
                st.markdown("""
                <div class="chart-card">
                    <h4>3D Scatter Plot</h4>
                    <p>Three-dimensional visualization of relationships</p>
                </div>
                """, unsafe_allow_html=True)
                
                x_3d = st.selectbox("Select X variable:", numeric_cols, key="3d_x")
                y_3d = st.selectbox("Select Y variable:", [col for col in numeric_cols if col != x_3d], key="3d_y")
                z_3d = st.selectbox("Select Z variable:", [col for col in numeric_cols if col not in [x_3d, y_3d]], key="3d_z")
                
                if x_3d and y_3d and z_3d:
                    fig_3d = px.scatter_3d(
                        df, 
                        x=x_3d, 
                        y=y_3d, 
                        z=z_3d,
                        title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}"
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("ℹ️ Need at least 3 numeric columns for multivariate analysis")
    
    # Tab 4: Dimensionality Reduction
    with tab4:
        st.subheader("📊 Dimensionality Reduction")
        st.write("**Reduce the number of variables while preserving important information.**")
        
        if len(numeric_cols) >= 2:
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # PCA
                st.markdown("""
                <div class="chart-card">
                    <h4>Principal Component Analysis (PCA)</h4>
                    <p>Reduce dimensions while preserving variance</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare data
                X = df[numeric_cols].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA
                n_components = min(3, len(numeric_cols))
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Explained variance
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # Plot explained variance
                fig_variance = go.Figure()
                fig_variance.add_trace(go.Bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=explained_variance,
                    name='Explained Variance'
                ))
                fig_variance.add_trace(go.Scatter(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=cumulative_variance,
                    name='Cumulative Variance',
                    mode='lines+markers'
                ))
                fig_variance.update_layout(
                    title="PCA Explained Variance",
                    xaxis_title="Principal Components",
                    yaxis_title="Explained Variance Ratio"
                )
                st.plotly_chart(fig_variance, use_container_width=True)
                
                # 2D PCA plot
                if n_components >= 2:
                    pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                    fig_pca = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2',
                        title="PCA: First Two Principal Components"
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                # Component loadings
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=numeric_cols
                )
                st.write("**Component Loadings:**")
                st.dataframe(loadings)
                
            except ImportError:
                st.error("❌ scikit-learn not available. Install with: `pip install scikit-learn`")
        else:
            st.info("ℹ️ Need at least 2 numeric columns for dimensionality reduction")
    
    # Tab 5: Summary Statistics
    with tab5:
        st.subheader("📋 Summary Statistics")
        st.write("**Comprehensive statistical summary of your dataset.**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="chart-card">
                <h4>Numeric Summary</h4>
                <p>Statistical summary of numeric variables</p>
            </div>
            """, unsafe_allow_html=True)
            
            if numeric_cols:
                numeric_summary = df[numeric_cols].describe()
                st.dataframe(numeric_summary)
            else:
                st.info("ℹ️ No numeric columns found")
        
        with col2:
            st.markdown("""
            <div class="chart-card">
                <h4>Categorical Summary</h4>
                <p>Summary of categorical variables</p>
            </div>
            """, unsafe_allow_html=True)
            
            if categorical_cols:
                categorical_summary = {}
                for col in categorical_cols:
                    categorical_summary[col] = {
                        'unique_count': df[col].nunique(),
                        'most_common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                        'missing_count': df[col].isnull().sum()
                    }
                
                cat_summary_df = pd.DataFrame(categorical_summary).T
                st.dataframe(cat_summary_df)
            else:
                st.info("ℹ️ No categorical columns found")

else:  # image data
    # Image EDA - Comprehensive Analysis
    if image_data is not None and len(image_data) > 0:
        st.markdown("""
        <div class="pipeline-step">
            <h3>🖼️ Image Exploratory Data Analysis</h3>
            <p>Comprehensive image analysis for machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🖼️ Image Visualization", "📊 Pixel Intensity Histogram", "🎨 Color Channel Distribution", "🔍 Sharpness & Noise Analysis", "🔄 Augmentation Preview"])
        
        with tab1:
            st.subheader("🖼️ Image Visualization")
            st.write("**Browse and visualize your image dataset.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h4>Image Gallery</h4>
                    <p>Browse through your uploaded images</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Number of images to show
                if image_data is not None and len(image_data) > 0:
                    n_images = min(8, len(image_data))
                else:
                    n_images = 0
                
                # Create image grid
                if image_data is not None and len(image_data) > 0:
                    cols = st.columns(4)
                    for i, img_info in enumerate(image_data[:n_images]):
                        with cols[i % 4]:
                            display_image_safely(img_info['data'], caption=img_info['name'])
                    
                    if len(image_data) > n_images:
                        st.info(f"Showing {n_images} of {len(image_data)} images. Use the slider to see more.")
                else:
                    st.warning("No image data available. Please upload images first.")
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h4>Image Statistics</h4>
                    <p>Statistical overview of your image dataset</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Image statistics
                    sizes = [img['size'] for img in image_data]
                    widths = [size[0] for size in sizes]
                    heights = [size[1] for size in sizes]
                    
                    st.write(f"**Total Images:** {len(image_data)}")
                    st.write(f"**Width Range:** {min(widths)} - {max(widths)} pixels")
                    st.write(f"**Height Range:** {min(heights)} - {max(heights)} pixels")
                    st.write(f"**Average Size:** {np.mean(widths):.0f} × {np.mean(heights):.0f} pixels")
                    
                    # Format distribution
                    formats = [img['format'] for img in image_data if img['format']]
                    if formats:
                        format_counts = pd.Series(formats).value_counts()
                        st.write("**Format Distribution:**")
                        for fmt, count in format_counts.items():
                            st.write(f"- {fmt}: {count} images")
                    
                    # Size consistency check
                    unique_sizes = set(sizes)
                    if len(unique_sizes) == 1:
                        st.success("✅ **All images have consistent size**")
                    else:
                        st.warning(f"⚠️ **Inconsistent sizes detected** ({len(unique_sizes)} different sizes)")
        
        with tab2:
            st.subheader("📊 Pixel Intensity Histogram")
            st.write("**Analyze the distribution of pixel values across your images.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h4>Histogram Analysis</h4>
                    <p>Distribution of pixel intensity values</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Select sample image for analysis
                if image_data is not None and len(image_data) > 0:
                    max_idx = max(0, len(image_data)-1)
                    if max_idx == 0:
                        sample_idx = 0
                        st.info("Only one image available")
                    else:
                        sample_idx = st.slider("Select image for analysis:", 0, max_idx, 0)
                else:
                    sample_idx = 0
                
                if image_data is not None and sample_idx < len(image_data):
                    img_array = image_data[sample_idx]['data']
                    
                    # Ensure proper data type for analysis
                    if img_array.dtype != np.uint8:
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    
                    # Convert to RGB if needed
                    if len(img_array.shape) == 3:
                        num_channels = img_array.shape[2]
                        
                        if num_channels == 3:  # RGB
                            # Color histogram
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            colors = ['red', 'green', 'blue']
                            channel_names = ['Red', 'Green', 'Blue']
                            
                            for i, (color, ax, name) in enumerate(zip(colors, axes, channel_names)):
                                ax.hist(img_array[:, :, i].flatten(), bins=50, color=color, alpha=0.7)
                                ax.set_title(f'{name} Channel')
                                ax.set_xlabel('Pixel Value')
                                ax.set_ylabel('Frequency')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # RGB scatter plot
                            # Sample pixels for visualization
                            sample_pixels = img_array[::10, ::10, :].reshape(-1, 3)
                            fig_rgb = px.scatter_3d(
                                x=sample_pixels[:, 0],
                                y=sample_pixels[:, 1],
                                z=sample_pixels[:, 2],
                                color=['RGB'] * len(sample_pixels),
                                title=f"RGB Color Space - {image_data[sample_idx]['name']}"
                            )
                            st.plotly_chart(fig_rgb, use_container_width=True)
                            
                        elif num_channels == 4:  # RGBA
                            # Color histogram for RGBA
                            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                            colors = ['red', 'green', 'blue', 'gray']
                            channel_names = ['Red', 'Green', 'Blue', 'Alpha']
                            
                            for i, (color, ax, name) in enumerate(zip(colors, axes, channel_names)):
                                ax.hist(img_array[:, :, i].flatten(), bins=50, color=color, alpha=0.7)
                                ax.set_title(f'{name} Channel')
                                ax.set_xlabel('Pixel Value')
                                ax.set_ylabel('Frequency')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.info("RGBA image detected. Alpha channel shows transparency information.")
                            
                        else:  # Other channel counts
                            # Generic histogram for any number of channels
                            fig, axes = plt.subplots(1, min(num_channels, 4), figsize=(5*min(num_channels, 4), 5))
                            if num_channels == 1:
                                axes = [axes]
                            
                            for i in range(min(num_channels, 4)):
                                axes[i].hist(img_array[:, :, i].flatten(), bins=50, color='blue', alpha=0.7)
                                axes[i].set_title(f'Channel {i}')
                                axes[i].set_xlabel('Pixel Value')
                                axes[i].set_ylabel('Frequency')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            if num_channels > 4:
                                st.info(f"Image has {num_channels} channels. Showing first 4 channels.")
                    else:
                        # Grayscale histogram
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(img_array.flatten(), bins=50, color='gray', alpha=0.7)
                        ax.set_title('Grayscale Histogram')
                        ax.set_xlabel('Pixel Value')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h4>Intensity Statistics</h4>
                    <p>Statistical analysis of pixel intensity</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and sample_idx < len(image_data):
                    img_array = image_data[sample_idx]['data']
                    
                    # Ensure proper data type for analysis
                    if img_array.dtype != np.uint8:
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    
                    if len(img_array.shape) == 3:
                        # Calculate color statistics
                        channel_means = np.mean(img_array, axis=(0, 1))
                        channel_stds = np.std(img_array, axis=(0, 1))
                        
                        # Handle different channel counts
                        if len(channel_means) == 3:  # RGB
                            r_mean, g_mean, b_mean = channel_means
                            r_std, g_std, b_std = channel_stds
                            
                            col_stats1, col_stats2 = st.columns(2)
                            
                            with col_stats1:
                                st.metric("Red Mean", f"{r_mean:.1f}")
                                st.metric("Green Mean", f"{g_mean:.1f}")
                                st.metric("Blue Mean", f"{b_mean:.1f}")
                            
                            with col_stats2:
                                st.metric("Red Std", f"{r_std:.1f}")
                                st.metric("Green Std", f"{g_std:.1f}")
                                st.metric("Blue Std", f"{b_std:.1f}")
                        elif len(channel_means) == 4:  # RGBA
                            r_mean, g_mean, b_mean, a_mean = channel_means
                            r_std, g_std, b_std, a_std = channel_stds
                            
                            col_stats1, col_stats2 = st.columns(2)
                            
                            with col_stats1:
                                st.metric("Red Mean", f"{r_mean:.1f}")
                                st.metric("Green Mean", f"{g_mean:.1f}")
                                st.metric("Blue Mean", f"{b_mean:.1f}")
                                st.metric("Alpha Mean", f"{a_mean:.1f}")
                            
                            with col_stats2:
                                st.metric("Red Std", f"{r_std:.1f}")
                                st.metric("Green Std", f"{g_std:.1f}")
                                st.metric("Blue Std", f"{b_std:.1f}")
                                st.metric("Alpha Std", f"{a_std:.1f}")
                        else:  # Other channel counts
                            st.write(f"**Image has {len(channel_means)} channels**")
                            for i, (mean_val, std_val) in enumerate(zip(channel_means, channel_stds)):
                                st.write(f"Channel {i}: Mean={mean_val:.1f}, Std={std_val:.1f}")
                        
                        # Brightness and contrast
                        brightness = np.mean(img_array)
                        contrast = np.std(img_array)
                        
                        st.metric("Overall Brightness", f"{brightness:.1f}")
                        st.metric("Overall Contrast", f"{contrast:.1f}")
                    else:
                        # Grayscale statistics
                        brightness = np.mean(img_array)
                        contrast = np.std(img_array)
                        
                        st.metric("Brightness", f"{brightness:.1f}")
                        st.metric("Contrast", f"{contrast:.1f}")
        
        with tab3:
            st.subheader("🎨 Color Channel Distribution")
            st.write("**Analyze color characteristics and distributions across your image dataset.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h4>Color Analysis</h4>
                    <p>Color channel distribution analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Analyze color distribution across all images
                    all_brightness = []
                    all_contrast = []
                    
                    for img_info in image_data:
                        img_array = img_info['data']
                        # Ensure proper data type for analysis
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        brightness = np.mean(img_array)
                        contrast = np.std(img_array)
                        all_brightness.append(brightness)
                        all_contrast.append(contrast)
                    
                    # Brightness distribution
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    ax1.hist(all_brightness, bins=30, color='skyblue', alpha=0.7)
                    ax1.set_title('Brightness Distribution Across Dataset')
                    ax1.set_xlabel('Brightness')
                    ax1.set_ylabel('Frequency')
                    
                    ax2.hist(all_contrast, bins=30, color='lightcoral', alpha=0.7)
                    ax2.set_title('Contrast Distribution Across Dataset')
                    ax2.set_xlabel('Contrast')
                    ax2.set_ylabel('Frequency')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h4>Color Statistics</h4>
                    <p>Statistical summary of color characteristics</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Calculate statistics
                    all_brightness = []
                    all_contrast = []
                    
                    for img in image_data:
                        img_array = img['data']
                        # Ensure proper data type for analysis
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        all_brightness.append(np.mean(img_array))
                        all_contrast.append(np.std(img_array))
                    
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        st.metric("Avg Brightness", f"{np.mean(all_brightness):.1f}")
                        st.metric("Brightness Std", f"{np.std(all_brightness):.1f}")
                        st.metric("Min Brightness", f"{np.min(all_brightness):.1f}")
                        st.metric("Max Brightness", f"{np.max(all_brightness):.1f}")
                    
                    with col_stats2:
                        st.metric("Avg Contrast", f"{np.mean(all_contrast):.1f}")
                        st.metric("Contrast Std", f"{np.std(all_contrast):.1f}")
                        st.metric("Min Contrast", f"{np.min(all_contrast):.1f}")
                        st.metric("Max Contrast", f"{np.max(all_contrast):.1f}")
                    
                    # Find brightest and darkest images
                    brightest_idx = np.argmax(all_brightness)
                    darkest_idx = np.argmin(all_brightness)
                    
                    st.write(f"**Brightest Image:** {image_data[brightest_idx]['name']} (Brightness: {all_brightness[brightest_idx]:.1f})")
                    st.write(f"**Darkest Image:** {image_data[darkest_idx]['name']} (Brightness: {all_brightness[darkest_idx]:.1f})")
        
        with tab4:
            st.subheader("🔍 Sharpness & Noise Analysis")
            st.write("**Analyze image quality metrics including sharpness and noise levels.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h4>Quality Metrics</h4>
                    <p>Sharpness and noise analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Calculate sharpness using Laplacian variance
                    sharpness_scores = []
                    
                    for img_info in image_data:
                        img_array = img_info['data']
                        
                        # Ensure proper data type for analysis
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        # Convert to grayscale if needed
                        if len(img_array.shape) == 3:
                            # Ensure we have RGB format for OpenCV conversion
                            if img_array.shape[2] == 3:
                                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            elif img_array.shape[2] == 4:
                                # Convert RGBA to RGB first, then to grayscale
                                rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                            else:
                                # For other channel counts, take the first channel as grayscale
                                gray = img_array[:, :, 0]
                        else:
                            gray = img_array
                        
                        # Calculate Laplacian variance (sharpness measure)
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        sharpness_scores.append(laplacian_var)
                    
                    # Sharpness distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(sharpness_scores, bins=30, color='green', alpha=0.7)
                    ax.set_title('Sharpness Distribution Across Dataset')
                    ax.set_xlabel('Sharpness Score (Laplacian Variance)')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h4>Quality Assessment</h4>
                    <p>Quality metrics and recommendations</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Calculate quality metrics
                    all_brightness = []
                    all_contrast = []
                    sharpness_scores = []
                    
                    for img_info in image_data:
                        img_array = img_info['data']
                        
                        # Ensure proper data type for analysis
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        # Calculate brightness and contrast
                        all_brightness.append(np.mean(img_array))
                        all_contrast.append(np.std(img_array))
                        
                        # Calculate sharpness
                        if len(img_array.shape) == 3:
                            # Ensure we have RGB format for OpenCV conversion
                            if img_array.shape[2] == 3:
                                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            elif img_array.shape[2] == 4:
                                # Convert RGBA to RGB first, then to grayscale
                                rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                            else:
                                # For other channel counts, take the first channel as grayscale
                                gray = img_array[:, :, 0]
                        else:
                            gray = img_array
                        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        sharpness_scores.append(laplacian_var)
                    
                    # Quality metrics
                    col_metrics1, col_metrics2 = st.columns(2)
                    
                    with col_metrics1:
                        st.metric("Avg Sharpness", f"{np.mean(sharpness_scores):.1f}")
                        st.metric("Sharpness Std", f"{np.std(sharpness_scores):.1f}")
                        st.metric("Min Sharpness", f"{np.min(sharpness_scores):.1f}")
                        st.metric("Max Sharpness", f"{np.max(sharpness_scores):.1f}")
                    
                    with col_metrics2:
                        st.metric("Avg Brightness", f"{np.mean(all_brightness):.1f}")
                        st.metric("Avg Contrast", f"{np.mean(all_contrast):.1f}")
                        st.metric("Brightness CV", f"{np.std(all_brightness)/np.mean(all_brightness):.2f}")
                        st.metric("Contrast CV", f"{np.std(all_contrast)/np.mean(all_contrast):.2f}")
                    
                    # Quality assessment
                    st.write("**🔍 Quality Assessment:**")
                    
                    # Check for issues
                    issues = []
                    if np.std(all_brightness) / np.mean(all_brightness) > 0.3:
                        issues.append("High brightness variability")
                    if np.std(all_contrast) / np.mean(all_contrast) > 0.3:
                        issues.append("High contrast variability")
                    if np.mean(sharpness_scores) < 100:
                        issues.append("Low overall sharpness")
                    
                    if issues:
                        st.warning("⚠️ **Potential quality issues detected:**")
                        for issue in issues:
                            st.write(f"- {issue}")
                    else:
                        st.success("✅ **Good overall image quality**")
                    
                    # Find sharpest and blurriest images
                    sharpest_idx = np.argmax(sharpness_scores)
                    blurriest_idx = np.argmin(sharpness_scores)
                    
                    st.write(f"**Sharpest Image:** {image_data[sharpest_idx]['name']} (Score: {sharpness_scores[sharpest_idx]:.1f})")
                    st.write(f"**Blurriest Image:** {image_data[blurriest_idx]['name']} (Score: {sharpness_scores[blurriest_idx]:.1f})")
        
        with tab5:
            st.subheader("🔄 Augmentation Preview")
            st.write("**Preview how data augmentation techniques would affect your images.**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="chart-card">
                    <h4>Augmentation Options</h4>
                    <p>Preview different augmentation techniques</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    # Select sample image
                    max_idx = max(0, len(image_data)-1)
                    if max_idx == 0:
                        sample_idx = 0
                        st.info("Only one image available")
                    else:
                        sample_idx = st.slider("Select image for augmentation preview:", 0, max_idx, 0, key="augmentation_preview")
                    
                    # Augmentation options
                    augmentation_type = st.selectbox(
                        "Augmentation type:",
                        ["Original", "Horizontal Flip", "Vertical Flip", "Rotation", "Brightness", "Contrast", "Blur"]
                    )
                    
                    if st.button("🔄 Apply Augmentation Preview"):
                        img_array = image_data[sample_idx]['data'].copy()
                        
                        # Ensure we're working with the right data type
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        if augmentation_type == "Horizontal Flip":
                            img_array = np.fliplr(img_array)
                        elif augmentation_type == "Vertical Flip":
                            img_array = np.flipud(img_array)
                        elif augmentation_type == "Rotation":
                            # Rotate 90 degrees clockwise
                            img_array = np.rot90(img_array, k=-1)
                        elif augmentation_type == "Brightness":
                            # Increase brightness
                            img_array = np.clip(img_array.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
                        elif augmentation_type == "Contrast":
                            # Increase contrast
                            mean_val = np.mean(img_array.astype(np.float32))
                            img_array = np.clip((img_array.astype(np.float32) - mean_val) * 1.5 + mean_val, 0, 255).astype(np.uint8)
                        elif augmentation_type == "Blur":
                            # Apply Gaussian blur
                            if len(img_array.shape) == 3:
                                img_array = cv2.GaussianBlur(img_array, (15, 15), 0)
                            else:
                                img_array = cv2.GaussianBlur(img_array, (15, 15), 0)
                        
                        # Store augmented image for display
                        st.session_state.augmented_preview = img_array
                        st.session_state.augmentation_type = augmentation_type
            
            with col2:
                st.markdown("""
                <div class="chart-card">
                    <h4>Augmentation Preview</h4>
                    <p>Compare original vs augmented image</p>
                </div>
                """, unsafe_allow_html=True)
                
                if image_data is not None and len(image_data) > 0:
                    if 'augmented_preview' in st.session_state:
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Original**")
                            display_image_safely(image_data[sample_idx]['data'], caption="Original")
                        
                        with col_b:
                            st.write(f"**{st.session_state.augmentation_type}**")
                            display_image_safely(st.session_state.augmented_preview, caption=st.session_state.augmentation_type)
                    else:
                        st.info("Click 'Apply Augmentation Preview' to see the effect of different augmentation techniques.")
    else:
        st.warning("⚠️ No image data available for analysis. Please upload images first.")

# Navigation section
st.header("🚀 What's Next?")

col1, col2 = st.columns(2)

with col1:
    st.info("**⚙️ Next Step: Model Training**")
    st.write("Train machine learning models with your analyzed data")
    if st.button("⚙️ Go to Model Training", type="primary", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")

with col2:
    st.info("**🔧 Alternative: Data Preprocessing**")
    st.write("Go back to clean and prepare your data")
    if st.button("🔧 Go to Data Preprocessing", use_container_width=True):
            safe_switch_page("pages/2_🔧_Data_Preprocessing.py")