# app/pages/3_📊_EDA.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ✅ Keep 'utilss' as you confirmed it's correct
from app.utilss.navigation import safe_switch_page
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(layout="wide", page_title="EDA", page_icon="📊")

# === CUSTOM CSS (Same as Train Model) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    .nav-icon {
        font-size: 1.5rem; margin-right: 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-value {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #718096; font-size: 0.9rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 12px;
        padding: 0.8rem 2rem; font-weight: 600; transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    .stSuccess, .stInfo, .stWarning { border-radius: 12px; border: none; }
</style>
""", unsafe_allow_html=True)

# === Main Container ===
with stylable_container("main_container", css_styles="""
    { background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 2rem;
      backdrop-filter: blur(10px); box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
"""):
    # === Header ===
    with stylable_container("page_header", css_styles="""
        { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
          color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; }
    """):
        st.markdown("""
        <div style="font-size: 3rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            <i class="fas fa-chart-network nav-icon"></i>Exploratory Data Analysis
        </div>
        <div style="font-size: 1.2rem; opacity: 0.9;">Uncover patterns, trends, and insights</div>
        """, unsafe_allow_html=True)

    # === Data Check ===
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("❌ No data loaded. Please go to Data Upload.")
        if st.button("📂 Go to Data Upload", use_container_width=True):
            safe_switch_page("pages/1_📂_Data_Upload.py")
        st.stop()

    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # === Dataset Overview ===
    colored_header("📊 Dataset Overview", "Key facts about your data", "blue-70")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container("metric_rows", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[0]:,}</div><div class='metric-label'>Rows</div>", unsafe_allow_html=True)
    with col2:
        with stylable_container("metric_cols", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{df.shape[1]:,}</div><div class='metric-label'>Columns</div>", unsafe_allow_html=True)
    with col3:
        mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        with stylable_container("metric_mem", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{mem:.1f}</div><div class='metric-label'>Memory (MB)</div>", unsafe_allow_html=True)
    with col4:
        missing = df.isnull().sum().sum()
        with stylable_container("metric_missing", css_styles="""{ text-align:center; padding:1.5rem; }"""):
            st.markdown(f"<div class='metric-value'>{missing:,}</div><div class='metric-label'>Missing Values</div>", unsafe_allow_html=True)

    st.markdown("---")

    # === Univariate Analysis ===
    colored_header("📈 Univariate Analysis", "Analyze one variable at a time", "violet-70")
    tab1, tab2 = st.tabs(["📊 Numeric", "🏷️ Categorical"])

    with tab1:
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            col = st.selectbox("Select numeric column", numeric_cols, key="eda_num_col")
            fig = px.histogram(df, x=col, nbins=50, marginal="violin", title=f"Distribution of {col}",
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)

            if df[col].nunique() <= 10:
                counts = df[col].value_counts()
                fig_pie = px.pie(values=counts.values, names=counts.index, title=f"Composition of {col}")
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        if not categorical_cols:
            st.info("No categorical columns found.")
        else:
            col = st.selectbox("Select categorical column", categorical_cols, key="eda_cat_col")
            top = df[col].value_counts().head(10)
            fig_bar = px.bar(top, x=top.index, y=top.values, title=f"Top 10 Categories in {col}",
                             color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(df, names=col, title=f"Category Distribution: {col}")
            st.plotly_chart(fig_pie, use_container_width=True)

    # === Multivariate Analysis ===
    colored_header("🔗 Multivariate Analysis", "Explore relationships between variables", "green-70")
    mv_tab1, mv_tab2, mv_tab3, mv_tab4 = st.tabs(["🧮 Correlation", "📉 Scatter", "🎻 Violin", "🧩 Clusters"])

    with mv_tab1:
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='Blues', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    with mv_tab2:
        if len(numeric_cols) < 2:
            st.info("Need 2+ numeric columns.")
        else:
            c1, c2 = st.columns(2)
            with c1: x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with c2: y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_color")
            fig = px.scatter(df, x=x_col, y=y_col, color=color_by if color_by != "None" else None,
                             title=f"{y_col} vs {x_col}", opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

    with mv_tab3:
        if not numeric_cols or not categorical_cols:
            st.info("Need numeric and categorical columns.")
        else:
            num_col = st.selectbox("Numeric variable", numeric_cols, key="violin_num")
            cat_col = st.selectbox("Category variable", categorical_cols, key="violin_cat")
            top_cats = df[cat_col].value_counts().index[:5]
            df_filtered = df[df[cat_col].isin(top_cats)]
            fig = px.violin(df_filtered, x=cat_col, y=num_col, box=True, points="outliers",
                            title=f"Violin Plot: {num_col} by {cat_col} (Top 5)")
            st.plotly_chart(fig, use_container_width=True)

    with mv_tab4:
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric features for clustering.")
        else:
            n_clusters = st.slider("Number of clusters", 2, 6, key="n_clusters_slider")
            sample_df = df[numeric_cols].dropna().sample(n=min(1000, len(df)), random_state=42)
            scaled_data = StandardScaler().fit_transform(sample_df)
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(scaled_data)
            fig = px.scatter(sample_df, x=numeric_cols[0], y=numeric_cols[1], color=labels.astype(str),
                             title=f"K-Means Clusters (k={n_clusters})")
            st.plotly_chart(fig, use_container_width=True)

    # === Dimensionality Reduction ===
    colored_header("🔄 Dimensionality Reduction", "Reduce features using PCA", "blue-70")
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for PCA.")
    else:
        n_comp = st.slider("Number of Principal Components", 1, min(10, len(numeric_cols)), 2, key="pca_n_comp")
        scale = st.checkbox("Standardize features before PCA", value=True)

        data_clean = df[numeric_cols].dropna()
        scaled_data = StandardScaler().fit_transform(data_clean) if scale else data_clean.values

        pca = PCA(n_components=n_comp)
        reduced_data = pca.fit_transform(scaled_data)
        explained = pca.explained_variance_ratio_

        st.write("✅ Explained Variance per Component:", [f"{v:.1%}" for v in explained])
        st.write("📈 Cumulative Variance:", f"**{sum(explained):.1%}**")

        reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_comp)])
        reduced_df['original_index'] = data_clean.index

        # PCA Visualization
        if n_comp >= 2:
            fig_pca = px.scatter(reduced_df, x="PC1", y="PC2", title="PCA: First Two Components",
                                 color=reduced_df.index, color_continuous_scale='Blues')
            st.plotly_chart(fig_pca, use_container_width=True)

        # Save & Continue Button (✅ This will now work)
        st.info("This reduced dataset will be used for training.")

        if st.button("✅ Save Reduced Data & Continue to Training", type="primary", use_container_width=True):
            st.session_state.df = reduced_df  # Replace with reduced data
            st.session_state.data_reduced = True
            st.session_state.pca_info = {
                "n_components": n_comp,
                "explained_variance": explained.tolist(),
                "cumulative": float(sum(explained))
            }
            safe_switch_page("pages/4_⚙️_Train_Model.py")  # ✅ Correct full path

    # === Final Navigation ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Go to Training", use_container_width=True):
            safe_switch_page("pages/4_⚙️_Train_Model.py")  # ✅ Full path
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            safe_switch_page("app.py")