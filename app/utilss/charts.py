# app/utils/charts.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Set plotly theme for better styling
import plotly.io as pio
pio.templates.default = "plotly_white"

# ---------- UNIVARIATE ANALYSIS ----------
def univariate_analysis(df):
    st.subheader("📊 **Univariate Analysis - Understanding Individual Variables**")
    st.info("""
    **Univariate analysis helps you understand each variable in isolation:**
    - **Distribution shape**: How values are spread out
    - **Central tendency**: Where most values cluster
    - **Variability**: How much values differ from each other
    - **Outliers**: Unusual or extreme values
    """)
    
    # Column selection with guidance
    col = st.selectbox(
        "🎯 **Select a column to analyze:**",
        df.columns,
        help="Choose a column to explore its distribution and characteristics"
    )
    
    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(f"**📈 Analyzing numeric column: {col}**")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[col].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[col].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[col].std():.2f}")
        with col4:
            st.metric("Range", f"{df[col].max() - df[col].min():.2f}")
        
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram with KDE', 'Box Plot', 'Violin Plot', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram with KDE
        fig.add_trace(
            go.Histogram(x=df[col], name="Histogram", nbinsx=30, opacity=0.7),
            row=1, col=1
        )
        
        # Add KDE line
        hist, bins = np.histogram(df[col].dropna(), bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        fig.add_trace(
            go.Scatter(x=bin_centers, y=hist, name="KDE", line=dict(color='red')),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[col], name="Box Plot", boxpoints="outliers"),
            row=1, col=2
        )
        
        # Violin plot
        fig.add_trace(
            go.Violin(y=df[col], name="Violin Plot", box_visible=True, meanline_visible=True),
            row=2, col=1
        )
        
        # Q-Q plot (simplified)
        sorted_data = np.sort(df[col].dropna())
        theoretical_quantiles = np.quantile(np.random.normal(0, 1, len(sorted_data)), np.linspace(0, 1, len(sorted_data)))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name="Q-Q Plot"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text=f"Comprehensive Analysis of {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("💡 **Key Insights**")
        skewness = df[col].skew()
        if abs(skewness) < 0.5:
            st.success("✅ **Distribution is approximately normal** (symmetric)")
        elif skewness > 0.5:
            st.info("📈 **Right-skewed distribution** (long tail to the right)")
        else:
            st.info("📉 **Left-skewed distribution** (long tail to the left)")
            
        # Outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            st.warning(f"⚠️ **{len(outliers)} outliers detected** (using IQR method)")
        else:
            st.success("✅ **No outliers detected** using standard methods")
            
    else:
        st.write(f"**📊 Analyzing categorical column: {col}**")
        
        # Value counts with percentages
        value_counts = df[col].value_counts()
        value_counts_pct = df[col].value_counts(normalize=True) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Counts:**")
            st.dataframe(value_counts.reset_index().rename(columns={'index': col, col: 'Count'}))
        
        with col2:
            st.write("**Percentages:**")
            st.dataframe(value_counts_pct.reset_index().rename(columns={'index': col, col: 'Percentage'}))
        
        # Visualizations
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bar Chart', 'Pie Chart'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name="Counts"),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.4),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text=f"Categorical Analysis of {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("💡 **Key Insights**")
        unique_count = df[col].nunique()
        if unique_count <= 5:
            st.success(f"✅ **Low cardinality** ({unique_count} unique values) - Good for most ML algorithms")
        elif unique_count <= 20:
            st.info(f"📊 **Medium cardinality** ({unique_count} unique values) - Consider encoding strategies")
        else:
            st.warning(f"⚠️ **High cardinality** ({unique_count} unique values) - May need feature engineering")

# ---------- BIVARIATE ANALYSIS ----------
def bivariate_analysis(df):
    st.subheader("🔗 **Bivariate Analysis - Exploring Relationships Between Variables**")
    st.info("""
    **Bivariate analysis reveals how two variables relate to each other:**
    - **Correlation**: How strongly variables move together
    - **Patterns**: Visual relationships and trends
    - **Outliers**: Unusual combinations of values
    - **Interaction effects**: How variables influence each other
    """)
    
    # Column selection
    col_x = st.selectbox("📊 **X-axis variable:**", df.columns, key="bivariate_x")
    col_y = st.selectbox("📈 **Y-axis variable:**", df.columns, key="bivariate_y")
    
    if col_x == col_y:
        st.warning("⚠️ Please select different variables for X and Y axes")
        return
    
    # Analysis based on data types
    if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
        st.write(f"**📊 Numeric-Numeric Analysis: {col_x} vs {col_y}**")
        
        # Correlation analysis
        correlation = df[col_x].corr(df[col_y])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        # Interpret correlation
        if abs(correlation) < 0.1:
            st.info("💡 **Very weak correlation** - Variables are essentially independent")
        elif abs(correlation) < 0.3:
            st.info("💡 **Weak correlation** - Slight relationship exists")
        elif abs(correlation) < 0.5:
            st.info("💡 **Moderate correlation** - Clear relationship present")
        elif abs(correlation) < 0.7:
            st.info("💡 **Strong correlation** - Strong relationship present")
        else:
            st.info("💡 **Very strong correlation** - Very strong relationship present")
        
        # Create comprehensive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scatter Plot', 'Scatter with Trend', 'Hexbin Plot', '2D Histogram'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=df[col_x], y=df[col_y], mode='markers', name="Data Points", opacity=0.6),
            row=1, col=1
        )
        
        # Scatter with trend line
        fig.add_trace(
            go.Scatter(x=df[col_x], y=df[col_y], mode='markers', name="Data Points", opacity=0.6),
            row=1, col=2
        )
        
        # Add trend line
        z = np.polyfit(df[col_x].dropna(), df[col_y].dropna(), 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=df[col_x], y=p(df[col_x]), mode='lines', name="Trend Line", line=dict(color='red')),
            row=1, col=2
        )
        
        # Hexbin plot for density
        fig.add_trace(
            go.Histogram2d(x=df[col_x], y=df[col_y], nbinsx=20, nbinsy=20, colorscale='Viridis'),
            row=2, col=1
        )
        
        # 2D histogram
        fig.add_trace(
            go.Histogram2d(x=df[col_x], y=df[col_y], nbinsx=20, nbinsy=20, colorscale='Blues'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text=f"Bivariate Analysis: {col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.write(f"**📊 Mixed/Categorical Analysis: {col_x} vs {col_y}**")
        
        # Determine which is categorical
        if pd.api.types.is_numeric_dtype(df[col_x]):
            cat_col, num_col = col_y, col_x
        else:
            cat_col, num_col = col_x, col_y
        
        # Box plot
        fig = px.box(df, x=cat_col, y=num_col, title=f"Distribution of {num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Violin plot
        fig = px.violin(df, x=cat_col, y=num_col, title=f"Detailed Distribution of {num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Group statistics
        st.subheader("📊 **Group Statistics**")
        group_stats = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(group_stats)

# ---------- MULTIVARIATE ANALYSIS ----------
def multivariate_analysis(df):
    st.subheader("🌐 **Multivariate Analysis - Complex Relationships and Patterns**")
    st.info("""
    **Multivariate analysis reveals complex patterns across multiple variables:**
    - **Correlation networks**: How all variables relate to each other
    - **Feature interactions**: Variables that work together
    - **Dimensional patterns**: Multi-dimensional relationships
    - **Clustering insights**: Natural groupings in your data
    """)
    
    # Correlation heatmap
    st.write("**🔥 Correlation Heatmap - Understanding Variable Relationships**")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    st.subheader("💡 **Correlation Insights**")
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if strong_corr:
        st.warning("⚠️ **Strong correlations detected** (|r| > 0.7):")
        for var1, var2, corr_val in strong_corr:
            st.write(f"- **{var1}** and **{var2}**: {corr_val:.3f}")
        st.info("💡 **Consider:** Feature selection to avoid multicollinearity")
    else:
        st.success("✅ **No strong correlations detected** - Good feature independence")
    
    # 3D Scatter plot
    if numeric_df.shape[1] >= 3:
        st.write("**🌌 3D Scatter Plot - Three-Dimensional Relationships**")
        cols = st.multiselect(
            "Select 3 columns for 3D visualization:",
            numeric_df.columns,
            default=numeric_df.columns[:3] if len(numeric_df.columns) >= 3 else numeric_df.columns
        )
        
        if len(cols) == 3:
            fig = px.scatter_3d(
                df, 
                x=cols[0], 
                y=cols[1], 
                z=cols[2],
                title=f"3D Scatter: {cols[0]} vs {cols[1]} vs {cols[2]}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance ranking (simplified)
    st.write("**📊 Feature Variance Analysis**")
    variance = numeric_df.var().sort_values(ascending=False)
    fig = px.bar(
        x=variance.index,
        y=variance.values,
        title="Feature Variance (Higher = More Variable)",
        labels={'x': 'Features', 'y': 'Variance'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------- DIMENSIONALITY REDUCTION ----------
def dimensionality_reduction(df):
    st.subheader("📉 **Dimensionality Reduction - Simplifying Complex Data**")
    st.info("""
    **Dimensionality reduction helps visualize and understand high-dimensional data:**
    - **Hidden structure**: Underlying patterns not visible in original dimensions
    - **Feature relationships**: How variables group and relate to each other
    - **Data visualization**: Making complex data easier to understand
    - **Model efficiency**: Reducing complexity while preserving information
    """)
    
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.shape[1] < 2:
        st.warning("⚠️ Need at least 2 numeric columns for dimensionality reduction")
        return
    
    if numeric_df.shape[1] < 3:
        st.info("ℹ️ **Low-dimensional data** - Dimensionality reduction may not be necessary")
        return
    
    # Method selection
    method = st.selectbox(
        "🔧 **Select reduction method:**",
        ["PCA", "t-SNE"],
        help="PCA: Linear reduction, preserves global structure. t-SNE: Non-linear, preserves local structure."
    )
    
    if method == "PCA":
        st.write("**📊 Principal Component Analysis (PCA) - Linear Dimensionality Reduction**")
        
        # PCA with different components
        n_components = st.slider("Number of components to show:", 2, min(10, numeric_df.shape[1]), 2)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PCA Components', 'Explained Variance'),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # PCA scatter
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0], 
                y=pca_result[:, 1], 
                mode='markers',
                marker=dict(size=6, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Explained variance bar chart
        fig.add_trace(
            go.Bar(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=explained_variance,
                name="Individual Variance"
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="PCA Analysis Results")
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("💡 **PCA Insights**")
        st.write(f"**First 2 components explain {cumulative_variance[1]:.1%} of total variance**")
        if cumulative_variance[1] > 0.8:
            st.success("✅ **Excellent!** First 2 components capture most of the data structure")
        elif cumulative_variance[1] > 0.6:
            st.info("📊 **Good!** First 2 components capture substantial structure")
        else:
            st.warning("⚠️ **Limited capture** - Consider more components or different method")
        
        # Feature contributions
        st.write("**🔍 Feature Contributions to First 2 Components:**")
        feature_contrib = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=numeric_df.columns
        )
        st.dataframe(feature_contrib.round(3))
        
    elif method == "t-SNE":
        st.write("**🎨 t-SNE - Non-linear Dimensionality Reduction**")
        st.info("**t-SNE is great for visualizing clusters and local structure**")
        
        # t-SNE parameters
        perplexity = st.slider("Perplexity (controls cluster size):", 5, 50, 30)
        learning_rate = st.slider("Learning rate:", 10, 1000, 200)
        
        with st.spinner("Running t-SNE (this may take a moment)..."):
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                learning_rate=learning_rate
            )
            tsne_result = tsne.fit_transform(numeric_df)
        
        # Create visualization
        fig = px.scatter(
            x=tsne_result[:, 0], 
            y=tsne_result[:, 1],
            title=f"t-SNE Visualization (Perplexity: {perplexity})",
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ **t-SNE completed!** Look for clusters and patterns in the visualization")

# ---------- TIME SERIES ANALYSIS ----------
def time_series_analysis(df):
    st.subheader("⏰ **Time Series & Specialized Analysis - Temporal and Advanced Patterns**")
    st.info("""
    **Time series analysis reveals patterns that change over time:**
    - **Trends**: Long-term changes or directions
    - **Seasonality**: Regular repeating patterns (daily, weekly, yearly)
    - **Cycles**: Irregular repeating patterns
    - **Anomalies**: Unusual time periods or events
    """)
    
    # Time column selection
    time_col = st.selectbox(
        "⏰ **Select time column:**",
        df.columns,
        help="Choose the column that represents time or dates"
    )
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            st.success("✅ **Successfully converted to datetime format**")
        except Exception as e:
            st.error(f"❌ **Could not convert to datetime:** {str(e)}")
            st.info("💡 **Tip:** Ensure your time column has consistent date formats")
            return
    
    # Numeric column selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("⚠️ **No numeric columns found** for time series analysis")
        return
    
    num_col = st.selectbox(
        "📊 **Select numeric column to analyze:**",
        numeric_cols,
        help="Choose the numeric variable to plot over time"
    )
    
    # Sort by time
    df_sorted = df.sort_values(time_col).dropna(subset=[time_col, num_col])
    
    if len(df_sorted) == 0:
        st.warning("⚠️ **No valid data points** after removing missing values")
        return
    
    # Time series visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series Line Plot', 'Rolling Statistics', 'Seasonal Decomposition', 'Autocorrelation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main time series plot
    fig.add_trace(
        go.Scatter(
            x=df_sorted[time_col], 
            y=df_sorted[num_col], 
            mode='lines+markers',
            name="Time Series"
        ),
        row=1, col=1
    )
    
    # Rolling statistics
    window = st.slider("Rolling window size:", 3, min(30, len(df_sorted)//4), 7)
    rolling_mean = df_sorted[num_col].rolling(window=window).mean()
    rolling_std = df_sorted[num_col].rolling(window=window).std()
    
    fig.add_trace(
        go.Scatter(x=df_sorted[time_col], y=rolling_mean, name=f"Rolling Mean ({window})", line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_sorted[time_col], y=rolling_std, name=f"Rolling Std ({window})", line=dict(color='green')),
        row=1, col=2
    )
    
    # Seasonal decomposition (simplified)
    # For now, just show the original data in the seasonal plot
    fig.add_trace(
        go.Scatter(x=df_sorted[time_col], y=df_sorted[num_col], mode='lines', name="Original Data"),
        row=2, col=1
    )
    
    # Autocorrelation (simplified)
    # Calculate lag-1 autocorrelation
    if len(df_sorted) > 1:
        lag1_corr = df_sorted[num_col].autocorr(lag=1)
        fig.add_trace(
            go.Bar(x=['Lag 1'], y=[lag1_corr], name="Autocorrelation"),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text=f"Time Series Analysis: {num_col} over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series insights
    st.subheader("💡 **Time Series Insights**")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Time Period", f"{df_sorted[time_col].max() - df_sorted[time_col].min()}")
    with col2:
        st.metric("Data Points", len(df_sorted))
    with col3:
        st.metric("Start Date", df_sorted[time_col].min().strftime("%Y-%m-%d"))
    with col4:
        st.metric("End Date", df_sorted[time_col].max().strftime("%Y-%m-%d"))
    
    # Trend analysis
    if len(df_sorted) > 1:
        # Simple linear trend
        x_trend = np.arange(len(df_sorted))
        y_trend = df_sorted[num_col].values
        z = np.polyfit(x_trend, y_trend, 1)
        trend_slope = z[0]
        
        if trend_slope > 0:
            st.success("📈 **Upward trend detected** - Values generally increase over time")
        elif trend_slope < 0:
            st.info("📉 **Downward trend detected** - Values generally decrease over time")
        else:
            st.info("➡️ **No clear trend** - Values are relatively stable over time")
