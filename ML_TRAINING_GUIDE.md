# InstaML Machine Learning Guide

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering & Selection](#feature-engineering-selection)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Deployment](#deployment)
8. [Best Practices](#best-practices)

## Platform Overview

InstaML provides a complete machine learning workflow through seven main modules, each designed to handle specific aspects of the ML pipeline.

### Module Structure

1. **Data Upload** - Import and validate datasets
2. **Data Preprocessing** - Clean and prepare data for modeling
3. **EDA (Exploratory Data Analysis)** - Understand data patterns and relationships
4. **Feature Engineering & Selection** - Construct powerful features and prune dimensions
5. **Train Model** - Build and optimize machine learning models
6. **Test Model** - Evaluate model performance and validate results
7. **Deploy Model** - Make models available for real-world predictions

## Data Preprocessing

### Overview Tab
- **Data Quality Assessment**: Identifies missing values, data types, and memory usage
- **Dataset Summary**: Provides statistical overview of numerical and categorical features
- **Data Validation**: Checks for common data quality issues

### Recommendations Tab
- **AI-Powered Suggestions**: Automated preprocessing recommendations based on data analysis
- **Feature Engineering**: Suggestions for new feature creation
- **Data Quality Improvements**: Recommendations for handling missing values and outliers

### Data Cleaning Tab
- **Duplicate Removal**: Identifies and removes duplicate records
- **Column Selection**: Choose relevant features for analysis
- **Data Type Conversion**: Convert columns to appropriate data types
- **Invalid Data Handling**: Manage inconsistent or invalid entries

### Scaling & Encoding Tab
- **Feature Scaling**: 
  - StandardScaler (z-score normalization)
  - MinMaxScaler (0-1 scaling)
  - RobustScaler (median-based scaling)
- **Categorical Encoding**:
  - Label Encoding for ordinal data
  - One-Hot Encoding for nominal data
  - Target Encoding for high-cardinality features

### Missing Values & Outliers Tab
- **Missing Value Strategies**:
  - Mean/Median imputation for numerical features
  - Mode imputation for categorical features
  - Forward/Backward fill for time series
  - Custom value imputation
- **Outlier Detection**:
  - IQR (Interquartile Range) method
  - Z-score analysis
  - Isolation Forest algorithm
  - Manual threshold setting

## Exploratory Data Analysis

### Visualization Categories

#### Univariate Analysis
- **Histograms**: Distribution of numerical features
- **Box Plots**: Quartile analysis and outlier identification
- **Violin Plots**: Distribution shape and density
- **Count Plots**: Frequency of categorical values
- **Density Plots**: Smooth distribution curves

#### Bivariate Analysis
- **Scatter Plots**: Relationship between two numerical variables
- **Correlation Heatmaps**: Feature correlation matrix
- **Joint Plots**: Combined scatter and distribution plots
- **Box Plots by Category**: Numerical distribution across categories
- **Cross-tabulation**: Relationship between categorical variables

#### Multivariate Analysis
- **Pair Plots**: Pairwise relationships between all features
- **Parallel Coordinates**: Multi-dimensional data visualization
- **Radar Charts**: Multi-attribute comparison
- **3D Scatter Plots**: Three-dimensional relationships
- **Correlation Networks**: Feature relationship networks

#### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction
- **t-SNE**: Non-linear embedding for visualization
- **UMAP**: Uniform manifold approximation and projection
- **Factor Analysis**: Identify underlying factors in data

#### Time Series Analysis
- **Time Series Plots**: Temporal trend visualization
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Autocorrelation**: Time series correlation analysis
- **Rolling Statistics**: Moving averages and statistics

### Statistical Insights
- **Descriptive Statistics**: Mean, median, standard deviation, skewness
- **Correlation Analysis**: Pearson, Spearman correlation coefficients
- **Feature Importance**: Relative importance of features for prediction
- **Distribution Testing**: Normality tests and distribution fitting

## Feature Engineering & Selection

Feature engineering and selection are crucial steps in building powerful, robust machine learning models. InstaML separates these into dedicated panels to give you granular control over input dimensions.

### Feature Engineering
Feature Engineering allows you to synthesize new dimensions from existing features, providing stronger learning signals to the models.

#### 1. Mathematical Operations
Create a new feature by combining two existing numerical features using basic arithmetic operators:
- **Addition ($+$)**: Useful for summing counts, parts, or magnitudes.
- **Subtraction ($-$)**: Calculates differences (e.g., age from birth year).
- **Multiplication ($*$)**: Captures interaction effects (e.g., Area $=$ Width $\times$ Height).
- **Division ($/$)**: Creates ratios (e.g., income-to-loan ratio). Handles division-by-zero errors safely by adding a minor epsilon ($1\times10^{-9}$).

#### 2. Mathematical Transformations
Modify skewness and distribution of numerical columns:
- **Logarithm ($\ln(x)$)**: Stabilizes variance and normalizes right-skewed data distributions (e.g., salary, price).
- **Square Root ($\sqrt{x}$)**: Dampens the effect of extreme values.
- **Square ($x^2$)**: Amplifies differences in high-value ranges.
- **Absolute Value ($|x|$)**: Disregards negative directions to capture magnitude.

#### 3. Continuous Binning
Discretizes continuous numerical values into ordinal categories (bins/quantiles). Helps algorithms like linear models capture non-linear steps.

---

### Feature Selection
Feature Selection is the process of reducing the number of input variables when developing a predictive model. It helps simplify the model, speed up training, and reduce overfitting by removing redundant or noisy columns.

#### 1. Variance Threshold
An unsupervised selection filter. It calculates the variance of each numerical feature and drops any column with variance below the threshold.
- **When to use**: To remove columns that are constant or near-constant across all samples.
- **Mathematical Form**: Drops feature $X$ if:
  $$\text{Var}(X) = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2 \le \theta$$
  where $\theta$ is the user-defined threshold (default is $0.01$).

#### 2. Select K Best (ANOVA)
A supervised univariate selection method. It scores each numeric feature against the target column and keeps the top $k$ highest-scoring columns.
- **Classification Tasks**: Uses ANOVA F-value (`f_classif`) to check if means of features differ significantly across classes.
- **Regression Tasks**: Uses univariate linear regression F-test (`f_regression`) to check for linear dependency with the continuous target.
- **When to use**: When you want a quick, model-agnostic ranking of features based on individual correlation with the target.

#### 3. Correlation-based Filter
A filter method designed to combat multi-collinearity. It computes the absolute Pearson correlation matrix of all features. If two features have a correlation coefficient exceeding the threshold, one of the features is removed.
- **When to use**: To drop highly redundant variables (e.g., keeping only one of `zip_code` and `postal_code`).
- **Standard Threshold**: $0.85$.

#### 4. Recursive Feature Elimination (RFE & RFECV)
A supervised wrapper method. RFE starts with all features, fits a model (Random Forest is used as the default), ranks features by importance, and removes the least important features recursively until the target number of features is reached.
- **RFE**: Drops features until a user-specified count $n$ is reached.
- **RFECV**: Performs RFE within a cross-validation loop ($3$-fold split) to automatically find the optimal number of features that yields the highest validation score.
- **When to use**: To leverage non-linear model weights and interaction effects to obtain the absolute best feature subset.

## Model Training

### Supported Algorithms

#### Classification
- **Logistic Regression**: Linear classification with probability outputs
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machine**: Margin-based classification
- **Gradient Boosting**: Sequential model improvement
- **XGBoost**: Optimized gradient boosting
- **Neural Networks**: Multi-layer perceptron

#### Regression
- **Linear Regression**: Simple linear relationship modeling
- **Ridge Regression**: L2 regularized linear regression
- **Lasso Regression**: L1 regularized with feature selection
- **Random Forest Regressor**: Ensemble regression
- **XGBoost Regressor**: Gradient boosting for regression
- **Support Vector Regression**: Non-linear regression

#### Object Detection (Planned)
- **YOLOv8**: Real-time object detection
- **Custom CNN**: Convolutional neural networks
- **Transfer Learning**: Pre-trained model fine-tuning

### Hyperparameter Optimization

#### Optuna Integration
- **Automatic Parameter Tuning**: AI-driven hyperparameter optimization
- **Multi-objective Optimization**: Balance accuracy and model complexity
- **Pruning**: Early stopping for inefficient trials
- **Visualization**: Optimization history and parameter importance

#### Search Strategies
- **Grid Search**: Exhaustive parameter combination testing
- **Random Search**: Random parameter sampling
- **Bayesian Optimization**: Probabilistic model-based optimization
- **Evolutionary Algorithms**: Population-based optimization

### Model Validation
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Train-Validation-Test Split**: Proper data partitioning
- **Stratified Sampling**: Maintain class distribution in splits
- **Time Series Validation**: Forward-chaining validation for temporal data

## Model Evaluation

### Classification Metrics
- **Accuracy**: Overall correct prediction percentage
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed prediction breakdown

### Regression Metrics
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **R-squared**: Coefficient of determination
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### Model Interpretation
- **Feature Importance**: Relative contribution of each feature
- **SHAP Values**: SHapley Additive exPlanations for prediction explanation
- **Partial Dependence Plots**: Feature effect on predictions
- **Learning Curves**: Training and validation performance over time

## Deployment

### Local Deployment
- **Model Serialization**: Save trained models using joblib/pickle
- **Prediction Interface**: Interactive web interface for new predictions
- **Batch Prediction**: Process multiple samples simultaneously
- **Model Comparison**: Compare different model performances

### API Generation
- **REST API**: Automatic FastAPI endpoint creation
- **Request Validation**: Input data validation and preprocessing
- **Response Formatting**: Structured prediction outputs
- **Documentation**: Auto-generated API documentation

### Production Considerations
- **Model Versioning**: Track model versions and metadata
- **Performance Monitoring**: Monitor prediction accuracy over time
- **Data Drift Detection**: Identify when model retraining is needed
- **Scalability**: Handle increasing prediction loads

## Best Practices

### Data Preparation
1. **Always perform EDA before preprocessing**: Understand your data first
2. **Handle missing values appropriately**: Consider the missing data mechanism
3. **Scale features for distance-based algorithms**: Normalization is crucial for SVM, k-NN
4. **Encode categorical variables properly**: Choose encoding based on cardinality
5. **Split data before preprocessing**: Prevent data leakage

### Model Selection
1. **Start with simple models**: Baseline with linear/logistic regression
2. **Consider problem type and data size**: Different algorithms for different scenarios
3. **Use cross-validation**: Get robust performance estimates
4. **Balance bias-variance trade-off**: Consider model complexity vs. performance
5. **Validate on unseen data**: Reserve test set for final evaluation

### Feature Engineering
1. **Domain knowledge is valuable**: Use business understanding for feature creation
2. **Feature selection matters**: Remove irrelevant or redundant features
3. **Consider feature interactions**: Create polynomial or interaction features
4. **Handle temporal features properly**: Extract relevant time-based features
5. **Normalize skewed distributions**: Consider log transformation for skewed data

### Model Evaluation
1. **Use appropriate metrics**: Choose metrics aligned with business objectives
2. **Consider class imbalance**: Use stratified sampling and appropriate metrics
3. **Validate across different data segments**: Check performance consistency
4. **Monitor for overfitting**: Compare training and validation performance
5. **Document model limitations**: Understand where the model fails

### Deployment
1. **Version control everything**: Models, data, and code
2. **Monitor model performance**: Set up alerts for performance degradation
3. **Plan for model updates**: Establish retraining pipelines
4. **Ensure reproducibility**: Document dependencies and environments
5. **Consider ethical implications**: Monitor for bias and fairness

---

This guide provides comprehensive coverage of InstaML's machine learning capabilities. For technical implementation details, refer to the API documentation and source code.
