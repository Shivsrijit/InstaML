# 🚀 InstaML — No-Code Machine Learning Platform

## 📌 Introduction
InstaML is a **no-code machine learning platform** that allows users to train, test, evaluate, and deploy AI models directly from a simple **Streamlit interface**.  
It is designed for students, researchers, and businesses who want to leverage **ML without writing complex code**.

---
## App Preview

<p align="center">
  <img src="images/Screenshot%20(95).png" alt="App Preview 1" width="200"/>
  <img src="images/Screenshot%20(94).png" alt="App Preview 2" width="200"/>
  <img src="images/Screenshot%20(97).png" alt="App Preview 3" width="200"/>
  <img src="images/Screenshot%20(98).png" alt="App Preview 4" width="200"/>
  <img src="images/Screenshot%20(99).png" alt="App Preview 5" width="200"/>
</p>

---

## 📜 Abstract
The goal of InstaML is to **democratize AI development** by enabling users to:
- Upload datasets
- **Preprocess and clean data** with automated recommendations
- Perform advanced **Exploratory Data Analysis (EDA)** with 75+ visualization options
- Train ML models for **classification, regression, and object detection**
- Test trained models with custom inputs
- Deploy models instantly for real-world use
- Manage model versions and metadata

The system integrates a **Streamlit frontend** with a **FastAPI backend** and supports multiple ML libraries like **scikit-learn, XGBoost, YOLOv8**, and more.

---

## 🛠 Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based interactive UI)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Machine Learning**: scikit-learn, XGBoost, YOLOv8, Optuna, MLflow
- **Data Visualization**: Seaborn, Matplotlib, Plotly
- **Data Handling**: Pandas, NumPy
- **Deployment**: Docker, GitHub Codespaces, Streamlit Cloud
- **Version Control**: Git + GitHub

---

## 🎯 Final Aim
To create a **complete no-code ML ecosystem** where:
- Users can train any model from **tabular, image, or time-series datasets**
- Fully automated **EDA and preprocessing**
- In-built **hyperparameter tuning**
- **Real-time deployment** with minimal effort
- Open-source and easily extensible

---

## 🚀 **NEW: Enhanced Data Pipeline**
The platform now features a **comprehensive data preprocessing workflow**:

### **Page Structure:**
1. **📂 Data Upload** - Upload CSV files or load from local paths
2. **🔧 Data Preprocessing** - **NEW!** Comprehensive data cleaning and transformation
3. **📊 EDA** - Exploratory Data Analysis with 75+ visualizations
4. **⚙️ Train Model** - ML model training with preprocessing validation
5. **🧪 Test Model** - Model evaluation and testing
6. **🚀 Deploy Model** - Model deployment and predictions

### **Preprocessing Features:**
- **🔍 Data Quality Analysis** - Missing values, data types, memory usage
- **💡 Smart Recommendations** - AI-powered preprocessing suggestions
- **🧹 Data Cleaning** - Duplicate removal, column selection, type conversion
- **📏 Scaling & Encoding** - StandardScaler, MinMaxScaler, Label/One-Hot encoding
- **🔢 Missing Values** - Multiple imputation strategies (mean, median, constant, drop)
- **📈 Outlier Detection** - IQR-based outlier identification and handling
- **💾 Data Persistence** - Save preprocessed data for consistent workflow

---

##  What Has Been Done Till Now
- **Data Upload module** (supports CSV)
- **🔧 NEW: Data Preprocessing module** with comprehensive cleaning capabilities
- **EDA module** with **75+ visualizations**
-  **Model training** for tabular datasets (classification/regression)
-  **Model evaluation** with performance metrics
-  **Project structure setup** for Streamlit + FastAPI
- **Utility functions** for charts and UI helpers
-  **Dependencies management** with complete requirements.txt
-  Docker-ready project setup

---

## Updates
- ✅ **Added comprehensive Data Preprocessing page** with 5 main tabs
- ✅ **Enhanced page navigation** with proper numbering and flow
- ✅ **Added preprocessing recommendations** based on data analysis
- ✅ **Improved data validation** and quality checks
- ✅ **Enhanced user experience** with better guidance and status indicators
- ✅ **Populated requirements.txt** with all necessary dependencies
- ✅ **Updated main app** to reflect new page structure
- ✅ **Enhanced EDA and Training pages** with preprocessing status checks
- Added **comprehensive EDA charts** (15+ in each category: Univariate, Bivariate, Multivariate, Dimensionality Reduction, Time Series)
- Fixed **import paths** and added `__init__.py` where necessary
- Added **modular chart functions** in `charts.py`
- Prepared **GitHub-ready folder structure**
- Created **README** for clear project onboarding

---

## Running the Project (Local)
```bash
# Navigate to the repo
cd InstaML/InstaML

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app on a specific port
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0



## Running on GitHub Codespaces (4 Core)
bash="gh codespace create --repo SAUNAK359/InstaML --machine standardLinux4Core --branch main"


## Pushing Code to GitHub
# Navigate to repo root
cd /workspaces/InstaML

# Stage all changes
git add .

# Commit changes with message
git commit -m "Updated EDA charts and fixed paths"

# Push to main branch
git push origin main

### Overwriting Old Code with Updated Scripts
# Force push updated code
git add .
git commit -m "Overwriting previous code with latest updates"
git push origin main --force
      
************   When committing, always write:
Summary of what changed
Files affected
Impact of changes
