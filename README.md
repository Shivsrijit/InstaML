# InstaML — No-Code Machine Learning Platform

## Overview

InstaML is a comprehensive no-code machine learning platform designed to democratize AI development. It provides an intuitive web-based interface that enables users to build, train, evaluate, and deploy machine learning models without writing code. The platform serves students learning ML concepts, researchers prototyping models, and business professionals solving real-world problems.


## App Preview

<p align="center">
  <img src="images/Screenshot%20(95).png" alt="App Preview 1" width="200"/>
  <img src="images/Screenshot%20(94).png" alt="App Preview 2" width="200"/>
  <img src="images/Screenshot%20(97).png" alt="App Preview 3" width="200"/>
  <img src="images/Screenshot%20(98).png" alt="App Preview 4" width="200"/>
  <img src="images/Screenshot%20(99).png" alt="App Preview 5" width="200"/>
</p>


## Core Features

**Data Management**
- Intelligent data upload and validation
- Comprehensive data preprocessing pipeline
- Advanced data quality assessment
- Automated feature engineering

**Exploratory Data Analysis**
- Interactive visualization dashboard
- Statistical analysis and insights
- Correlation and feature importance analysis
- Multi-dimensional data exploration

**Machine Learning**
- Multiple algorithm support (Classification, Regression, Object Detection)
- Automated hyperparameter optimization
- Model performance evaluation and comparison
- Cross-validation and metrics analysis

**Deployment**
- One-click model deployment
- REST API generation
- Real-time prediction interface
- Model versioning and management

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | FastAPI | API services and model serving |
| **ML Frameworks** | scikit-learn, XGBoost, YOLOv8 | Model training and inference |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Plotly, Seaborn, Matplotlib | Charts and interactive graphs |
| **Optimization** | Optuna | Hyperparameter tuning |
| **MLOps** | MLflow | Experiment tracking and versioning |
| **Deployment** | Docker, GitHub Codespaces | Containerization and cloud deployment |

## Platform Workflow

1. **Data Upload** → Upload datasets and perform initial validation
2. **Data Preprocessing** → Clean, transform, and prepare data for modeling
3. **Exploratory Analysis** → Understand data patterns through visualizations
4. **Model Training** → Select algorithms and train models with optimization
5. **Model Testing** → Evaluate performance and validate results
6. **Model Deployment** → Deploy models for real-world predictions

## Quick Start

### Local Installation

```bash
git clone https://github.com/shivsrijit/InstaML.git
cd InstaML/InstaML
pip install -r requirements.txt
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

Access the application at: **[http://localhost:8501](http://localhost:8501)**

---

## Development Status

| Component          | Status         | Description                                          |
|--------------------|---------------|------------------------------------------------------|
| Data Upload        | ✅ Complete   | CSV support with comprehensive validation            |
| Data Preprocessing | ✅ Complete   | 5-stage preprocessing workflow                       |
| EDA Dashboard      | ✅ Complete   | Lots of visualization options                        |
| Model Training     | ✅ Complete   | Classification and regression algorithms             |
| Model Testing      | ✅ Complete   | Performance evaluation and validation                |
| Model Deployment   | 🔄 In Progress| Local deployment infrastructure                      |
| Computer Vision    | ✅ Complete   | YOLO integration for object detection                |
| Time Series        | 📋 Planned    | Forecasting and trend analysis                       |
| API Backend        | 📋 Planned    | FastAPI service architecture                         |


---

## Project Vision

**InstaML** aims to bridge the gap between complex machine learning concepts and practical implementation.  

By providing a **no-code interface**, we enable users to focus on **problem-solving and insights** rather than technical implementation details.  

The platform maintains the **flexibility required for advanced use cases** while ensuring **accessibility for beginners**.

---

## Contributing

We welcome contributions to enhance InstaML's capabilities.  
Please review our contributing guidelines and feel free to submit **issues, feature requests, or pull requests**.

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

## Documentation

- 📘 **ML Guide** – Comprehensive machine learning documentation  
- ⚙️ **API Reference** – Backend API documentation  
- 🚀 **Deployment Guide** – Production deployment instructions  

---

✨ **Built for the AI community** | ⭐ **Star this repository if you find it helpful**

