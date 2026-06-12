# InstaML Developer Continuation & Extension Guide

This guide describes how to extend, debug, and contribute to the InstaML codebase. It serves as an onboarding reference for new developers looking to introduce new features, data transformations, or machine learning algorithms.

---

## 1. Codebase Structure Walkthrough

Before writing code, familiarize yourself with the structural locations of each component:

```
InstaML/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI Endpoint routers (auth, data, training, prediction)
│   │   ├── core/            # App business logic (config, data_handler, worker thread manager)
│   │   └── db/              # Database models, schemas (Pydantic), and database connections
│   ├── core/                # Data Science and ML Training packages (unified_trainer, preprocess)
│   │   └── ML_models/       # Modality-specific engines (tabular_data, image_data, text_data, etc.)
│   └── storage/             # Locally cached user files (Parquet files and .pkl models)
├── frontend/
│   └── src/
│       ├── components/      # Common UI parts (Sidebar, GuideDrawer)
│       ├── pages/           # Pipeline page views (EDA, Preprocessing, TrainModel, TestModel, Versions)
│       └── services/        # API communication wrappers (axios instance)
└── docs/                    # Walkthroughs, architecture diagrams, and REST APIs
```

---

## 2. Extending the Preprocessing & Feature Selection Pipelines

If you want to add a new preprocessing operation (e.g., standardizing, filling missing data, or encoding):

### Step 2.1: Add the Logic in `data_handler.py`
Open [data_handler.py](file:///c:/Users/SSN/OneDrive - Shiv Nadar University - Chennai/Desktop/instaml/backend/app/core/data_handler.py) and locate `apply_preprocessing_operations`. 

Add a conditional check block for your new operation:

```python
elif op_type == "my_custom_transform":
    cols = op.get("columns", [])
    # Apply standard pandas/numpy transformations
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x * 10)
```

### Step 2.2: Register Target Invertibility (If applicable)
If the transformation is applied to target variables, you must register it in the `TransformRegistry` so that inference predictions can be inverse-solved.

```python
# In data_handler.py
TransformRegistry.register(
    name="multiply_10",
    forward_fn=lambda val, metadata: val * 10,
    inverse_fn=lambda val, metadata: val / 10,
    invertible=True
)
```

---

## 3. Integrating a New Machine Learning Algorithm

To add a new tabular model (e.g. LightGBM, AdaBoost, or ElasticNet):

### Step 3.1: Declare the Model in `tabular_data.py`
Open [tabular_data.py](file:///c:/Users/SSN/OneDrive - Shiv Nadar University - Chennai/Desktop/instaml/backend/core/ML_models/tabular_data.py) and import the scikit-learn estimator. Locate `_initialize_models` and append your model to the classifier/regressor dictionary:

```python
# inside _initialize_models() under classification
self.models["AdaBoost"] = AdaBoostClassifier(random_state=self.random_state)
```

### Step 3.2: Configure Hyperparameter Tuning (Optuna)
Locate the `objective` function inside `train_model` and add trial suggestion boundaries for the model:

```python
elif model_name == "AdaBoost":
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
    }
```

### Step 3.3: Expose the Option in FastAPI Router
Open [training.py](file:///c:/Users/SSN/OneDrive - Shiv Nadar University - Chennai/Desktop/instaml/backend/app/api/training.py) and append the new model name to the allowed algorithm list:

```python
@router.get("/training/options")
def get_training_options(...):
    if project.data_type == "tabular":
        return {
            "classification": [
                "Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression",
                "SVM", "KNN", "Decision Tree", "Naive Bayes", "MLP", "AdaBoost"
            ],
            ...
        }
```

---

## 4. Running the Test Suites

When modifying core utilities, verify that you didn't introduce target leakage or route conflicts by running the tests. Make sure your virtual environment is active.

### 4.1 Running API & Training tests (Tabular + CV + MLP)
```bash
python -m unittest backend/test_cv_and_mlp.py
```

### 4.2 Running Full E2E Modality tests (Text, Image, Audio)
Make sure the main gateway is running on `http://127.0.0.1:8000` and then execute:
```bash
python backend/test_modalities_e2e.py
```

### 4.3 Running Preprocessing Pipeline tests
```bash
python -m unittest backend/test_pipeline.py
```
