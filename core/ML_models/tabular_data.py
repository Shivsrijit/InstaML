import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
from core.preprocess import create_preprocessing_pipeline
from core.evaluator import evaluate_classification
import warnings
warnings.filterwarnings('ignore')

class TabularModelTrainer:
    """Comprehensive trainer for tabular data models."""
    
    def __init__(self, df, target_col, task_type="auto", test_size=0.2, random_state=42, scaling="standard"):
        """
        Initialize the trainer.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            task_type: "classification", "regression", or "auto"
            test_size: Test set size
            random_state: Random seed
            scaling: Scaling method ("standard", "minmax", "robust", "none")
        """
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaling = scaling
        
        # Auto-detect task type if not specified
        if task_type == "auto":
            self.task_type = self._detect_task_type()
        else:
            self.task_type = task_type
            
        # Prepare data
        self._prepare_data()
        
        # Initialize models
        self._initialize_models()
        
    def _detect_task_type(self):
        """Auto-detect if this is classification or regression."""
        target_dtype = self.df[self.target_col].dtype
        unique_values = self.df[self.target_col].nunique()
        
        if target_dtype in ['object', 'category'] or unique_values <= 10:
            return "classification"
        else:
            return "regression"
    
    def _prepare_data(self):
        """Prepare features and target."""
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create preprocessing pipeline
        self.preprocessor, self.num_cols, self.cat_cols = create_preprocessing_pipeline(
            self.df, self.target_col, scaling=self.scaling
        )
    
    def _initialize_models(self):
        """Initialize model dictionaries."""
        if self.task_type == "classification":
            self.models = {
                "Random Forest": RandomForestClassifier(random_state=self.random_state),
                "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
                "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "SVM": SVC(random_state=self.random_state, probability=True),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Naive Bayes": GaussianNB()
            }
            
            self.param_grids = {
                "Random Forest": {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10]
                },
                "XGBoost": {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [3, 6, 9],
                    'model__learning_rate': [0.01, 0.1, 0.3]
                },
                "Gradient Boosting": {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [3, 6, 9],
                    'model__learning_rate': [0.01, 0.1, 0.3]
                },
                "Logistic Regression": {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l1', 'l2']
                },
                "SVM": {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear']
                },
                "KNN": {
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance']
                },
                "Decision Tree": {
                    'model__max_depth': [3, 5, 7, None],
                    'model__min_samples_split': [2, 5, 10]
                }
            }
        else:  # Regression
            self.models = {
                "Random Forest": RandomForestRegressor(random_state=self.random_state),
                "XGBoost": XGBRegressor(random_state=self.random_state),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=self.random_state),
                "Lasso": Lasso(random_state=self.random_state),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state)
            }
            
            self.param_grids = {
                "Random Forest": {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10]
                },
                "XGBoost": {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [3, 6, 9],
                    'model__learning_rate': [0.01, 0.1, 0.3]
                }
            }
    
    def train_model(self, model_name, use_hyperparameter_tuning=True):
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            use_hyperparameter_tuning: Whether to use GridSearchCV
            
        Returns:
            Trained model and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Create pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", model)
        ])
        
        # Hyperparameter tuning
        if use_hyperparameter_tuning and model_name in self.param_grids:
            grid_search = GridSearchCV(
                pipeline, 
                self.param_grids[model_name], 
                cv=5, 
                scoring='accuracy' if self.task_type == "classification" else 'r2',
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_model = pipeline
            best_model.fit(self.X_train, self.y_train)
            best_params = {}
        
        # Evaluate model
        if self.task_type == "classification":
            metrics = self._evaluate_classification(best_model)
        else:
            metrics = self._evaluate_regression(best_model)
        
        # Store results
        self.trained_model = best_model
        self.model_name = model_name
        self.best_params = best_params
        self.metrics = metrics
        
        return best_model, metrics, best_params
    
    def _evaluate_classification(self, model):
        """Evaluate classification model."""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_test, y_pred, average='weighted'),
            "f1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "classification_report": classification_report(self.y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Add ROC AUC if possible
        if y_pred_proba is not None and len(np.unique(self.y_test)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            except:
                pass
        
        return metrics
    
    def _evaluate_regression(self, model):
        """Evaluate regression model."""
        y_pred = model.predict(self.X_test)
        
        metrics = {
            "mse": mean_squared_error(self.y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
            "mae": mean_absolute_error(self.y_test, y_pred),
            "r2": r2_score(self.y_test, y_pred),
            "mape": np.mean(np.abs((self.y_test - y_pred) / np.where(self.y_test != 0, self.y_test, 1))) * 100
        }
        
        return metrics
    
    def cross_validate(self, model_name, cv=5):
        """Perform cross-validation."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", model)
        ])
        
        scoring = 'accuracy' if self.task_type == "classification" else 'r2'
        scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring=scoring)
        
        return {
            "cv_scores": scores.tolist(),
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "cv_folds": cv
        }
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        
        # Get feature names after preprocessing
        feature_names = []
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        
        # Extract feature importance
        if hasattr(self.trained_model.named_steps['model'], 'feature_importances_'):
            importance = self.trained_model.named_steps['model'].feature_importances_
        elif hasattr(self.trained_model.named_steps['model'], 'coef_'):
            importance = np.abs(self.trained_model.named_steps['model'].coef_)
            if len(importance.shape) > 1:
                importance = np.mean(importance, axis=0)
        else:
            return None
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def get_available_models(self):
        """Get list of available models for the current task type."""
        if self.task_type == "classification":
            return {
                "classification": list(self.models.keys())
            }
        else:
            return {
                "regression": list(self.models.keys())
            }
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        
        joblib.dump(self.trained_model, filepath)
        
    def load_model(self, filepath):
        """Load a saved model."""
        self.trained_model = joblib.load(filepath)
        return self.trained_model

# Convenience functions for backward compatibility
def train_tabular_model(df, target_col, model_type="random_forest", scaling="standard", test_size=0.2, random_state=42):
    """Legacy function for backward compatibility."""
    trainer = TabularModelTrainer(df, target_col, test_size=test_size, random_state=random_state)
    
    # Map legacy model types to new names
    model_mapping = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost"
    }
    
    model_name = model_mapping.get(model_type, "Random Forest")
    model, metrics, _ = trainer.train_model(model_name, use_hyperparameter_tuning=False)
    
    return model, metrics

def get_available_models():
    """Get list of available models."""
    return {
        "classification": [
            "Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression",
            "SVM", "KNN", "Decision Tree", "Naive Bayes"
        ],
        "regression": [
            "Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression",
            "Ridge", "Lasso", "SVR", "KNN", "Decision Tree"
        ]
    }
