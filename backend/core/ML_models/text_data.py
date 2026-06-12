import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class TextModelTrainer:
    """Comprehensive trainer for raw text classification tasks using TF-IDF."""
    
    def __init__(self, df, text_col, target_col, test_size=0.2, random_state=42):
        """
        Initialize the text trainer.
        
        Args:
            df: Input DataFrame
            text_col: Column containing raw text feature
            target_col: Target column name
            test_size: Test set size
            random_state: Random seed
        """
        self.df = df.copy()
        self.text_col = text_col
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        # Impute missing values in text column
        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)
        
        # Split data
        self.X = self.df[self.text_col]
        self.y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize models
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Naive Bayes": MultinomialNB(),
            "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        }
        
        # Param grids for grid search tuning
        self.param_grids = {
            "Logistic Regression": {
                'model__C': [0.1, 1.0, 10.0],
                'vectorizer__max_features': [1000, 5000]
            },
            "Naive Bayes": {
                'model__alpha': [0.1, 0.5, 1.0],
                'vectorizer__max_features': [1000, 5000]
            }
        }
        
    def train_model(self, model_name, use_hyperparameter_tuning=True):
        """
        Train a text classification model pipeline.
        
        Args:
            model_name: Name of model to train
            use_hyperparameter_tuning: Whether to use GridSearchCV
            
        Returns:
            Trained pipeline, evaluation metrics, best params
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        
        # Create pipeline
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words='english')),
            ("model", model)
        ])
        
        # Hyperparameter tuning
        if use_hyperparameter_tuning and model_name in self.param_grids:
            grid_search = GridSearchCV(
                pipeline,
                self.param_grids[model_name],
                cv=3,
                scoring='accuracy',
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
        metrics = self._evaluate_classification(best_model)
        
        # Store results
        self.trained_model = best_model
        self.model_name = model_name
        self.best_params = best_params
        self.metrics = metrics
        
        return best_model, metrics, best_params
        
    def _evaluate_classification(self, model):
        """Evaluate classification model metrics."""
        y_pred = model.predict(self.X_test)
        
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted'),
            "recall": recall_score(self.y_test, y_pred, average='weighted'),
            "f1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "classification_report": classification_report(self.y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        return metrics
        
    def get_feature_importance(self):
        """Extract feature significance (top word coefficients/importances)."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
            
        vectorizer = self.trained_model.named_steps['vectorizer']
        model = self.trained_model.named_steps['model']
        feature_names = vectorizer.get_feature_names_out()
        
        importance = None
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if len(importance.shape) > 1:
                # For multi-class, average coefficient absolute values across classes
                importance = np.mean(importance, axis=0)
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
        if importance is not None:
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance_df
            
        return None
        
    def get_available_models(self):
        """Get list of available text models."""
        return {
            "classification": list(self.models.keys())
        }
        
    def save_model(self, filepath):
        """Save the trained model pipeline."""
        if not hasattr(self, 'trained_model'):
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.trained_model, filepath)
        
    def load_model(self, filepath):
        """Load a saved model pipeline."""
        self.trained_model = joblib.load(filepath)
        return self.trained_model
