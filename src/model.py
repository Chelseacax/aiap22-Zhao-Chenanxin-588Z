import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
    
    def initialize_models(self):
        """Initialize three diverse models as required"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
        }
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'label'):
        """Train all models with cross-validation"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.initialize_models()
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
            self.cv_results[name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std()
            }
            
            # Full training
            model.fit(X, y)
            self.models[name] = model
            
            print(f"{name} - CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.models
    
    def save_models(self, path: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
            print(f"Saved {name} to {path}/{name}.pkl")