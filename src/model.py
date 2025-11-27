import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.cv_scores = {}
        
    def train_models(self, df: pd.DataFrame) -> dict:
        """Train multiple models and return performance"""
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Define models (at least 3 as required)
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=10
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True
            )
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            self.cv_scores[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            # Train final model
            model.fit(X, y)
            self.models[name] = model
            
            print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.models
    
    def get_best_model(self) -> tuple:
        """Get the best performing model based on CV scores"""
        best_name = max(self.cv_scores.keys(), 
                       key=lambda x: self.cv_scores[x]['mean'])
        return best_name, self.models[best_name]
    
    def save_models(self, path: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
        print(f"Models saved to {path}")