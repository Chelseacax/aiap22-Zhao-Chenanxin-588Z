import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
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
        
        # 1. Split into train+val (80%) and test (20%) FIRST
        X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        # Store test set for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
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
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
        }
        
  # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.cv = cv
        
        for name, model in models.items():
            # Cross-validation scores (accuracy)
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            # Cross-validation scores (f1)
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            self.cv_scores[name] = {
                'accuracy_mean': acc_scores.mean(),
                'accuracy_std': acc_scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std()
            }
            
            # Train final model
            model.fit( X_train_val, y_train_val)
            self.models[name] = model
            
            print(f"{name}: CV Accuracy = {acc_scores.mean():.4f} (+/- {acc_scores.std() * 2:.4f}), CV F1 = {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        
        return self.models
    
    # def get_best_model(self) -> tuple:
    #     """Get the best performing model based on f1 score"""
    #     best_name = max(self.cv_scores.keys(), 
    #                key=lambda x: self.cv_scores[x].get('f1_mean', -np.inf))
    #     return best_name, self.models[best_name]
    
    
    
    def save_models(self, path: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
        print(f"Models saved to {path}")