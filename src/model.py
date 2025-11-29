import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, random_state: int = 42,tune_hyperparams: bool = True):
        self.random_state = random_state
        self.tune_hyperparams = tune_hyperparams
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
    def train_models(self, df: pd.DataFrame) -> dict:
        """Train models with hyperparameter tuning (no parallel processing)"""
        X = df.drop('label', axis=1)
        y = df['label']
        
        base_models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state)
        }
        
        # Simplified parameter grids (fewer combinations)
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2]
            }
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name, base_model in base_models.items():
            if self.tune_hyperparams:
                print(f"\n🔧 Tuning {name}...")
                
                grid_search = GridSearchCV(
                    base_model, 
                    param_grids[name], 
                    cv=cv, 
                    scoring='f1',
                    n_jobs=1,  
                    verbose=1   
                )
                
                grid_search.fit(X, y)
                self.models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                self.cv_scores[name] = {
                    'f1_mean': grid_search.best_score_,
                    'best_params': grid_search.best_params_
                }
                
                print(f"✅ {name} - Best F1: {grid_search.best_score_:.4f}")
                
            else:
                # No tuning - much faster
                print(f"\n⚡ Training {name} with default parameters...")
                model = base_model
                model.fit(X, y)
                self.models[name] = model
                
                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                self.cv_scores[name] = {
                    'f1_mean': f1_scores.mean(),
                    'best_params': 'default'
                }
                print(f"✅ {name} - CV F1: {f1_scores.mean():.4f}")
        
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