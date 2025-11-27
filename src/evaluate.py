from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


class ModelEvaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
    
    def evaluate_models(self, models: dict, df: pd.DataFrame, target_col: str = 'label'):
        """Comprehensive model evaluation"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print(f"{'='*50}")
            
            # Train on training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc
            }
            
            # Print detailed report
            print(f"Accuracy:  {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1-Score:  {f1:.3f}")
            print(f"AUC-ROC:   {auc_roc:.3f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate']))
        
        return results
    
    def plot_feature_importance(self, model, feature_names, top_n=10):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(top_n), importances[indices[:top_n]])
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
            plt.tight_layout()
            plt.show()