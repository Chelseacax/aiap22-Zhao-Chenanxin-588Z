import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_models(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Comprehensive model evaluation"""
        evaluation_results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation_results[name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"\n=== {name} Evaluation ===")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        self.results = evaluation_results
        return evaluation_results
    
    def plot_feature_importance(self, model, feature_names: list, top_n: int = 10):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(min(top_n, len(importances))), 
                   importances[indices[:top_n]])
            plt.xticks(range(min(top_n, len(importances))), 
                      [feature_names[i] for i in indices[:top_n]], rotation=45)
            plt.tight_layout()
            plt.show()
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        report_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            report_data.append(row)
        
        return pd.DataFrame(report_data)