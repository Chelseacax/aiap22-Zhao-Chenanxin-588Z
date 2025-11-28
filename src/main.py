from dataloader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import ModelTrainer
from evaluate import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd

def run_pipeline():
    """Main pipeline execution"""
    print("=== PHISHING DETECTION PIPELINE ===")
    
    # 1. Load Data
    print("\n1. Loading data...")
    loader = DataLoader(
        db_url="https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"
    )
    df = loader.load_data()
    
    # 2. Preprocess Data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df)
    
    # 3. Feature Engineering
    print("\n3. Engineering features...")
    feature_engineer = FeatureEngineer()
    df_processed = feature_engineer.fit_transform(df_clean)
    
    # 4. Split Data
    X = df_processed.drop('label', axis=1)
    y = df_processed['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train Models
    print("\n4. Training models...")
    trainer = ModelTrainer()
    models = trainer.train_models(pd.concat([X_train, y_train], axis=1))
    
    # 6. Evaluate Models
    print("\n5. Evaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(models, X_test, y_test)
    
    # 7. Generate Report
    report = evaluator.generate_report()
    print("\n=== FINAL RESULTS ===")
    print(report)
    
    # 8. Save best model
    best_name, best_model = evaluator.get_best_model()
    print(f"\nBest model: {best_name}")
    
    return results, best_model

if __name__ == "__main__":
    results, best_model = run_pipeline()