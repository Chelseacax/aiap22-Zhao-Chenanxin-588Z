from dataloader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import ModelTrainer
from evaluate import ModelEvaluator

def main():
    # Configuration
    DB_URL = "https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"
    
    # Execute pipeline
    print("=== ML PIPELINE EXECUTION ===")
    
    # 1. Data Loading
    loader = DataLoader(DB_URL)
    df = loader.load_data()
    
    # 2. Preprocessing
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df)
    
    # 3. Feature Engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(df_clean)
    
    # 4. Model Training
    trainer = ModelTrainer()
    models = trainer.train_models(df_features)
    trainer.save_models()
    
    # 5. Evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(models, df_features)
    
    print("\n=== PIPELINE COMPLETED ===")
    return results

if __name__ == "__main__":
    main()