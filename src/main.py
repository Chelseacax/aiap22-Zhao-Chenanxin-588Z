# config.yaml - UPDATED BASED ON EDA
data:
  db_url: "https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"
  local_path: "data/phishing.db"
  table_name: "phishing_data"

preprocessing:
  remove_negative_images: true
  cap_extreme_values: true
  domain_limits:
    NoOfiFrame: 30  # Based on EDA domain knowledge
    NoOfImage: 5000
    LargestLineLength: 20000
    NoOfPopup: 50

feature_engineering:
  selected_features:
    numerical: ["DomainAgeMonths", "NoOfiFrame"]  # ONLY TOP 2 NUMERICAL
    categorical: ["HostingProvider", "IsResponsive"]  # ONLY TOP 2 CATEGORICAL
  create_missing_indicators: true
  encode_categorical: true

# need to understand the reasons for choosing these models and their hyperparameters
models:
  models_to_train: ["random_forest", "xgboost", "logistic_regression"]
  test_size: 0.2
  random_state: 42
  cv_folds: 5

random_forest:
  n_estimators: 100
  max_depth: 10
  random_state: 42

xgboost:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  random_state: 42

logistic_regression:
  C: 1.0
  penalty: "l2"
  random_state: 42

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  confidence_level: 0.95