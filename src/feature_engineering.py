import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self):
        # Features will be discovered at fit time
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features using all available columns (except label)"""
        if 'label' not in df.columns:
            raise ValueError("Input dataframe must contain a 'label' column")
        
        X = df.drop('label', axis=1).copy()
        y = df['label']
        
        # Auto-detect numerical and categorical features
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        print(f"Detected {len(self.numerical_features)} numerical features: {self.numerical_features}")
        print(f"Detected {len(self.categorical_features)} categorical features: {self.categorical_features}")
        
        # Create preprocessing pipeline dynamically
        transformers = []
        if len(self.numerical_features) > 0:
            transformers.append(('num', StandardScaler(), self.numerical_features))
        if len(self.categorical_features) > 0:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        
        X_processed = self.preprocessor.fit_transform(X)
        feature_names = self._get_feature_names()
        
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df['label'] = y.reset_index(drop=True)
        
        print(f"Created {len(feature_names)} features after transformation")
        return processed_df
    
    def _get_feature_names(self) -> list:
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if transformer == 'drop':
                continue
            if transformer == 'passthrough':
                feature_names.extend(features)
                continue
            
            # Try to get feature names from transformer
            try:
                names = transformer.get_feature_names_out(features)
                feature_names.extend(names.tolist() if isinstance(names, np.ndarray) else list(names))
            except Exception:
                # Fallback for transformers that don't support get_feature_names_out
                feature_names.extend(list(features))
        
        return feature_names
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        if 'label' not in df.columns:
            raise ValueError("Input dataframe must contain a 'label' column")
        
        X = df.drop('label', axis=1).copy()
        X_processed = self.preprocessor.transform(X)
        feature_names = self._get_feature_names()
        
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df['label'] = df['label'].reset_index(drop=True)
        
        return processed_df