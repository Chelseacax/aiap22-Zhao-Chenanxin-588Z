import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self):
        self.numerical_features = ['DomainAgeMonths', 'NoOfiFrame']
        self.categorical_features = ['HostingProvider', 'IsResponsive']
        self.preprocessor = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        feature_names = self._get_feature_names()
        
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df['label'] = y.reset_index(drop=True)
        
        print(f"Created {len(feature_names)} features: {feature_names}")
        return processed_df
    
    def _get_feature_names(self) -> list:
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                encoder = transformer
                cat_features = encoder.get_feature_names_out(features)
                feature_names.extend(cat_features)
        
        return feature_names
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X = df.drop('label', axis=1)
        X_processed = self.preprocessor.transform(X)
        feature_names = self._get_feature_names()
        
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df['label'] = df['label'].reset_index(drop=True)
        
        return processed_df