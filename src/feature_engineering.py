import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self, selected_features=None):
        self.selected_features = selected_features or [
            'DomainAgeMonths', 'HostingProvider', 'NoOfiFrame', 'IsResponsive',
            'LineOfCode', 'NoOfExternalRef', 'Robots', 'Industry'
        ]
        self.scalers = {}
        self.encoders = {}
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        categorical_features = ['HostingProvider', 'Industry']
        binary_features = ['Robots', 'IsResponsive']
        
        # Label encoding for high-cardinality categorical
        for feature in categorical_features:
            if feature in self.selected_features:
                self.encoders[feature] = LabelEncoder()
                df_encoded[feature] = self.encoders[feature].fit_transform(
                    df_encoded[feature].astype(str)
                )
        
        # Ensure binary features are numeric
        for feature in binary_features:
            if feature in self.selected_features:
                df_encoded[feature] = df_encoded[feature].astype(int)
        
        return df_encoded
    
    def scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        numerical_features = [
            'DomainAgeMonths', 'NoOfiFrame', 'LineOfCode', 'NoOfExternalRef'
        ]
        
        for feature in numerical_features:
            if feature in self.selected_features:
                self.scalers[feature] = StandardScaler()
                df_scaled[feature] = self.scalers[feature].fit_transform(
                    df_scaled[[feature]]
                )
        
        return df_scaled
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features based on EDA insights"""
        df_interaction = df.copy()
        
        # Domain age bins
        df_interaction['DomainAge_Young'] = (df_interaction['DomainAgeMonths'] < 12).astype(int)
        df_interaction['DomainAge_Mature'] = (df_interaction['DomainAgeMonths'] >= 24).astype(int)
        
        # High iframe count indicator
        df_interaction['High_iFrames'] = (df_interaction['NoOfiFrame'] > 10).astype(int)
        
        return df_interaction
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full feature engineering pipeline"""
        print("Starting feature engineering...")
        
        df = self.encode_categorical(df)
        df = self.scale_numerical(df)
        df = self.create_interaction_features(df)
        
        # Select final features
        final_features = [f for f in self.selected_features if f in df.columns]
        final_features.extend(['DomainAge_Young', 'DomainAge_Mature', 'High_iFrames'])
        
        # Ensure target is included
        if 'label' in df.columns and 'label' not in final_features:
            final_features.append('label')
        
        print(f"Final feature set: {len(final_features)} features")
        return df[final_features]