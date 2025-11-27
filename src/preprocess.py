import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.feature_columns = ['DomainAgeMonths', 'HostingProvider', 'NoOfiFrame', 'IsResponsive', 'label']
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df_clean = self._select_features(df)
        df_clean = self._handle_negative_images(df_clean)
        df_clean = self._cap_extreme_values(df_clean)
        df_clean = self._validate_data(df_clean)
        return df_clean
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the top 4 features + target"""
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        return df[self.feature_columns].copy()
    
    def _handle_negative_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove impossible negative NoOfImage values"""
        initial_count = len(df)
        df_clean = df[df['NoOfiFrame'] >= 0].copy()
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            print(f"Removed {removed_count} records with negative NoOfiFrame values")
        return df_clean
    
    def _cap_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme values for numerical features"""
        df_capped = df.copy()
        
        # Cap NoOfiFrame at 99th percentile
        iframe_cap = df_capped['NoOfiFrame'].quantile(0.99)
        extreme_iframes = (df_capped['NoOfiFrame'] > iframe_cap).sum()
        if extreme_iframes > 0:
            df_capped['NoOfiFrame'] = np.where(
                df_capped['NoOfiFrame'] > iframe_cap, iframe_cap, df_capped['NoOfiFrame']
            )
            print(f"Capped {extreme_iframes} extreme NoOfiFrame values at {iframe_cap}")
        
        return df_capped
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data validation"""
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values after preprocessing")
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing")
        print(f"Final dataset shape: {df.shape}")
        return df