import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.feature_columns = None
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df_clean = self._select_all_features(df)
        #remove unnamed columns if exist
        df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._handle_negative_images(df_clean)
        # df_clean = self._cap_extreme_values(df_clean)
        df_clean = self._validate_data(df_clean)
        return df_clean
    
    def _select_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select all available features + target"""
        # Keep all columns that exist in the dataframe
        self.feature_columns = df.columns.tolist()
        print(f"Using all {len(self.feature_columns)} attributes: {self.feature_columns}")
        return df[self.feature_columns].copy()
    
   
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Report missing values before handling
        missing_summary = df_clean.isnull().sum()
        if missing_summary.sum() > 0:
            print("\nMissing values detected:")
            print(missing_summary[missing_summary > 0])
        
        # Special handling for LineOfCode: use phishing median
        if 'LineOfCode' in df_clean.columns:
            missing_loc_count = df_clean['LineOfCode'].isnull().sum()
            if missing_loc_count > 0:
                # Calculate median of LineOfCode for phishing websites (label == 0)
                phishing_median = df_clean[df_clean['label'] == 0]['LineOfCode'].median()
                df_clean.loc[df_clean['LineOfCode'].isnull(), 'LineOfCode'] = phishing_median
                print(f"Filled {missing_loc_count} missing LineOfCode values with phishing median ({phishing_median:.2f})")
        
        # # Handle missing values in other numerical columns with their overall median
        # numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # if 'label' in numerical_cols:
        #     numerical_cols.remove('label')
        
        # for col in numerical_cols:
        #     if col == 'LineOfCode':
        #         # Already handled above, skip
        #         continue
        #     if df_clean[col].isnull().sum() > 0:
        #         median_val = df_clean[col].median()
        #         df_clean[col].fillna(median_val, inplace=True)
        #         print(f"Filled {df_clean[col].isnull().sum()} missing {col} values with median ({median_val:.2f})")
        
        # # Handle missing values in categorical columns with mode
        # categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # for col in categorical_cols:
        #     if df_clean[col].isnull().sum() > 0:
        #         mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
        #         df_clean[col].fillna(mode_val, inplace=True)
        #         print(f"Filled {df_clean[col].isnull().sum()} missing {col} values with mode ({mode_val})")
        
        return df_clean
 
    
    def _handle_negative_images(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Check if NoOfImage column exists
        if 'NoOfImage' in df_clean.columns:
            negative_count = (df_clean['NoOfImage'] < 0).sum()
            if negative_count > 0:
                df_clean['NoOfImage'] = np.where(
                    df_clean['NoOfImage'] < 0, 
                    0, 
                    df_clean['NoOfImage']
                )
                print(f"Replaced {negative_count} negative NoOfImage values with 0")
        return df_clean
    # def _cap_extreme_values(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Cap extreme values for numerical features"""
    #     df_capped = df.copy()
        
    #     # Cap NoOfiFrame at 99th percentile
    #     iframe_cap = df_capped['NoOfiFrame'].quantile(0.99)
    #     extreme_iframes = (df_capped['NoOfiFrame'] > iframe_cap).sum()
    #     if extreme_iframes > 0:
    #         df_capped['NoOfiFrame'] = np.where(
    #             df_capped['NoOfiFrame'] > iframe_cap, iframe_cap, df_capped['NoOfiFrame']
    #         )
    #         print(f"Capped {extreme_iframes} extreme NoOfiFrame values at {iframe_cap}")
        
    #     return df_capped
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data validation"""
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values after preprocessing")
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing")
        print(f"Final dataset shape: {df.shape}")
        return df