# src/feature_engineering.py - ENHANCED WITH EDA INSIGHTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from typing import Dict, Any, List

class FeatureEngineer:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select ONLY the top 4 features identified in EDA"""
        selected_numerical = self.config['selected_features']['numerical']
        selected_categorical = self.config['selected_features']['categorical']
        
        # Combine top 4 features + label
        top_features = selected_numerical + selected_categorical + ['label']
        
        # Ensure features exist
        available_features = [f for f in top_features if f in df.columns]
        missing_features = set(top_features) - set(available_features)
        
        if missing_features:
            print(f"Warning: Missing top features: {missing_features}")
            # Fallback to all available features
            available_features = [f for f in top_features if f in df.columns] + ['label']
        
        self.feature_names = [f for f in available_features if f != 'label']
        print(f"Selected top {len(self.feature_names)} features from EDA: {self.feature_names}")
        
        return df[available_features]
    
    def create_domain_age_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df_enhanced = df.copy()
        
        # Bin domain age based on EDA insights (phishing sites are much newer)
        df_enhanced['DomainAge_Binned'] = pd.cut(
            df_enhanced['DomainAgeMonths'],
            bins=[0, 6, 12, 24, 60, np.inf],
            labels=['0-6m', '6-12m', '1-2y', '2-5y', '5y+']
        )
        
        # Create binary flag for very new domains (high phishing risk)
        df_enhanced['Domain_Very_New'] = (df_enhanced['DomainAgeMonths'] < 6).astype(int)
        
        # Log transformation for skewed distribution
        df_enhanced['DomainAge_Log'] = np.log1p(df_enhanced['DomainAgeMonths'])
        
        self.feature_names.extend(['DomainAge_Binned', 'Domain_Very_New', 'DomainAge_Log'])
        return df_enhanced
    
    def create_iframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering for NoOfiFrame based on EDA"""
        df_enhanced = df.copy()
        
        # Create tiers based on EDA analysis
        df_enhanced['iFrame_Tier1'] = (df_enhanced['NoOfiFrame'] <= 15).astype(int)
        df_enhanced['iFrame_Tier2'] = ((df_enhanced['NoOfiFrame'] > 15) & 
                                     (df_enhanced['NoOfiFrame'] <= 30)).astype(int)
        df_enhanced['iFrame_Tier3'] = (df_enhanced['NoOfiFrame'] > 30).astype(int)
        
        # Binary flag for excessive iframes (potential malicious use)
        df_enhanced['Excessive_iFrames'] = (df_enhanced['NoOfiFrame'] > 30).astype(int)
        
        self.feature_names.extend(['iFrame_Tier1', 'iFrame_Tier2', 'iFrame_Tier3', 'Excessive_iFrames'])
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features based on EDA relationships"""
        df_interaction = df.copy()
        
        # Interaction: New domain + no responsiveness = high phishing risk
        if 'DomainAgeMonths' in df.columns and 'IsResponsive' in df.columns:
            df_interaction['NewDomain_NotResponsive'] = (
                (df_interaction['DomainAgeMonths'] < 6) & 
                (df_interaction['IsResponsive'] == 0)
            ).astype(int)
            self.feature_names.append('NewDomain_NotResponsive')
        
        # Interaction: Excessive iframes + specific hosting providers
        if 'NoOfiFrame' in df.columns and 'HostingProvider' in df.columns:
            # Mark suspicious hosting providers with high iframe counts
            suspicious_providers = ['Unknown Provider', 'Freehostia']  # From EDA
            df_interaction['Suspicious_Hosting_iFrames'] = (
                (df_interaction['HostingProvider'].isin(suspicious_providers)) &
                (df_interaction['NoOfiFrame'] > 15)
            ).astype(int)
            self.feature_names.append('Suspicious_Hosting_iFrames')
        
        return df_interaction
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart encoding based on EDA cardinality insights"""
        df_encoded = df.copy()
        categorical_features = self.config['selected_features']['categorical']
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                # HostingProvider: High cardinality -> targeted encoding
                if feature == 'HostingProvider':
                    # Group less frequent providers based on EDA
                    provider_counts = df_encoded['HostingProvider'].value_counts()
                    top_providers = provider_counts[provider_counts > 100].index.tolist()
                    df_encoded['HostingProvider_Grouped'] = df_encoded['HostingProvider'].apply(
                        lambda x: x if x in top_providers else 'Other'
                    )
                    # One-hot encode the grouped version
                    dummies = pd.get_dummies(df_encoded['HostingProvider_Grouped'], prefix='Hosting')
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(['HostingProvider', 'HostingProvider_Grouped'], axis=1, inplace=True)
                    
                    self.feature_names.remove('HostingProvider')
                    self.feature_names.extend(dummies.columns.tolist())
                
                # IsResponsive: Binary -> keep as is or one-hot
                elif feature == 'IsResponsive':
                    # Already binary, no encoding needed
                    pass
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using RobustScaler (handles outliers)"""
        df_scaled = df.copy()
        numerical_features = self.config['selected_features']['numerical']
        
        for feature in numerical_features:
            if feature in df_scaled.columns:
                scaler = RobustScaler()  # Robust to outliers identified in EDA
                df_scaled[feature] = scaler.fit_transform(df_scaled[[feature]])
                self.scalers[feature] = scaler
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering pipeline based on EDA"""
        print("Starting EDA-informed feature engineering...")
        
        # Select only top 4 features
        df = self.select_features(df)
        
        # Enhanced feature engineering for top features
        df = self.create_domain_age_features(df)
        df = self.create_iframe_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale numerical features
        df = self.scale_numerical_features(df)
        
        print(f"Final feature set ({len(self.feature_names)} features): {self.feature_names}")
        print("EDA-informed feature engineering completed")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names