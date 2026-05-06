import pandas as pd
import numpy as np
from typing import List
from .base import BASIC_FEATURE_COLS


class FeatureEngineer:
    def __init__(self, feature_level: str = 'basic'):
        self.feature_level = feature_level
        self.basic_cols = BASIC_FEATURE_COLS.copy()
        
    def add_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['times'].dt.hour
        df['minute'] = df['times'].dt.minute
        df['dayofweek'] = df['times'].dt.dayofweek
        df['month'] = df['times'].dt.month
        return df
    
    def add_enhanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_basic_time_features(df)
        
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['quarter'] = df['times'].dt.quarter
        df['dayofyear'] = df['times'].dt.dayofyear
        
        hour = df['hour']
        df['is_morning_peak'] = ((hour >= 7) & (hour <= 11)).astype(int)
        df['is_evening_peak'] = ((hour >= 17) & (hour <= 21)).astype(int)
        df['is_night'] = ((hour >= 0) & (hour < 6)).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 96, 96*7]) -> pd.DataFrame:
        df = df.copy()
        for col in self.basic_cols:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [4, 12, 24, 96]) -> pd.DataFrame:
        df = df.copy()
        for col in self.basic_cols:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
        return df
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        if self.feature_level == 'basic':
            df = self.add_basic_time_features(df)
            feature_cols = self.basic_cols + ['hour', 'minute', 'dayofweek', 'month']
        elif self.feature_level == 'enhanced':
            df = self.add_enhanced_time_features(df)
            
            if is_train:
                df = self.add_lag_features(df)
                df = self.add_rolling_features(df)
                df = df.dropna()
            
            enhanced_time_cols = [
                'hour', 'minute', 'dayofweek', 'month',
                'is_weekend', 'quarter', 'dayofyear',
                'is_morning_peak', 'is_evening_peak', 'is_night',
                'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos'
            ]
            
            if is_train:
                lag_cols = [f'{col}_lag{lag}' for col in self.basic_cols for lag in [1, 2, 3, 96, 96*7]]
                roll_cols = [f'{col}_roll_mean_{w}' for col in self.basic_cols for w in [4, 12, 24, 96]]
                roll_cols += [f'{col}_roll_std_{w}' for col in self.basic_cols for w in [4, 12, 24, 96]]
                feature_cols = self.basic_cols + enhanced_time_cols + lag_cols + roll_cols
            else:
                feature_cols = self.basic_cols + enhanced_time_cols
        
        return df, feature_cols
