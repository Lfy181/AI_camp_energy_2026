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
        df['weekofyear'] = df['times'].dt.isocalendar().week.astype(int)
        
        hour = df['hour']
        df['is_morning_peak'] = ((hour >= 7) & (hour <= 11)).astype(int)
        df['is_evening_peak'] = ((hour >= 17) & (hour <= 21)).astype(int)
        df['is_night'] = ((hour >= 0) & (hour < 6)).astype(int)
        df['is_afternoon'] = ((hour >= 12) & (hour < 17)).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        if lags is None:
            lags = [1, 2, 3, 4, 8, 16, 24, 48, 96, 96*2, 96*7]
        df = df.copy()
        for col in self.basic_cols:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        if windows is None:
            windows = [4, 8, 16, 24, 48, 96, 192]
        df = df.copy()
        for col in self.basic_cols:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                df[f'{col}_roll_median_{window}'] = df[col].rolling(window=window, min_periods=1).median()
        return df
    
    def add_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.basic_cols:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_4'] = df[col].diff(4)
            df[f'{col}_diff_8'] = df[col].diff(8)
            df[f'{col}_pct_change_1'] = df[col].pct_change(1)
            df[f'{col}_pct_change_4'] = df[col].pct_change(4)
        return df
    
    def add_aggregate_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        if 'hour' not in df.columns:
            df['hour'] = df['times'].dt.hour
        if 'dayofweek' not in df.columns:
            df['dayofweek'] = df['times'].dt.dayofweek
        
        for col in self.basic_cols:
            df[f'{col}_hour_mean'] = df.groupby('hour')[col].transform('mean')
            df[f'{col}_hour_std'] = df.groupby('hour')[col].transform('std')
            
            df[f'{col}_dow_mean'] = df.groupby('dayofweek')[col].transform('mean')
            df[f'{col}_dow_std'] = df.groupby('dayofweek')[col].transform('std')
            
            df[f'{col}_dev_from_hour_mean'] = df[col] - df[f'{col}_hour_mean']
            df[f'{col}_dev_from_dow_mean'] = df[col] - df[f'{col}_dow_mean']
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        feature_pairs = [
            ('系统负荷预测值', '风光总加预测值'),
            ('系统负荷预测值', '风电预测值'),
            ('系统负荷预测值', '光伏预测值'),
            ('风光总加预测值', '风电预测值'),
            ('风光总加预测值', '光伏预测值'),
            ('风电预测值', '光伏预测值'),
        ]
        
        for col1, col2 in feature_pairs:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].abs() + 1e-8)
        
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
                df = self.add_rate_of_change_features(df)
                df = self.add_aggregate_features(df, is_train)
                df = self.add_interaction_features(df)
                
                # 清理无穷大和过大的值
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].clip(lower=-1e10, upper=1e10)
                
                df = df.dropna()
            
            enhanced_time_cols = [
                'hour', 'minute', 'dayofweek', 'month',
                'is_weekend', 'quarter', 'dayofyear', 'weekofyear',
                'is_morning_peak', 'is_evening_peak', 'is_night', 'is_afternoon',
                'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos',
                'dayofyear_sin', 'dayofyear_cos'
            ]
            
            if is_train:
                lag_cols = [f'{col}_lag{lag}' for col in self.basic_cols for lag in [1, 2, 3, 4, 8, 16, 24, 48, 96, 96*2, 96*7]]
                roll_cols = []
                for col in self.basic_cols:
                    for window in [4, 8, 16, 24, 48, 96, 192]:
                        roll_cols.extend([
                            f'{col}_roll_mean_{window}',
                            f'{col}_roll_std_{window}',
                            f'{col}_roll_min_{window}',
                            f'{col}_roll_max_{window}',
                            f'{col}_roll_median_{window}'
                        ])
                diff_cols = []
                for col in self.basic_cols:
                    diff_cols.extend([
                        f'{col}_diff_1', f'{col}_diff_4', f'{col}_diff_8',
                        f'{col}_pct_change_1', f'{col}_pct_change_4'
                    ])
                agg_cols = []
                for col in self.basic_cols:
                    agg_cols.extend([
                        f'{col}_hour_mean', f'{col}_hour_std',
                        f'{col}_dow_mean', f'{col}_dow_std',
                        f'{col}_dev_from_hour_mean', f'{col}_dev_from_dow_mean'
                    ])
                interaction_cols = []
                feature_pairs = [
                    ('系统负荷预测值', '风光总加预测值'),
                    ('系统负荷预测值', '风电预测值'),
                    ('系统负荷预测值', '光伏预测值'),
                    ('风光总加预测值', '风电预测值'),
                    ('风光总加预测值', '光伏预测值'),
                    ('风电预测值', '光伏预测值'),
                ]
                for col1, col2 in feature_pairs:
                    interaction_cols.extend([
                        f'{col1}_x_{col2}',
                        f'{col1}_div_{col2}'
                    ])
                
                feature_cols = (
                    self.basic_cols + 
                    enhanced_time_cols + 
                    lag_cols + 
                    roll_cols + 
                    diff_cols + 
                    agg_cols + 
                    interaction_cols
                )
            else:
                feature_cols = self.basic_cols + enhanced_time_cols
        
        return df, feature_cols
