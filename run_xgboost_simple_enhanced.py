#!/usr/bin/env python
"""
修复策略：XGBoost + 简化增强特征 + 暴力搜索
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import DataLoader
from experiments.models import ModelFactory
from experiments.strategies import StrategyGenerator

class SimpleEnhancedFeatures:
    """简化版增强特征 - 训练和测试都可用"""
    
    def __init__(self):
        self.basic_cols = [
            '系统负荷预测值', '风光总加预测值', '联络线预测值',
            '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
        ]
    
    def transform(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        
        # 时间特征
        df['hour'] = df['times'].dt.hour
        df['minute'] = df['times'].dt.minute
        df['dayofweek'] = df['times'].dt.dayofweek
        df['month'] = df['times'].dt.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['quarter'] = df['times'].dt.quarter
        df['dayofyear'] = df['times'].dt.dayofyear
        
        # 时段标记
        hour = df['hour']
        df['is_morning_peak'] = ((hour >= 7) & (hour <= 11)).astype(int)
        df['is_evening_peak'] = ((hour >= 17) & (hour <= 21)).astype(int)
        df['is_night'] = ((hour >= 0) & (hour < 6)).astype(int)
        df['is_afternoon'] = ((hour >= 12) & (hour < 17)).astype(int)
        
        # 周期编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 交互特征（乘法）
        df['负荷_x_风光'] = df['系统负荷预测值'] * df['风光总加预测值']
        df['负荷_x_风电'] = df['系统负荷预测值'] * df['风电预测值']
        df['负荷_x_光伏'] = df['系统负荷预测值'] * df['光伏预测值']
        df['风光_x_风电'] = df['风光总加预测值'] * df['风电预测值']
        df['风光_x_光伏'] = df['风光总加预测值'] * df['光伏预测值']
        
        # 清理
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].clip(-1e10, 1e10)
        
        feature_cols = self.basic_cols + [
            'hour', 'minute', 'dayofweek', 'month', 'is_weekend', 'quarter', 'dayofyear',
            'is_morning_peak', 'is_evening_peak', 'is_night', 'is_afternoon',
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos',
            '负荷_x_风光', '负荷_x_风电', '负荷_x_光伏',
            '风光_x_风电', '风光_x_光伏'
        ]
        
        return df, feature_cols


def run_xgboost_fixed():
    print("="*80)
    print("修复策略: XGBoost + 简化增强特征 + 暴力搜索".center(80))
    print("="*80)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader()
    df_train = data_loader.load_train_data()
    df_test = data_loader.load_test_data()
    print(f"  训练数据: {df_train.shape}")
    print(f"  测试数据: {df_test.shape}")
    
    # 特征工程
    print("\n[2/4] 特征工程 (简化增强特征)...")
    fe = SimpleEnhancedFeatures()
    df_train_processed, feature_cols = fe.transform(df_train)
    df_test_processed, _ = fe.transform(df_test)
    print(f"  特征数量: {len(feature_cols)}")
    
    # 准备数据
    X = df_train_processed[feature_cols].values
    y = df_train_processed['A'].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    
    # 训练XGBoost模型
    print("\n[3/4] 训练XGBoost模型...")
    model = ModelFactory.create_model('xgboost')
    val_metrics = model.fit(X_train, y_train, X_val, y_val)
    print(f"  验证 RMSE: {val_metrics['rmse']:.6f}")
    
    # 预测
    print("\n[4/4] 预测和策略生成...")
    X_test = df_test_processed[feature_cols].values
    y_test_pred = model.predict(X_test)
    
    df_price_pred = pd.DataFrame({
        'times': df_test_processed['times'],
        'A': y_test_pred
    })
    
    # 分析预测价格分布
    print(f"\n预测价格分布:")
    print(f"  最小值: {y_test_pred.min():.2f}")
    print(f"  最大值: {y_test_pred.max():.2f}")
    print(f"  平均值: {y_test_pred.mean():.2f}")
    print(f"  负数比例: {(y_test_pred < 0).mean() * 100:.1f}%")
    print(f"  正数比例: {(y_test_pred > 0).mean() * 100:.1f}%")
    
    # 生成策略
    strategy = StrategyGenerator.create_strategy('brute')
    df_result, total_profit = strategy.generate(df_price_pred)
    
    # 计算真实收益
    df_result['profit'] = df_result['power'] * df_result['实时价格'] * 0.25
    
    print(f"\n" + "="*80)
    print("策略结果".center(80))
    print("="*80)
    print(f"模型: XGBoost")
    print(f"特征: 简化增强特征 ({len(feature_cols)}个)")
    print(f"策略: 暴力搜索")
    print(f"\n收益分析:")
    print(f"  总收益: {df_result['profit'].sum():,.2f}")
    print(f"  充电收益: {df_result[df_result['power'] < 0]['profit'].sum():,.2f}")
    print(f"  放电收益: {df_result[df_result['power'] > 0]['profit'].sum():,.2f}")
    print(f"\n策略统计:")
    print(f"  充电时间点: {(df_result['power'] < 0).sum()}")
    print(f"  放电时间点: {(df_result['power'] > 0).sum()}")
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    df_result[['times', '实时价格', 'power']].to_csv(os.path.join(output_dir, 'output.csv'), index=False)
    print(f"\n✓ 已更新 output.csv")
    
    return df_result, total_profit

if __name__ == '__main__':
    run_xgboost_fixed()
