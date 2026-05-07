#!/usr/bin/env python
"""
修复策略：使用XGBoost模型重新生成最优策略
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import DataLoader
from experiments.features import FeatureEngineer
from experiments.models import ModelFactory
from experiments.strategies import StrategyGenerator

def run_xgboost_fixed():
    print("="*80)
    print("修复策略: XGBoost + 增强特征 + 暴力搜索".center(80))
    print("="*80)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader()
    df_train = data_loader.load_train_data()
    df_test = data_loader.load_test_data()
    print(f"  训练数据: {df_train.shape}")
    print(f"  测试数据: {df_test.shape}")
    
    # 特征工程 (增强特征)
    print("\n[2/4] 特征工程...")
    fe = FeatureEngineer(feature_level='enhanced')
    df_train_processed, feature_cols = fe.transform(df_train, is_train=True)
    df_test_processed, _ = fe.transform(df_test, is_train=False)
    print(f"  特征数量: {len(feature_cols)}")
    
    # 准备数据
    X = df_train_processed[feature_cols].values
    y = df_train_processed['A'].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
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
    
    # 生成策略
    strategy = StrategyGenerator.create_strategy('brute')
    df_result, total_profit = strategy.generate(df_price_pred)
    
    # 计算真实收益
    df_result['profit'] = df_result['power'] * df_result['实时价格'] * 0.25
    
    print(f"\n" + "="*80)
    print("策略结果".center(80))
    print("="*80)
    print(f"模型: XGBoost")
    print(f"特征: 增强特征")
    print(f"策略: 暴力搜索")
    print(f"\\n收益分析:")
    print(f"  总收益: {df_result['profit'].sum():,.2f}")
    print(f"  充电收益: {df_result[df_result['power'] < 0]['profit'].sum():,.2f}")
    print(f"  放电收益: {df_result[df_result['power'] > 0]['profit'].sum():,.2f}")
    print(f"\\n策略统计:")
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
