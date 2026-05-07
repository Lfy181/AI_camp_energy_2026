#!/usr/bin/env python
"""
运行最佳实验: MLP + 基础特征 + 暴力搜索
收益提升: +20.27%
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import DataLoader
from experiments.features import FeatureEngineer
from experiments.models import ModelFactory
from experiments.strategies import StrategyGenerator

def main():
    print("="*80)
    print("运行最佳方案: MLP (Exp5)".center(80))
    print("="*80)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader()
    df_train = data_loader.load_train_data()
    df_test = data_loader.load_test_data()
    print(f"  训练数据: {df_train.shape}")
    print(f"  测试数据: {df_test.shape}")
    
    # 特征工程 (基础特征)
    print("\n[2/4] 特征工程 (基础特征)...")
    fe = FeatureEngineer(feature_level='basic')
    df_train_processed, feature_cols = fe.transform(df_train, is_train=True)
    df_test_processed, _ = fe.transform(df_test, is_train=False)
    print(f"  特征数量: {len(feature_cols)}")
    
    # 准备训练数据
    X = df_train_processed[feature_cols].values
    y = df_train_processed['A'].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    
    # 训练 MLP 模型
    print("\n[3/4] 训练 MLP 模型...")
    model = ModelFactory.create_model('mlp')
    val_metrics = model.fit(X_train, y_train, X_val, y_val)
    print(f"  验证 RMSE: {val_metrics['rmse']:.6f}")
    print(f"  验证 MAE: {val_metrics['mae']:.6f}")
    
    # 预测
    print("\n[4/4] 预测和策略生成...")
    X_test = df_test_processed[feature_cols].values
    y_test_pred = model.predict(X_test)
    
    df_price_pred = pd.DataFrame({
        'times': df_test_processed['times'],
        'A': y_test_pred
    })
    
    # 生成策略
    strategy = StrategyGenerator.create_strategy('brute')
    df_result, total_profit = strategy.generate(df_price_pred)
    
    print(f"\n" + "="*80)
    print("实验结果".center(80))
    print("="*80)
    print(f"模型: MLP (多层感知机)")
    print(f"特征: 基础特征 ({len(feature_cols)}个)")
    print(f"策略: 暴力搜索")
    print(f"总收益: {total_profit:,.2f}")
    print(f"平均日收益: {total_profit/df_result['times'].dt.date.nunique():,.2f}")
    print(f"验证 RMSE: {val_metrics['rmse']:.6f}")
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.csv')
    df_result.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "="*80)
    print("充放电策略示例 (前96个时间点/第一天)".center(80))
    print("="*80)
    
    df_sample = df_result.head(96).copy()
    df_sample['power_action'] = df_sample['power'].apply(
        lambda x: '充电' if x < 0 else ('放电' if x > 0 else '-')
    )
    print(df_sample[['times', '实时价格', 'power', 'power_action']].to_string(index=False))
    
    # 统计充放电时间
    charge_points = df_result[df_result['power'] < 0]
    discharge_points = df_result[df_result['power'] > 0]
    print(f"\n充电时间点数量: {len(charge_points)}")
    print(f"放电时间点数量: {len(discharge_points)}")

if __name__ == '__main__':
    main()
