#!/usr/bin/env python
"""
新实验: MLP + 增强特征 + 暴力搜索
探索MLP在增强特征下的表现
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

def run_mlp_enhanced_experiment():
    print("="*80)
    print("新实验: MLP + 增强特征 + 暴力搜索".center(80))
    print("="*80)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader()
    df_train = data_loader.load_train_data()
    df_test = data_loader.load_test_data()
    print(f"  训练数据: {df_train.shape}")
    print(f"  测试数据: {df_test.shape}")
    
    # 特征工程 (增强特征)
    print("\n[2/4] 特征工程 (增强特征)...")
    fe = FeatureEngineer(feature_level='enhanced')
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
    print("\n[3/4] 训练 MLP 模型 (增强特征)...")
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
    
    # 生成策略 (暴力搜索)
    print("\n  使用暴力搜索策略...")
    strategy_brute = StrategyGenerator.create_strategy('brute')
    df_result_brute, total_profit_brute = strategy_brute.generate(df_price_pred)
    
    print(f"\n" + "="*80)
    print("MLP + 增强特征 + 暴力搜索 实验结果".center(80))
    print("="*80)
    print(f"模型: MLP (多层感知机)")
    print(f"特征: 增强特征 ({len(feature_cols)}个)")
    print(f"策略: 暴力搜索")
    print(f"总收益: {total_profit_brute:,.2f}")
    print(f"平均日收益: {total_profit_brute/df_result_brute['times'].dt.date.nunique():,.2f}")
    print(f"验证 RMSE: {val_metrics['rmse']:.6f}")
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'MLP_Enhanced_Brute_output.csv')
    df_result_brute.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 现在测试DP策略
    print("\n" + "="*80)
    print("新实验: MLP + 增强特征 + DP优化".center(80))
    print("="*80)
    
    print("\n  使用DP优化策略...")
    strategy_dp = StrategyGenerator.create_strategy('dp')
    df_result_dp, total_profit_dp = strategy_dp.generate(df_price_pred)
    
    print(f"\n" + "="*80)
    print("MLP + 增强特征 + DP优化 实验结果".center(80))
    print("="*80)
    print(f"模型: MLP (多层感知机)")
    print(f"特征: 增强特征 ({len(feature_cols)}个)")
    print(f"策略: DP优化")
    print(f"总收益: {total_profit_dp:,.2f}")
    print(f"平均日收益: {total_profit_dp/df_result_dp['times'].dt.date.nunique():,.2f}")
    print(f"验证 RMSE: {val_metrics['rmse']:.6f}")
    
    # 保存DP结果
    output_path_dp = os.path.join(output_dir, 'MLP_Enhanced_DP_output.csv')
    df_result_dp.to_csv(output_path_dp, index=False)
    print(f"\n结果已保存到: {output_path_dp}")
    
    # 与之前的最佳结果对比
    print("\n" + "="*80)
    print("与之前最佳结果对比".center(80))
    print("="*80)
    
    baseline_profit = 696353.72  # GradientBoost 基线
    mlp_basic_profit = 837523.25  # MLP + 基础特征
    
    print(f"\n模型对比:")
    print(f"{'实验':<30} {'总收益':>15} {'提升':>10}")
    print("-" * 60)
    print(f"{'GradientBoost (基线)':<30} {baseline_profit:>15,.2f} {'-':>10}")
    print(f"{'MLP + 基础特征 + 暴力':<30} {mlp_basic_profit:>15,.2f} {'+20.27%':>10}")
    print(f"{'MLP + 增强特征 + 暴力':<30} {total_profit_brute:>15,.2f} {(total_profit_brute-baseline_profit)/baseline_profit*100:>+9.2f}%")
    print(f"{'MLP + 增强特征 + DP':<30} {total_profit_dp:>15,.2f} {(total_profit_dp-baseline_profit)/baseline_profit*100:>+9.2f}%")
    
    # 确定最佳策略
    profits = {
        'MLP + 基础特征 + 暴力': mlp_basic_profit,
        'MLP + 增强特征 + 暴力': total_profit_brute,
        'MLP + 增强特征 + DP': total_profit_dp
    }
    best_strategy = max(profits, key=profits.get)
    best_profit = profits[best_strategy]
    
    print(f"\n最佳策略: {best_strategy}")
    print(f"最佳收益: {best_profit:,.2f}")
    print(f"相对基线提升: {(best_profit-baseline_profit)/baseline_profit*100:+.2f}%")
    
    # 更新output.csv为最佳策略
    if best_strategy == 'MLP + 增强特征 + 暴力':
        df_result_brute.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
        print(f"\n✓ 已更新 output.csv 为最佳策略结果")
    elif best_strategy == 'MLP + 增强特征 + DP':
        df_result_dp.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
        print(f"\n✓ 已更新 output.csv 为最佳策略结果")
    
    return {
        'mlp_enhanced_brute': total_profit_brute,
        'mlp_enhanced_dp': total_profit_dp,
        'rmse': val_metrics['rmse'],
        'best_strategy': best_strategy,
        'best_profit': best_profit
    }

if __name__ == '__main__':
    results = run_mlp_enhanced_experiment()
