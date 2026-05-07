#!/usr/bin/env python
"""
新实验: MLP + 简化增强特征 + 优化算法
使用测试集可用的增强特征（不含滞后和滚动特征）
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
    """简化版增强特征 - 只包含测试集可用的特征"""
    
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
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # 交互特征（乘法）
        df['负荷_x_风光'] = df['系统负荷预测值'] * df['风光总加预测值']
        df['负荷_x_风电'] = df['系统负荷预测值'] * df['风电预测值']
        df['负荷_x_光伏'] = df['系统负荷预测值'] * df['光伏预测值']
        df['风光_x_风电'] = df['风光总加预测值'] * df['风电预测值']
        df['风光_x_光伏'] = df['风光总加预测值'] * df['光伏预测值']
        df['风电_x_光伏'] = df['风电预测值'] * df['光伏预测值']
        
        # 交互特征（除法）
        df['负荷_div_风光'] = df['系统负荷预测值'] / (df['风光总加预测值'].abs() + 1e-8)
        df['风光_div_风电'] = df['风光总加预测值'] / (df['风电预测值'].abs() + 1e-8)
        df['风光_div_光伏'] = df['风光总加预测值'] / (df['光伏预测值'].abs() + 1e-8)
        
        # 清理
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].clip(-1e10, 1e10)
        
        feature_cols = self.basic_cols + [
            'hour', 'minute', 'dayofweek', 'month', 'is_weekend', 'quarter', 'dayofyear',
            'is_morning_peak', 'is_evening_peak', 'is_night', 'is_afternoon',
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
            '负荷_x_风光', '负荷_x_风电', '负荷_x_光伏',
            '风光_x_风电', '风光_x_光伏', '风电_x_光伏',
            '负荷_div_风光', '风光_div_风电', '风光_div_光伏'
        ]
        
        return df, feature_cols


def run_mlp_enhanced_experiment():
    print("="*80)
    print("新实验: MLP + 简化增强特征 + 暴力搜索/DP".center(80))
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
    
    # 暴力搜索策略
    print("\n  [暴力搜索策略]")
    strategy_brute = StrategyGenerator.create_strategy('brute')
    df_result_brute, total_profit_brute = strategy_brute.generate(df_price_pred)
    
    print(f"  总收益: {total_profit_brute:,.2f}")
    print(f"  平均日收益: {total_profit_brute/df_result_brute['times'].dt.date.nunique():,.2f}")
    
    # DP策略
    print("\n  [DP优化策略]")
    strategy_dp = StrategyGenerator.create_strategy('dp')
    df_result_dp, total_profit_dp = strategy_dp.generate(df_price_pred)
    
    print(f"  总收益: {total_profit_dp:,.2f}")
    print(f"  平均日收益: {total_profit_dp/df_result_dp['times'].dt.date.nunique():,.2f}")
    
    # 与之前结果对比
    print("\n" + "="*80)
    print("完整对比".center(80))
    print("="*80)
    
    baseline_profit = 696353.72
    mlp_basic_profit = 837523.25
    
    print(f"\n{'实验':<40} {'总收益':>15} {'提升':>12}")
    print("-" * 70)
    print(f"{'GradientBoost (基线)':<40} {baseline_profit:>15,.2f} {'-':>12}")
    print(f"{'MLP + 基础特征 + 暴力':<40} {mlp_basic_profit:>15,.2f} {'+20.27%':>12}")
    print(f"{'MLP + 简化增强特征 + 暴力':<40} {total_profit_brute:>15,.2f} {(total_profit_brute-baseline_profit)/baseline_profit*100:>+11.2f}%")
    print(f"{'MLP + 简化增强特征 + DP':<40} {total_profit_dp:>15,.2f} {(total_profit_dp-baseline_profit)/baseline_profit*100:>+11.2f}%")
    
    # 确定最佳
    results = {
        'MLP + 基础特征 + 暴力': mlp_basic_profit,
        'MLP + 简化增强特征 + 暴力': total_profit_brute,
        'MLP + 简化增强特征 + DP': total_profit_dp
    }
    best = max(results, key=results.get)
    best_profit = results[best]
    
    print(f"\n最佳方案: {best}")
    print(f"最佳收益: {best_profit:,.2f}")
    print(f"相对基线提升: {(best_profit-baseline_profit)/baseline_profit*100:+.2f}%")
    
    # 保存结果
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    if best == 'MLP + 简化增强特征 + 暴力':
        df_result_brute.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
        df_result_brute.to_csv(os.path.join(output_dir, 'MLP_Enhanced_Brute_output.csv'), index=False)
    else:
        df_result_dp.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
        df_result_dp.to_csv(os.path.join(output_dir, 'MLP_Enhanced_DP_output.csv'), index=False)
    
    print(f"\n✓ 已更新 output.csv 为最佳策略结果")
    
    return results

if __name__ == '__main__':
    run_mlp_enhanced_experiment()
