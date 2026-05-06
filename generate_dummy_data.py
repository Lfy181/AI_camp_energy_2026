#!/usr/bin/env python
"""
生成模拟数据用于测试消融实验框架
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_data(start_date, num_days, is_test=False):
    # 生成时间序列（15分钟间隔）
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    times = []
    for i in range(num_days * 96):
        times.append(start_dt + timedelta(minutes=15 * i))
    
    # 生成边界条件特征
    np.random.seed(42)
    n = len(times)
    hour = np.array([t.hour for t in times])
    
    # 基础负荷模式（早晚高峰）
    load_base = 50 + 20 * np.sin(2 * np.pi * (hour - 8) / 24) + 15 * np.sin(2 * np.pi * (hour - 18) / 12)
    
    data = {
        'times': times,
        '系统负荷实际值': load_base + np.random.normal(0, 3, n),
        '系统负荷预测值': load_base + np.random.normal(0, 5, n),
        '风光总加实际值': 30 + 20 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 5, n),
        '风光总加预测值': 30 + 20 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 7, n),
        '联络线实际值': 10 + np.random.normal(0, 5, n),
        '联络线预测值': 10 + np.random.normal(0, 6, n),
        '风电实际值': 15 + 10 * np.sin(2 * np.pi * hour / 24 + np.random.normal(0, 0.5, n)) + np.random.normal(0, 3, n),
        '风电预测值': 15 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 4, n),
        '光伏实际值': np.maximum(0, 20 * np.sin(2 * np.pi * (hour - 12) / 12) + np.random.normal(0, 3, n)),
        '光伏预测值': np.maximum(0, 20 * np.sin(2 * np.pi * (hour - 12) / 12) + np.random.normal(0, 4, n)),
        '水电实际值': 20 + np.random.normal(0, 3, n),
        '水电预测值': 20 + np.random.normal(0, 4, n),
        '非市场化机组实际值': 25 + np.random.normal(0, 3, n),
        '非市场化机组预测值': 25 + np.random.normal(0, 4, n),
    }
    
    df_boundary = pd.DataFrame(data)
    
    # 生成节点电价
    price_base = 100 + 50 * np.sin(2 * np.pi * (hour - 8) / 24) + 30 * np.sin(2 * np.pi * (hour - 18) / 12)
    price_noise = np.random.normal(0, 10, n)
    
    df_price = pd.DataFrame({
        'times': times,
        'A': price_base + price_noise
    })
    
    return df_boundary, df_price


def main():
    # 创建目录
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    print("Generating training data...")
    df_train_boundary, df_train_price = generate_data('2023-01-01', 30, is_test=False)
    df_train_boundary.to_csv('data/train/mengxi_boundary_anon_filtered.csv', index=False)
    df_train_price.to_csv('data/train/mengxi_node_price_selected.csv', index=False)
    print(f"  Train boundary: {df_train_boundary.shape}")
    print(f"  Train price: {df_train_price.shape}")
    
    print("\nGenerating test data...")
    df_test_boundary, _ = generate_data('2023-02-01', 7, is_test=True)
    df_test_boundary.to_csv('data/test/test_in_feature_ori.csv', index=False)
    print(f"  Test boundary: {df_test_boundary.shape}")
    
    print("\nDummy data generated successfully!")
    print("\nData structure:")
    print("data/")
    print("├── train/")
    print("│   ├── mengxi_boundary_anon_filtered.csv")
    print("│   └── mengxi_node_price_selected.csv")
    print("└── test/")
    print("    └── test_in_feature_ori.csv")


if __name__ == '__main__':
    main()
