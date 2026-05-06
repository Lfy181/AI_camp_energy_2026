# 消融实验模块

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行实验

确保数据文件已放置在 `data/` 目录下：

```
data/
├── train/
│   ├── mengxi_boundary_anon_filtered.csv
│   └── mengxi_node_price_selected.csv
└── test/
    └── test_in_feature_ori.csv
```

运行所有实验：

```bash
python run.py
```

或直接：

```bash
python -m experiments.run_experiments
```

## 实验说明

| 实验编号 | 模型          | 特征工程 | 超参数优化 | 策略优化 |
|----------|---------------|----------|------------|----------|
| Exp 0    | GradientBoost | 基础     | 无         | 暴力搜索 |
| Exp 1    | XGBoost       | 基础     | 无         | 暴力搜索 |
| Exp 2    | XGBoost       | 增强     | 无         | 暴力搜索 |
| Exp 3    | XGBoost       | 增强     | Optuna     | 暴力搜索 |
| Exp 4    | XGBoost       | 增强     | Optuna     | DP优化   |

## 输出

结果将保存在 `output/experiments/` 目录下：
- `experiment_results_*.csv` - 所有实验的汇总指标
- `Exp*_output.csv` - 每个实验的充放电策略输出
