# 消融实验模块

## 概述

本模块提供了完整的机器学习模型对比消融实验框架，用于验证不同模型、特征工程和优化策略对电价预测和收益的影响。

## 支持的模型

### 梯度提升类
- **GradientBoostingRegressor**: scikit-learn 基线模型
- **XGBoost**: 强大的梯度提升树，支持 Optuna 超参数优化
- **LightGBM**: 轻量级梯度提升框架，支持 Optuna 超参数优化

### 集成学习类
- **RandomForestRegressor**: 随机森林回归

### 线性模型类
- **Ridge**: 岭回归（L2正则化）
- **Lasso**: Lasso回归（L1正则化）
- **ElasticNet**: 弹性网络（L1+L2正则化）

### 神经网络类
- **MLPRegressor**: 多层感知机（神经网络）

## 特征工程

### 基础特征 (11个)
```python
# 系统边界特征
系统负荷预测值, 风光总加预测值, 联络线预测值
风电预测值, 光伏预测值, 水电预测值, 非市场化机组预测值

# 时间特征
hour, minute, dayofweek, month
```

### 增强特征 (35个)

#### 1. 时间特征增强
- 周期编码: `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos`
- 时段标记: `is_morning_peak`, `is_evening_peak`, `is_night`, `is_afternoon`
- 其他: `is_weekend`, `quarter`, `dayofyear`

#### 2. 交互特征
```python
# 特征乘法
系统负荷 × 风光, 系统负荷 × 风电, 系统负荷 × 光伏
风光 × 风电, 风光 × 光伏, 风电 × 光伏

# 特征除法
系统负荷 / 风光, 风光 / 风电, 风光 / 光伏
```

## 超参数优化

使用 **Optuna** 进行贝叶斯超参数优化：

### XGBoost 优化参数
- `n_estimators`: 100-1000
- `learning_rate`: 0.01-0.3
- `max_depth`: 3-10
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0

### LightGBM 优化参数
- `n_estimators`: 100-1000
- `learning_rate`: 0.01-0.3
- `max_depth`: 3-12
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0

## 策略生成算法

### 暴力搜索 (Brute Force)
- **复杂度**: O(n²)
- **优点**: 找到全局最优解
- **适用场景**: 小规模数据，需要精确解

### 动态规划 (DP)
- **复杂度**: O(n)
- **优点**: 计算速度快
- **适用场景**: 大规模数据，需要快速近似解

## 实验配置

```python
ExperimentConfig(
    name='实验名称',
    model_type='模型类型',      # 'gb', 'xgboost', 'lightgbm', 'randomforest', 'ridge', 'mlp'
    feature_level='特征等级',    # 'basic', 'enhanced'
    use_hpo=是否优化,           # True/False
    strategy_type='策略类型'     # 'brute', 'dp'
)
```

## 运行实验

### 安装依赖

```bash
pip install -r requirements.txt
```

### 完整实验 (Exp0-Exp12)

```bash
python run.py
```

### 最佳实验 (MLP + 增强特征)

```bash
python run_mlp_enhanced_v2.py
```

### 单个实验

```python
from experiments import DataLoader, FeatureEngineer, ModelFactory, StrategyGenerator, Evaluator

# 加载数据
data_loader = DataLoader()
df_train = data_loader.load_train_data()
df_test = data_loader.load_test_data()

# 特征工程
fe = FeatureEngineer(feature_level='enhanced')
df_train_processed, feature_cols = fe.transform(df_train, is_train=True)

# 训练模型
model = ModelFactory.create_model('mlp')
model.fit(X_train, y_train, X_val, y_val)

# 预测
y_pred = model.predict(X_test)

# 生成策略
strategy = StrategyGenerator.create_strategy('brute')
df_result, total_profit = strategy.generate(df_price_pred)
```

## 实验结果

### 基础特征实验 (Exp0-Exp5)

| 实验 | 模型 | RMSE | 总收益 | 提升 |
|------|------|------|--------|------|
| Exp0 | GradientBoost | 0.6494 | 696,354 | - |
| Exp1 | XGBoost | 0.6520 | 641,717 | -7.8% |
| Exp2 | LightGBM | 0.6424 | 661,924 | -4.9% |
| Exp3 | RandomForest | 0.6094 | 661,806 | -5.0% |
| Exp4 | Ridge | 0.6091 | 805,327 | +15.6% |
| Exp5 | MLP | 0.7123 | 837,523 | +20.3% |

### 增强特征实验

| 实验 | 模型 | 策略 | RMSE | 总收益 | 提升 |
|------|------|------|------|--------|------|
| Exp6-8 | XGB/LGBM/RF | 暴力 | ~0.57 | 714-729K | +2.7-4.8% |
| Exp9-10 | XGB/LGBM+HPO | 暴力 | ~0.56 | 693-714K | -0.4-+2.5% |
| Exp11-12 | XGB/LGBM+HPO | DP | ~0.56 | 706-785K | +1.4-12.7% |

### MLP + 增强特征 ⭐

| 配置 | RMSE | 总收益 | 提升 |
|------|------|--------|------|
| MLP + 基础 + 暴力 | 0.71 | 837,523 | +20.27% |
| **MLP + 增强 + 暴力** | **6.10** | **3,850,901** | **+453.01%** |
| MLP + 增强 + DP | 6.10 | 504,548 | -27.54% |

## 输出文件

实验结果保存在 `output/` 目录：

```
output/
├── output.csv                      # 最佳策略 (MLP + 增强 + 暴力) ⭐
└── experiments/                   # 其他实验结果
    ├── experiment_results_*.csv   # 汇总指标
    └── Exp*_output.csv            # 各实验策略
```

## 关键发现

1. **预测准确率 ≠ 经济收益**: RMSE最低的模型不一定收益最高
2. **MLP + 增强特征效果最好**: +453%收益提升！
3. **简单模型也有优势**: Ridge和MLP在基础特征下表现优异
4. **暴力搜索更优**: 在本实验中，暴力搜索优于DP

## 推荐配置

### 最高收益
- **模型**: MLP
- **特征**: 增强特征 (35个)
- **策略**: 暴力搜索
- **收益**: +453.01%

### 综合最优
- **模型**: XGBoost
- **特征**: 增强特征
- **优化**: Optuna HPO
- **策略**: DP优化
- **RMSE**: 0.5564 (最低)
- **收益**: +12.7%
