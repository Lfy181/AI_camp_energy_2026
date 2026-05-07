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

### 增强特征 (700+个)

#### 1. 时间特征增强
- 周期编码: `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos`
- 时段标记: `is_morning_peak`, `is_evening_peak`, `is_night`, `is_afternoon`
- 其他: `is_weekend`, `quarter`, `dayofyear`, `weekofyear`

#### 2. 滞后特征
```python
lags = [1, 2, 3, 4, 8, 16, 24, 48, 96, 192, 672]  # 1分钟到7天
```

#### 3. 滚动统计特征
```python
windows = [4, 8, 16, 24, 48, 96, 192]  # 1小时到2天
stats = ['mean', 'std', 'min', 'max', 'median']
```

#### 4. 变化率特征
- 差分: `diff_1`, `diff_4`, `diff_8`
- 百分比变化: `pct_change_1`, `pct_change_4`

#### 5. 聚合特征
- 按小时聚合: `hour_mean`, `hour_std`
- 按星期聚合: `dow_mean`, `dow_std`
- 偏离度: `dev_from_hour_mean`, `dev_from_dow_mean`

#### 6. 交互特征
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
- `min_child_weight`: 1-10
- `gamma`: 0.0-1.0
- `reg_alpha`: 0.0-1.0
- `reg_lambda`: 0.0-1.0

### LightGBM 优化参数
- `n_estimators`: 100-1000
- `learning_rate`: 0.01-0.3
- `max_depth`: 3-12
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0
- `min_child_samples`: 1-20
- `reg_alpha`: 0.0-1.0
- `reg_lambda`: 0.0-1.0

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

### 仅增强特征实验 (Exp6-Exp12)

```bash
python run_remaining_experiments.py
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
model = ModelFactory.create_model('xgboost', use_hpo=True)
model.fit(X_train, y_train, X_val, y_val)

# 预测
y_pred = model.predict(X_test)

# 生成策略
strategy = StrategyGenerator.create_strategy('dp')
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
| **Exp5** | **MLP** | **0.7123** | **837,523** | **+20.3%** |

### 增强特征实验 (Exp6-Exp12)

| 实验 | 模型 | HPO | 策略 | RMSE | 总收益 | 提升 |
|------|------|-----|------|------|--------|------|
| Exp6 | XGBoost | ❌ | 暴力 | 0.5783 | 724,060 | +4.0% |
| Exp7 | LightGBM | ❌ | 暴力 | 0.5682 | 714,898 | +2.7% |
| Exp8 | RandomForest | ❌ | 暴力 | 0.5700 | 729,861 | +4.8% |
| Exp9 | XGBoost | ✅ | 暴力 | 0.5617 | 693,334 | -0.4% |
| Exp10 | LightGBM | ✅ | 暴力 | 0.5698 | 713,976 | +2.5% |
| **Exp11** | **XGBoost** | **✅** | **DP** | **0.5564** | **784,857** | **+12.7%** |
| Exp12 | LightGBM | ✅ | DP | 0.5696 | 706,212 | +1.4% |

## 输出文件

实验结果保存在 `output/experiments/` 目录：

```
output/experiments/
├── experiment_results_YYYYMMDD_HHMMSS.csv  # 汇总指标
├── Exp0_Baseline_GB_output.csv             # 各实验的策略输出
├── Exp1_XGBoost_BasicFeatures_output.csv
├── ...
└── Exp12_Full_LightGBM_HPO_DP_output.csv
```

## 关键发现

1. **预测准确率 ≠ 经济收益**: RMSE最低的模型不一定收益最高
2. **特征工程至关重要**: 增强特征平均降低RMSE约13%
3. **简单模型也有优势**: Ridge和MLP在基础特征下表现优异
4. **DP策略效率高**: O(n)复杂度获得接近暴力搜索的结果

## 推荐配置

### 最高收益
- **模型**: MLP
- **特征**: 基础特征
- **策略**: 暴力搜索
- **收益**: +20.27%

### 综合最优
- **模型**: XGBoost
- **特征**: 增强特征
- **优化**: Optuna HPO
- **策略**: DP优化
- **RMSE**: 0.5564 (最低)
- **收益**: +12.7%
