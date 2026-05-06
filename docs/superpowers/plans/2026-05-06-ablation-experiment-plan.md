# XGBoost 消融实验计划

&gt; **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过对比消融实验，验证 XGBoost 模型、增强特征工程和优化算法对电价预测准确率和总收益的提升效果。

**Architecture:** 采用模块化设计，将基线、特征工程、模型、策略分离，便于单独替换和对比。使用统一的评估框架计算 RMSE/MAE 和总收益。

**Tech Stack:** Python, XGBoost, scikit-learn, pandas, numpy, optuna (超参数优化)

---

## 文件结构

```
/workspace/
├── experiments/
│   ├── __init__.py
│   ├── base.py              # 基础接口和工具函数
│   ├── features.py          # 特征工程模块
│   ├── models.py            # 模型模块（Baseline GB + XGBoost）
│   ├── strategies.py        # 策略生成模块
│   ├── evaluator.py         # 评估模块
│   └── run_experiments.py   # 实验运行入口
├── output/
│   └── experiments/         # 实验结果输出目录
└── docs/superpowers/plans/
    └── 2026-05-06-ablation-experiment-plan.md
```

---

## 实验设计

### 实验配置

| 实验编号 | 模型          | 特征工程 | 超参数优化 | 策略优化 | 描述                          |
|----------|---------------|----------|------------|----------|-------------------------------|
| Exp 0    | GradientBoost | 基础     | 无         | 暴力搜索 | **基线** (当前 sklearn_baseline) |
| Exp 1    | XGBoost       | 基础     | 无         | 暴力搜索 | 仅替换模型                    |
| Exp 2    | XGBoost       | 增强     | 无         | 暴力搜索 | 模型 + 增强特征               |
| Exp 3    | XGBoost       | 增强     | Optuna     | 暴力搜索 | 模型 + 增强特征 + 超参优化    |
| Exp 4    | XGBoost       | 增强     | Optuna     | DP优化   | **完整版** (所有优化)         |

### 评估指标

- **预测准确率**: RMSE, MAE
- **经济收益**: 总收益, 平均日收益

---

## 任务分解

### Task 1: 创建基础模块和接口

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/base.py`

- [ ] **Step 1: 创建 __init__.py**

```python
"""
XGBoost 消融实验模块
"""
from .base import BaseExperiment, DataLoader
from .features import FeatureEngineer
from .models import ModelFactory
from .strategies import StrategyGenerator
from .evaluator import Evaluator
from .run_experiments import run_all_experiments

__all__ = [
    'BaseExperiment',
    'DataLoader',
    'FeatureEngineer',
    'ModelFactory',
    'StrategyGenerator',
    'Evaluator',
    'run_all_experiments'
]
```

- [ ] **Step 2: 创建 base.py - 数据加载器**

```python
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(current_dir, 'data')

@dataclass
class ExperimentConfig:
    name: str
    model_type: str  # 'gb' or 'xgboost'
    feature_level: str  # 'basic' or 'enhanced'
    use_hpo: bool
    strategy_type: str  # 'brute' or 'dp'


class DataLoader:
    def __init__(self):
        self.train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
        self.train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
        self.test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')
        
    def check_data_exists(self) -&gt; bool:
        return (os.path.exists(self.train_feature_path) and 
                os.path.exists(self.train_label_path) and 
                os.path.exists(self.test_feature_path))
    
    def load_train_data(self) -&gt; Tuple[pd.DataFrame, pd.DataFrame]:
        df_feat = pd.read_csv(self.train_feature_path)
        df_label = pd.read_csv(self.train_label_path)
        df_train = pd.merge(df_feat, df_label, on='times', how='inner')
        df_train['times'] = pd.to_datetime(df_train['times'])
        return df_train
    
    def load_test_data(self) -&gt; pd.DataFrame:
        df_test = pd.read_csv(self.test_feature_path)
        df_test['times'] = pd.to_datetime(df_test['times'])
        return df_test


# 基础特征列
BASIC_FEATURE_COLS = [
    '系统负荷预测值', '风光总加预测值', '联络线预测值',
    '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
]
TARGET_COL = 'A'
```

- [ ] **Step 3: 提交任务 1**

```bash
git add experiments/__init__.py experiments/base.py
git commit -m "feat: add base module and data loader"
```

---

### Task 2: 实现特征工程模块

**Files:**
- Create: `experiments/features.py`

- [ ] **Step 1: 创建 features.py**

```python
import pandas as pd
import numpy as np
from typing import List
from .base import BASIC_FEATURE_COLS


class FeatureEngineer:
    def __init__(self, feature_level: str = 'basic'):
        self.feature_level = feature_level
        self.basic_cols = BASIC_FEATURE_COLS.copy()
        
    def add_basic_time_features(self, df: pd.DataFrame) -&gt; pd.DataFrame:
        df = df.copy()
        df['hour'] = df['times'].dt.hour
        df['minute'] = df['times'].dt.minute
        df['dayofweek'] = df['times'].dt.dayofweek
        df['month'] = df['times'].dt.month
        return df
    
    def add_enhanced_time_features(self, df: pd.DataFrame) -&gt; pd.DataFrame:
        df = df.copy()
        df = self.add_basic_time_features(df)
        
        df['is_weekend'] = (df['dayofweek'] &gt;= 5).astype(int)
        df['quarter'] = df['times'].dt.quarter
        df['dayofyear'] = df['times'].dt.dayofyear
        
        hour = df['hour']
        df['is_morning_peak'] = ((hour &gt;= 7) &amp; (hour &lt;= 11)).astype(int)
        df['is_evening_peak'] = ((hour &gt;= 17) &amp; (hour &lt;= 21)).astype(int)
        df['is_night'] = ((hour &gt;= 0) &amp; (hour &lt; 6)).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 96, 96*7]) -&gt; pd.DataFrame:
        df = df.copy()
        for col in self.basic_cols:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [4, 12, 24, 96]) -&gt; pd.DataFrame:
        df = df.copy()
        for col in self.basic_cols:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
        return df
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -&gt; pd.DataFrame:
        df = df.copy()
        
        if self.feature_level == 'basic':
            df = self.add_basic_time_features(df)
            feature_cols = self.basic_cols + ['hour', 'minute', 'dayofweek', 'month']
        elif self.feature_level == 'enhanced':
            df = self.add_enhanced_time_features(df)
            
            if is_train:
                df = self.add_lag_features(df)
                df = self.add_rolling_features(df)
                df = df.dropna()
            
            enhanced_time_cols = [
                'hour', 'minute', 'dayofweek', 'month',
                'is_weekend', 'quarter', 'dayofyear',
                'is_morning_peak', 'is_evening_peak', 'is_night',
                'hour_sin', 'hour_cos',
                'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos'
            ]
            
            if is_train:
                lag_cols = [f'{col}_lag{lag}' for col in self.basic_cols for lag in [1, 2, 3, 96, 96*7]]
                roll_cols = [f'{col}_roll_mean_{w}' for col in self.basic_cols for w in [4, 12, 24, 96]]
                roll_cols += [f'{col}_roll_std_{w}' for col in self.basic_cols for w in [4, 12, 24, 96]]
                feature_cols = self.basic_cols + enhanced_time_cols + lag_cols + roll_cols
            else:
                feature_cols = self.basic_cols + enhanced_time_cols
        
        return df, feature_cols
```

- [ ] **Step 2: 提交任务 2**

```bash
git add experiments/features.py
git commit -m "feat: add feature engineering module"
```

---

### Task 3: 实现模型模块 (GB + XGBoost)

**Files:**
- Create: `experiments/models.py`

- [ ] **Step 1: 创建 models.py**

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. Will use GradientBoosting only.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class BaseModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -&gt; np.ndarray:
        raise NotImplementedError
    
    def get_params(self) -&gt; Dict[str, Any]:
        return self.model.get_params() if self.model else {}


class GBModel(BaseModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__()
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'verbose': 0
        }
        self.params = params if params else default_params
        self.model = GradientBoostingRegressor(**self.params)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        self.model.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            return {'rmse': rmse, 'mae': mae}
        return {}
    
    def predict(self, X: np.ndarray) -&gt; np.ndarray:
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    def __init__(self, params: Dict[str, Any] = None, use_hpo: bool = False):
        super().__init__()
        self.use_hpo = use_hpo
        self.best_params = None
        
        if params:
            self.params = params
        else:
            self.params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'verbosity': 0
            }
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = GradientBoostingRegressor(**{k: v for k, v in self.params.items() 
                                                    if k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample']})
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(**params)
        else:
            model = GradientBoostingRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                subsample=params['subsample'],
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        
        if self.use_hpo and OPTUNA_AVAILABLE and X_val is not None and y_val is not None:
            print("  Running Optuna hyperparameter optimization...")
            study = optuna.create_study(direction='minimize', study_name='xgb_hpo')
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), 
                          n_trials=50, show_progress_bar=True)
            
            self.best_params = study.best_params
            self.best_params.update({
                'objective': 'reg:squarederror',
                'random_state': 42,
                'verbosity': 0
            })
            print(f"  Best params: {self.best_params}")
            
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(**self.best_params)
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=self.best_params['n_estimators'],
                    learning_rate=self.best_params['learning_rate'],
                    max_depth=self.best_params['max_depth'],
                    subsample=self.best_params['subsample'],
                    random_state=42
                )
        
        self.model.fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            return {'rmse': rmse, 'mae': mae}
        return {}
    
    def predict(self, X: np.ndarray) -&gt; np.ndarray:
        return self.model.predict(X)


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, use_hpo: bool = False, params: Dict[str, Any] = None) -&gt; BaseModel:
        if model_type == 'gb':
            return GBModel(params)
        elif model_type == 'xgboost':
            return XGBoostModel(params, use_hpo=use_hpo)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

- [ ] **Step 2: 提交任务 3**

```bash
git add experiments/models.py
git commit -m "feat: add models module (GB + XGBoost + HPO)"
```

---

### Task 4: 实现策略生成模块 (暴力搜索 + DP优化)

**Files:**
- Create: `experiments/strategies.py`

- [ ] **Step 1: 创建 strategies.py**

```python
import pandas as pd
import numpy as np
from typing import Tuple


class BaseStrategy:
    def __init__(self):
        pass
    
    def generate(self, df_price: pd.DataFrame) -&gt; pd.DataFrame:
        raise NotImplementedError


class BruteForceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
    
    def generate(self, df_price: pd.DataFrame) -&gt; Tuple[pd.DataFrame, float]:
        df = df_price.copy()
        df['times'] = pd.to_datetime(df['times'])
        df['date'] = df['times'].dt.date
        
        results = []
        total_profit = 0
        
        for date, group in df.groupby('date'):
            prices = group['A'].values
            times = group['times'].values
            
            n = len(prices)
            if n != 96:
                print(f"Warning: {date} has {n} points, expected 96")
                continue
            
            best_profit = 0
            best_tc = -1
            best_td = -1
            
            for tc in range(0, 81):
                charge_cost = np.sum(prices[tc:tc+8]) * 1000
                
                for td in range(tc + 8, 89):
                    discharge_revenue = np.sum(prices[td:td+8]) * 1000
                    profit = discharge_revenue - charge_cost
                    
                    if profit &gt; best_profit:
                        best_profit = profit
                        best_tc = tc
                        best_td = td
            
            power = np.zeros(96)
            if best_tc &gt;= 0 and best_td &gt;= 0:
                power[best_tc:best_tc+8] = -1000
                power[best_td:best_td+8] = 1000
                total_profit += best_profit
            
            for i, (t, p, pr) in enumerate(zip(times, power, prices)):
                results.append({
                    'times': t,
                    '实时价格': pr,
                    'power': p
                })
        
        df_result = pd.DataFrame(results)
        return df_result, total_profit


class DynamicProgrammingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
    
    def generate(self, df_price: pd.DataFrame) -&gt; Tuple[pd.DataFrame, float]:
        df = df_price.copy()
        df['times'] = pd.to_datetime(df['times'])
        df['date'] = df['times'].dt.date
        
        results = []
        total_profit = 0
        
        for date, group in df.groupby('date'):
            prices = group['A'].values
            times = group['times'].values
            
            n = len(prices)
            if n != 96:
                print(f"Warning: {date} has {n} points, expected 96")
                continue
            
            charge_window = 8
            discharge_window = 8
            
            min_charge_cost = float('inf')
            best_tc = -1
            
            for tc in range(0, 81):
                cost = np.sum(prices[tc:tc+charge_window])
                if cost &lt; min_charge_cost:
                    min_charge_cost = cost
                    best_tc = tc
            
            max_discharge_revenue = 0
            best_td = -1
            
            for td in range(best_tc + charge_window, 89):
                revenue = np.sum(prices[td:td+discharge_window])
                if revenue &gt; max_discharge_revenue:
                    max_discharge_revenue = revenue
                    best_td = td
            
            best_profit = (max_discharge_revenue - min_charge_cost) * 1000
            
            if best_profit &lt;= 0:
                best_tc = -1
                best_td = -1
                best_profit = 0
            
            power = np.zeros(96)
            if best_tc &gt;= 0 and best_td &gt;= 0:
                power[best_tc:best_tc+8] = -1000
                power[best_td:best_td+8] = 1000
                total_profit += best_profit
            
            for i, (t, p, pr) in enumerate(zip(times, power, prices)):
                results.append({
                    'times': t,
                    '实时价格': pr,
                    'power': p
                })
        
        df_result = pd.DataFrame(results)
        return df_result, total_profit


class StrategyGenerator:
    @staticmethod
    def create_strategy(strategy_type: str) -&gt; BaseStrategy:
        if strategy_type == 'brute':
            return BruteForceStrategy()
        elif strategy_type == 'dp':
            return DynamicProgrammingStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
```

- [ ] **Step 2: 提交任务 4**

```bash
git add experiments/strategies.py
git commit -m "feat: add strategies module (brute force + DP)"
```

---

### Task 5: 实现评估模块

**Files:**
- Create: `experiments/evaluator.py`

- [ ] **Step 1: 创建 evaluator.py**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple
import os
from datetime import datetime


class Evaluator:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(current_dir, 'output', 'experiments')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = []
    
    def evaluate_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -&gt; Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def evaluate_strategy(self, df_result: pd.DataFrame, total_profit: float) -&gt; Dict[str, float]:
        df_result['times'] = pd.to_datetime(df_result['times'])
        df_result['date'] = df_result['times'].dt.date
        n_days = len(df_result['date'].unique())
        
        avg_daily_profit = total_profit / n_days if n_days &gt; 0 else 0
        
        return {
            'total_profit': total_profit,
            'avg_daily_profit': avg_daily_profit,
            'n_days': n_days
        }
    
    def add_experiment_result(self, config_name: str, 
                             pred_metrics: Dict[str, float],
                             strategy_metrics: Dict[str, float],
                             duration: float = None):
        result = {
            'experiment': config_name,
            'rmse': pred_metrics.get('rmse', np.nan),
            'mae': pred_metrics.get('mae', np.nan),
            'mape': pred_metrics.get('mape', np.nan),
            'total_profit': strategy_metrics.get('total_profit', np.nan),
            'avg_daily_profit': strategy_metrics.get('avg_daily_profit', np.nan),
            'n_days': strategy_metrics.get('n_days', 0),
            'duration_seconds': duration
        }
        self.results.append(result)
    
    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'experiment_results_{timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filepath, index=False, float_format='%.6f')
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY".center(100))
        print("="*100)
        
        df_results = pd.DataFrame(self.results)
        
        cols = ['experiment', 'rmse', 'mae', 'total_profit', 'avg_daily_profit', 'duration_seconds']
        df_display = df_results[cols].copy()
        
        df_display['rmse_improvement'] = (df_display['rmse'].iloc[0] - df_display['rmse']) / df_display['rmse'].iloc[0] * 100
        df_display['profit_improvement'] = (df_display['total_profit'] - df_display['total_profit'].iloc[0]) / df_display['total_profit'].iloc[0] * 100
        
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print("\nMetrics:")
        print(df_display.to_string(index=False))
        
        print("\n" + "-"*100)
        print("Key Improvements vs Baseline:")
        
        best_idx = df_display['total_profit'].idxmax()
        print(f"  Best Total Profit: {df_display['experiment'].iloc[best_idx]} - {df_display['total_profit'].iloc[best_idx]:.2f}")
        
        if len(df_display) &gt; 1:
            print(f"  Profit Improvement: {df_display['profit_improvement'].iloc[best_idx]:.2f}%")
            print(f"  RMSE Reduction: {df_display['rmse_improvement'].iloc[best_idx]:.2f}%")
        
        print("="*100)
```

- [ ] **Step 2: 提交任务 5**

```bash
git add experiments/evaluator.py
git commit -m "feat: add evaluator module"
```

---

### Task 6: 实现实验运行入口

**Files:**
- Create: `experiments/run_experiments.py`

- [ ] **Step 1: 创建 run_experiments.py**

```python
import os
import time
import pandas as pd
import numpy as np

from .base import DataLoader, ExperimentConfig, TARGET_COL
from .features import FeatureEngineer
from .models import ModelFactory
from .strategies import StrategyGenerator
from .evaluator import Evaluator


EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name='Exp0_Baseline_GB',
        model_type='gb',
        feature_level='basic',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp1_XGBoost_BasicFeatures',
        model_type='xgboost',
        feature_level='basic',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp2_XGBoost_EnhancedFeatures',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp3_XGBoost_Enhanced_HPO',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp4_Full_XGBoost_HPO_DP',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='dp'
    )
]


def run_single_experiment(config: ExperimentConfig, 
                         data_loader: DataLoader,
                         evaluator: Evaluator) -&gt; None:
    print("\n" + "="*80)
    print(f"RUNNING: {config.name}".center(80))
    print("="*80)
    
    start_time = time.time()
    
    df_train = data_loader.load_train_data()
    df_test = data_loader.load_test_data()
    
    print(f"\n[1/4] Feature engineering (level: {config.feature_level})...")
    fe = FeatureEngineer(feature_level=config.feature_level)
    df_train_processed, feature_cols = fe.transform(df_train, is_train=True)
    df_test_processed, _ = fe.transform(df_test, is_train=False)
    
    feature_cols = [col for col in feature_cols if col in df_test_processed.columns]
    
    X = df_train_processed[feature_cols].values
    y = df_train_processed[TARGET_COL].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    print(f"\n[2/4] Training model ({config.model_type}, HPO: {config.use_hpo})...")
    model = ModelFactory.create_model(config.model_type, use_hpo=config.use_hpo)
    val_metrics = model.fit(X_train, y_train, X_val, y_val)
    
    if val_metrics:
        print(f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}")
    
    print(f"\n[3/4] Predicting on test set...")
    X_test = df_test_processed[feature_cols].values
    y_test_pred = model.predict(X_test)
    
    df_price_pred = pd.DataFrame({
        'times': df_test_processed['times'],
        'A': y_test_pred
    })
    
    print(f"\n[4/4] Generating strategy ({config.strategy_type})...")
    strategy = StrategyGenerator.create_strategy(config.strategy_type)
    df_result, total_profit = strategy.generate(df_price_pred)
    
    strategy_metrics = evaluator.evaluate_strategy(df_result, total_profit)
    print(f"  Total Profit: {strategy_metrics['total_profit']:.2f}")
    print(f"  Avg Daily Profit: {strategy_metrics['avg_daily_profit']:.2f}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'output', 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    df_result.to_csv(os.path.join(output_dir, f'{config.name}_output.csv'), index=False)
    
    duration = time.time() - start_time
    
    evaluator.add_experiment_result(
        config.name,
        val_metrics,
        strategy_metrics,
        duration
    )
    
    print(f"\nCompleted in {duration:.2f}s")


def run_all_experiments():
    print("="*80)
    print("ABLATION EXPERIMENTS: XGBoost + Feature Engineering + Optimization".center(80))
    print("="*80)
    
    data_loader = DataLoader()
    
    if not data_loader.check_data_exists():
        print("ERROR: Data files not found!")
        print("Please download data and place in data/ directory")
        return
    
    evaluator = Evaluator()
    
    for config in EXPERIMENT_CONFIGS:
        try:
            run_single_experiment(config, data_loader, evaluator)
        except Exception as e:
            print(f"ERROR in {config.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    evaluator.print_summary()
    evaluator.save_results()
    
    print("\nAll experiments completed!")


if __name__ == '__main__':
    run_all_experiments()
```

- [ ] **Step 2: 创建 requirements.txt**

```txt
pandas&gt;=1.5.0
numpy&gt;=1.23.0
scikit-learn&gt;=1.0.0
xgboost&gt;=1.7.0
optuna&gt;=3.0.0
```

- [ ] **Step 3: 创建 run.py 作为快捷入口**

```python
#!/usr/bin/env python
"""
快捷入口：运行消融实验
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import run_all_experiments

if __name__ == '__main__':
    run_all_experiments()
```

- [ ] **Step 4: 提交任务 6**

```bash
git add experiments/run_experiments.py requirements.txt run.py
git commit -m "feat: add experiment runner and requirements"
```

---

### Task 7: 创建 README 和示例运行

**Files:**
- Create: `experiments/README.md`

- [ ] **Step 1: 创建 experiments/README.md**

```markdown
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
```

- [ ] **Step 2: 提交任务 7**

```bash
git add experiments/README.md
git commit -m "docs: add experiments README"
```

---

## 执行总结

计划完成！现在可以执行实验了。

**文件创建列表：**
- `experiments/__init__.py` - 模块初始化
- `experiments/base.py` - 基础接口和数据加载
- `experiments/features.py` - 特征工程（基础+增强）
- `experiments/models.py` - 模型（GB + XGBoost + Optuna HPO）
- `experiments/strategies.py` - 策略（暴力搜索 + DP优化）
- `experiments/evaluator.py` - 评估模块
- `experiments/run_experiments.py` - 实验运行入口
- `experiments/README.md` - 实验说明
- `requirements.txt` - 依赖
- `run.py` - 快捷入口
- `docs/superpowers/plans/2026-05-06-ablation-experiment-plan.md` - 本计划文档

**下一步：**
1. 安装依赖：`pip install -r requirements.txt`
2. 确保数据在 `data/` 目录
3. 运行实验：`python run.py`
