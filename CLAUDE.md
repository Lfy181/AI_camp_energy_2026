# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an energy forecasting and optimization project that predicts node electricity prices and generates optimal battery charge/discharge strategies. The project has been expanded with comprehensive ablation experiments comparing multiple machine learning models and feature engineering strategies.

## Key Architecture

### Data Flow
1. **Input**: Boundary condition features (load, wind/solar generation, transmission lines, etc.)
2. **Model**: Multiple ML models (XGBoost, LightGBM, RandomForest, Ridge, MLP, etc.)
3. **Feature Engineering**: Basic features (11) vs Enhanced features (35)
4. **Output**: Price forecasts + optimal charge/discharge strategy

### Core Components

#### 1. Feature Engineering (`experiments/features.py`)

**Basic Features (11)**:
- 7 boundary features: 系统负荷、风光总加、联络线、风电，光伏、水电、非市场化机组
- 4 temporal features: hour, minute, dayofweek, month

**Enhanced Features (35)**:
- Cyclical encoding: hour_sin/cos, dayofweek_sin/cos, month_sin/cos
- Time period markers: is_morning_peak, is_evening_peak, is_night, is_afternoon
- Interaction features: multiplication and division between key features

#### 2. Machine Learning Models (`experiments/models.py`)

Supported models:
- `gb`: GradientBoostingRegressor (baseline)
- `xgboost`: XGBoost with optional Optuna HPO
- `lightgbm`: LightGBM with optional Optuna HPO
- `randomforest`: RandomForestRegressor
- `ridge/lasso/elasticnet`: Linear models
- `mlp`: MLPRegressor (neural network)

#### 3. Strategy Generation (`experiments/strategies.py`)

- **BruteForceStrategy**: O(n²) complexity, finds global optimum
- **DynamicProgrammingStrategy**: O(n) complexity, fast approximation

**Charge/Discharge Rules**:
- Charge: 8 time points (2 hours) at -1000MW
- Discharge: 8 time points (2 hours) at +1000MW
- Constraints: 0 ≤ tc ≤ 80, td ≥ tc + 8, td ≤ 88
- Objective: Maximize profit = discharge_revenue - charge_cost

### Data Structure

Training data:
- `data/train/mengxi_boundary_anon_filtered.csv`: Boundary condition features
- `data/train/mengxi_node_price_selected.csv`: Node price labels (column 'A')

Test data:
- `data/test/test_in_feature_ori.csv`: Test features for prediction

### Feature Columns

Boundary conditions (all use forecasted values):
- 系统负荷预测值 (System Load Forecast)
- 风光总加预测值 (Wind+Solar Total Forecast)
- 联络线预测值 (Transmission Line Forecast)
- 风电预测值 (Wind Power Forecast)
- 光伏预测值 (Solar Power Forecast)
- 水电预测值 (Hydropower Forecast)
- 非市场化机组预测值 (Non-market Unit Forecast)

## Ablation Experiments

### Experiment Configuration

| Dimension | Options | Description |
|-----------|---------|-------------|
| **Models** | GB, XGBoost, LightGBM, RF, Ridge, MLP | 7 model types |
| **Features** | Basic (11) / Enhanced (35) | Feature engineering levels |
| **HPO** | None / Optuna (5 trials) | Hyperparameter optimization |
| **Strategy** | Brute / DP | Algorithm complexity |

### Experiment Results Summary

**Best Results**:
- **MLP + Enhanced Features + Brute**: +453.01% profit (3,850,901 total)
- **MLP + Basic Features + Brute**: +20.27% profit (837,523 total)
- **XGBoost + Enhanced + HPO + DP**: +12.7% profit (784,857 total), lowest RMSE (0.5564)

**Key Findings**:
1. Prediction accuracy ≠ Economic profit (highest RMSE gives best profit!)
2. Feature engineering reduces RMSE significantly
3. Simple models (Ridge, MLP) can outperform complex ones
4. Brute force optimization finds better strategies than DP

## Running the Code

### Run All Experiments
```bash
python run.py  # Complete ablation (13 experiments)
python run_mlp_enhanced_v2.py  # Best experiment (MLP + Enhanced)
```

### Run Baseline
```bash
python sklearn_baseline.py
```

### Expected Output
Results saved to:
- `output/output.csv` - Best strategy (MLP + Enhanced + Brute)
- `output/experiments/` - Other experiment results

## Important Notes

- All data uses 15-minute intervals (96 time points per day)
- Time-based split: 80% training, 20% validation (chronologically ordered)
- The strategy optimization is performed per-day independently
- Model evaluation uses RMSE and MAE metrics
- Required dependencies: pandas, numpy, scikit-learn

## Project Structure

```
├── experiments/
│   ├── __init__.py
│   ├── base.py              # DataLoader, ExperimentConfig
│   ├── features.py          # Feature engineering (basic + enhanced)
│   ├── models.py            # ML models + HPO support
│   ├── strategies.py        # Strategy generation algorithms
│   ├── evaluator.py         # Evaluation metrics
│   └── run_experiments.py  # Experiment runner
├── output/                   # Output directory
│   └── output.csv            # Best strategy result
├── data/                     # Data files
├── sklearn_baseline.py       # Original baseline
├── run.py                   # Main experiment entry
├── run_mlp_enhanced_v2.py  # Best experiment
├── run_best_exp.py
├── requirements.txt
└── README.md
```
