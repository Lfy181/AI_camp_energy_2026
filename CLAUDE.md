# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an energy forecasting and optimization project that predicts node electricity prices and generates optimal battery charge/discharge strategies. The project uses time-series prediction of energy system boundary conditions to forecast real-time electricity prices, then determines optimal charge/discharge timing to maximize profit.

## Key Architecture

### Data Flow
1. **Input**: Boundary condition features (load, wind/solar generation, transmission lines, etc.)
2. **Model**: Scikit-learn GradientBoostingRegressor for price prediction
3. **Output**: Price forecasts + optimal charge/discharge strategy

### Core Components

- **Feature Engineering** (`add_time_features`): Extracts temporal features (hour, minute, day of week, month) from timestamps
- **Price Prediction**: Uses gradient boosting to predict node price 'A' from boundary condition forecasts
- **Strategy Generation** (`generate_strategy`): Implements brute-force optimization to find optimal charge/discharge timing
  - Charge: 8 time points (2 hours) at -1000MW
  - Discharge: 8 time points (2 hours) at +1000MW
  - Constraints: 0 ≤ tc ≤ 80, td ≥ tc + 8, td ≤ 88
  - Objective: Maximize profit = discharge_revenue - charge_cost

### Data Structure

Training data:
- `data/train/mengxi_boundary_anon_filtered.csv`: Boundary condition features
- `data/train/mengxi_node_price_selected.csv`: Node price labels

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

## Running the Code

Run the baseline model:
```bash
python sklearn_baseline.py
```

This will:
1. Load training data and merge by timestamp
2. Train a GradientBoostingRegressor (200 estimators, max_depth=6)
3. Predict prices on test set
4. Generate optimal charge/discharge strategy
5. Save results to `output/output.csv`

## Important Notes

- All data uses 15-minute intervals (96 time points per day)
- Time-based split: 80% training, 20% validation (chronologically ordered)
- The strategy optimization is performed per-day independently
- Model evaluation uses RMSE and MAE metrics
- Required dependencies: pandas, numpy, scikit-learn
