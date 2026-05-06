#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import DataLoader, ExperimentConfig
from experiments.features import FeatureEngineer
from experiments.models import ModelFactory
from experiments.strategies import StrategyGenerator
from experiments.evaluator import Evaluator

REMAINING_EXPERIMENTS = [
    ExperimentConfig(
        name='Exp6_XGBoost_EnhancedFeatures',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp7_LightGBM_EnhancedFeatures',
        model_type='lightgbm',
        feature_level='enhanced',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp8_RandomForest_EnhancedFeatures',
        model_type='randomforest',
        feature_level='enhanced',
        use_hpo=False,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp9_XGBoost_Enhanced_HPO',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp10_LightGBM_Enhanced_HPO',
        model_type='lightgbm',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='brute'
    ),
    ExperimentConfig(
        name='Exp11_Full_XGBoost_HPO_DP',
        model_type='xgboost',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='dp'
    ),
    ExperimentConfig(
        name='Exp12_Full_LightGBM_HPO_DP',
        model_type='lightgbm',
        feature_level='enhanced',
        use_hpo=True,
        strategy_type='dp'
    )
]

def run_single_experiment(config, data_loader, evaluator):
    import time
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
    import pandas as pd
    y = df_train_processed['A'].values
    
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

def main():
    print("="*80)
    print("RUNNING REMAINING EXPERIMENTS".center(80))
    print("="*80)
    
    data_loader = DataLoader()
    
    if not data_loader.check_data_exists():
        print("ERROR: Data files not found!")
        return
    
    evaluator = Evaluator()
    
    for config in REMAINING_EXPERIMENTS:
        try:
            run_single_experiment(config, data_loader, evaluator)
        except Exception as e:
            print(f"ERROR in {config.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    evaluator.print_summary()
    evaluator.save_results()
    
    print("\nAll remaining experiments completed!")

if __name__ == '__main__':
    main()
