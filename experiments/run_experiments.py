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
                         evaluator: Evaluator) -> None:
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
