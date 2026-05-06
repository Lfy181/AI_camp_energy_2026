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
    
    def evaluate_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def evaluate_strategy(self, df_result: pd.DataFrame, total_profit: float) -> Dict[str, float]:
        df_result['times'] = pd.to_datetime(df_result['times'])
        df_result['date'] = df_result['times'].dt.date
        n_days = len(df_result['date'].unique())
        
        avg_daily_profit = total_profit / n_days if n_days > 0 else 0
        
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
        
        if len(df_display) > 1:
            print(f"  Profit Improvement: {df_display['profit_improvement'].iloc[best_idx]:.2f}%")
            print(f"  RMSE Reduction: {df_display['rmse_improvement'].iloc[best_idx]:.2f}%")
        
        print("="*100)
