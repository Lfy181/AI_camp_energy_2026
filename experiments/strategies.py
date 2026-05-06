import pandas as pd
import numpy as np
from typing import Tuple


class BaseStrategy:
    def __init__(self):
        pass
    
    def generate(self, df_price: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BruteForceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
    
    def generate(self, df_price: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
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
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_tc = tc
                        best_td = td
            
            power = np.zeros(96)
            if best_tc >= 0 and best_td >= 0:
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
    
    def generate(self, df_price: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
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
                if cost < min_charge_cost:
                    min_charge_cost = cost
                    best_tc = tc
            
            max_discharge_revenue = 0
            best_td = -1
            
            for td in range(best_tc + charge_window, 89):
                revenue = np.sum(prices[td:td+discharge_window])
                if revenue > max_discharge_revenue:
                    max_discharge_revenue = revenue
                    best_td = td
            
            best_profit = (max_discharge_revenue - min_charge_cost) * 1000
            
            if best_profit <= 0:
                best_tc = -1
                best_td = -1
                best_profit = 0
            
            power = np.zeros(96)
            if best_tc >= 0 and best_td >= 0:
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
    def create_strategy(strategy_type: str) -> BaseStrategy:
        if strategy_type == 'brute':
            return BruteForceStrategy()
        elif strategy_type == 'dp':
            return DynamicProgrammingStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
