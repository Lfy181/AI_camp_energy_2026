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
        
    def check_data_exists(self) -> bool:
        return (os.path.exists(self.train_feature_path) and 
                os.path.exists(self.train_label_path) and 
                os.path.exists(self.test_feature_path))
    
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_feat = pd.read_csv(self.train_feature_path)
        df_label = pd.read_csv(self.train_label_path)
        df_train = pd.merge(df_feat, df_label, on='times', how='inner')
        df_train['times'] = pd.to_datetime(df_train['times'])
        return df_train
    
    def load_test_data(self) -> pd.DataFrame:
        df_test = pd.read_csv(self.test_feature_path)
        df_test['times'] = pd.to_datetime(df_test['times'])
        return df_test


# 基础特征列
BASIC_FEATURE_COLS = [
    '系统负荷预测值', '风光总加预测值', '联络线预测值',
    '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值'
]
TARGET_COL = 'A'
