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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, use_hpo: bool = False, params: Dict[str, Any] = None) -> BaseModel:
        if model_type == 'gb':
            return GBModel(params)
        elif model_type == 'xgboost':
            return XGBoostModel(params, use_hpo=use_hpo)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
