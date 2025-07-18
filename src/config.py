# src/config.py (Estrutura sugerida)
from pydantic_settings import BaseSettings
from typing import List, Dict, Any

class ModelConfig(BaseSettings):
    TARGET_VARIABLE: str = 'fiscal_stability_index'
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    MODELS_TO_RUN: List[str] = ["lgbm", "xgb", "ebm"]
    # ... outras configs

class MLFlowConfig(BaseSettings):
    TRACKING_URI: str = "http://127.0.0.1:5001"  # Alterado de 5000 para 5001
    EXPERIMENT_NAME: str = "FiscalStabilityPrediction"

class TrainingConfig(BaseSettings):
    # Parâmetros de tuning
    TUNING_N_ITER: int = 20
    TUNING_CV: int = 5

    # Features
    NUMERIC_FEATURES: List[str] = [
        'Inflation (CPI %)', 'GDP Growth (% Annual)', 'Current Account Balance (% GDP)',
        'Government Revenue (% of GDP)', 'Public Debt (% of GDP)', 'Interest Rate (Real, %)'
    ]

    # Hiperparâmetros
    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
        "lgbm": {
            'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40, 50], 'max_depth': [-1, 10, 20],
            'reg_alpha': [0.1, 0.5, 1.0], 'reg_lambda': [0.1, 0.5, 1.0],
        },
        "xgb": {
            'n_estimators': [100, 200, 300, 400], 'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9], 'gamma': [0, 0.1, 0.5],
            'subsample': [0.7, 0.8, 1.0], 'colsample_bytree': [0.7, 0.8, 1.0],
        },
        "ebm": {
            'outer_bags': [8, 12, 16], 'inner_bags': [4, 8],
            'learning_rate': [0.01, 0.05, 0.1],
        }
    }

class AppConfig(BaseSettings):
    model: ModelConfig = ModelConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    training: TrainingConfig = TrainingConfig()
    raw_data_path: str = "data/01_raw/world_bank_data_2025.csv"
    raw_data_with_target_path: str = "data/01_raw/world_bank_data_with_target.csv"
    processed_data_path: str = "data/02_processed/processed_data.csv"
    final_data_path: str = "data/03_final/final_data.csv"
    output_model_path: str = "notebooks/models/fiscal_stability_model.joblib"