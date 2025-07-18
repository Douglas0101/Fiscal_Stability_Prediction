# src/config.py
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import randint, uniform

class ModelConfig(BaseSettings):
    """Configurações gerais do projeto e dados."""
    TARGET_VARIABLE: str = 'fiscal_stability_index'
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

class MLFlowConfig(BaseSettings):
    """Configurações para o MLflow."""
    TRACKING_URI: str = "http://127.0.0.1:5001"
    EXPERIMENT_NAME: str = "FiscalStabilityPrediction"

class TrainingConfig(BaseSettings):
    """Configurações para o pipeline de treino, incluindo a definição completa dos modelos."""
    TUNING_N_ITER: int = 20
    TUNING_CV: int = 5

    NUMERIC_FEATURES: List[str] = [
        'Inflation (CPI %)', 'GDP Growth (% Annual)', 'Current Account Balance (% GDP)',
        'Government Revenue (% of GDP)', 'Public Debt (% of GDP)', 'Interest Rate (Real, %)'
    ]

    # ARQUITETURA FINAL: Um dicionário único que define tudo sobre cada modelo.
    # Isto elimina a necessidade de if/elif no código de treino.
    MODELS: Dict[str, Dict[str, Any]] = {
        "lgbm": {
            "class": LGBMClassifier,
            "static_params": {"random_state": 42, "n_jobs": -1},
            # MELHORIA: Usar distribuições para RandomizedSearchCV é mais eficiente
            # do que usar listas fixas.
            "hyperparameters": {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.1),
                'num_leaves': randint(20, 50),
                'max_depth': randint(10, 30),
                'reg_alpha': uniform(0.1, 1.0),
                'reg_lambda': uniform(0.1, 1.0),
            }
        },
        "xgb": {
            "class": XGBClassifier,
            "static_params": {
                "random_state": 42, "n_jobs": -1, "use_label_encoder": False, "eval_metric": "logloss"
            },
            "hyperparameters": {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.1),
                'max_depth': randint(3, 10),
                'gamma': uniform(0, 0.5),
                'subsample': uniform(0.7, 0.3),  # (loc, scale) -> de 0.7 a 1.0
                'colsample_bytree': uniform(0.7, 0.3),
            }
        },
        "ebm": {
            "class": ExplainableBoostingClassifier,
            "static_params": {"random_state": 42, "n_jobs": -1},
            "hyperparameters": {
                'outer_bags': randint(8, 16),
                'inner_bags': randint(0, 8),
                'learning_rate': uniform(0.01, 0.1),
            }
        }
    }

class AppConfig(BaseSettings):
    """Configuração raiz que agrega todas as outras."""
    model: ModelConfig = ModelConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    training: TrainingConfig = TrainingConfig()

    # Caminhos dos dados
    raw_data_path: str = "data/01_raw/world_bank_data_2025.csv"
    raw_data_with_target_path: str = "data/01_raw/world_bank_data_with_target.csv"
    processed_data_path: str = "data/02_processed/processed_data.csv"
    final_data_path: str = "data/03_final/final_data.csv"
    output_model_path: str = "notebooks/models/fiscal_stability_model.joblib"