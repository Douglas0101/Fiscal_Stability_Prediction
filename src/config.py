import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any


# Sub-classe para configurações específicas do modelo
class ModelConfig(BaseModel):
    """Configurações relacionadas ao modelo e às features."""
    TARGET_VARIABLE: str = 'fiscal_stability_index'
    ENTITY_COLUMN: str = 'country_id'
    YEAR_COLUMN: str = 'year'
    DROP_COLUMNS: List[str] = ['country_name']

    LEAKY_FEATURES: List[str] = ['Public Debt (% of GDP)', 'Government Revenue (% of GDP)']

    CATEGORICAL_FEATURES: List[str] = ['country_id']

    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        'rf': {
            "random_state": 42, "n_jobs": -1, "n_estimators": 100, "class_weight": "balanced"
        },
        'lgbm': {
            "random_state": 42, "n_jobs": -1, "n_estimators": 100, "scale_pos_weight": 8.5
        },
        'xgb': {
            "random_state": 42, "n_jobs": -1, "n_estimators": 100, "eval_metric": "logloss", "scale_pos_weight": 8.5
        },
        'pytorch': {
            "hidden_size_1": 256,  # Aumentada a primeira camada
            "hidden_size_2": 128,  # Nova camada
            "learning_rate": 0.001,
            "epochs": 50,  # Aumentado o número de épocas
            "batch_size": 64,
            "weight_decay": 0.01,  # Parâmetro para o otimizador AdamW
            "dropout_rate": 0.4  # Aumentado o dropout para a rede maior
        }
    }


class Settings(BaseSettings):
    """
    Configurações centralizadas do projeto.
    """
    DEBUG: bool = False
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH: str = os.path.join(BASE_DIR, 'data/01_raw/world_bank_data_2025.csv')
    RAW_DATA_WITH_TARGET_PATH: str = os.path.join(BASE_DIR, 'data/01_raw/world_bank_data_with_target.csv')
    PROCESSED_DATA_PATH: str = os.path.join(BASE_DIR, 'data/02_processed/processed_data.csv')
    MODEL_PATH: str = os.path.join(BASE_DIR, 'models/')
    REPORTS_PATH: str = os.path.join(BASE_DIR, 'reports/')
    LOG_FILE_PATH: str = os.path.join(BASE_DIR, 'app.log')
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    model: ModelConfig = ModelConfig()

    class Config:
        case_sensitive = True


settings = Settings()
