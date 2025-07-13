import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any


class ModelConfig(BaseModel):
    """Configurações relacionadas ao modelo e às features."""
    TARGET_VARIABLE: str = 'fiscal_stability_index'
    ENTITY_COLUMN: str = 'country_id'
    YEAR_COLUMN: str = 'year'
    DROP_COLUMNS: List[str] = ['country_name']
    LEAKY_FEATURES: List[str] = ['Public Debt (% of GDP)', 'Government Revenue (% of GDP)']
    CATEGORICAL_FEATURES: List[str] = ['country_id']
    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        'rf': {"random_state": 42, "n_jobs": -1, "n_estimators": 100, "class_weight": "balanced"},
        'lgbm': {"random_state": 42, "n_jobs": -1, "n_estimators": 100, "scale_pos_weight": 8.5},
        'xgb': {"random_state": 42, "n_jobs": -1, "n_estimators": 100, "eval_metric": "logloss",
                "scale_pos_weight": 8.5},
        'pytorch': {"hidden_size_1": 256, "hidden_size_2": 128, "learning_rate": 0.001, "epochs": 50, "batch_size": 64,
                    "weight_decay": 0.01, "dropout_rate": 0.4}
    }


class AppConfig(BaseSettings):
    """Configurações centralizadas do projeto."""
    DEBUG: bool = False

    # Define o diretório raiz do projeto de forma robusta
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Caminhos para os scripts executados via docker-compose
    data_processing_script_path: str = "src/data_processing.py"
    train_script_path: str = "src/train.py"

    # Caminhos para os dados
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'data/01_raw/world_bank_data_2025.csv')
    raw_data_with_target_path: str = os.path.join(PROJECT_ROOT, 'data/01_raw/world_bank_data_with_target.csv')
    processed_data_path: str = os.path.join(PROJECT_ROOT, 'data/02_processed/processed_data.csv')

    # Caminhos para modelos
    model_dir: str = os.path.join(PROJECT_ROOT, 'src/models')

    # Configurações do modelo aninhadas (nome corrigido)
    model: ModelConfig = Field(default_factory=ModelConfig)
    default_model: str = 'xgb'

    # Configurações da API e MLflow
    api_port: int = 8000
    mlflow_port: int = 5000

    class Config:
        # Remove o aviso do Pydantic sobre namespaces protegidos
        protected_namespaces = ()
        case_sensitive = True

