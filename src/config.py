# src/config.py
import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any

class ModelConfig(BaseModel):
    TARGET_VARIABLE: str = 'fiscal_stability_index'
    ENTITY_COLUMN: str = 'country_id'
    YEAR_COLUMN: str = 'year'
    DROP_COLUMNS: List[str] = ['country_name']
    CATEGORICAL_FEATURES: List[str] = ['country_id']
    MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
        'xgb': {"random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"},
        'ebm': {"random_state": 42},
    }

class AppConfig(BaseSettings):
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_processing_script_path: str = "src/data_processing.py"
    train_script_path: str = "src/train.py"
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'data/01_raw/world_bank_data_2025.csv')
    raw_data_with_target_path: str = os.path.join(PROJECT_ROOT, 'data/01_raw/world_bank_data_with_target.csv')
    processed_data_path: str = os.path.join(PROJECT_ROOT, 'data/02_processed/processed_data.csv')
    model_dir: str = os.path.join(PROJECT_ROOT, 'src/models')
    models_to_run: List[str] = ['xgb', 'ebm']
    model: ModelConfig = Field(default_factory=ModelConfig)
    api_port: int = 8000
    mlflow_port: int = 5000
    class Config:
        protected_namespaces = ()
