# ==============================================================================
# CONFIGURAÇÃO CENTRALIZADA DO PROJETO
# ------------------------------------------------------------------------------
# Este ficheiro utiliza Pydantic para carregar, validar e centralizar todas
# as configurações da aplicação, desde caminhos de ficheiros até
# hiperparâmetros de modelos.
# ==============================================================================

import os
from functools import lru_cache
from typing import Dict, Any

from pydantic_settings import BaseSettings

# --- VERIFICAÇÃO DE SANIDADE ---
if "SECRET_KEY" not in os.environ:
    raise EnvironmentError(
        "ERRO CRÍTICO: Variáveis de ambiente não foram carregadas.\n"
        "Certifique-se de que o ficheiro '.env' existe na raiz do projeto e que\n"
        "o ponto de entrada da aplicação (main.py) está a usar `load_dotenv()`."
    )

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ApiSettings(BaseSettings):
    """Configurações relacionadas com a API e segurança."""
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


class PathSettings(BaseSettings):
    """Configurações de caminhos para dados, modelos e relatórios."""
    RAW_DATA_PATH: str = os.path.join(BASE_DIR, 'data', '01_raw')
    PROCESSED_DATA_PATH: str = os.path.join(BASE_DIR, 'data', '02_processed')
    MODELS_PATH: str = os.path.join(BASE_DIR, 'models')
    REPORTS_PATH: str = os.path.join(BASE_DIR, 'reports')


class ModelParams(BaseSettings):
    """Parâmetros para o treinamento e avaliação do modelo."""
    SPLIT_YEAR: int = 2018
    # CORREÇÃO: Alterado de "Year" para "year" para corresponder aos dados reais.
    YEAR_COLUMN: str = "year"
    ENTITY_COLUMN: str = "Country Name"
    TARGET_COLUMN: str = "fiscal_stability_index"

    LGBM_PARAMS: Dict[str, Any] = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 1000,
        'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
        'random_state': 42, 'n_jobs': -1, 'colsample_bytree': 0.8, 'subsample': 0.8,
    }


class Settings(BaseSettings):
    """Classe principal que agrega todas as configurações da aplicação."""
    APP_NAME: str = "Fiscal Stability Prediction API"
    DEBUG: bool = False
    DATABASE_URL: str
    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT_NAME: str = "fiscal_stability_prediction"

    api: ApiSettings = ApiSettings()
    paths: PathSettings = PathSettings()
    model: ModelParams = ModelParams()


@lru_cache()
def get_settings() -> Settings:
    """Retorna uma instância singleton da classe de configurações."""
    settings_obj = Settings()
    os.makedirs(settings_obj.paths.MODELS_PATH, exist_ok=True)
    os.makedirs(settings_obj.paths.REPORTS_PATH, exist_ok=True)
    return settings_obj


settings = get_settings()
