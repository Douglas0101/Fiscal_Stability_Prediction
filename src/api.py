# src/api.py (Versão Final, Corrigida e Resiliente)

import os
import joblib
import json
import pandas as pd
import mlflow
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text  # <-- CORREÇÃO: Importado para health check do DB
from pydantic_settings import BaseSettings
from mlflow.exceptions import MlflowException  # <-- CORREÇÃO: Importado para tratamento de erro específico

# Importações locais do projeto
from .database import get_db
from . import models, schemas
from .logger_config import get_logger

logger = get_logger(__name__)


# --- 1. CONFIGURAÇÃO CENTRALIZADA (OO) ---
class APISettings(BaseSettings):
    """ Carrega e valida as configurações da API a partir de variáveis de ambiente. """
    API_KEY: str
    MODEL_NAME: str = "fiscal-stability-xgb"
    MODEL_STAGE: str = "production"  # <-- CORREÇÃO: Padronizado para minúsculas

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = APISettings()


# --- 2. CLASSE DE LÓGICA DE ML (ENCAPSULAMENTO) ---
class MLModelHandler:
    """
    Encapsula todo o estado (modelo, scaler) e comportamento (carregar, prever)
    relacionado ao modelo de Machine Learning.
    """

    def __init__(self, model_name: str, model_stage: str):
        self.model_name = model_name
        self.model_stage = model_stage  # Usado como alias
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        """ Carrega todos os artefatos necessários do MLflow de forma resiliente. """
        logger.info(f"Tentando carregar o modelo '{self.model_name}' com alias '@{self.model_stage}' do MLflow...")
        try:
            # CORREÇÃO 1: O model_uri com @ está CORRETO para aliases.
            model_uri = f"models:/{self.model_name}@{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)

            client = mlflow.tracking.MlflowClient()

            # CORREÇÃO 2: Substituir a função obsoleta pela correta para aliases.
            # Troca de 'get_latest_versions' por 'get_model_version_by_alias'.
            latest_version = client.get_model_version_by_alias(name=self.model_name, alias=self.model_stage)
            run_id = latest_version.run_id

            # O resto do código para baixar artefatos continua igual.
            local_preprocessor_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor")
            self.scaler = joblib.load(os.path.join(local_preprocessor_path, "scaler.joblib"))

            local_features_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="features")
            with open(os.path.join(local_features_path, "feature_names.json"), 'r') as f:
                self.feature_names = json.load(f)

            logger.info(f"SUCESSO: Modelo versão {latest_version.version} e artefatos foram carregados.")

        except MlflowException as e:
            # CORREÇÃO 3: Tratar o erro de "recurso não encontrado" de forma graciosa.
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.warning(
                    f"AVISO: Modelo '{self.model_name}' com alias '@{self.model_stage}' não encontrado. "
                    "A API iniciará em modo degradado. Treine e defina o alias de um modelo para carregar."
                )
                self.model = self.scaler = None
                self.feature_names = []
            else:
                logger.critical(f"FALHA CRÍTICA AO CARREGAR ARTEFATOS DO ML: {e}", exc_info=True)
                raise RuntimeError(f"Startup falhou: Erro inesperado do MLflow. Erro: {e}")
        except Exception as e:
            logger.critical(f"FALHA CRÍTICA AO CARREGAR ARTEFATOS DO ML: {e}", exc_info=True)
            raise RuntimeError(f"Startup falhou: Não foi possível carregar os artefatos de ML. Erro: {e}")

    def predict(self, request_data: schemas.PredictionRequest) -> schemas.PredictionResponse:
        """ Executa a previsão usando os artefatos carregados. """
        if not self.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Serviço indisponível: Modelo não carregado. Treine e promova um modelo para 'production'."
            )

        input_df = pd.DataFrame([request_data.model_dump(by_alias=True)])

        # Reordenar colunas para garantir a correspondência com a ordem do treinamento
        input_df = input_df[self.feature_names]

        # A lógica de scaling precisa ser cuidadosa aqui
        numeric_features = [col for col in self.feature_names if not col.startswith('country_id_')]
        input_df[numeric_features] = self.scaler.transform(input_df[numeric_features])

        prediction = self.model.predict(input_df)[0]
        probability = max(self.model.predict_proba(input_df)[0])
        return schemas.PredictionResponse(prediction=int(prediction), probability=float(probability))

    def is_ready(self) -> bool:
        """ Verifica se todos os componentes do modelo estão prontos. """
        return all([self.model, self.scaler, self.feature_names])


# --- 3. CICLO DE VIDA E SETUP DA API FASTAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """ No startup, cria uma instância única do handler do modelo e a anexa ao estado da app. """
    app.state.model_handler = MLModelHandler(
        model_name=settings.MODEL_NAME,
        model_stage=settings.MODEL_STAGE
    )
    yield
    app.state.model_handler = None
    logger.info("Recursos de ML limpos. Aplicação encerrada.")


app = FastAPI(
    title="API de Previsão de Estabilidade Fiscal (Final)",
    description="Uma API que utiliza um modelo de ML para prever risco fiscal.",
    version="3.0.0",
    lifespan=lifespan
)

# --- Dependências ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida ou ausente.")


# --- 4. ROTAS DA API (CAMADA WEB) ---
@app.get("/api/v1/health", response_model=schemas.HealthStatus, tags=["Monitoramento"])
async def health_check(request: Request, db: AsyncSession = Depends(get_db)):
    handler: MLModelHandler = request.app.state.model_handler
    db_ok = False
    try:
        # --- CORREÇÃO 4: Usar text() para executar uma consulta SQL literal. ---
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check do DB falhou: {e}")

    return schemas.HealthStatus(
        model_loaded=handler.is_ready(),
        database_status="ok" if db_ok else "error"
    )


@app.post("/api/v1/predict", response_model=schemas.PredictionResponse, dependencies=[Depends(get_api_key)],
          tags=["Previsão"])
async def predict(
        request_data: schemas.PredictionRequest,
        request: Request
        # db: AsyncSession = Depends(get_db) # Removido se o DB não for usado aqui
):
    handler: MLModelHandler = request.app.state.model_handler
    try:
        prediction_response = handler.predict(request_data)
        return prediction_response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Erro no endpoint de previsão: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a previsão.")