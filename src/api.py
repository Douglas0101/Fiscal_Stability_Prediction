# src/api.py

import os
import uuid
import time
import joblib
import json
import pandas as pd
import mlflow
from fastapi import FastAPI, Depends, HTTPException, Security, BackgroundTasks, Request
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_settings import BaseSettings

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
    MODEL_STAGE: str = "Production"

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
        self.model_stage = model_stage
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        """ Carrega todos os artefatos necessários do MLflow. """
        logger.info(f"Carregando modelo '{self.model_name}' (stage: {self.model_stage}) do MLflow...")
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)

            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(name=self.model_name, stages=[self.model_stage])[0]
            run_id = latest_version.run_id

            local_preprocessor_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor")
            self.scaler = joblib.load(os.path.join(local_preprocessor_path, "scaler.joblib"))

            local_features_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="features")
            with open(os.path.join(local_features_path, "feature_names.json"), 'r') as f:
                self.feature_names = json.load(f)

            logger.info("Modelo e todos os artefatos carregados com sucesso.")
        except Exception as e:
            logger.critical(f"FALHA CRÍTICA AO CARREGAR ARTEFATOS DO ML: {e}", exc_info=True)
            raise RuntimeError(f"Startup falhou: Não foi possível carregar os artefatos de ML. Erro: {e}")

    def predict(self, request_data: schemas.PredictionRequest) -> schemas.PredictionResponse:
        """ Executa a previsão usando os artefatos carregados. """
        if not all([self.model, self.scaler, self.feature_names]):
            raise RuntimeError("Modelo ou artefatos não estão carregados.")

        input_df = pd.DataFrame([request_data.model_dump()], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_df)

        prediction = self.model.predict(input_scaled)[0]
        probability = max(self.model.predict_proba(input_scaled)[0])

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
    # No shutdown, limpa os recursos se necessário (aqui não é preciso)
    app.state.model_handler = None
    logger.info("Recursos de ML limpos. Aplicação encerrada.")


app = FastAPI(
    title="API de Previsão de Estabilidade Fiscal (OO Refactored)",
    description="Uma API que utiliza um modelo de ML para prever risco fiscal.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Dependências ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida ou ausente.")


# --- 4. ROTAS DA API (CAMADA WEB) ---
# A lógica das rotas agora é mínima. Elas apenas orquestram chamadas
# para o handler do modelo e para o banco de dados.

@app.get("/api/v1/health", response_model=schemas.HealthStatus, tags=["Monitoramento"])
async def health_check(request: Request, db: AsyncSession = Depends(get_db)):
    handler: MLModelHandler = request.app.state.model_handler
    db_ok = False
    try:
        await db.execute("SELECT 1")
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
        request: Request,  # Para acessar o estado da app
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    handler: MLModelHandler = request.app.state.model_handler
    try:
        prediction_response = handler.predict(request_data)
        # Tarefa em background para salvar no DB
        # background_tasks.add_task(save_prediction_to_db, request_data, prediction_response, db)
        return prediction_response
    except Exception as e:
        logger.error(f"Erro no endpoint de previsão: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a previsão.")

# Middleware e função de background permanecem os mesmos...