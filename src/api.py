# src/api.py (Versão Final, Resiliente e com Logging Aprimorado)

import os
import joblib
import json
import pandas as pd
import mlflow
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic_settings import BaseSettings
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .database import get_db
from . import schemas
from .logger_config import get_logger

logger = get_logger(__name__)


class APISettings(BaseSettings):
    API_KEY: str
    MODEL_NAME: str = "fiscal-stability-xgb"
    MODEL_STAGE: str = "production"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = APISettings()


class MLModelHandler:
    def __init__(self, model_name: str, model_stage: str):
        self.model_name = model_name
        self.model_stage = model_stage
        self.model = self.scaler = self.feature_names = self.model_version = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Tenta carregar os artefactos do MLflow de forma segura e com logging detalhado."""
        try:
            logger.info(f"Conectando ao MLflow em: {settings.MLFLOW_TRACKING_URI}")
            client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

            logger.info(f"Buscando o modelo '{self.model_name}' no stage '@{self.model_stage}'...")
            latest_version = client.get_latest_versions(name=self.model_name, stages=[self.model_stage])[0]
            self.model_version = latest_version.version
            run_id = latest_version.run_id
            logger.info(
                f"Versão {self.model_version} encontrada (Run ID: {run_id}). Fazendo download dos artefactos...")

            # Download do Modelo
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("SUCESSO: Modelo carregado.")

            # Download do Scaler e Features
            local_artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

            scaler_path = os.path.join(local_artifacts_path, "preprocessor/scaler.joblib")
            self.scaler = joblib.load(scaler_path)
            logger.info("SUCESSO: Scaler carregado.")

            features_path = os.path.join(local_artifacts_path, "features/feature_names.json")
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info("SUCESSO: Nomes das features carregados.")

        except MlflowException as e:
            # Erro específico se o modelo ou versão não existe
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.warning(
                    f"AVISO: O modelo '{self.model_name}' com stage '@{self.model_stage}' não foi encontrado no MLflow. "
                    "A API iniciará em modo degradado. Treine e promova um modelo para 'Production'."
                )
            else:
                logger.error(
                    f"FALHA NO MLFLOW: Não foi possível carregar os artefactos. Verifique a conexão e o estado do servidor MLflow. Erro: {e}",
                    exc_info=True)
            self.model = self.scaler = self.feature_names = self.model_version = None
        except Exception as e:
            # Captura outras exceções (ex: falha de rede, ficheiro de artefacto em falta)
            logger.critical(f"FALHA CRÍTICA AO CARREGAR ARTEFACTOS DO ML: {e}", exc_info=True)
            self.model = self.scaler = self.feature_names = self.model_version = None

    def predict(self, request_data: schemas.PredictionRequest) -> schemas.PredictionResponse:
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Serviço indisponível: Modelo de ML não carregado.")

        # Garante a ordem correta das colunas e o pré-processamento
        input_df = pd.DataFrame([request_data.model_dump(by_alias=True)])
        input_df_ordered = input_df[self.feature_names]

        input_df_ordered = self.scaler.transform(input_df_ordered)

        prediction = self.model.predict(input_df_ordered)[0]
        probability = max(self.model.predict_proba(input_df_ordered)[0])

        return schemas.PredictionResponse(prediction=int(prediction), probability=float(probability))

    def is_ready(self) -> bool:
        """Verifica se todos os componentes necessários do modelo estão carregados."""
        return all([self.model, self.scaler, self.feature_names])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_handler = MLModelHandler(
        model_name=settings.MODEL_NAME, model_stage=settings.MODEL_STAGE
    )
    yield
    app.state.model_handler = None
    logger.info("Recursos de ML limpos.")


app = FastAPI(title="API de Previsão de Estabilidade Fiscal", version="4.0.0", lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida.")


@app.get("/api/v1/health", response_model=schemas.HealthStatus, tags=["Monitoramento"])
async def health_check(request: Request, db: AsyncSession = Depends(get_db)):
    handler: MLModelHandler = request.app.state.model_handler
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check do DB falhou: {e}")

    return schemas.HealthStatus(
        model_status="loaded" if handler.is_ready() else "degraded",
        model_name=handler.model_name,
        model_version=handler.model_version,
        database_status="ok" if db_ok else "error"
    )


@app.post(
    "/api/v1/predict",
    response_model=schemas.PredictionResponse,
    dependencies=[Depends(get_api_key)],
    tags=["Previsão"]
)
async def predict(request_data: schemas.PredictionRequest, request: Request):
    handler: MLModelHandler = request.app.state.model_handler
    try:
        return handler.predict(request_data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Erro no endpoint de previsão: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno do servidor ao processar a previsão.")