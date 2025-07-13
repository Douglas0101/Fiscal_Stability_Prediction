# src/api.py
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# CORRIGIDO: Importa a nova função de inicialização
from .database import init_db
from .logger_config import get_logger

logger = get_logger(__name__)

# --- Configurações da Aplicação ---
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
MODEL_NAME = "fiscal-stability-xgb"
ml_model = None


def load_model_with_retry(max_retries=10, delay_seconds=10):
    """Tenta carregar a versão mais recente do modelo do MLflow, com várias tentativas."""
    global ml_model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentativa {attempt}/{max_retries}: A carregar modelo '{MODEL_NAME}' de '{model_uri}'...")
            ml_model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Modelo '{MODEL_NAME}' carregado com sucesso!")
            return
        except Exception as e:
            logger.warning(f"Tentativa {attempt} falhou ao carregar modelo: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)

    logger.error("Não foi possível carregar o modelo do MLflow após várias tentativas.")
    raise RuntimeError("Falha crítica: Modelo de ML não pôde ser carregado.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Funções a serem executadas durante o ciclo de vida da API."""
    logger.info("API iniciando... Inicializando a base de dados e carregando o modelo de ML.")

    # CORRIGIDO: Chama a nova função de inicialização robusta
    await init_db()

    load_model_with_retry()
    yield
    logger.info("API encerrando.")


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: int


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "API de Previsão de Estabilidade Fiscal está online."}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):
    """Recebe um conjunto de features e retorna a previsão de estabilidade fiscal."""
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de Machine Learning não está disponível no momento."
        )

    try:
        input_df = pd.DataFrame([request.features])
        prediction = ml_model.predict(input_df)
        result = int(prediction[0])
        return PredictionResponse(prediction=result)
    except Exception as e:
        logger.error(f"Erro durante a previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
