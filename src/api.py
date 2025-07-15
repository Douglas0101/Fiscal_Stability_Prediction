# src/api.py
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .logger_config import get_logger
from .config import AppConfig
from pydantic import BaseModel

logger = get_logger(__name__)
config = AppConfig()
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
ml_models: Dict[str, Any] = {}

def load_model(model_name: str, max_retries=5, delay_seconds=5):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    registered_model_name = f"fiscal-stability-{model_name}"
    model_uri = f"models:/{registered_model_name}/latest"
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentativa {attempt}/{max_retries}: A carregar modelo '{registered_model_name}'...")
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            ml_models[model_name] = loaded_model
            logger.info(f"Modelo '{registered_model_name}' carregado com sucesso como '{model_name}'.")
            return
        except Exception as e:
            logger.warning(f"Tentativa {attempt} falhou ao carregar o modelo '{registered_model_name}': {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
    logger.error(f"Não foi possível carregar o modelo '{registered_model_name}' após várias tentativas.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API iniciando...")
    await init_db()
    logger.info(f"A carregar os seguintes modelos: {config.models_to_run}")
    for model_name in config.models_to_run:
        load_model(model_name)
    yield
    logger.info("API encerrando.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_used: str
    prediction: int

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"API de Previsão de Estabilidade Fiscal está online. Modelos carregados: {list(ml_models.keys())}"}

@app.post("/predict/{model_name}", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest, model_name: str = Path(..., description=f"Modelos disponíveis: {config.models_to_run}")):
    if model_name not in ml_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' não encontrado. Modelos disponíveis: {list(ml_models.keys())}")
    model = ml_models[model_name]
    try:
        input_df = pd.DataFrame([request.features])
        prediction = model.predict(input_df)
        result = int(prediction[0])
        return PredictionResponse(model_used=model_name, prediction=result)
    except Exception as e:
        logger.error(f"Erro durante a previsão com o modelo '{model_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")
