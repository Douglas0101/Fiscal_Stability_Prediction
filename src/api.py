# src/api.py (Orquestrador da Aplicação)

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from contextlib import asynccontextmanager
import os
import logging
import pandas as pd
import mlflow
import asyncio

# Importações relativas para os módulos da nossa aplicação.
from .database import get_db, create_db_and_tables
from . import schemas, crud

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"
api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# CORREÇÃO: Usar uma constante para o nome do modelo para garantir consistência.
# Este nome TEM de corresponder ao nome usado no registo em train.py (ex: "fiscal-stability-xgb")
REGISTERED_MODEL_NAME = "fiscal-stability-xgb"

async def get_api_key(api_key: str = Security(api_key_scheme)):
    """Valida a chave de API presente no cabeçalho da requisição."""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key


# --- Gestor de Ciclo de Vida (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gere o arranque e o encerramento da aplicação."""
    logger.info("API iniciando... Carregando modelo de ML.")
    # CORREÇÃO: Usar a constante para o nome do modelo.
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    max_retries = 10
    retry_delay = 5

    # Define o tracking URI para o servidor MLflow dentro do Docker.
    mlflow.set_tracking_uri("http://mlflow-server:5000")

    for attempt in range(max_retries):
        try:
            app.state.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Modelo '{model_uri}' carregado com sucesso.")
            break
        except Exception as e:
            logger.warning(f"Tentativa {attempt + 1} falhou ao carregar modelo: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Não foi possível carregar o modelo do MLflow após várias tentativas.")
                app.state.model = None

    await create_db_and_tables()
    logger.info("Tabelas prontas. API operacional.")
    yield
    logger.info("API encerrando.")


app = FastAPI(
    title="API de Previsão de Estabilidade Fiscal",
    description="Uma API para prever a estabilidade fiscal de clientes.",
    version="2.0.2", # Versão com correção do nome do modelo
    lifespan=lifespan
)


# --- Lógica de Negócio ---
def run_model_prediction(model: any, data: schemas.FinancialDataCreate) -> schemas.PredictionResult:
    """Executa a predição e retorna o resultado estruturado."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de predição não disponível.")

    input_df = pd.DataFrame([data.model_dump()])
    prediction_value = model.predict_proba(input_df)[0][1]

    risk = "high"
    if prediction_value > 0.7:
        risk = "low"
    elif prediction_value > 0.4:
        risk = "medium"

    return schemas.PredictionResult(
        client_id=data.client_id,
        prediction=prediction_value,
        risk_level=risk
    )


# --- Endpoints da API ---
@app.post(
    "/predict/",
    response_model=schemas.PredictionResult,
    dependencies=[Depends(get_api_key)],
    summary="Realiza uma nova predição de estabilidade fiscal"
)
async def create_prediction(data: schemas.FinancialDataCreate, db: AsyncSession = Depends(get_db)):
    """
    Recebe dados financeiros, realiza uma predição e guarda o resultado.
    """
    try:
        prediction_result = run_model_prediction(app.state.model, data)
        await crud.create_prediction_record(db, data, prediction_result)
        return prediction_result
    except Exception as e:
        logger.error(f"Erro no endpoint de predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get(
    "/predictions/",
    response_model=List[schemas.FinancialDataResponse],
    dependencies=[Depends(get_api_key)],
    summary="Lista o histórico de predições"
)
async def read_predictions(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """
    Retorna uma lista paginada de todas as predições guardadas na base de dados.
    """
    return await crud.get_all_predictions(db, skip=skip, limit=limit)

@app.get("/health", summary="Verifica a saúde da API")
def health_check():
    """Endpoint simples para verificar se a API está a responder."""
    return {"status": "ok", "model_loaded": app.state.model is not None}
