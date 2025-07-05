import os
import pandas as pd
import shap
import logging
import mlflow
import re
import joblib
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

# Importações relativas que funcionarão dentro do Docker
from . import models, schemas, auth
from .database import engine, get_db

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_MODEL_NAME = os.getenv("API_MODEL_NAME", "fiscal-stability-xgb")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


# --- Classe para Gerenciar Artefatos de ML ---
class MLModelHandler:
    """Encapsula o modelo de ML, o scaler e outros artefatos."""

    def __init__(self, model: Any, scaler: Any, explainer: Any, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.explainer = explainer
        self.feature_names = feature_names


# --- Ciclo de Vida da Aplicação (Lifespan) ---
def create_db_and_tables():
    models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Iniciando ciclo de vida da aplicação...")
    create_db_and_tables()
    logging.info("Tabelas do banco de dados criadas/verificadas.")

    logging.info(f"Conectando ao MLflow ({MLFLOW_TRACKING_URI}) para carregar o modelo de produção...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_handler = None
    try:
        model_uri = f"models:/{API_MODEL_NAME}/Production"
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_latest_versions(name=API_MODEL_NAME, stages=["Production"])[0]
        run_id = model_version_details.run_id

        logging.info(f"Carregando modelo da URI: {model_uri} (Run ID: {run_id})")

        # Carrega o modelo sklearn
        model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        unwrapped_model = joblib.load(os.path.join(model_path, "model.pkl"))

        # Carrega o scaler associado
        scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessor/scaler.joblib")
        scaler = joblib.load(scaler_path)

        logging.info("Modelo e scaler carregados com sucesso.")

        # Cria o explainer e prepara os nomes das features
        explainer = shap.TreeExplainer(unwrapped_model)
        feature_names_raw = list(schemas.PredictionFeatures.model_fields.keys())
        feature_names_sanitized = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in feature_names_raw]

        model_handler = MLModelHandler(
            model=unwrapped_model,
            scaler=scaler,
            explainer=explainer,
            feature_names=feature_names_sanitized
        )
        logging.info("Handler de ML pronto para uso.")

    except Exception as e:
        logging.error(
            f"Erro crítico ao carregar modelo ou artefatos do MLflow: {e}. A API iniciará sem um modelo funcional.",
            exc_info=True)

    app.state.ml_handler = model_handler
    yield
    app.state.ml_handler = None
    logging.info("Recursos de ML liberados.")


# --- Criação da Aplicação FastAPI ---
app = FastAPI(
    title="API de Previsão de Estabilidade Fiscal",
    description="Uma API para prever a estabilidade fiscal usando um modelo de ML carregado via MLflow, com persistência e explicabilidade.",
    version="3.0.0",
    lifespan=lifespan
)


# --- Dependências ---
def get_ml_handler(request: Request) -> MLModelHandler:
    if not hasattr(request.app.state, 'ml_handler') or request.app.state.ml_handler is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Modelo de ML não foi carregado.")
    return request.app.state.ml_handler


# --- Rotas da API ---
# Incluindo o roteador de autenticação
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])


@app.post("/predict", summary="Realiza e armazena uma nova predição", response_model=schemas.PredictionResponse)
async def predict(
        features: schemas.PredictionFeatures,
        ml_handler: MLModelHandler = Depends(get_ml_handler),
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    try:
        input_df = pd.DataFrame([features.model_dump()])
        input_scaled = ml_handler.scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=ml_handler.feature_names)

        prediction = int(ml_handler.model.predict(input_scaled_df)[0])
        prediction_proba = ml_handler.model.predict_proba(input_scaled_df)[0]

        shap_values = ml_handler.explainer.shap_values(input_scaled_df)[1]

        db_prediction = crud.create_prediction(
            db=db,
            owner_id=current_user.id,
            features=features.model_dump(),
            prediction=prediction,
            prediction_probability=prediction_proba[prediction]
        )

        return schemas.PredictionResponse(
            prediction_id=db_prediction.id,
            prediction="Instável" if prediction == 1 else "Estável",
            prediction_code=prediction,
            probability_stable=float(prediction_proba[0]),
            probability_unstable=float(prediction_proba[1]),
            shap_values=dict(zip(ml_handler.feature_names, shap_values.tolist()))
        )
    except Exception as e:
        logging.error(f"Erro durante a predição: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Ocorreu um erro ao processar sua requisição.")


@app.get("/predictions/me", summary="Lista o histórico de predições do usuário",
         response_model=List[schemas.PredictionRecord])
async def get_user_predictions(
        db: Session = Depends(get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.get_predictions_by_user(db=db, user_id=current_user.id)


@app.get("/health", summary="Verifica a saúde da API")
def health_check(request: Request):
    model_status = "carregado" if hasattr(request.app.state,
                                          'ml_handler') and request.app.state.ml_handler is not None else "não carregado"
    return {"status": "ok", "model_status": model_status}
