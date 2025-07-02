import os
import joblib
import pandas as pd
import numpy as np
import shap
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# --- Configuração ---
# Configura um logger para exibir informações úteis no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lê os caminhos dos artefatos a partir de variáveis de ambiente.
# Se não definidas, usa os caminhos padrão.
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/best_xgb_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "src/models/scaler.pkl")


# --- Modelo de Dados de Entrada (Pydantic) ---
# Define a estrutura e os tipos de dados esperados na requisição
class PredictionFeatures(BaseModel):
    taxa_de_juros: float = Field(..., example=5.0, description="Taxa de Juros Anual (%)")
    inflacao_anual: float = Field(..., example=2.5, description="Taxa de Inflação Anual (%)")
    crescimento_pib: float = Field(..., example=3.0, description="Crescimento do PIB (%)")
    divida_publica_pib: float = Field(..., example=60.0, description="Dívida Pública como % do PIB")
    balanca_comercial: float = Field(..., example=10.0, description="Balança Comercial (em Bilhões de USD)")
    investimento_estrangeiro: float = Field(..., example=20.0,
                                            description="Investimento Estrangeiro Direto (em Bilhões de USD)")

    class Config:
        # Garante que a documentação da API mostre o exemplo correto
        schema_extra = {
            "example": {
                "taxa_de_juros": 5.0,
                "inflacao_anual": 2.5,
                "crescimento_pib": 3.0,
                "divida_publica_pib": 60.0,
                "balanca_comercial": 10.0,
                "investimento_estrangeiro": 20.0
            }
        }


# --- Ciclo de Vida da Aplicação ---
# Dicionário global para armazenar os artefatos de ML carregados
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código executado na inicialização da API
    logging.info("Carregando artefatos de Machine Learning...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Para modelos baseados em árvores como XGBoost, TreeExplainer é muito mais eficiente
        explainer = shap.TreeExplainer(model)

        ml_artifacts["model"] = model
        ml_artifacts["scaler"] = scaler
        ml_artifacts["explainer"] = explainer

        # Armazena a ordem correta das features, extraída do próprio modelo
        # Isso evita erros se a ordem dos dados de entrada mudar
        ml_artifacts["feature_names"] = list(model.get_booster().feature_names)

        logging.info("Artefatos carregados com sucesso.")
    except FileNotFoundError as e:
        logging.error(
            f"Erro fatal: Arquivo de modelo ou scaler não encontrado em '{MODEL_PATH}' ou '{SCALER_PATH}'. A API não poderá servir predições.")
        ml_artifacts["model"] = None  # Marca o modelo como não disponível

    yield  # A API fica em execução aqui

    # Código executado no encerramento da API
    ml_artifacts.clear()
    logging.info("Recursos de ML liberados com sucesso.")


# --- Criação da Aplicação FastAPI ---
app = FastAPI(
    title="API de Previsão de Estabilidade Fiscal",
    description="Uma API para prever a estabilidade fiscal de um país usando um modelo XGBoost.",
    version="1.0.0",
    lifespan=lifespan
)

# Adiciona o middleware CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para o domínio do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints da API ---
@app.post("/predict", summary="Realiza uma nova predição de estabilidade fiscal")
async def predict(features: PredictionFeatures) -> Dict:
    """
    Recebe as features de um país, realiza a predição e retorna o resultado
    juntamente com a interpretabilidade via SHAP.
    """
    if not ml_artifacts.get("model"):
        raise HTTPException(
            status_code=503,
            detail="Serviço indisponível: O modelo de ML não pôde ser carregado. Verifique os logs do servidor."
        )

    try:
        # Converte os dados de entrada para um DataFrame do pandas
        # Garante que a ordem das colunas é a mesma usada no treinamento
        feature_names = ml_artifacts["feature_names"]
        input_df = pd.DataFrame([features.dict()], columns=feature_names)

        # Padroniza os dados usando o scaler carregado
        scaler = ml_artifacts["scaler"]
        input_scaled = scaler.transform(input_df)

        # Realiza a predição
        model = ml_artifacts["model"]
        prediction = int(model.predict(input_scaled)[0])
        prediction_proba = model.predict_proba(input_scaled)

        # Calcula os valores SHAP para interpretabilidade
        explainer = ml_artifacts["explainer"]
        shap_values = explainer.shap_values(input_scaled)

        # Formata a resposta
        return {
            "prediction": "Instável" if prediction == 1 else "Estável",
            "prediction_code": prediction,
            "probability_stable": float(prediction_proba[0][0]),
            "probability_unstable": float(prediction_proba[0][1]),
            "shap_values": dict(zip(feature_names, shap_values[1][0].tolist()))
        }
    except Exception as e:
        logging.error(f"Erro durante a predição: {e}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar a requisição: {e}")


@app.get("/health", summary="Verifica a saúde da API")
def health_check():
    """Endpoint para verificar se a API está online e se o modelo foi carregado."""
    model_status = "carregado" if ml_artifacts.get("model") else "não carregado"
    return {"status": "ok", "model_status": model_status}
