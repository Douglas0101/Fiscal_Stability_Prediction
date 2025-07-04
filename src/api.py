import os
import joblib
import pandas as pd
import numpy as np
import shap
import logging
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Dict
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine, get_db
from .auth import authenticate_user, create_access_token, get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES

# --- Configuração ---
# Configura um logger para exibir informações úteis no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lê os caminhos dos artefatos a partir de variáveis de ambiente.
# Se não definidas, usa os caminhos padrão.
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/best_xgb_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "src/models/scaler.pkl")
FEATURED_DATA_PATH = os.getenv("FEATURED_DATA_PATH", "data/04_features/featured_data.csv")
PREDICTION_RESULTS_PATH = os.getenv("PREDICTION_RESULTS_PATH", "notebooks/reports/resultados_modelo.json")


# --- Modelo de Dados de Entrada (Pydantic) ---
# Define a estrutura e os tipos de dados esperados na requisição
class PredictionFeatures(BaseModel):
    taxa_de_juros: float = Field(..., example=5.0, description="Taxa de Juros Anual (%)")
    inflacao_anual: float = Field(..., example=2.5, description="Taxa de Inflação Anual (%) ")
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

def create_db_and_tables():
    models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código executado na inicialização da API
    logging.info("Criando tabelas do banco de dados...")
    create_db_and_tables()
    logging.info("Tabelas do banco de dados criadas/verificadas.")

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

        # Carrega dados de features e resultados de predição
        ml_artifacts["featured_data"] = pd.read_csv(FEATURED_DATA_PATH).to_dict(orient="records")
        ml_artifacts["prediction_results"] = pd.read_json(PREDICTION_RESULTS_PATH).to_dict(orient="records")

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

# --- Rotas de Autenticação ---
auth_router = APIRouter()

@auth_router.post("/register", response_model=schemas.User, summary="Registra um novo usuário")
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email já registrado")
    return crud.create_user(db=db, user=user)

@auth_router.post("/token", response_model=schemas.Token, summary="Obtém um token de acesso para autenticação")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/users/me/", response_model=schemas.User, summary="Obtém informações do usuário logado")
async def read_users_me(current_user: schemas.User = Depends(get_current_active_user)):
    return current_user

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])


# --- Endpoints da API ---
@app.post("/predict", summary="Realiza uma nova predição de estabilidade fiscal")
async def predict(features: PredictionFeatures, current_user: schemas.User = Depends(get_current_active_user)) -> Dict:
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


@app.get("/data/features", summary="Retorna os dados de features históricos")
async def get_features_data(current_user: schemas.User = Depends(get_current_active_user)):
    if "featured_data" not in ml_artifacts:
        raise HTTPException(status_code=500, detail="Dados de features não carregados.")
    return ml_artifacts["featured_data"]

@app.get("/data/predictions", summary="Retorna os resultados das predições históricas")
async def get_predictions_data(current_user: schemas.User = Depends(get_current_active_user)):
    if "prediction_results" not in ml_artifacts:
        raise HTTPException(status_code=500, detail="Resultados de predição não carregados.")
    return ml_artifacts["prediction_results"]

@app.get("/data/summary", summary="Retorna métricas de resumo do modelo")
async def get_summary_data(current_user: schemas.User = Depends(get_current_active_user)):
    if "prediction_results" not in ml_artifacts:
        raise HTTPException(status_code=500, detail="Resultados de predição não carregados para o resumo.")
    
    df_predictions = pd.DataFrame(ml_artifacts["prediction_results"])
    
    total_predictions = len(df_predictions)
    stable_predictions = df_predictions[df_predictions["prediction_code"] == 0].shape[0]
    unstable_predictions = df_predictions[df_predictions["prediction_code"] == 1].shape[0]
    
    # Exemplo de cálculo de média de confiança (se houver uma coluna de confiança)
    # if "confidence" in df_predictions.columns:
    #     avg_confidence = df_predictions["confidence"].mean()
    # else:
    #     avg_confidence = None

    return {
        "total_predictions": total_predictions,
        "stable_predictions": stable_predictions,
        "unstable_predictions": unstable_predictions,
        # "average_confidence": avg_confidence
    }

@app.get("/data/historical", summary="Retorna dados históricos para visualização")
async def get_historical_data(current_user: schemas.User = Depends(get_current_active_user)):
    # Este é um exemplo. Em um cenário real, você carregaria dados de um banco de dados ou arquivo.
    # Por simplicidade, estou retornando dados mockados.
    return [
        {"year": 2018, "gdp_growth": 2.5, "inflation": 3.0, "public_debt": 55.0},
        {"year": 2019, "gdp_growth": 2.8, "inflation": 3.2, "public_debt": 57.0},
        {"year": 2020, "gdp_growth": -4.0, "inflation": 2.0, "public_debt": 70.0},
        {"year": 2021, "gdp_growth": 5.0, "inflation": 5.0, "public_debt": 68.0},
        {"year": 2022, "gdp_growth": 3.5, "inflation": 6.0, "public_debt": 65.0},
    ]

@app.get("/health", summary="Verifica a saúde da API")
def health_check():
    """Endpoint para verificar se a API está online e se o modelo foi carregado."""
    model_status = "carregado" if ml_artifacts.get("model") else "não carregado"
    return {"status": "ok", "model_status": model_status}
