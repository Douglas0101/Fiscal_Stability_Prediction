# File: src/main.py
# (Ficheiro Atualizado)
# Ponto de entrada principal da API. Integra todos os módulos.

from fastapi import FastAPI
from .database import engine
from .auth import auth as auth_router, models as auth_models
# Se tiver outros routers (ex: para predições), importe-os aqui.
# from .prediction import router as prediction_router

# Este comando cria as tabelas no banco de dados (ex: a tabela 'users')
# se elas ainda não existirem. É seguro executar a cada inicialização.
auth_models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Fiscal Stability Prediction API",
    description="API para prever estabilidade fiscal e gerir usuários.",
    version="1.0.0"
)

# Inclui os endpoints de autenticação (/auth/login, /auth/register) na sua API.
app.include_router(auth_router.router)
# app.include_router(prediction_router) # Adicione outros routers aqui.

@app.get("/", tags=["Root"])
def read_root():
    """Endpoint raiz para verificar se a API está a funcionar."""
    return {"message": "Welcome to the Fiscal Stability Prediction API"}
