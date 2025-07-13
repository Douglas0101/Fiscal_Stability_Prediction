# src/schemas.py (Focado nos Contratos da API)

from pydantic import BaseModel
from datetime import datetime
from typing import List

# --- Modelos Pydantic (Camada de Interface) ---
# Estes modelos definem a estrutura dos dados que entram e saem da API.

class FinancialDataCreate(BaseModel):
    """Schema para os dados de ENTRADA do endpoint de predição."""
    client_id: str
    income: float
    expenses: float
    debt: float
    assets: float

class PredictionResult(BaseModel):
    """Schema para a SAÍDA do endpoint de predição."""
    client_id: str
    prediction: float
    risk_level: str

class FinancialDataResponse(BaseModel):
    """
    Schema para os dados de SAÍDA do endpoint de leitura.
    Define a estrutura completa de um registo a ser retornado pela API.
    """
    id: int
    client_id: str
    income: float
    expenses: float
    debt: float
    assets: float
    prediction: float | None = None
    risk_level: str | None = None
    timestamp: datetime

    class Config:
        # Permite que o Pydantic leia os dados diretamente de um objeto SQLAlchemy.
        from_attributes = True
