# src/models.py (Focado na Camada de Persistência)

from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

# Importação relativa da Base declarativa.
from .database import Base

class FinancialData(Base):
    """
    Modelo ORM (SQLAlchemy) que mapeia a tabela 'financial_data' no banco de dados.
    Esta classe define a estrutura da tabela.
    """
    __tablename__ = "financial_data"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, index=True, unique=True)
    income = Column(Float)
    expenses = Column(Float)
    debt = Column(Float)
    assets = Column(Float)
    prediction = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
