# src/models.py
from sqlalchemy import Column, Integer, String, Float, JSON
from .database import Base

class Prediction(Base):
    """
    Modelo ORM para a tabela de previsões no banco de dados.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_data = Column(JSON)
    prediction = Column(String)
    probability = Column(Float)