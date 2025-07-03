# File: src/database.py
# (Ficheiro Novo)
# Este módulo centraliza a configuração e a gestão da sessão do banco de dados.

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Carrega a URL do banco de dados a partir de uma variável de ambiente para flexibilidade.
# O valor padrão é para desenvolvimento local fora do Docker.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost:5432/fiscaldb")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependência do FastAPI que cria uma sessão do banco de dados por requisição
    e garante que ela seja fechada ao final, mesmo em caso de erro.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
