# src/database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@db-api:5432/apidb")

engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Base para os modelos declarativos
Base = declarative_base()

# Criador de sessão assíncrona
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def create_db_and_tables():
    """
    Cria todas as tabelas no banco de dados que herdam de Base.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    """
    Função de dependência para obter uma sessão do banco de dados.
    """
    async with async_session() as session:
        yield session