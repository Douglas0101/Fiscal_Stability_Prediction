# src/database.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic_settings import BaseSettings

# --- 1. Configuração Robusta com Pydantic (Sem alterações) ---
class DatabaseSettings(BaseSettings):
    DATABASE_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = DatabaseSettings()

# --- 2. Engine e Sessão Assíncronos (Aprimorado) ---
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    # --- APRIMORAMENTO: Configuração explícita do pool de conexões para produção ---
    pool_size=10,  # Número de conexões a manter abertas no pool.
    max_overflow=20,  # Conexões extras que podem ser abertas além do pool_size.
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# --- APRIMORAMENTO: Base declarativa para ser usada pelo Alembic para migrações ---
# O Alembic (ferramenta de migração) usará esta 'Base' para detectar
# mudanças nos seus modelos (como em 'models.py') e gerar os scripts de migração.
Base = declarative_base()


# --- 3. Função de Dependência Assíncrona para FastAPI (Sem alterações) ---
async def get_db() -> AsyncSession:
    """
    Fornece uma sessão de base de dados assíncrona para cada request.
    Garante que a sessão é sempre fechada corretamente.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()