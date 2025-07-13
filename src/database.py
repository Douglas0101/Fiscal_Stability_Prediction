# src/database.py
import logging
import time
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from .logger_config import get_logger

logger = get_logger(__name__)

# --- Detalhes da Conexão ---
DB_USER = "user"
DB_PASSWORD = "password"
DB_HOST = "db_api"
DB_NAME = "fiscal_stability_db"

# URL para a base de dados da aplicação
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
# URL para a base de dados de manutenção 'postgres' (que existe sempre)
MAINTENANCE_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/postgres"

# Engine para a aplicação principal
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)


async def ensure_database_exists():
    """
    Conecta-se à base de dados 'postgres' para garantir que a base de dados da aplicação
    existe, criando-a se for necessário.
    """
    logger.info(f"Verificando a existência da base de dados '{DB_NAME}'...")
    maintenance_engine = create_async_engine(MAINTENANCE_DATABASE_URL, isolation_level="AUTOCOMMIT")

    try:
        async with maintenance_engine.connect() as conn:
            db_exists_result = await conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'")
            )
            if not db_exists_result.scalar_one_or_none():
                logger.info(f"A base de dados '{DB_NAME}' não existe. A criar...")
                await conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
                logger.info(f"Base de dados '{DB_NAME}' criada com sucesso.")
            else:
                logger.info(f"A base de dados '{DB_NAME}' já existe.")
    finally:
        await maintenance_engine.dispose()


async def create_tables_in_db():
    """Conecta-se à base de dados da aplicação e cria todas as tabelas necessárias."""
    logger.info("Verificando/criando tabelas na base de dados da aplicação...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Tabelas verificadas/criadas com sucesso.")


async def init_db(max_retries=5, delay_seconds=5):
    """
    Inicializa a base de dados: garante que ela existe e depois cria as tabelas.
    Inclui uma lógica de "retry" para robustez.
    """
    for attempt in range(max_retries):
        try:
            await ensure_database_exists()
            await create_tables_in_db()
            return  # Sucesso, sai da função
        except Exception as e:
            logger.warning(f"Falha ao inicializar a base de dados na tentativa {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"A aguardar {delay_seconds} segundos antes de tentar novamente...")
                time.sleep(delay_seconds)
            else:
                logger.error("Não foi possível inicializar a base de dados após várias tentativas.")
                raise


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependência do FastAPI para obter uma sessão da base de dados."""
    async with AsyncSessionLocal() as session:
        yield session
