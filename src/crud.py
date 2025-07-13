# src/crud.py (Centraliza as Operações de Banco de Dados)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

# Importações relativas para os modelos e schemas.
from . import models, schemas


async def get_all_predictions(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.FinancialData]:
    """
    Busca no banco de dados uma lista de todos os registros de predições.

    Args:
        db: A sessão da base de dados.
        skip: O número de registros a pular (para paginação).
        limit: O número máximo de registros a retornar.

    Returns:
        Uma lista de objetos SQLAlchemy FinancialData.
    """
    result = await db.execute(select(models.FinancialData).offset(skip).limit(limit))
    return result.scalars().all()


async def create_prediction_record(
        db: AsyncSession,
        data: schemas.FinancialDataCreate,
        prediction_result: schemas.PredictionResult
) -> models.FinancialData:
    """
    Cria e salva um novo registro de predição no banco de dados.

    Args:
        db: A sessão da base de dados.
        data: Os dados de entrada originais.
        prediction_result: O resultado da predição do modelo.

    Returns:
        O objeto SQLAlchemy FinancialData que foi criado e salvo.
    """
    # Cria uma instância do modelo de banco de dados.
    db_financial_data = models.FinancialData(
        **data.model_dump(),
        prediction=prediction_result.prediction,
        risk_level=prediction_result.risk_level
    )

    # Adiciona, confirma e atualiza o registro.
    db.add(db_financial_data)
    await db.commit()
    await db.refresh(db_financial_data)

    return db_financial_data
