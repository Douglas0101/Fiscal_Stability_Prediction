import pandas as pd
import logging
from typing import List, Optional
from src.config import settings  # CORREÇÃO: Importa o objeto 'settings'

logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Verifica se todas as colunas esperadas estão presentes no DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame a ser validado.
        expected_columns (List[str]): A lista de nomes de colunas esperadas.

    Returns:
        bool: True se todas as colunas estiverem presentes, False caso contrário.
    """
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Validação de colunas falhou. Colunas ausentes: {sorted(list(missing_cols))}")
        return False

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        logger.warning(f"Validação de colunas: Colunas extras encontradas: {sorted(list(extra_cols))}")

    logger.info("Validação de colunas passou com sucesso.")
    return True


def check_for_missing_values(df: pd.DataFrame):
    """
    Verifica e loga a quantidade e percentagem de valores ausentes por coluna.
    """
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]

    if not missing_per_column.empty:
        logger.warning("Foram encontrados valores ausentes nas seguintes colunas:")
        missing_df = pd.DataFrame({
            'Contagem Faltante': missing_per_column,
            'Porcentagem (%)': (missing_per_column / len(df)) * 100
        })
        logger.warning(f"\n{missing_df.round(2)}")
    else:
        logger.info("Nenhuma coluna com dados faltantes foi encontrada.")


def validate_data_integrity(df: pd.DataFrame):
    """
    Executa um conjunto de validações de integridade nos dados.
    """
    logger.info("--- Iniciando validação de integridade dos dados ---")

    # Validação de duplicatas
    num_duplicatas = df.duplicated().sum()
    if num_duplicatas > 0:
        logger.warning(f"Encontradas {num_duplicatas} linhas completamente duplicadas.")
    else:
        logger.info("Nenhuma linha duplicada encontrada.")

    # Validação de valores ausentes
    check_for_missing_values(df)

    logger.info("--- Validação de integridade concluída ---")


# CORREÇÃO: Função para validar o ficheiro processado, usando 'settings'
def validate_processed_data(file_path: str = settings.PROCESSED_DATA_PATH) -> bool:
    """
    Carrega e valida o ficheiro de dados já processado.
    Esta função é um exemplo e pode ser expandida com regras mais específicas.
    """
    try:
        logger.info(f"Iniciando a validação do ficheiro processado: {file_path}")
        df = pd.read_csv(file_path)

        # Verifica se a coluna alvo existe após o processamento
        if settings.TARGET_VARIABLE not in df.columns:
            logger.error(
                f"Validação falhou: A coluna alvo '{settings.TARGET_VARIABLE}' não está no ficheiro processado.")
            return False

        logger.info(f"Validação do ficheiro {file_path} concluída com sucesso.")
        return True
    except FileNotFoundError:
        logger.error(f"Ficheiro de validação não encontrado: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Erro durante a validação do ficheiro {file_path}: {e}")
        return False
