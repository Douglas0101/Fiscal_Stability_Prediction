import pandas as pd
import logging
from src.config import settings  # CORREÇÃO: Importa o objeto 'settings'

logger = logging.getLogger(__name__)

# CORREÇÃO: O argumento padrão agora usa o objeto 'settings'.
def load_data(file_path: str = settings.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.

    Args:
        file_path (str): O caminho para o arquivo CSV. O padrão é o caminho dos dados brutos.

    Returns:
        pd.DataFrame: O DataFrame carregado.
    """
    try:
        logger.info(f"Carregando dados de: {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado em: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
        raise
