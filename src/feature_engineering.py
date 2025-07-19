import logging
import os
import pandas as pd
# --- CORREÇÃO 1: Importar a classe AppConfig em vez do objeto 'settings' ---
from src.config import AppConfig

logger = logging.getLogger(__name__)

# --- CORREÇÃO 2: Instanciar a classe de configuração ---
config = AppConfig()

def create_target_variable(input_path: str, output_path: str) -> None:
    """
    Carrega os dados brutos, cria a variável alvo com base em regras de negócio
    e remove a coluna fonte para prevenir o vazamento de dados (data leakage).
    """
    try:
        logger.info(f"Carregando dados brutos de: {input_path}")
        df = pd.read_csv(input_path)

        debt_column = 'Public Debt (% of GDP)'
        if debt_column not in df.columns:
            raise ValueError(f"A coluna '{debt_column}' necessária para criar o alvo não foi encontrada.")

        # --- CORREÇÃO 3: Usar a instância 'config' para aceder às configurações ---
        target_col = config.model.TARGET_VARIABLE
        logger.info(f"Criando a coluna alvo '{target_col}' com base na regra de negócio.")

        df[debt_column] = pd.to_numeric(df[debt_column], errors='coerce')
        debt_median = df[debt_column].median()
        df[debt_column].fillna(debt_median, inplace=True)

        df[target_col] = df.apply(
            lambda row: 0 if row[debt_column] > 90 else 1, axis=1
        )

        target_distribution = df[target_col].value_counts(normalize=True)
        logger.info(f"Distribuição da variável alvo criada:\n{target_distribution}")

        # Remove a coluna original usada para criar o alvo para evitar data leakage
        logger.info(f"Removendo a coluna fonte '{debt_column}' para prevenir vazamento de dados.")
        df.drop(columns=[debt_column], inplace=True, errors='ignore')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Salvando dados com a nova variável alvo em: {output_path}")
        df.to_csv(output_path, index=False)

    except FileNotFoundError:
        logger.error(f"Ficheiro de dados brutos não encontrado em {input_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Um erro ocorreu durante a engenharia de features: {e}", exc_info=True)
        raise