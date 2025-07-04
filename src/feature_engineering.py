# ==============================================================================
# ENGENHARIA DE FEATURES E CRIAÇÃO DA VARIÁVEL ALVO
# ------------------------------------------------------------------------------
# Este script é responsável por criar a variável que queremos prever.
# Ele carrega os dados brutos e aplica uma lógica de negócio para gerar
# a coluna 'fiscal_stability_index'.
# ==============================================================================

import logging
import os
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


def create_target_variable(input_path: str, output_path: str) -> None:
    """
    Carrega os dados brutos e cria a variável alvo com base em regras de negócio.

    Args:
        input_path (str): Caminho para o ficheiro de dados brutos.
        output_path (str): Caminho para salvar o ficheiro com a nova coluna alvo.
    """
    try:
        logger.info(f"Carregando dados brutos de: {input_path}")
        df = pd.read_csv(input_path)

        # --- LÓGICA DE NEGÓCIO PARA CRIAR A VARIÁVEL ALVO ---
        # Exemplo: Um país é considerado instável (0) se a sua dívida pública
        # ultrapassar 90% do PIB. Caso contrário, é estável (1).
        #
        # IMPORTANTE: Esta é a regra que você deve ajustar com base no seu
        # conhecimento e nos objetivos do projeto.
        debt_column = 'Public Debt (% of GDP)'
        if debt_column not in df.columns:
            raise ValueError(f"A coluna '{debt_column}' necessária para criar o alvo não foi encontrada.")

        logger.info(f"Criando a coluna alvo '{settings.model.TARGET_COLUMN}' com base na regra de negócio.")

        # Garante que a coluna de dívida é numérica, tratando erros
        df[debt_column] = pd.to_numeric(df[debt_column], errors='coerce')

        # Preenche valores de dívida ausentes com a mediana para não perder dados
        debt_median = df[debt_column].median()
        df[debt_column].fillna(debt_median, inplace=True)
        logger.warning(f"Valores ausentes na coluna de dívida foram preenchidos com a mediana ({debt_median:.2f}).")

        df[settings.model.TARGET_COLUMN] = df.apply(
            lambda row: 0 if row[debt_column] > 90 else 1, axis=1
        )

        # ---------------------------------------------------------

        # Verifica a distribuição da nova coluna
        target_distribution = df[settings.model.TARGET_COLUMN].value_counts(normalize=True)
        logger.info(f"Distribuição da variável alvo criada:\n{target_distribution}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Salvando dados com a nova variável alvo em: {output_path}")
        df.to_csv(output_path, index=False)

    except FileNotFoundError:
        logger.error(f"Ficheiro de dados brutos não encontrado em {input_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Um erro ocorreu durante a engenharia de features: {e}", exc_info=True)
        raise
