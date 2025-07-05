import pandas as pd
import numpy as np
import logging
import os
from src import config

# Configura o logging
logger = logging.getLogger(__name__)


def generate_target_variable(input_path: str, output_path: str):
    """
    Carrega os dados brutos, cria a variável alvo 'fiscal_stability_index'
    e salva o resultado em um novo arquivo CSV.

    Args:
        input_path (str): Caminho para o arquivo de dados brutos.
        output_path (str): Caminho para salvar o arquivo com a nova coluna alvo.
    """
    try:
        logger.info(f"Carregando dados brutos de: {input_path}")
        df = pd.read_csv(input_path)

        # --- LÓGICA DE EXEMPLO PARA CRIAR A VARIÁVEL ALVO ---
        # ATENÇÃO: Substitua esta lógica pela sua regra de negócio definitiva.
        # Exemplo: um país é considerado estável (1) se a dívida pública for
        # menor que 60% do PIB e a receita do governo for maior que 20% do PIB.
        # Caso contrário, é instável (0).
        logger.info("Criando a variável alvo 'fiscal_stability_index' com base em regras de negócio.")
        conditions = [
            (df['Public Debt (% of GDP)'] < 60) & (df['Government Revenue (% of GDP)'] > 20),
            (df['Public Debt (% of GDP)'] >= 60) | (df['Government Revenue (% of GDP)'] <= 20)
        ]
        outcomes = [1, 0]  # 1 para estável, 0 para instável
        df['fiscal_stability_index'] = np.select(conditions, outcomes, default=0)

        # Verifica se a coluna foi criada
        if 'fiscal_stability_index' not in df.columns:
            raise RuntimeError("Falha ao criar a coluna alvo 'fiscal_stability_index'.")

        # Garante que o diretório de saída exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Salvando dados com a coluna alvo em: {output_path}")
        df.to_csv(output_path, index=False)

        logger.info("Criação da variável alvo concluída com sucesso.")
        logger.info(f"Distribuição da variável alvo:\n{df['fiscal_stability_index'].value_counts(normalize=True)}")

    except FileNotFoundError:
        logger.error(f"Erro: O arquivo de entrada não foi encontrado em {input_path}")
        raise
    except KeyError as e:
        logger.error(f"Erro: Uma coluna necessária para criar o alvo não foi encontrada: {e}")
        raise
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a criação da variável alvo: {e}")
        raise


if __name__ == '__main__':
    # Para testes diretos do script
    from src.logger_config import setup_logging

    setup_logging()
    generate_target_variable(
        input_path=config.RAW_DATA_PATH,
        output_path=config.RAW_DATA_WITH_TARGET_PATH
    )
