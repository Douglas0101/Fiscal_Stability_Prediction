import pandas as pd
import numpy as np
import logging
import os

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_target_variable(input_path: str, output_path: str):
    """
    Carrega os dados processados, cria a variável alvo 'fiscal_stability_index'
    e salva o resultado no diretório final.
    """
    try:
        logger.info(f"Carregando dados de: {input_path}")
        if not os.path.exists(input_path):
            logger.error(f"Arquivo de entrada não encontrado em {input_path}")
            raise FileNotFoundError

        df = pd.read_csv(input_path)

        # --- LÓGICA PARA CRIAR A VARIÁVEL ALVO ---
        # Regra: Estável (1) se a dívida pública < 90% do PIB, senão Instável (0).
        # Esta é a regra do seu feature_engineering.py, que parece ser a mais recente.
        debt_column = 'Public Debt (% of GDP)'
        if debt_column not in df.columns:
            logger.error(f"A coluna '{debt_column}' necessária para criar o alvo não foi encontrada.")
            raise KeyError(f"Coluna '{debt_column}' ausente.")

        logger.info(f"Criando a variável alvo 'fiscal_stability_index' com base na coluna '{debt_column}'.")

        # Garante que a coluna é numérica, tratando erros
        df[debt_column] = pd.to_numeric(df[debt_column], errors='coerce')

        # Preenche valores NaN na coluna de dívida com a mediana para não perder dados
        debt_median = df[debt_column].median()
        df[debt_column].fillna(debt_median, inplace=True)
        logger.info(f"Valores ausentes em '{debt_column}' preenchidos com a mediana ({debt_median:.2f}).")

        df['fiscal_stability_index'] = np.where(df[debt_column] < 90, 1, 0)

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
        logger.error(f"Ocorreu um erro inesperado: {e}")
        raise


if __name__ == '__main__':
    # Define os caminhos de entrada e saída.
    # O script vai ler os dados processados e salvar os dados finais.
    INPUT_DATA_PATH = "data/02_processed/processed_data.csv"
    OUTPUT_DATA_PATH = "data/03_final/final_data.csv"

    generate_target_variable(
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_DATA_PATH
    )
