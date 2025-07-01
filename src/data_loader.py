# src/data_loader.py

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: O DataFrame carregado.
    """
    try:
        # Adicionado sep=',' para garantir a leitura correta de arquivos CSV padrão.
        df = pd.read_csv(file_path, sep=',')
        print(f"Dados carregados com sucesso de: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo não foi encontrado no caminho: {file_path}")
        raise
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo: {e}")
        raise
