# src/data_loader.py
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.

    Args:
        file_path (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: O DataFrame carregado.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo {file_path} não foi encontrado.")
        raise

