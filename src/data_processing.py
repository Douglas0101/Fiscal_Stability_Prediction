# src/data_processing.py
import pandas as pd


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento e limpeza inicial dos dados.
    Esta função é o local ideal para encapsular a lógica dos seus scripts
    de validação, tratamento de dados ausentes e imputação.

    Args:
        df (pd.DataFrame): O DataFrame bruto.

    Returns:
        pd.DataFrame: O DataFrame processado e limpo.
    """
    print("Executando limpeza e pré-processamento de dados...")

    # Exemplo de operação: Ordenar os dados, crucial para séries temporais
    df_processed = df.sort_values(by=["country_id", "year"]).reset_index(drop=True)

    # Exemplo: Remover colunas que não serão usadas (se houver)
    # df_processed = df_processed.drop(columns=['some_useless_column'], errors='ignore')

    # TODO: Adicione aqui a lógica dos seus scripts:
    # data_validation.py
    # missing_data_eda.py
    # data_preprocessing.py
    # data_imputation.py

    print("Dados processados com sucesso.")
    return df_processed
