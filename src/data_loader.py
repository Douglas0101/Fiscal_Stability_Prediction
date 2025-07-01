import pandas as pd
import numpy as np
from typing import Optional
import os


def carregar_e_validar_dataset(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Carrega um dataset a partir de um ficheiro CSV, realiza uma validação inicial
    detalhada e retorna um DataFrame do Pandas.
    """
    print(f"Iniciando o carregamento e validação para o ficheiro: "
          f"{caminho_arquivo}\n")

    try:
        df = pd.read_csv(caminho_arquivo)
        print("Ficheiro CSV carregado com sucesso na memória.\n")
    except FileNotFoundError:
        print(
            f"ERRO: O ficheiro no caminho especificado não foi encontrado: "
            f"{caminho_arquivo}"
        )
        return None
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado durante o carregamento: {e}")
        return None

    print("=" * 50)
    print("RELATÓRIO DE VALIDAÇÃO INICIAL DO DATASET")
    print("=" * 50)

    nome_arquivo = os.path.basename(caminho_arquivo)
    print(f"Arquivo Carregado: {nome_arquivo}")

    total_celulas = np.prod(df.shape)
    num_duplicatas = df.duplicated().sum()
    total_celulas_vazias = df.isnull().sum().sum()
    porcentagem_faltante = (
        (total_celulas_vazias / total_celulas) * 100
        if total_celulas > 0
        else 0
    )

    print("\n--- Métricas Gerais ---")
    print(f"- Dimensões: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"- Total de Células: {total_celulas}")
    print(f"- Linhas Duplicadas: {num_duplicatas}")

    print("\n--- Análise de Dados Faltantes ---")
    print(f"- Total de Células Vazias: {total_celulas_vazias}")
    print(f"- Porcentagem de Dados Faltantes: {porcentagem_faltante:.2f}%")

    print("\nVALIDAÇÃO CONCLUÍDA. O dataset foi carregado com sucesso.")
    print("=" * 50)

    return df
