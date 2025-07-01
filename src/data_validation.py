import pandas as pd
import numpy as np
from typing import Optional
import os


def carregar_e_validar_dataset(caminho_arquivo: str) -> Optional[pd.DataFrame]:
    """
    Carrega um dataset a partir de um ficheiro CSV, realiza uma validação inicial
    detalhada e retorna um DataFrame do Pandas.

    Esta função verifica dimensões, tipos de dados, valores ausentes, duplicatas e
    fornece estatísticas descritivas, consolidando tudo num relatório final.

    Args:
        caminho_arquivo (str): O caminho completo para o ficheiro CSV a ser carregado.

    Returns:
        Optional[pd.DataFrame]: Um DataFrame do Pandas com os dados carregados se o
                                processo for bem-sucedido; caso contrário, retorna None.
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
    except pd.errors.ParserError as e:
        print(f"ERRO: Falha ao fazer o parsing do ficheiro CSV. Verifique a "
              f"estrutura do ficheiro. Detalhe: {e}")
        return None
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado durante o carregamento: {e}")
        return None

    print("=" * 50)
    print("INÍCIO DA VALIDAÇÃO DO DATASET")
    print("=" * 50)

    # a. Inspeção Inicial
    print("\n--- 1. Inspeção Inicial ---")
    print(f"a. Dimensões do Dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    print("\nb. Amostra dos Dados (Primeiras 5 linhas):")
    print(df.head())
    print("\nc. Tipos de Dados e Uso de Memória:")
    df.info(verbose=True)

    # b. Análise de Dados Faltantes (Missing Values)
    print("\n--- 2. Análise de Dados Faltantes ---")
    total_celulas_vazias = df.isnull().sum().sum()
    total_celulas = np.prod(df.shape)
    porcentagem_faltante = (total_celulas_vazias / total_celulas) * 100 \
        if total_celulas > 0 else 0

    print(f"a. Quantidade total de células vazias: {total_celulas_vazias}")
    print(f"b. Porcentagem total de dados faltantes: {porcentagem_faltante:.2f}%")

    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0].sort_values(
        ascending=False)
    if not missing_per_column.empty:
        print("\nc. Detalhamento por coluna (contagem e %):")
        missing_df = pd.DataFrame({
            'Contagem Faltante': missing_per_column,
            'Porcentagem (%)': (missing_per_column / len(df)) * 100
        })
        print(missing_df.round(2))
    else:
        print("\nc. Nenhuma coluna com dados faltantes foi encontrada.")

    # c. Verificação de Duplicatas
    print("\n--- 3. Verificação de Linhas Duplicadas ---")
    num_duplicatas = df.duplicated().sum()
    print(f"a. Número de linhas completamente duplicadas: {num_duplicatas}")

    # d. Análise Descritiva Básica
    print("\n--- 4. Análise Descritiva ---")
    # Colunas Numéricas
    df_numericas = df.select_dtypes(include=np.number)
    if not df_numericas.empty:
        print("\na. Estatísticas para colunas numéricas:")
        print(df_numericas.describe().round(2))
    else:
        print("\na. Nenhuma coluna numérica encontrada.")

    # Colunas Categóricas/Object
    df_categoricas = df.select_dtypes(include=['object', 'category'])
    if not df_categoricas.empty:
        print("\nb. Estatísticas para colunas categóricas/object:")
        print(df_categoricas.describe())
    else:
        print("\nb. Nenhuma coluna categórica/object encontrada.")

    # --- Relatório Final Consolidado ---
    print("\n" + "=" * 50)
    print("RELATÓRIO DE VALIDAÇÃO INICIAL DO DATASET")
    print("=" * 50)

    nome_arquivo = os.path.basename(caminho_arquivo)
    print(f"Ficheiro Carregado: {nome_arquivo}")

    print("\n--- Métricas Gerais ---")
    print(f"- Dimensões: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"- Total de Células: {total_celulas}")
    print(f"- Linhas Duplicadas: {num_duplicatas}")

    print("\n--- Análise de Dados Faltantes ---")
    print(f"- Total de Células Vazias: {total_celulas_vazias}")
    print(f"- Porcentagem de Dados Faltantes: {porcentagem_faltante:.2f}%")
    colunas_criticas = missing_per_column[
        missing_per_column / len(df) > 0.1].index.tolist()
    if colunas_criticas:
        print(f"- Colunas com mais de 10% de dados faltantes: {colunas_criticas}")
    else:
        print("- Nenhuma coluna com mais de 10% de dados faltantes.")

    print("\n--- Tipos de Dados ---")
    print(f"- Colunas Numéricas: {df_numericas.columns.tolist()}")
    print(f"- Colunas Categóricas/Object: {df_categoricas.columns.tolist()}")

    print("\nVALIDAÇÃO CONCLUÍDA. O dataset foi carregado com sucesso.")
    print("=" * 50)

    return df