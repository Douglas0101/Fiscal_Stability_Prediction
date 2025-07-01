import pandas as pd
import numpy as np
import os

# Importar configurações
from . import config

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho}")
    return pd.read_csv(caminho)

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa os dados, removendo anos e países com muitos valores ausentes."""
    df_limpo = df.copy()
    # Lógica de limpeza baseada nos scripts anteriores
    anos_para_remover = [2024, 2025]
    df_limpo = df_limpo[~df_limpo["year"].isin(anos_para_remover)]

    ausencia_por_pais = df_limpo.groupby("country_name")[config.TARGET_VARIABLE].apply(
        lambda x: x.isnull().mean())
    paises_a_remover = set(ausencia_por_pais[ausencia_por_pais > 0.9].index)
    df_limpo = df_limpo[~df_limpo["country_name"].isin(paises_a_remover)]

    return df_limpo

def imputar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche valores ausentes usando interpolação e preenchimento."""
    df_imputado = df.sort_values(by=["country_name", "year"])
    colunas_numericas = df_imputado.select_dtypes(include="number").columns.drop("year")

    df_imputado[colunas_numericas] = df_imputado.groupby("country_name")[colunas_numericas].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
    )
    # Fallback para colunas que ainda possam ter NaNs
    df_imputado.fillna(df_imputado.median(numeric_only=True), inplace=True)
    return df_imputado

def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features para o modelo."""
    df_featured = df.copy()
    df_featured = df_featured.sort_values(by=["country_name", "year"])

    # Balanço Fiscal
    df_featured["fiscal_balance_gdp"] = df_featured["Government Revenue (% of GDP)"] - \
        df_featured["Government Expense (% of GDP)"]

    # Lags
    for feature in config.LAG_FEATURES:
        df_featured[f"{feature}_lag1"] = df_featured.groupby("country_name")[feature].shift(1)

    # Rolling Features
    for feature in config.ROLLING_FEATURES:
        df_featured[f"{feature}_rolling_mean_3y"] = df_featured.groupby("country_name")[feature].transform(lambda x: x.rolling(3, 1).mean())
        df_featured[f"{feature}_rolling_std_3y"] = df_featured.groupby("country_name")[feature].transform(lambda x: x.rolling(3, 1).std())

    # Rate of Change
    for feature in config.ROLLING_FEATURES:
        df_featured[f"{feature}_pct_change_1y"] = df_featured.groupby("country_name")[feature].pct_change()

    # Limpar NaNs e Infinitos gerados
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_featured.dropna(inplace=True)
    
    return df_featured

def pipeline_completo_de_dados():
    """Executa o pipeline completo de processamento de dados."""
    print("Iniciando pipeline de dados...")
    df_raw = carregar_dados(config.RAW_DATA_FILE)
    df_clean = limpar_dados(df_raw)
    df_imputed = imputar_dados(df_clean)
    df_featured = criar_features(df_imputed)

    # Salvar o dataset final
    os.makedirs(os.path.dirname(config.FEATURED_DATA_FILE), exist_ok=True)
    df_featured.to_csv(config.FEATURED_DATA_FILE, index=False)
    print(f"Pipeline de dados concluído. Dataset final salvo em: {config.FEATURED_DATA_FILE}")
    return df_featured

if __name__ == "__main__":
    # Se o script for executado diretamente, ele gera o dataset de features
    pipeline_completo_de_dados()