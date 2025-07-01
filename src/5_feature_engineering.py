import pandas as pd
from typing import List

# Importar configurações
from . import config


def criar_features(
        df: pd.DataFrame,
        coluna_pais: str,
        lag_features: List[str],
        rolling_features: List[str],
        lag_period: int = 1,
        rolling_window: int = 3
) -> pd.DataFrame:
    """
    Enriquece o dataset criando novas variáveis (features) baseadas em
    conhecimento de domínio.

    As features criadas incluem:
    1. Balanço Fiscal (% do PIB).
    2. Variáveis temporais defasadas (lags).
    3. Estatísticas móveis (rolling statistics).
    4. Taxas de variação (rate of change).

    Args:
        df (pd.DataFrame): O DataFrame completo e imputado.
        coluna_pais (str): O nome da coluna que identifica o país.
        lag_features (List[str]): Uma lista de nomes de colunas para criar os lags.
        rolling_features (List[str]): Uma lista de nomes de colunas para criar
                                      estatísticas móveis.
        lag_period (int, optional): O número de períodos (anos) para a defasagem.
                                    Padrão é 1.
        rolling_window (int, optional): A janela para as estatísticas móveis.
                                      Padrão é 3.

    Returns:
        pd.DataFrame: O DataFrame final, enriquecido com as novas features e sem
                      valores NaN.
    """
    print("\n" + "=" * 60)
    print("INÍCIO DA ENGENHARIA DE VARIÁVEIS (FEATURE ENGINEERING)")
    print("=" * 60)

    df_featured = df.copy()
    df_featured = df_featured.sort_values(by=[coluna_pais, 'year'])

    # --- a. Balanço Fiscal ---
    print("1. Criando a feature 'fiscal_balance_gdp'...")
    df_featured['fiscal_balance_gdp'] = df_featured['Government Revenue (% of GDP)'] - \
        df_featured['Government Expense (% of GDP)']
    print("   -> Feature 'fiscal_balance_gdp' criada com sucesso.")

    # --- b. Variáveis Defasadas (Lags) ---
    print(f"\n2. Criando variáveis defasadas (lag de {lag_period} ano) para "
          f"{len(lag_features)} colunas...")
    for feature in lag_features:
        new_col_name = f"{feature}_lag{lag_period}"
        df_featured[new_col_name] = df_featured.groupby(coluna_pais)[feature].shift(lag_period)
        print(f"   -> Feature '{new_col_name}' criada.")

    # --- c. Estatísticas Móveis ---
    print(f"\n3. Criando estatísticas móveis (janela de {rolling_window} anos) para "
          f"{len(rolling_features)} colunas...")
    for feature in rolling_features:
        df_featured[f'{feature}_rolling_mean_{rolling_window}y'] = \
            df_featured.groupby(coluna_pais)[feature].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
        df_featured[f'{feature}_rolling_std_{rolling_window}y'] = \
            df_featured.groupby(coluna_pais)[feature].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).std())
        print(f"   -> Features de média e desvio padrão móvel para '{feature}' criadas.")

    # --- d. Taxas de Variação ---
    print(f"\n4. Criando taxas de variação para {len(rolling_features)} colunas...")
    for feature in rolling_features:
        df_featured[f'{feature}_pct_change_1y'] = \
            df_featured.groupby(coluna_pais)[feature].pct_change()
        print(f"   -> Feature de taxa de variação para '{feature}' criada.")

    # --- e. Tratamento de Novos Valores Ausentes ---
    linhas_antes_drop = len(df_featured)
    print(f"\n5. Removendo linhas com valores NaN resultantes da criação das features...")
    df_featured.dropna(inplace=True)
    linhas_depois_drop = len(df_featured)
    print(
        f"   -> {linhas_antes_drop - linhas_depois_drop} linhas foram removidas.")

    print("\nENGENHARIA DE VARIÁVEIS CONCLUÍDA.")
    print("=" * 60)

    return df_featured