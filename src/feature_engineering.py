# src/feature_engineering.py

import pandas as pd
from typing import List


def create_temporal_features(
        df: pd.DataFrame,
        target_col: str,
        lag_features: List[str],
        entity_col: str,
        lags: List[int] = [1, 2, 3],
        rolling_windows: List[int] = [3, 5]
) -> pd.DataFrame:
    """
    Cria features temporais de forma robusta, agrupando por entidade
    e tratando NaNs com imputação para frente e para trás.
    """
    print("Iniciando a criação de features temporais...")

    df_featured = df.sort_values([entity_col, 'year']).copy()
    df_featured['year_feature'] = df_featured['year']
    available_features = [col for col in lag_features if col in df_featured.columns]

    for col in available_features:
        print(f"Processando features para a coluna: '{col}'")
        for lag in lags:
            df_featured[f'{col}_lag{lag}'] = df_featured.groupby(entity_col)[col].shift(lag)
        for window in rolling_windows:
            df_featured[f'{col}_rolling_mean_{window}'] = df_featured.groupby(entity_col)[col].shift(1).rolling(
                window=window, min_periods=1).mean()
        df_featured[f'{col}_diff_1'] = df_featured.groupby(entity_col)[col].diff(1)

    # --- Tratamento Robusto de NaNs ---
    print(f"\nDataFrame shape antes da imputação de NaNs: {df_featured.shape}")

    df_featured = df_featured.groupby(entity_col, group_keys=False).apply(lambda group: group.ffill().bfill())
    df_featured.dropna(inplace=True)

    print(f"DataFrame shape após a imputação: {df_featured.shape}")

    # --- Verificação de Sanidade ---
    if df_featured.empty:
        raise ValueError(
            "O DataFrame ficou vazio após a engenharia de features e tratamento de NaNs. Verifique a qualidade dos dados de entrada.")

    if target_col in df_featured.columns:
        df_featured.rename(columns={target_col: 'target'}, inplace=True)

    return df_featured
