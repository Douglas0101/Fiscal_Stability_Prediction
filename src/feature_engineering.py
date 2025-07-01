# src/feature_engineering.py
import pandas as pd
from src import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features (lags e estatísticas móveis) no DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame processado.

    Returns:
        pd.DataFrame: O DataFrame com as novas features.
    """
    df_feat = df.copy()

    # Agrupar por país para criar features temporais corretamente
    grouped = df_feat.groupby("country_id")

    # Criar Lags
    for col in config.LAG_FEATURES:
        if col in df_feat.columns:
            df_feat[f'{col}_lag1'] = grouped[col].shift(1)
            df_feat[f'{col}_lag2'] = grouped[col].shift(2)

    # Criar Estatísticas Móveis
    for col in config.ROLLING_FEATURES:
        if col in df_feat.columns:
            # Garante que o índice seja resetado para alinhar corretamente após o rolling
            rolling_mean = grouped[col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_std = grouped[col].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
            df_feat[f'{col}_roll_mean3'] = rolling_mean
            df_feat[f'{col}_roll_std3'] = rolling_std

    # Remover linhas com NaN gerados pelos lags/rolling e resetar o índice
    df_featured = df_feat.dropna().reset_index(drop=True)

    return df_featured
