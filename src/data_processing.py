# ==============================================================================
# PIPELINE DE PROCESSAMENTO DE DADOS (VERSÃO ROBUSTA)
# ------------------------------------------------------------------------------
# Este script contém uma lógica de pré-processamento robusta que:
# 1. Separa colunas de ID, Alvo e Features.
# 2. Verifica se a coluna Alvo existe antes de tentar usá-la.
# 3. Processa apenas as colunas de features.
# 4. Reconstrói o dataframe final de forma segura.
# ==============================================================================

import logging
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Usa imports relativos para referenciar módulos no mesmo pacote (src)
from .config import settings

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Encapsula as etapas de pré-processamento APENAS para as colunas de features.
    - Aplica StandardScaler em features numéricas.
    - Aplica One-Hot Encoding em features categóricas.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        if not numeric_features:
            logger.warning("Nenhuma feature numérica foi fornecida para escalonamento.")
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()

    def fit_transform(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta o processador às features e as transforma.
        Assume que o DataFrame de entrada contém APENAS as features a serem processadas.
        """
        logger.info("Iniciando o processo de fit e transform das features.")
        df_processed = df_features.copy()

        if self.categorical_features:
            logger.info(f"Aplicando one-hot encoding em: {self.categorical_features}")
            df_processed = pd.get_dummies(df_processed, columns=self.categorical_features, drop_first=True, dtype=int)

        # Identifica as colunas numéricas que ainda existem (e não foram transformadas em dummies)
        current_numeric_features = [col for col in self.numeric_features if col in df_processed.columns]

        if current_numeric_features:
            logger.info(f"Aplicando StandardScaler em: {current_numeric_features}")
            df_processed[current_numeric_features] = self.scaler.fit_transform(df_processed[current_numeric_features])

        logger.info("Processamento de features concluído.")
        return df_processed


def process_data(input_path: str, output_path: str) -> None:
    """
    Orquestra o pipeline completo de processamento de dados, separando
    features, IDs e alvo, processando apenas as features.
    """
    try:
        logger.info(f"Carregando dados de {input_path}")
        raw_df = pd.read_csv(input_path)

        # --- 1. Separação de Colunas ---
        id_cols_config = [settings.model.ENTITY_COLUMN, settings.model.YEAR_COLUMN]

        # Guarda as colunas que não serão processadas (IDs e Alvo)
        cols_to_keep = []
        for col in id_cols_config:
            if col in raw_df.columns:
                cols_to_keep.append(col)

        if settings.model.TARGET_COLUMN in raw_df.columns:
            logger.info(f"Coluna alvo '{settings.model.TARGET_COLUMN}' encontrada.")
            cols_to_keep.append(settings.model.TARGET_COLUMN)
        else:
            logger.warning(
                f"Coluna alvo '{settings.model.TARGET_COLUMN}' não encontrada. O script continuará, assumindo que é um dataset para predição.")

        non_feature_df = raw_df[cols_to_keep]
        features_df = raw_df.drop(columns=cols_to_keep, errors='ignore')

        # --- 2. Processamento das Features ---
        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

        processor = DataProcessor(numeric_features=numeric_features, categorical_features=categorical_features)

        processed_features_df = processor.fit_transform(features_df)

        # --- 3. Reconstrução do DataFrame Final ---
        final_df = pd.concat([non_feature_df.reset_index(drop=True), processed_features_df.reset_index(drop=True)],
                             axis=1)

        logger.info(f"Salvando dados processados em {output_path}")
        final_df.to_csv(output_path, index=False)

    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado em {input_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Um erro ocorreu durante o processamento dos dados: {e}", exc_info=True)
        raise
