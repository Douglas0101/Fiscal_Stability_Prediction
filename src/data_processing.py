import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.config import settings

logger = logging.getLogger(__name__)


def process_data(input_path: str, output_path: str):
    """
    Carrega os dados, aplica pré-processamento (one-hot, scaling) e salva o resultado.
    """
    try:
        logger.info(f"Carregando dados de {input_path}")
        df = pd.read_csv(input_path)

        target_col = settings.model.TARGET_VARIABLE
        if target_col in df.columns:
            y = df[[target_col]]
            df = df.drop(columns=[target_col])
        else:
            y = None

        # --- CORREÇÃO DE VAZAMENTO ---
        # Remove as colunas que foram usadas para criar o alvo
        leaky_features = settings.model.LEAKY_FEATURES
        existing_leaky_features = [col for col in leaky_features if col in df.columns]
        if existing_leaky_features:
            df = df.drop(columns=existing_leaky_features)
            logger.info(f"Colunas de vazamento removidas para evitar 'colar': {existing_leaky_features}")
        # ---------------------------

        cols_to_drop = settings.model.DROP_COLUMNS
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            logger.info(f"Colunas desnecessárias removidas: {existing_cols_to_drop}")

        categorical_features = [col for col in settings.model.CATEGORICAL_FEATURES if col in df.columns]
        numerical_features = [col for col in df.columns if col not in categorical_features]

        logger.info(f"Features Categóricas para One-Hot Encoding: {categorical_features}")
        logger.info(f"Features Numéricas para Scaling: {numerical_features}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )

        logger.info("Iniciando o processo de fit e transform das features.")
        processed_features = preprocessor.fit_transform(df)

        new_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        processed_cols = numerical_features + list(new_cat_features)

        df_processed = pd.DataFrame(processed_features, columns=processed_cols, index=df.index)
        logger.info("Processamento de features concluído.")

        if y is not None:
            df_final = pd.concat([y.reset_index(drop=True), df_processed.reset_index(drop=True)], axis=1)
        else:
            df_final = df_processed

        logger.info(f"Salvando dados processados em {output_path}")
        df_final.to_csv(output_path, index=False)

    except Exception as e:
        logger.error(f"Um erro inesperado ocorreu durante o processamento dos dados: {e}", exc_info=True)
        raise
