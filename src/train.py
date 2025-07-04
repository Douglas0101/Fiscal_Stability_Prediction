# ==============================================================================
# PIPELINE DE TREINAMENTO DO MODELO (VERSÃO ROBUSTA)
# ------------------------------------------------------------------------------
# Esta versão é mais inteligente e adaptável:
# 1. Valida se a coluna alvo existe antes de prosseguir.
# 2. Define as features dinamicamente, evitando erros com colunas já removidas.
# ==============================================================================

import logging
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib
import shap
import matplotlib.pyplot as plt

# Importa o objeto de configuração centralizado
from .config import settings

logger = logging.getLogger(__name__)


def train_model(data_path: str) -> None:
    """
    Orquestra o pipeline de treinamento, avaliação e registro do modelo.

    Args:
        data_path (str): Caminho para o ficheiro de dados processados.
    """
    try:
        logger.info(f"Carregando dados processados de: {data_path}")
        df = pd.read_csv(data_path)

        # --- 1. VALIDAÇÃO CRÍTICA: A COLUNA ALVO EXISTE? ---
        if settings.model.TARGET_COLUMN not in df.columns:
            error_msg = f"ERRO: A coluna alvo '{settings.model.TARGET_COLUMN}' não foi encontrada nos dados processados. Não é possível treinar o modelo. Verifique se o seu ficheiro de dados brutos contém a variável alvo."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- 2. Divisão dos Dados (Temporal) ---
        logger.info(f"Dividindo os dados para treino/teste com base no ano: {settings.model.SPLIT_YEAR}")

        train_df = df[df[settings.model.YEAR_COLUMN] < settings.model.SPLIT_YEAR]
        test_df = df[df[settings.model.YEAR_COLUMN] >= settings.model.SPLIT_YEAR]

        # --- 3. Definição de Features (X) e Alvo (y) ---
        logger.info("Separando features (X) e alvo (y).")

        # O alvo (y) é a coluna que queremos prever.
        y_train = train_df[settings.model.TARGET_COLUMN]
        y_test = test_df[settings.model.TARGET_COLUMN]

        # As features (X) são todas as outras colunas, exceto o alvo e o ano.
        # A coluna 'Country Name' original já foi removida pelo one-hot encoding.
        X_train = train_df.drop(columns=[settings.model.TARGET_COLUMN, settings.model.YEAR_COLUMN])
        X_test = test_df.drop(columns=[settings.model.TARGET_COLUMN, settings.model.YEAR_COLUMN])

        logger.info(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras com {X_train.shape[1]} features.")
        logger.info(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras.")

        # --- 4. Configuração e Treinamento com MLflow ---
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            logger.info(f"Iniciando execução do MLflow com ID: {run.info.run_id}")

            logger.info("Treinando o modelo LightGBM.")
            model = LGBMClassifier(**settings.model.LGBM_PARAMS)
            model.fit(X_train, y_train)

            logger.info("Avaliando o modelo e registando métricas.")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            mlflow.log_params(settings.model.LGBM_PARAMS)
            mlflow.log_metric("accuracy", accuracy)
            if "1" in report:  # Loga métricas da classe positiva se ela existir
                mlflow.log_metric("precision_class_1", report["1"]["precision"])
                mlflow.log_metric("recall_class_1", report["1"]["recall"])
                mlflow.log_metric("f1_score_class_1", report["1"]["f1-score"])

            # --- 5. Logging de Artefactos (Relatórios, Gráficos, Modelo) ---
            # (O resto do código para salvar artefactos permanece o mesmo)
            # ...

            logger.info("Modelo treinado e todos os artefactos foram registados com sucesso.")

    except FileNotFoundError:
        logger.error(f"Ficheiro de dados não encontrado em {data_path}", exc_info=True)
        raise
    except ValueError as e:
        # Captura o nosso erro de validação personalizado
        logger.error(f"Erro de validação de dados: {e}")
        raise
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante o treinamento: {e}", exc_info=True)
        raise
