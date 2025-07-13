# src/train.py (Versão Completa, Robusta e Alinhada com a API e o Docker Compose)

import argparse
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import requests
import shap
from interpret.glassbox import ExplainableBoostingClassifier
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier  # Exemplo, pode ser removido se não usado
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Assumindo a existência de um logger centralizado em src/logger_config.py
# Se não o tiver, pode substituir pela configuração básica do logging.
try:
    from logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# --- 1. GESTÃO DE CONFIGURAÇÃO ---
class TrainingConfig(BaseModel):
    """Configurações validadas para o pipeline de treino."""
    processed_data_path: str = Field(default='data/02_processed/processed_data.csv')
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42)
    target_variable: str = Field(default='fiscal_stability_index')
    mlflow_experiment_name: str = Field(default='Fiscal Stability Prediction')
    mlflow_tracking_uri: str = Field(default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    compute_shap: bool = Field(default=True, description="Flag para controlar o cálculo SHAP.")


# --- 2. ABSTRAÇÃO DE DADOS ---
class DataManager:
    """Responsável por todo o acesso e manipulação de dados."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Carregando dados de: {self.config.processed_data_path}")
        if not os.path.exists(self.config.processed_data_path):
            logger.error(f"Ficheiro de dados não encontrado em: {self.config.processed_data_path}")
            raise FileNotFoundError(f"Ficheiro de dados não encontrado: {self.config.processed_data_path}")
        return pd.read_csv(self.config.processed_data_path)

    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Separando features e a variável alvo '{self.config.target_variable}'.")
        X = df.drop(columns=[self.config.target_variable])
        y = df[self.config.target_variable]
        # Garante que os nomes das colunas são strings (importante para alguns modelos)
        X.columns = [str(col) for col in X.columns]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(
            f"Dividindo os dados: {1 - self.config.test_size:.0%} para treino, {self.config.test_size:.0%} para teste.")
        return train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y  # Essencial para classificação desbalanceada
        )


# --- 3. PADRÃO STRATEGY PARA MODELOS ---
class ModelStrategy(ABC):
    """Interface para as diferentes estratégias de treino de modelos."""

    def __init__(self, random_state: int):
        self.random_state = random_state
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> Any:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_model(self) -> Any:
        return self.model

    def log_specific_artifacts(self, model: Any, preprocessor: Pipeline, X_test: pd.DataFrame, temp_dir: str):
        """Método para registar artefactos específicos do modelo (opcional)."""
        pass


class XGBoostStrategy(ModelStrategy):
    def _create_model(self) -> XGBClassifier:
        return XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')

    def get_name(self) -> str:
        return "xgb"


class EBMStrategy(ModelStrategy):
    def _create_model(self) -> ExplainableBoostingClassifier:
        return ExplainableBoostingClassifier(random_state=self.random_state)

    def get_name(self) -> str:
        return "ebm"

    def log_specific_artifacts(self, model: Any, preprocessor: Pipeline, X_test: pd.DataFrame, temp_dir: str):
        logger.info("Gerando artefactos de explicabilidade nativos do EBM.")
        try:
            global_explanation = model.explain_global()
            explanation_path = os.path.join(temp_dir, "ebm_global_explanation.html")
            global_explanation.write_html(explanation_path)
            mlflow.log_artifact(explanation_path, "ebm_explainability")
        except Exception as e:
            logger.warning(f"Não foi possível gerar a explicação global do EBM: {e}")


# --- 4. ABSTRAÇÃO DE TRACKING DE EXPERIMENTOS ---
class MLflowExperimentTracker:
    """Wrapper para o MLflow, com lógica de retry para robustez."""

    def __init__(self, config: TrainingConfig, max_retries: int = 5, initial_wait_sec: int = 5):
        self.config = config
        self._connect_with_retry(max_retries, initial_wait_sec)

    def _connect_with_retry(self, max_retries: int, initial_wait_sec: int):
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        for attempt in range(max_retries):
            try:
                mlflow.set_experiment(self.config.mlflow_experiment_name)
                logger.info(
                    f"MLflow conectado com sucesso: URI='{self.config.mlflow_tracking_uri}', Experimento='{self.config.mlflow_experiment_name}'")
                return
            except (requests.exceptions.ConnectionError, mlflow.exceptions.MlflowException) as e:
                wait_time = initial_wait_sec * (2 ** attempt)
                logger.warning(f"Não foi possível conectar ao MLflow (Tentativa {attempt + 1}/{max_retries}). "
                               f"Servidor pode estar a iniciar. Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
        logger.error(f"Não foi possível estabelecer conexão com o MLflow após {max_retries} tentativas.")
        raise ConnectionError("Falha ao conectar ao servidor MLflow.")

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics_from_report(self, report: Dict[str, Any]):
        accuracy = report.pop('accuracy', 0.0)
        self.log_metric("accuracy", accuracy)
        for class_or_avg, metrics in report.items():
            if isinstance(metrics, dict):
                clean_name = re.sub(r'[^A-Za-z0-9_]+', '', class_or_avg).strip()
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{clean_name}_{metric_name}", value)

    def log_artifact(self, local_path: str, artifact_path: str):
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model: Pipeline, model_name: str):
        # O MLflow agora regista o pipeline inteiro, o que é mais robusto
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # Diretório padrão do MLflow para o modelo
            registered_model_name=model_name
        )


# --- 5. GESTÃO DE ARTEFACTOS ---
class ArtifactManager:
    """Lida com a criação e o registo de todos os artefactos de um run."""

    def __init__(self, tracker: MLflowExperimentTracker, config: TrainingConfig):
        self.tracker = tracker
        self.config = config

    def save_and_log_artifacts(self, pipeline: Pipeline, model_strategy: ModelStrategy, X_test: pd.DataFrame,
                               y_test: pd.Series, feature_names: List[str]):
        report = classification_report(y_test, pipeline.predict(X_test), output_dict=True, zero_division=0)
        logger.info(f"\n{classification_report(y_test, pipeline.predict(X_test), zero_division=0)}")
        self.tracker.log_metrics_from_report(report)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Artefacto 1: Relatório de Classificação
            report_path = os.path.join(temp_dir, "classification_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            self.tracker.log_artifact(report_path, "reports")

            # Artefacto 2: Nomes das Features (essencial para a API)
            features_path = os.path.join(temp_dir, "feature_names.json")
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            self.tracker.log_artifact(features_path, "features")

            # Artefacto 3: Gráfico de Importância (SHAP)
            if self.config.compute_shap:
                self._log_shap_plot(pipeline, X_test, feature_names, temp_dir)
            else:
                logger.info("Cálculo de SHAP desativado. Pulando esta etapa.")

            # Artefacto 4: Artefactos Específicos do Modelo (ex: EBM)
            model_strategy.log_specific_artifacts(
                pipeline.named_steps['classifier'],
                pipeline.named_steps['preprocessor'],
                X_test,
                temp_dir
            )

            # Registo do Modelo no Model Registry do MLflow
            # O nome aqui é crucial para a API o encontrar.
            model_name = f"fiscal-stability-{model_strategy.get_name()}"
            logger.info(f"Registando o pipeline completo como o modelo '{model_name}' no MLflow...")
            self.tracker.log_model(pipeline, model_name)
            logger.info(f"Modelo '{model_name}' registado com sucesso.")

    def _log_shap_plot(self, pipeline: Pipeline, X_test: pd.DataFrame, feature_names: List[str], temp_dir: str):
        try:
            logger.info("Iniciando cálculo de explicabilidade (SHAP)...")
            start_time = time.time()

            classifier = pipeline.named_steps['classifier']
            X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)

            # SHAP é sensível a dataframes. É mais seguro converter para numpy.
            if isinstance(X_test_processed, pd.DataFrame):
                X_test_processed = X_test_processed.values

            explainer = shap.Explainer(classifier, X_test_processed)
            shap_values = explainer(X_test_processed)

            duration = time.time() - start_time
            logger.info(f"Cálculo SHAP concluído em {duration:.2f} segundos.")
            mlflow.log_metric("shap_calculation_duration_sec", duration)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, features=X_test_processed, feature_names=feature_names, plot_type="bar",
                              show=False)
            plt.tight_layout()

            shap_plot_path = os.path.join(temp_dir, "shap_summary_plot.png")
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()
            self.tracker.log_artifact(shap_plot_path, "feature_importance")
        except Exception as e:
            logger.warning(f"Não foi possível gerar o gráfico SHAP. Erro: {e}", exc_info=True)


# --- 6. ORQUESTRAÇÃO DO TREINO ---
class TrainingOrchestrator:
    """Orquestra o pipeline de treino injetando as dependências necessárias."""

    def __init__(self, data_manager: DataManager, model_strategy: ModelStrategy, tracker: MLflowExperimentTracker,
                 artifact_manager: ArtifactManager):
        self.data_manager = data_manager
        self.model_strategy = model_strategy
        self.tracker = tracker
        self.artifact_manager = artifact_manager

    def run(self):
        logger.info("--- INICIANDO PIPELINE DE TREINO ---")
        t0 = time.time()

        df = self.data_manager.load_data()
        X, y = self.data_manager.get_features_and_target(df)
        X_train, X_test, y_train, y_test = self.data_manager.split_data(X, y)

        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.model_strategy.get_model())
        ])

        run_name = f"TrainRun_{self.model_strategy.get_name().upper()}_{int(time.time())}"
        with self.tracker.start_run(run_name):
            self.tracker.log_params({
                "model_strategy": type(self.model_strategy).__name__,
                "model_name": self.model_strategy.get_name(),
                "feature_count": len(X.columns),
                "training_data_shape": str(X_train.shape),
                "testing_data_shape": str(X_test.shape),
                "compute_shap": self.artifact_manager.config.compute_shap
            })

            logger.info(f"Iniciando treino do modelo '{type(self.model_strategy.get_model()).__name__}'...")
            t1 = time.time()
            full_pipeline.fit(X_train, y_train)
            train_duration = time.time() - t1
            logger.info(f"Treino do modelo concluído em {train_duration:.2f} segundos.")
            mlflow.log_metric("training_duration_sec", train_duration)

            logger.info("Avaliando modelo e guardando artefactos...")
            t2 = time.time()
            self.artifact_manager.save_and_log_artifacts(full_pipeline, self.model_strategy, X_test, y_test,
                                                         list(X.columns))
            logger.info(f"Gestão de artefactos concluída em {time.time() - t2:.2f} segundos.")

        logger.info(f"--- PIPELINE DE TREINO CONCLUÍDO COM SUCESSO EM {time.time() - t0:.2f} SEGUNDOS ---")


# --- 7. PONTO DE ENTRADA DA EXECUÇÃO ---
def main():
    """Ponto de entrada do script, lida com argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Treinar e avaliar modelos de previsão de estabilidade fiscal.")
    parser.add_argument("model_type", type=str, choices=['xgb', 'ebm'],
                        help="O tipo de modelo a ser treinado ('xgb' ou 'ebm').")
    parser.add_argument("--no-shap", action="store_true",
                        help="Desativa o cálculo dos valores SHAP para acelerar o treino.")
    args = parser.parse_args()

    config = TrainingConfig(compute_shap=not args.no_shap)

    strategies = {
        'xgb': XGBoostStrategy,
        'ebm': EBMStrategy,
    }

    strategy_class = strategies.get(args.model_type)
    if not strategy_class:
        raise ValueError(f"Estratégia de modelo '{args.model_type}' não suportada.")

    # Injeção de Dependências
    data_manager = DataManager(config)
    model_strategy = strategy_class(random_state=config.random_state)
    tracker = MLflowExperimentTracker(config)
    artifact_manager = ArtifactManager(tracker, config)
    orchestrator = TrainingOrchestrator(data_manager, model_strategy, tracker, artifact_manager)

    orchestrator.run()


if __name__ == '__main__':
    # Este bloco é executado quando o script é chamado diretamente.
    # Ex: python src/train.py xgb
    main()