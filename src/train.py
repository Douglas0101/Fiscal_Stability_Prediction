import argparse
import logging
import random
from pathlib import Path
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from src.config import AppConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Sementes de aleatoriedade definidas como: {seed}")

class DataManager:
    def __init__(self, config: AppConfig):
        self.config = config.model
        self.data_path = config.final_data_path

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Carregando dados de '{self.data_path}'...")
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado em '{self.data_path}'")
        df = pd.read_csv(self.data_path)
        if self.config.TARGET_VARIABLE not in df.columns:
            raise ValueError(f"A coluna alvo '{self.config.TARGET_VARIABLE}' não foi encontrada.")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Dividindo os dados em treino ({1 - self.config.TEST_SIZE:.0%}) e teste ({self.config.TEST_SIZE:.0%})...")
        X = df.drop(columns=[self.config.TARGET_VARIABLE])
        y = df[self.config.TARGET_VARIABLE].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE, stratify=y)
        logger.info(f"Divisão concluída. Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras.")
        return X_train, X_test, y_train, y_test

class ModelTrainer:
    def __init__(self, model_name: str, config: AppConfig, y_train: pd.Series):
        self.model_name = model_name
        self.config = config
        self.pipeline = None
        model_config = self.config.training.MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Configurações para o modelo '{model_name}' não encontradas no arquivo de configuração.")
        if 'class' not in model_config or 'static_params' not in model_config or 'hyperparameters' not in model_config:
            raise KeyError(f"A configuração para '{model_name}' está incompleta. Requer 'class', 'static_params' e 'hyperparameters'.")
        self.param_grid = model_config['hyperparameters']
        self.model = self._instantiate_model(model_config, y_train)

    def _instantiate_model(self, model_config: dict, y_train: pd.Series) -> Any:
        model_class = model_config['class']
        static_params = model_config['static_params'].copy()
        if self.model_name in ['lgbm', 'xgb']:
            counts = np.bincount(y_train)
            scale_pos_weight = counts[0] / counts[1] if counts[1] > 0 else 1
            logger.info(f"Calculado 'scale_pos_weight' para o desbalanceamento: {scale_pos_weight:.2f}")
            static_params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Instanciando o modelo: {model_class.__name__}")
        return model_class(**static_params)

    def tune_and_train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        mlflow.set_tracking_uri(self.config.mlflow.TRACKING_URI)
        mlflow.set_experiment(self.config.mlflow.EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"train_{self.model_name}") as run:
            logger.info(f"Iniciando run do MLflow: {run.info.run_id}")
            mlflow.log_param("model_name", self.model_name)
            if 'scale_pos_weight' in self.model.get_params():
                mlflow.log_param("scale_pos_weight", f"{self.model.get_params()['scale_pos_weight']:.2f}")
            preprocessor = self._build_preprocessor()
            X_train_resampled, y_train_resampled = self._apply_preprocessing_and_smote(X_train, y_train, preprocessor)
            random_search = self._tune_hyperparameters(X_train_resampled, y_train_resampled)
            self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', random_search.best_estimator_)])
            self.evaluate(X_test, y_test)
            self._plot_roc_curve(X_test, y_test, "roc_curve.png")
            self._plot_feature_importance("feature_importance.png")
            self.save_model_mlflow()
            logger.info(f"Run {run.info.run_id} concluída. Verifique no UI do MLflow.")

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_features = self.config.training.NUMERIC_FEATURES
        return ColumnTransformer(transformers=[('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features)], remainder='drop')

    def _apply_preprocessing_and_smote(self, X_train, y_train, preprocessor) -> Tuple[np.ndarray, pd.Series]:
        logger.info("Aplicando pré-processamento e SMOTE...")
        X_train_processed = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=self.config.model.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        logger.info(f"Tamanho do dataset antes/depois do SMOTE: {X_train_processed.shape[0]}/{X_train_resampled.shape[0]}")
        mlflow.log_metric("training_samples_before_smote", X_train_processed.shape[0])
        mlflow.log_metric("training_samples_after_smote", X_train_resampled.shape[0])
        return X_train_resampled, y_train_resampled

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: pd.Series) -> RandomizedSearchCV:
        logger.info(f"Iniciando RandomizedSearchCV para {self.model_name.upper()}...")
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_grid, n_iter=self.config.training.TUNING_N_ITER, cv=self.config.training.TUNING_CV, scoring='roc_auc', n_jobs=-1, random_state=self.config.model.RANDOM_STATE, verbose=1)
        random_search.fit(X_train, y_train)
        logger.info(f"Melhores hiperparâmetros: {random_search.best_params_}")
        logger.info(f"Melhor score (AUC-ROC CV): {random_search.best_score_:.4f}")
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_roc_auc", random_search.best_score_)
        return random_search

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        logger.info("Avaliando o modelo no conjunto de teste...")
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info("Relatório de Classificação:\n" + classification_report(y_test, y_pred))
        mlflow.log_metric("accuracy", report['accuracy'])
        for class_label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_label}_{metric_name}", value)

    def _plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, output_path: str):
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - Modelo {self.model_name.upper()}'); plt.legend(loc="lower right")
        plt.savefig(output_path)
        mlflow.log_artifact(output_path)
        logger.info(f"Curva ROC salva e logada em '{output_path}'"); plt.close()

    def _plot_feature_importance(self, output_path: str):
        try:
            feature_names = self.config.training.NUMERIC_FEATURES
            classifier = self.pipeline.named_steps['classifier']
            if not hasattr(classifier, 'feature_importances_'):
                logger.warning(f"O modelo {self.model_name} não tem 'feature_importances_'. A pular o gráfico.")
                return
            importances = classifier.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(15)
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Importância das Features - {self.model_name.upper()}'); plt.tight_layout()
            plt.savefig(output_path)
            mlflow.log_artifact(output_path)
            logger.info(f"Gráfico de importância das features salvo e logado em '{output_path}'"); plt.close()
        except Exception as e:
            logger.error(f"Não foi possível gerar o gráfico de importância das features: {e}")

    def save_model_mlflow(self):
        if not self.pipeline:
            raise RuntimeError("O modelo deve ser treinado antes de ser salvo.")
        logger.info(f"Registrando o modelo '{self.model_name}' no MLflow...")
        mlflow.sklearn.log_model(sk_model=self.pipeline, artifact_path=f"model_{self.model_name}", registered_model_name=f"fiscal-stability-{self.model_name}")
        logger.info("Modelo registrado com sucesso no MLflow.")

def main():
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Script de treino para modelos de estabilidade fiscal.")
    parser.add_argument("--model", type=str, required=True, choices=config.training.MODELS.keys(), help="O tipo de modelo a ser treinado.")
    args = parser.parse_args()
    set_seeds(config.model.RANDOM_STATE)
    data_manager = DataManager(config)
    df = data_manager.load_data()
    X_train, X_test, y_train, y_test = data_manager.split_data(df)
    trainer = ModelTrainer(model_name=args.model, config=config, y_train=y_train)
    trainer.tune_and_train(X_train, y_train, X_test, y_test)
    logger.info("Pipeline de treino concluído com sucesso!")

if __name__ == "__main__":
    main()