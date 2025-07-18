# -*- coding: utf-8 -*-
"""
Script de Treino Modular para Classificação Binária de Estabilidade Fiscal.

Este script implementa um pipeline completo de Machine Learning para treinar,
avaliar e otimizar diferentes modelos de classificação (LGBM, XGBoost, EBM)
para prever a estabilidade fiscal.
"""
import argparse
import logging
import os
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, classification_report, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Responsável por carregar, dividir e preparar os dados.
    """

    def __init__(self, data_path: str, target: str, test_size: float, random_state: int):
        self.data_path = data_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados a partir do caminho especificado e valida a presença da coluna alvo."""
        logger.info(f"Carregando dados de '{self.data_path}'...")
        if not os.path.exists(self.data_path):
            logger.error(f"Arquivo não encontrado em '{self.data_path}'")
            raise FileNotFoundError(f"Arquivo não encontrado em '{self.data_path}'")

        df = pd.read_csv(self.data_path)

        if self.target not in df.columns:
            error_msg = (
                f"A coluna alvo '{self.target}' não foi encontrada no arquivo de dados "
                f"'{self.data_path}'. Verifique se os scripts de pré-processamento "
                f"(como create_target.py) foram executados corretamente."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide os dados em conjuntos de treino e teste."""
        logger.info(f"Dividindo os dados em treino ({1 - self.test_size:.0%}) e teste ({self.test_size:.0%})...")
        X = df.drop(columns=[self.target])
        y = df[self.target].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Divisão concluída. Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras.")
        return X_train, X_test, y_train, y_test


class ModelTrainer:
    """
    Orquestra o pipeline de treino, otimização e avaliação do modelo.
    """

    def __init__(self, model_name: str, config: Dict[str, Any], random_state: int):
        self.model_name = model_name
        self.config = config
        self.random_state = random_state
        self.pipeline = None

    def _get_model(self, y_train: pd.Series) -> Any:
        """Instancia o modelo de classificação com base no nome."""
        counts = np.bincount(y_train)
        scale_pos_weight = counts[0] / counts[1] if counts[1] > 0 else 1
        logger.info(f"Calculado 'scale_pos_weight' para o desbalanceamento: {scale_pos_weight:.2f}")

        if self.model_name == 'lgbm':
            return LGBMClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1
            )
        elif self.model_name == 'xgb':
            return XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        elif self.model_name == 'ebm':
            return ExplainableBoostingClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            raise ValueError(f"Modelo '{self.model_name}' não suportado.")

    def tune_and_train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Aplica pré-processamento, SMOTE, e depois treina o modelo com busca de hiperparâmetros.
        """
        numeric_features = self.config['features']['numeric']

        # 1. Define o pré-processador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ],
            remainder='drop'
        )

        # 2. Aplica o pré-processamento aos dados de treino
        logger.info("Aplicando pré-processamento aos dados de treino...")
        X_train_processed = preprocessor.fit_transform(X_train)

        # 3. Aplica SMOTE para rebalancear os dados de treino *após* o pré-processamento
        logger.info("Aplicando SMOTE para rebalancear as classes...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        logger.info(f"Tamanho do dataset após SMOTE: {X_train_resampled.shape[0]} amostras.")

        # 4. Define o modelo e o pipeline final (sem SMOTE, pois já foi aplicado)
        model = self._get_model(y_train)  # Passa o y_train original para calcular o peso

        param_grid = self.config['hyperparameters']

        logger.info(f"Iniciando a busca de hiperparâmetros para o modelo {self.model_name.upper()}...")

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=self.config['tuning']['n_iter'],
            cv=self.config['tuning']['cv'],
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        # Treina o RandomizedSearchCV com os dados já rebalanceados
        random_search.fit(X_train_resampled, y_train_resampled)

        logger.info(f"Melhores hiperparâmetros encontrados: {random_search.best_params_}")
        logger.info(f"Melhor score (AUC-ROC) na validação cruzada: {random_search.best_score_:.4f}")

        # Constrói o pipeline final para inferência (pré-processador + melhor modelo)
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', random_search.best_estimator_)
        ])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Avalia o modelo final no conjunto de teste."""
        if not self.pipeline:
            raise RuntimeError("O modelo deve ser treinado antes da avaliação.")

        logger.info("Avaliando o melhor modelo no conjunto de teste...")
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        logger.info("Relatório de Classificação:\n" + classification_report(y_test, y_pred))

        self._plot_roc_curve(y_test, y_proba)

    def _plot_roc_curve(self, y_test: pd.Series, y_proba: np.ndarray):
        """Gera e salva o gráfico da Curva ROC."""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - Modelo {self.model_name.upper()}')
        plt.legend(loc="lower right")

        output_path = "roc_curve.png"
        plt.savefig(output_path)
        logger.info(f"Curva ROC salva em '{output_path}'")
        plt.close()

    def save_model(self, output_path: str):
        """Salva o pipeline treinado em um arquivo."""
        logger.info(f"Salvando o modelo treinado em '{output_path}'...")
        joblib.dump(self.pipeline, output_path)
        logger.info("Modelo salvo com sucesso.")


def get_model_config() -> Dict[str, Any]:
    """Retorna as configurações para cada modelo."""
    numeric_features = [
        'Inflation (CPI %)', 'GDP Growth (% Annual)', 'Current Account Balance (% GDP)',
        'Government Revenue (% of GDP)', 'Public Debt (% of GDP)', 'Interest Rate (Real, %)'
    ]

    return {
        "features": {"numeric": numeric_features},
        "tuning": {"n_iter": 15, "cv": 5},
        "lgbm": {
            "hyperparameters": {
                'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 40], 'max_depth': [-1, 10, 20],
                'reg_alpha': [0.1, 0.5], 'reg_lambda': [0.1, 0.5],
            }
        },
        "xgb": {
            "hyperparameters": {
                'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7], 'gamma': [0, 0.1, 0.5],
                'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0],
            }
        },
        "ebm": {
            # --- CORREÇÃO APLICADA ---
            # Grade de hiperparâmetros do EBM simplificada para acelerar o treino.
            "hyperparameters": {
                'outer_bags': [8, 12],
                'learning_rate': [0.01, 0.05],
            }
        }
    }


def main():
    """Ponto de entrada principal para o script de treino."""
    parser = argparse.ArgumentParser(
        description="Script de treino para modelos de classificação de estabilidade fiscal.")
    parser.add_argument("--model", type=str, choices=['lgbm', 'xgb', 'ebm'], required=True,
                        help="O tipo de modelo a ser treinado.")
    parser.add_argument("--data-path", type=str, default="data/03_final/final_data.csv",
                        help="Caminho para o arquivo de dados processados.")
    parser.add_argument("--output-model-path", type=str, default="notebooks/models/fiscal_stability_model.joblib",
                        help="Caminho para salvar o modelo treinado.")
    args = parser.parse_args()

    RANDOM_STATE, TEST_SIZE, TARGET_VARIABLE = 42, 0.2, 'fiscal_stability_index'

    full_config = get_model_config()
    model_specific_config = {**full_config[args.model],
                             **{k: v for k, v in full_config.items() if k not in ['lgbm', 'xgb', 'ebm']}}

    data_manager = DataManager(data_path=args.data_path, target=TARGET_VARIABLE, test_size=TEST_SIZE,
                               random_state=RANDOM_STATE)

    df = data_manager.load_data()
    X_train, X_test, y_train, y_test = data_manager.split_data(df)

    trainer = ModelTrainer(model_name=args.model, config=model_specific_config, random_state=RANDOM_STATE)

    trainer.tune_and_train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.save_model(args.output_model_path)

    logger.info("Pipeline de treino concluído com sucesso!")


if __name__ == "__main__":
    main()
