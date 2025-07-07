# src/train.py (Versão Completa e Final)

import pandas as pd
import logging
import joblib
import os
import torch
import mlflow
import json
import re
import matplotlib.pyplot as plt
import shap
import numpy as np
import tempfile
import optuna
import argparse
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ValidationError

from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# --- Módulos do PyTorch ---
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X if isinstance(X, np.ndarray) else X.values, dtype=torch.float32)
        self.y = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes=3, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# --- Configurações e Validação de Dados ---
class Settings:
    PROCESSED_DATA_PATH = 'data/02_processed/processed_data.csv'
    REPORTS_PATH = 'reports'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    class model:
        # Nome exato da sua coluna alvo
        TARGET_VARIABLE = 'fiscal_stability_index'
        MODEL_PARAMS = {
            'rf': {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 2, 'random_state': 42},
            'lgbm': {'n_estimators': 200, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
            'xgb': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8, 'random_state': 42,
                    'use_label_encoder': False, 'eval_metric': 'mlogloss'},
            'pytorch': {'epochs': 20, 'lr': 0.001, 'batch_size': 32, 'hidden_size': 64}
        }


settings = Settings()


# Classe DataRow agora corresponde EXATAMENTE às colunas do seu CSV
# Usamos `Field(alias='...')` para nomes de colunas com espaços ou caracteres especiais.
class DataRow(BaseModel):
    fiscal_stability_index: int
    year: float
    inflation_cpi: float = Field(alias='Inflation (CPI %)')
    gdp_current_usd: float = Field(alias='GDP (Current USD)')
    gdp_per_capita: float = Field(alias='GDP per Capita (Current USD)')
    unemployment_rate: float = Field(alias='Unemployment Rate (%)')
    interest_rate_real: float = Field(alias='Interest Rate (Real, %)')
    inflation_gdp_deflator: float = Field(alias='Inflation (GDP Deflator, %)')
    gdp_growth_annual: float = Field(alias='GDP Growth (% Annual)')
    current_account_balance_gdp: float = Field(alias='Current Account Balance (% GDP)')
    government_expense_gdp: float = Field(alias='Government Expense (% of GDP)')
    tax_revenue_gdp: float = Field(alias='Tax Revenue (% of GDP)')
    gross_national_income_usd: float = Field(alias='Gross National Income (USD)')

    class Config:
        extra = 'allow'  # Permite as colunas 'country_id_*' sem precisar declará-las
        populate_by_name = True  # Permite que o Pydantic use os aliases para popular os campos


def validate_data(df: pd.DataFrame) -> bool:
    try:
        _ = [DataRow.model_validate(row) for row in df.to_dict(orient='records')]
        logger.info("Validação de dados concluída com sucesso. O esquema está correto.")
        return True
    except ValidationError as e:
        logger.error(f"Validação de dados falhou. Erros encontrados:\n{e}")
        return False


# --- Logger e Configuração do MLflow ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_mlflow():
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Fiscal Stability Prediction")
    logger.info(f"MLflow configurado: URI='{mlflow_tracking_uri}', Experimento='Fiscal Stability Prediction'")


def log_classification_metrics(report_dict: dict):
    if 'accuracy' in report_dict:
        mlflow.log_metric("accuracy", report_dict.pop('accuracy'))
    for class_or_avg, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                metric_name = re.sub(r'[^A-Za-z0-9_]+', '', class_or_avg).strip()
                mlflow.log_metric(f"{metric_name}_{metric}", value)


# --- Classes de Treinadores de Modelo ---
class ModelTrainer(ABC):
    def __init__(self, X_train, y_train, X_test, y_test, params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.model = None

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def evaluate(self): pass

    @abstractmethod
    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str): pass

    def get_report(self, y_true, y_pred) -> dict:
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


class SklearnTrainer(ModelTrainer):
    def __init__(self, model_class, **kwargs):
        super().__init__(**kwargs)
        # Limpa os nomes das colunas para compatibilidade com LGBM/XGB
        self.X_train.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in self.X_train.columns]
        self.X_test.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in self.X_test.columns]
        self.model = model_class(**self.params)

    def train(self):
        logger.info(f"Treinando {self.model.__class__.__name__}...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("Treinamento completo.")

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return self.get_report(self.y_test, y_pred)

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        logger.info("Gerando e logando gráfico de importância de features (SHAP).")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False, max_display=20,
                          class_names=['Estável', 'Alerta', 'Instável'])
        shap_plot_path = os.path.join(temp_dir, "feature_importance.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_plot_path, "feature_importance")


class PyTorchTrainer(ModelTrainer):
    def train(self):
        global loss
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Treinando modelo PyTorch em '{device}'.")

        input_size = self.X_train.shape[1]
        self.model = SimpleMLP(input_size, num_classes=len(self.y_train.unique())).to(device)

        train_dataset = TabularDataset(self.X_train.values, self.y_train.values)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.params['batch_size'], shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])

        self.model.train()
        for epoch in range(self.params['epochs']):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.params['epochs']}], Loss: {loss.item():.4f}")

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_dataset = TabularDataset(self.X_test.values, self.y_test.values)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.params['batch_size'])

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = self.model(batch_X)
                _, predicted_labels = torch.max(outputs, 1)
                all_preds.extend(predicted_labels.cpu().numpy())
        return self.get_report(self.y_test, np.array(all_preds))

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        logger.info("Gerando e logando gráfico SHAP para PyTorch.")
        device = next(self.model.parameters()).device
        background_data = torch.tensor(self.X_train.values[:100], dtype=torch.float32).to(device)
        test_data_tensor = torch.tensor(self.X_test.values[:100], dtype=torch.float32).to(device)
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(test_data_tensor)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test.head(100), plot_type="bar", show=False, max_display=20,
                          class_names=['Estável', 'Alerta', 'Instável'])
        shap_plot_path = os.path.join(temp_dir, "feature_importance_pytorch.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_plot_path, "feature_importance")


# --- Função Principal de Orquestração ---
def orchestrate_training(model_choice: str, n_trials: int = 10):
    setup_mlflow()

    logger.info(f"Carregando dados processados de: {settings.PROCESSED_DATA_PATH}")
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)

    if not validate_data(df):
        raise ValueError("Pipeline interrompido devido a falha na validação dos dados.")

    target_col = settings.model.TARGET_VARIABLE
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )

    # Identificar features numéricas para scaling (excluindo as colunas 'country_id_*')
    numeric_features = [col for col in X_train.columns if not col.startswith('country_id_')]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    def objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            if model_choice == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': settings.RANDOM_STATE,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                }
                model_class = XGBClassifier
            else:  # Exemplo para RF
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': settings.RANDOM_STATE
                }
                model_class = RandomForestClassifier

            mlflow.log_params(params)
            trainer = SklearnTrainer(model_class, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                     params=params)
            trainer.train()
            report = trainer.evaluate()
            accuracy = report['accuracy']
            mlflow.log_metric('accuracy', accuracy)
            return accuracy

    with mlflow.start_run(run_name=f"Train_Run_{model_choice}") as parent_run:
        logger.info(f"Iniciando run principal do MLflow '{parent_run.info.run_name}' para o modelo: {model_choice}")

        best_params = settings.model.MODEL_PARAMS.get(model_choice, {})
        is_optimizable = model_choice in ['rf', 'lgbm', 'xgb']

        if is_optimizable and n_trials > 0:
            logger.info(f"Iniciando otimização com Optuna ({n_trials} trials)...")
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params.update(study.best_params)
            mlflow.log_params({"best_optuna_params": study.best_params})
            mlflow.log_metric("best_optuna_accuracy", study.best_value)
            logger.info(f"Otimização concluída. Melhor acurácia: {study.best_value:.4f}")
        else:
            logger.info("Pulando otimização de hiperparâmetros.")

        mlflow.log_param("model_type", model_choice)
        mlflow.log_params(best_params)

        logger.info("Treinando modelo final com os melhores parâmetros...")

        trainers = {
            'rf': (SklearnTrainer, {'model_class': RandomForestClassifier, 'X_train': X_train, 'X_test': X_test}),
            'lgbm': (SklearnTrainer, {'model_class': LGBMClassifier, 'X_train': X_train, 'X_test': X_test}),
            'xgb': (SklearnTrainer, {'model_class': XGBClassifier, 'X_train': X_train, 'X_test': X_test}),
            'pytorch': (PyTorchTrainer, {'model_class': None, 'X_train': X_train_scaled, 'X_test': X_test_scaled})
        }

        TrainerClass, extra_args = trainers[model_choice]
        model_class = extra_args.pop('model_class', None)

        final_trainer = TrainerClass(
            model_class=model_class,
            y_train=y_train, y_test=y_test,
            params=best_params, **extra_args
        )
        final_trainer.train()

        report_dict = final_trainer.evaluate()
        log_classification_metrics(report_dict)

        with tempfile.TemporaryDirectory() as temp_dir:
            scaler_path = os.path.join(temp_dir, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, "preprocessor")

            final_feature_names = list(X_train.columns)
            features_path = os.path.join(temp_dir, "feature_names.json")
            with open(features_path, 'w') as f:
                json.dump(final_feature_names, f)
            mlflow.log_artifact(features_path, "features")

            registered_model_name = f"fiscal-stability-{model_choice}"
            final_trainer.log_model_and_artifacts(registered_model_name, temp_dir)

            report_json_path = os.path.join(temp_dir, "classification_report.json")
            with open(report_json_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
            mlflow.log_artifact(report_json_path, "reports")

        logger.info(f"Run para o modelo {model_choice} concluído e logado com sucesso no MLflow.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinar e avaliar modelos de previsão de estabilidade fiscal.")
    parser.add_argument("model", type=str, choices=['rf', 'lgbm', 'xgb', 'pytorch'],
                        help="O tipo de modelo a ser treinado.")
    parser.add_argument("--optimize", type=int, default=0, metavar='N_TRIALS',
                        help="Número de trials para a otimização com Optuna. Defina como 0 para desativar.")
    args = parser.parse_args()

    orchestrate_training(model_choice=args.model, n_trials=args.optimize)