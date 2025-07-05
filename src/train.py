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
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# Supondo que estes módulos existam na sua estrutura de projeto
# from .pytorch_models import TabularDataset, SimpleMLP
# from .config import settings

# --- Módulos de Exemplo (para o código ser executável) ---
class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class Settings:
    PROCESSED_DATA_PATH = 'processed_data.csv'  # Path de exemplo
    REPORTS_PATH = 'reports'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    class model:
        TARGET_VARIABLE = 'target'
        MODEL_PARAMS = {
            'rf': {'n_estimators': 100, 'random_state': 42},
            'lgbm': {'n_estimators': 100, 'random_state': 42},
            'xgb': {'n_estimators': 100, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'},
            'pytorch': {'epochs': 10, 'lr': 0.001, 'batch_size': 32, 'hidden_size': 64}
        }


settings = Settings()
# ---------------------------------------------------------


# Configuração do Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Configura a conexão com o servidor MLflow."""
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    mlflow.set_experiment("Fiscal Stability Prediction")
    logger.info("MLflow experiment set to 'Fiscal Stability Prediction'")


def log_classification_metrics(report_dict: dict):
    """Extrai e loga métricas do relatório de classificação para o MLflow."""
    if 'accuracy' in report_dict:
        mlflow.log_metric("accuracy", report_dict['accuracy'])
    for class_or_avg, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                # Evita logar a métrica de suporte com nome de classe numérico
                metric_name = re.sub(r'[^A-Za-z_]+', '', class_or_avg)
                mlflow.log_metric(f"{metric_name}_{metric}", value)


class ModelTrainer(ABC):
    """Classe base abstrata para treinadores de modelo."""

    def __init__(self, X_train, y_train, X_test, y_test, params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.model = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        pass

    def get_report(self, y_true, y_pred) -> dict:
        """Gera um relatório de classificação."""
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


class SklearnTrainer(ModelTrainer):
    """Treinador para modelos compatíveis com Scikit-learn."""

    def __init__(self, model_class, **kwargs):
        super().__init__(**kwargs)
        self.model = model_class(**self.params)
        # Sanitiza nomes de colunas para LGBM e XGB
        if model_class in [LGBMClassifier, XGBClassifier]:
            self.X_train.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in self.X_train.columns]
            self.X_test.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in self.X_test.columns]

    def train(self):
        logger.info(f"Training {self.model.__class__.__name__}...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training complete.")

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return self.get_report(self.y_test, y_pred)

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        # Loga o modelo usando o formato pyfunc para maior portabilidade
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path=f"model",
            registered_model_name=registered_model_name
        )

        # Gera e loga o gráfico de importância de features (SHAP)
        logger.info("Generating and logging SHAP feature importance plot.")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)

        # Para classificação binária, use shap_values[1] para a classe positiva
        shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure()
        shap.summary_plot(shap_values_for_plot, self.X_test, plot_type="bar", show=False, max_display=20)
        shap_plot_path = os.path.join(temp_dir, f"feature_importance.png")
        plt.savefig(shap_plot_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_plot_path, "feature_importance")


class PyTorchTrainer(ModelTrainer):
    """Treinador para o modelo SimpleMLP com PyTorch."""

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training PyTorch model on {device}.")

        input_size = self.X_train.shape[1]
        self.model = SimpleMLP(input_size, hidden_size=self.params['hidden_size']).to(device)

        train_dataset = TabularDataset(self.X_train.values, self.y_train.values)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.params['batch_size'], shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
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
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())

        return self.get_report(self.y_test, np.array(all_preds))

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        logger.warning("SHAP plot for PyTorch models is not implemented in this script.")


def orchestrate_training(model_choice: str):
    """
    Orquestra o processo completo: carrega dados, treina o modelo escolhido
    e loga todos os resultados e artefatos no MLflow.
    """
    setup_mlflow()

    # Criar dados de exemplo se não existirem
    if not os.path.exists(settings.PROCESSED_DATA_PATH):
        logger.info("Creating dummy processed data file.")
        n_samples, n_features = 1000, 20
        X, y = np.random.rand(n_samples, n_features), np.random.randint(0, 2, n_samples)
        df_dummy = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df_dummy[settings.model.TARGET_VARIABLE] = y
        df_dummy.to_csv(settings.PROCESSED_DATA_PATH, index=False)

    logger.info(f"Loading processed data from: {settings.PROCESSED_DATA_PATH}")
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)

    target_col = settings.model.TARGET_VARIABLE
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )

    # Pré-processamento com StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    trainers = {
        'rf': (SklearnTrainer, {'model_class': RandomForestClassifier}),
        'lgbm': (SklearnTrainer, {'model_class': LGBMClassifier}),
        'xgb': (SklearnTrainer, {'model_class': XGBClassifier}),
        'pytorch': (PyTorchTrainer, {})
    }

    if model_choice not in trainers:
        raise ValueError(f"Model '{model_choice}' is not a valid option. Choose from {list(trainers.keys())}")

    with mlflow.start_run(run_name=f"train_{model_choice}") as run:
        logger.info(f"Starting MLflow run '{run.info.run_name}' for model: {model_choice}")
        mlflow.log_param("model_type", model_choice)

        params = settings.model.MODEL_PARAMS.get(model_choice, {})
        mlflow.log_params(params)

        # Usar um diretório temporário para artefatos
        with tempfile.TemporaryDirectory() as temp_dir:
            # Logar o scaler
            scaler_path = os.path.join(temp_dir, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, "preprocessor")

            # Seleciona o X apropriado (scaled para pytorch, original para árvores)
            X_train_final = X_train_scaled if model_choice == 'pytorch' else X_train
            X_test_final = X_test_scaled if model_choice == 'pytorch' else X_test

            # Instancia e treina o modelo
            TrainerClass, extra_args = trainers[model_choice]
            trainer_instance = TrainerClass(
                X_train=X_train_final, y_train=y_train,
                X_test=X_test_final, y_test=y_test,
                params=params,
                **extra_args
            )

            trainer_instance.train()
            report_dict = trainer_instance.evaluate()
            log_classification_metrics(report_dict)

            # Loga modelo e artefatos específicos
            registered_model_name = f"fiscal-stability-{model_choice}"
            trainer_instance.log_model_and_artifacts(registered_model_name, temp_dir)

            # Loga o relatório de classificação
            report_json_path = os.path.join(temp_dir, "classification_report.json")
            with open(report_json_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
            mlflow.log_artifact(report_json_path, "reports")

        logger.info(f"Run for model {model_choice} completed and logged to MLflow.")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        model_to_train = sys.argv[1]
        orchestrate_training(model_to_train)
    else:
        logger.warning("No model specified. Training XGBoost by default. Use 'rf', 'lgbm', 'xgb', or 'pytorch'.")
        orchestrate_training('xgb')