# src/train.py (Versão Final com EBM Corrigido de Verdade)

# Standard library imports
import argparse
import json
import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod

# Third-party imports
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field, ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

# --- CORREÇÃO FINAL E VERIFICADA ---
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import save  # A função 'save' está no namespace principal.


# ... (Todo o resto do seu código permanece exatamente o mesmo, pois já estava correto) ...
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray): self.X = torch.tensor(X,
                                                                            dtype=torch.float32); self.y = torch.tensor(
        y, dtype=torch.long)

    def __len__(self) -> int: return len(self.y)

    def __getitem__(self, idx: int): return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 3, hidden_size: int = 64):
        super().__init__();
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)


class Settings:
    PROCESSED_DATA_PATH: str = 'data/02_processed/processed_data.csv';
    TEST_SIZE: float = 0.2;
    RANDOM_STATE: int = 42;
    SHAP_SAMPLE_SIZE: int = 100

    class model:
        TARGET_VARIABLE: str = 'fiscal_stability_index'
        MODEL_PARAMS: dict = {'rf': {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 2, 'random_state': 42},
                              'lgbm': {'n_estimators': 200, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': 42},
                              'xgb': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.8,
                                      'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'mlogloss'},
                              'pytorch': {'epochs': 20, 'lr': 0.001, 'batch_size': 32, 'hidden_size': 64},
                              'ebm': {'interactions': 0, 'random_state': 42}}


settings = Settings()


class DataRow(BaseModel):
    fiscal_stability_index: int;
    year: float;
    inflation_cpi: float = Field(alias='Inflation (CPI %)');
    gdp_current_usd: float = Field(alias='GDP (Current USD)');
    gdp_per_capita: float = Field(alias='GDP per Capita (Current USD)');
    unemployment_rate: float = Field(alias='Unemployment Rate (%)');
    interest_rate_real: float = Field(alias='Interest Rate (Real, %)');
    inflation_gdp_deflator: float = Field(alias='Inflation (GDP Deflator, %)');
    gdp_growth_annual: float = Field(alias='GDP Growth (% Annual)');
    current_account_balance_gdp: float = Field(alias='Current Account Balance (% GDP)');
    government_expense_gdp: float = Field(alias='Government Expense (% of GDP)');
    tax_revenue_gdp: float = Field(alias='Tax Revenue (% of GDP)');
    gross_national_income_usd: float = Field(alias='Gross National Income (USD)')

    class Config: extra = 'allow'; populate_by_name = True


def validate_data(df: pd.DataFrame) -> bool:
    try:
        _ = [DataRow.model_validate(row) for row in df.to_dict(orient='records')]; logger.info(
            "Validação de dados concluída com sucesso."); return True
    except ValidationError as e:
        logger.error(f"Validação de dados falhou:\n{e}"); return False


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy();
    df_copy.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df_copy.columns];
    return df_copy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s');
logger = logging.getLogger(__name__)


def setup_mlflow():
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000");
    mlflow.set_tracking_uri(mlflow_tracking_uri);
    mlflow.set_experiment("Fiscal Stability Prediction");
    logger.info(f"MLflow configurado: URI='{mlflow_tracking_uri}'")


def log_classification_metrics(report_dict: dict):
    if 'accuracy' in report_dict: mlflow.log_metric("accuracy", report_dict.pop('accuracy'))
    for class_or_avg, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items(): mlflow.log_metric(
                f"{re.sub(r'[^A-Za-z0-9_]+', '', class_or_avg).strip()}_{metric}", value)


class ModelTrainer(ABC):
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                 params: dict):
        self.X_train, self.y_train, self.X_test, self.y_test, self.params = X_train, y_train, X_test, y_test, params;
        self.model = None

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def evaluate(self) -> dict: pass

    @abstractmethod
    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str): pass

    def get_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict: return classification_report(y_true, y_pred,
                                                                                                       output_dict=True,
                                                                                                       zero_division=0)


class SklearnTrainer(ModelTrainer):
    def __init__(self, model_class, use_smote: bool = False, **kwargs):
        super().__init__(**kwargs)
        pipeline_steps = [('imputer', SimpleImputer(strategy='median'))]
        if use_smote: pipeline_steps.append(('smote', SMOTE(random_state=settings.RANDOM_STATE)))
        pipeline_steps.append(('classifier', model_class(**self.params)))
        self.model = Pipeline(pipeline_steps)
        logger.info(f"SklearnTrainer configurado com o pipeline: {[step[0] for step in self.model.steps]}")

    def train(self):
        logger.info(f"Treinando Pipeline: {[step[0] for step in self.model.steps]}...");
        self.model.fit(self.X_train, self.y_train);
        logger.info("Treinamento completo.")

    def evaluate(self) -> dict:
        return self.get_report(self.y_test, self.model.predict(self.X_test))

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        mlflow.sklearn.log_model(sk_model=self.model, artifact_path="model",
                                 registered_model_name=registered_model_name)
        logger.info("Gerando e logando gráfico de importância de features (SHAP).")
        trained_classifier = self.model.named_steps['classifier']
        X_test_imputed = pd.DataFrame(self.model.named_steps['imputer'].transform(self.X_test),
                                      columns=self.X_test.columns)
        try:
            explainer = shap.TreeExplainer(trained_classifier);
            shap_values = explainer.shap_values(X_test_imputed)
            plt.figure(figsize=(10, 8));
            shap.summary_plot(shap_values, X_test_imputed, plot_type="bar", show=False, max_display=20,
                              class_names=['Estável', 'Alerta', 'Instável'])
            shap_plot_path = os.path.join(temp_dir, "feature_importance.png");
            plt.savefig(shap_plot_path, bbox_inches='tight');
            plt.close()
            mlflow.log_artifact(shap_plot_path, "feature_importance")
        except Exception as e:
            logger.warning(
                f"Não foi possível gerar o gráfico SHAP TreeExplainer para {trained_classifier.__class__.__name__}. Pode não ser um modelo de árvore. Erro: {e}")


class EBMTrainer(SklearnTrainer):
    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        super().log_model_and_artifacts(registered_model_name, temp_dir)
        logger.info("Gerando e logando artefatos de explicabilidade nativos do EBM.")
        ebm_model = self.model.named_steps['classifier']
        global_explanation = ebm_model.explain_global()
        explanation_html_path = os.path.join(temp_dir, "ebm_global_explanation.html")
        # --- CORREÇÃO FINAL: Chamada da função correta 'save' ---
        save(global_explanation, file_name=explanation_html_path)
        mlflow.log_artifact(explanation_html_path, "ebm_explainability")
        logger.info(f"Painel de explicabilidade global do EBM salvo e logado em: {explanation_html_path}")


class PyTorchTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs);
        self.model = None

    def train(self):
        imputer = SimpleImputer(strategy='median').fit(self.X_train);
        X_train_imputed = imputer.transform(self.X_train)
        device = "cuda" if torch.cuda.is_available() else "cpu";
        logger.info(f"Treinando modelo PyTorch em '{device}'.")
        input_size, num_classes = X_train_imputed.shape[1], len(self.y_train.unique())
        self.model = SimpleMLP(input_size, num_classes=num_classes, hidden_size=self.params['hidden_size']).to(device)
        train_dataset = TabularDataset(X_train_imputed, self.y_train.values)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.model.train()
        for epoch in range(self.params['epochs']):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device);
                optimizer.zero_grad()
                outputs = self.model(batch_X);
                loss = criterion(outputs, batch_y);
                loss.backward();
                optimizer.step()
            if (epoch + 1) % 5 == 0: logger.info(
                f"Epoch [{epoch + 1}/{self.params['epochs']}], Loss: {loss.item():.4f}")

    def evaluate(self) -> dict:
        imputer = SimpleImputer(strategy='median').fit(self.X_train);
        X_test_imputed = imputer.transform(self.X_test)
        device = "cuda" if torch.cuda.is_available() else "cpu";
        test_dataset = TabularDataset(X_test_imputed, self.y_test.values)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.params['batch_size']);
        self.model.eval();
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device);
                outputs = self.model(batch_X);
                _, predicted_labels = torch.max(outputs, 1)
                all_preds.extend(predicted_labels.cpu().numpy())
        return self.get_report(self.y_test.values, np.array(all_preds))

    def log_model_and_artifacts(self, registered_model_name: str, temp_dir: str):
        mlflow.pytorch.log_model(pytorch_model=self.model, artifact_path="model",
                                 registered_model_name=registered_model_name)


def orchestrate_training(model_choice: str, n_trials: int = 10, use_smote: bool = False):
    setup_mlflow();
    logger.info(f"Carregando dados processados de: {settings.PROCESSED_DATA_PATH}")
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    if not validate_data(df): raise ValueError("Pipeline interrompido: falha na validação dos dados.")

    X, y = df.drop(columns=[settings.model.TARGET_VARIABLE]), df[settings.model.TARGET_VARIABLE]
    X = sanitize_column_names(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE,
                                                        random_state=settings.RANDOM_STATE, stratify=y)

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            model_class, model_params, extra_params = None, {}, {'random_state': settings.RANDOM_STATE}
            if model_choice == 'ebm':
                model_class, model_params = ExplainableBoostingClassifier, {
                    'interactions': trial.suggest_int('interactions', 0, 5),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'max_bins': trial.suggest_int('max_bins', 128, 512)}
            elif model_choice == 'rf':
                model_class, model_params = RandomForestClassifier, {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)}
            elif model_choice == 'lgbm':
                model_class, model_params = LGBMClassifier, {'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                                                             'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                                                             'learning_rate': trial.suggest_float('learning_rate', 0.01,
                                                                                                  0.3, log=True)}
            elif model_choice == 'xgb':
                model_class, model_params = XGBClassifier, {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)}
                extra_params.update({'use_label_encoder': False, 'eval_metric': 'mlogloss'})
            pipeline_steps = [('imputer', SimpleImputer(strategy='median'))]
            if use_smote: pipeline_steps.append(('smote', SMOTE(random_state=settings.RANDOM_STATE)))
            pipeline_steps.append(('classifier', model_class(**extra_params)))
            model_to_train = Pipeline(pipeline_steps)
            pipeline_params = {f'classifier__{k}': v for k, v in model_params.items()}
            model_to_train.set_params(**pipeline_params)
            mlflow.log_params(model_params)
            model_to_train.fit(X_train, y_train)
            accuracy = classification_report(y_test, model_to_train.predict(X_test), output_dict=True)['accuracy']
            mlflow.log_metric('accuracy', accuracy)
            return accuracy

    run_name = f"Train_Run_{model_choice}" + ("_SMOTE" if use_smote else "")
    with mlflow.start_run(run_name=run_name) as parent_run:
        logger.info(f"Iniciando run principal do MLflow '{parent_run.info.run_name}'")
        best_params = settings.model.MODEL_PARAMS.get(model_choice, {}).copy()
        is_optimizable = model_choice in ['rf', 'lgbm', 'xgb', 'ebm']
        if is_optimizable and n_trials > 0:
            logger.info(f"Iniciando otimização com Optuna ({n_trials} trials)...")
            study = optuna.create_study(direction='maximize', study_name=f"{model_choice}_opt_{use_smote}")
            study.optimize(objective, n_trials=n_trials)
            best_params.update(study.best_params);
            mlflow.log_params({"best_optuna_params": study.best_params});
            mlflow.log_metric("best_optuna_accuracy", study.best_value)
            logger.info(
                f"Otimização concluída. Melhor acurácia: {study.best_value:.4f}, Melhores parâmetros: {study.best_params}")
        else:
            logger.info("Pulando otimização. Usando parâmetros padrão.")

        mlflow.log_params({"final_model_params": best_params});
        mlflow.log_param("model_type", model_choice);
        mlflow.log_param("used_smote", use_smote)

        logger.info("Treinando modelo final com os melhores parâmetros...")
        trainers = {
            'rf': (SklearnTrainer, {'model_class': RandomForestClassifier, 'use_smote': use_smote}),
            'lgbm': (SklearnTrainer, {'model_class': LGBMClassifier, 'use_smote': use_smote}),
            'xgb': (SklearnTrainer, {'model_class': XGBClassifier, 'use_smote': use_smote}),
            'ebm': (EBMTrainer, {'model_class': ExplainableBoostingClassifier, 'use_smote': use_smote}),
            'pytorch': (PyTorchTrainer, {})
        }
        TrainerClass, trainer_kwargs = trainers[model_choice]
        # PyTorch lida com seus próprios dados, os outros usam os dados brutos
        X_train_final, X_test_final = (X_train, X_test)
        if model_choice == 'pytorch':
            # PyTorch precisa de um tratamento especial de dados (imputação + scaling)
            imputer = SimpleImputer(strategy='median').fit(X_train);
            X_train_imputed = imputer.transform(X_train);
            X_test_imputed = imputer.transform(X_test)
            scaler = StandardScaler().fit(X_train_imputed);
            X_train_final = scaler.transform(X_train_imputed);
            X_test_final = scaler.transform(X_test_imputed)

        final_trainer = TrainerClass(X_train=X_train_final, y_train=y_train, X_test=X_test_final, y_test=y_test,
                                     params=best_params, **trainer_kwargs)
        final_trainer.train()

        report_dict = final_trainer.evaluate()
        log_classification_metrics(report_dict)

        with tempfile.TemporaryDirectory() as temp_dir:
            features_path = os.path.join(temp_dir, "feature_names.json")
            with open(features_path, 'w') as f: json.dump(list(X.columns), f)
            mlflow.log_artifact(features_path, "features")
            registered_model_name = f"fiscal-stability-{model_choice}" + ("-smote" if use_smote else "")
            final_trainer.log_model_and_artifacts(registered_model_name, temp_dir)
            report_json_path = os.path.join(temp_dir, "classification_report.json")
            with open(report_json_path, 'w') as f: json.dump(report_dict, f, indent=4)
            mlflow.log_artifact(report_json_path, "reports")

        logger.info(f"Run para o modelo {model_choice} concluído e logado com sucesso no MLflow.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinar e avaliar modelos de previsão de estabilidade fiscal.")
    parser.add_argument("model", type=str, choices=['rf', 'lgbm', 'xgb', 'pytorch', 'ebm'],
                        help="O tipo de modelo a ser treinado.")
    parser.add_argument("--optimize", type=int, default=0, metavar='N_TRIALS',
                        help="Número de trials para a otimização com Optuna (0 para desativar).")
    parser.add_argument("--smote", action='store_true',
                        help="Ativar o uso de SMOTE para balanceamento de classes (apenas para modelos sklearn).")
    args = parser.parse_args()

    if args.model == 'pytorch' and args.smote:
        logger.warning("SMOTE não está implementado para o PyTorchTrainer. A flag --smote será ignorada.")
        args.smote = False

    orchestrate_training(model_choice=args.model, n_trials=args.optimize, use_smote=args.smote)