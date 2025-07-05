import pandas as pd
import logging
import joblib
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import re
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.pytorch_models import TabularDataset, SimpleMLP
from src.config import settings

logger = logging.getLogger(__name__)


def train_model(data_path: str, model_choice: str):
    """
    Carrega os dados, seleciona, treina, avalia, explica e salva o modelo e os seus relatórios.
    """
    try:
        logger.info(f"Carregando dados processados de: {data_path}")
        df = pd.read_csv(data_path)

        target_col = settings.model.TARGET_VARIABLE
        if target_col not in df.columns:
            raise ValueError(f"ERRO: A coluna alvo '{target_col}' não foi encontrada.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
        )

        if model_choice in ['lgbm', 'xgb']:
            # (Lógica existente sem alterações)
            logger.info(f"Limpando nomes das colunas para compatibilidade com o modelo {model_choice.upper()}...")

            def clean_col_names(df_to_clean):
                cols = df_to_clean.columns
                new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in cols]
                df_to_clean.columns = new_cols
                return df_to_clean

            X_train = clean_col_names(X_train)
            X_test = clean_col_names(X_test)
            logger.info("Nomes das colunas limpos com sucesso.")

        model = None
        model_save_path = ""
        report_dict = {}

        if model_choice == 'pytorch':
            params = settings.model.MODEL_PARAMS.get('pytorch', {})

            logger.info("Padronizando os dados de entrada para o PyTorch (Média 0, Desvio Padrão 1).")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            y_train_np = y_train.values
            y_test_np = y_test.values

            # --- OTIMIZAÇÃO DE MEMÓRIA E PROCESSAMENTO PARALELO ---
            # num_workers > 0 ativa o carregamento de dados em subprocessos
            # pin_memory=True acelera a transferência de dados para a GPU (se disponível)
            train_loader = DataLoader(
                TabularDataset(X_train_scaled, y_train_np),
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=2,  # Ajuste este número com base nos cores da sua CPU
                pin_memory=True
            )
            test_loader = DataLoader(
                TabularDataset(X_test_scaled, y_test_np),
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            input_size = X_train_scaled.shape[1]
            pytorch_model = SimpleMLP(
                input_size=input_size,
                hidden_size_1=params['hidden_size_1'],
                hidden_size_2=params['hidden_size_2'],
                dropout_rate=params['dropout_rate']
            )

            num_neg = np.bincount(y_train_np)[0]
            num_pos = np.bincount(y_train_np)[1]
            pos_weight = torch.tensor([(num_neg / num_pos)], dtype=torch.float32)
            logger.info(f"Classes desbalanceadas. Aplicando peso de {pos_weight.item():.2f} para a classe positiva.")

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # --- OTIMIZADOR E AGENDADOR DE TAXA DE APRENDIZAGEM ---
            optimizer = optim.AdamW(pytorch_model.parameters(), lr=params['learning_rate'],
                                    weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

            logger.info(f"Iniciando o treinamento do modelo: {pytorch_model.__class__.__name__}")

            for epoch in range(params['epochs']):
                pytorch_model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = pytorch_model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)

                    loss.backward()
                    # --- GRADIENT CLIPPING ---
                    torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(), max_norm=1.0)

                    optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_loader)
                scheduler.step(avg_epoch_loss)  # Atualiza o agendador com base na perda da época

                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {avg_epoch_loss:.4f}")

            # (Lógica de avaliação permanece a mesma)
            pytorch_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = torch.sigmoid(pytorch_model(batch_X).squeeze())
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(batch_y.numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            model_save_path = os.path.join(settings.MODEL_PATH, "trained_model_pytorch.pt")
            torch.save(pytorch_model.state_dict(), model_save_path)
        else:
            # (Lógica para os outros modelos permanece a mesma)
            models = {
                'rf': RandomForestClassifier(**settings.model.MODEL_PARAMS.get('rf', {})),
                'lgbm': LGBMClassifier(**settings.model.MODEL_PARAMS.get('lgbm', {})),
                'xgb': XGBClassifier(**settings.model.MODEL_PARAMS.get('xgb', {}))
            }
            model = models.get(model_choice)
            if model is None:
                raise ValueError(f"Modelo '{model_choice}' não é uma opção válida.")

            logger.info(f"Iniciando o treinamento do modelo: {model.__class__.__name__}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            model_filename = f"trained_model_{model_choice}.joblib"
            model_save_path = os.path.join(settings.MODEL_PATH, model_filename)
            joblib.dump(model, model_save_path)

        # (Lógica de salvar relatórios e SHAP permanece a mesma)
        report_str = classification_report(y_test, model.predict(X_test),
                                           zero_division=0) if model else classification_report(all_labels, all_preds,
                                                                                                zero_division=0)
        logger.info(f"--- Resultados para o modelo: {model_choice.upper()} ---")
        logger.info(f"Acurácia: {accuracy:.4f}")
        logger.info(f"Relatório de Classificação:\n{report_str}")

        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        logger.info(f"Salvando o modelo treinado em: {model_save_path}")
        logger.info("Modelo salvo com sucesso.")

        logger.info("Salvando o relatório de classificação em formato JSON.")
        os.makedirs(settings.REPORTS_PATH, exist_ok=True)
        report_json_filename = f"classification_report_{model_choice}.json"
        report_json_save_path = os.path.join(settings.REPORTS_PATH, report_json_filename)
        with open(report_json_save_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"Relatório JSON salvo em: {report_json_save_path}")

        if model and model_choice != 'pytorch':
            logger.info("Iniciando a análise de importância de features com SHAP.")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure()
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
            report_filename = f"feature_importance_{model_choice}.png"
            report_save_path = os.path.join(settings.REPORTS_PATH, report_filename)
            plt.savefig(report_save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de importância de features salvo em: {report_save_path}")

    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante o treinamento: {e}", exc_info=True)
        raise
