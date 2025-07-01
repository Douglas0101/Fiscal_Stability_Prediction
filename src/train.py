# src/train.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# CORREÇÃO SOFISTICADA: Importa a função recomendada para RMSE, eliminando o FutureWarning.
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import re  # Importa a biblioteca de expressões regulares para a sanitização
from src.config import SPLIT_YEAR, ENTITY_COLUMN, RESULTS_JSON_PATH, REPORT_MD_PATH


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa os nomes das colunas de um DataFrame para serem compatíveis com o LightGBM.
    Substitui todos os caracteres que não são letras, números ou underscore por '_'.
    Esta é a abordagem padrão e recomendada pela indústria.
    """
    print("\nSanitizando nomes de colunas para compatibilidade com o modelo...")

    clean_columns = {}
    # Pega um nome de coluna de exemplo para mostrar a transformação
    sample_original_col = next((col for col in df.columns if re.search(r'[^A-Za-z0-9_]', col)), None)

    for col in df.columns:
        # Substitui qualquer caractere problemático por um underscore
        clean_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        clean_columns[col] = clean_col

    if sample_original_col:
        print(f"  - Exemplo de transformação: '{sample_original_col}' -> '{clean_columns[sample_original_col]}'")

    df = df.rename(columns=clean_columns)
    print("Sanitização de nomes de colunas concluída.")
    return df


def train_model(data_path: str, model_output_path: str, report_output_path: str):
    """
    Carrega os dados, sanitiza nomes de colunas, aplica One-Hot Encoding,
    treina um modelo LightGBM, avalia e salva os resultados.
    """
    print("Iniciando o processo de treinamento...")
    df = pd.read_csv(data_path)

    print(f"Aplicando One-Hot Encoding na coluna '{ENTITY_COLUMN}'...")
    df_encoded = pd.get_dummies(df, columns=[ENTITY_COLUMN], prefix=ENTITY_COLUMN)

    # --- CORREÇÃO DEFINITIVA: Sanitiza todos os nomes de colunas ANTES de treinar ---
    df_sanitized = sanitize_feature_names(df_encoded)

    # Divisão temporal
    train_df = df_sanitized[df_sanitized['year'] < SPLIT_YEAR]
    test_df = df_sanitized[df_sanitized['year'] >= SPLIT_YEAR]

    if train_df.empty or test_df.empty:
        raise ValueError(f"O conjunto de treino ou teste está vazio após a divisão no ano {SPLIT_YEAR}.")

    print(f"\nDados divididos: {len(train_df)} amostras de treino, {len(test_df)} amostras de teste.")

    # A coluna 'target' já é um nome limpo e não será alterada pela sanitização.
    X_train = train_df.drop(columns=['target', 'year'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target', 'year'])
    y_test = test_df['target']

    # Treinamento com GridSearchCV
    lgb_reg = lgb.LGBMRegressor(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50]}
    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(estimator=lgb_reg, param_grid=param_grid, cv=tscv, scoring='r2', n_jobs=-1,
                               error_score='raise')

    print("\nIniciando a busca de hiperparâmetros com GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # --- AVALIAÇÃO COM TÉCNICAS ATUAIS ---
    # Usa a função recomendada 'root_mean_squared_error' para evitar o FutureWarning.
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\nMétricas de Avaliação:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}')

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"\nModelo salvo em: {model_output_path}")
