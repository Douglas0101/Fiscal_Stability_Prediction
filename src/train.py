# src/train.py (Versão Final Corrigida)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from src import config


def train_model(df: pd.DataFrame):
    """
    Treina, avalia e salva o modelo de machine learning.

    Args:
        df (pd.DataFrame): O DataFrame final com todas as features.
    """
    # --- Preparação para Modelagem ---
    print("--- Preparando Dados para Modelagem ---")

    # CORREÇÃO: Sanitizar nomes das colunas para o LightGBM
    original_target_name = config.TARGET_VARIABLE
    sanitized_columns = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns}
    df.rename(columns=sanitized_columns, inplace=True)

    target_variable_sanitized = sanitized_columns[original_target_name]
    print(f"Nomes das colunas sanitizados. Novo nome da variável alvo: {target_variable_sanitized}")

    # =============================================================================
    # CORREÇÃO CRÍTICA: Remover TODAS as colunas não-numéricas e de ID
    # antes de definir X e y, utilizando a lista COLS_TO_DROP do config.
    # =============================================================================

    # Mapeia os nomes originais das colunas a serem dropadas para os seus nomes sanitizados
    cols_to_drop_sanitized = [sanitized_columns.get(col) for col in config.COLS_TO_DROP if col in sanitized_columns]

    # Garante que a variável alvo também está na lista de colunas a serem removidas de X
    if target_variable_sanitized not in cols_to_drop_sanitized:
        cols_to_drop_sanitized.append(target_variable_sanitized)

    print(f"Colunas a serem removidas do conjunto de features: {cols_to_drop_sanitized}")

    # Define features (X) e variável alvo (y)
    y = df[target_variable_sanitized]
    X = df.drop(columns=cols_to_drop_sanitized)

    # Verifica se alguma coluna não-numérica ainda existe em X
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        raise TypeError(f"Erro: Colunas não-numéricas encontradas no conjunto de features: {non_numeric_cols}")

    # Dividir em treino e teste com base no ano
    train_mask = df['year'] < config.TEST_SPLIT_YEAR
    test_mask = df['year'] >= config.TEST_SPLIT_YEAR

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

    # --- Treinamento do Modelo ---
    print("\n--- Treinando o Modelo LightGBM ---")
    lgbm = lgb.LGBMRegressor(random_state=config.RANDOM_STATE)
    lgbm.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")

    # Salvar o modelo treinado
    joblib.dump(lgbm, config.MODEL_FILE)

    # --- Avaliação do Modelo ---
    print("\n--- Avaliando o Modelo ---")
    y_pred = lgbm.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # --- Salvando Artefatos ---
    print("\n--- Salvando Artefatos Finais ---")

    # Salvar resultados em JSON
    feature_importance_list = [int(x) for x in lgbm.feature_importances_]
    results = {
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'parameters': {'test_split_year': config.TEST_SPLIT_YEAR, 'random_state': config.RANDOM_STATE},
        'feature_importance': dict(zip(X.columns, feature_importance_list))
    }
    with open(config.JSON_REPORT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    # Salvar gráfico de importância das features
    feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, X.columns)), columns=['Valor', 'Feature'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Valor", y="Feature", data=feature_imp.sort_values(by="Valor", ascending=False))
    plt.title('Importância das Features', fontsize=16)
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{config.REPORTS_DIR}/feature_importance.png")
    plt.close()

    # Criar e salvar relatório em Markdown
    report_md = f"""
# Relatório de Modelo - Predição de Estabilidade Fiscal
## Resumo do Projeto
Este relatório documenta o processo de treinamento e avaliação de um modelo para prever a `{original_target_name}`.
## Métricas de Avaliação
| Métrica | Valor |
|---|---|
| MAE | {mae:.2f} |
| RMSE | {rmse:.2f} |
| R² | {r2:.2f} |
## Importância das Features
![Feature Importance](feature_importance.png)
"""
    with open(config.MD_REPORT_FILE, 'w') as f:
        f.write(report_md)
