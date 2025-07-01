# src/train.py

import pandas as pd
import lightgbm as lgb
import optuna
import functools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import re
from datetime import datetime
from src.config import SPLIT_YEAR, ENTITY_COLUMN


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa os nomes das colunas de um DataFrame para serem compatíveis com o LightGBM.
    Substitui todos os caracteres que não são letras, números ou underscore por '_'.
    """
    print("\nSanitizando nomes de colunas para compatibilidade com o modelo...")

    clean_columns = {}
    sample_original_col = next((col for col in df.columns if re.search(r'[^A-Za-z0-9_]', col)), None)

    for col in df.columns:
        clean_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        clean_columns[col] = clean_col

    if sample_original_col:
        print(f"  - Exemplo de transformação: '{sample_original_col}' -> '{clean_columns[sample_original_col]}'")

    df = df.rename(columns=clean_columns)
    print("Sanitização de nomes de colunas concluída.")
    return df


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Função objetivo que o Optuna tentará otimizar (minimizar o RMSE).
    Define o espaço de busca dos hiperparâmetros e avalia o modelo usando
    validação cruzada temporal.
    """
    # Adicionada uma lógica para busca inteligente de hiperparâmetros
    max_depth = trial.suggest_int('max_depth', 4, 12)

    param = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 400, 2500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'max_depth': max_depth,
        # Restringe num_leaves para um valor válido e eficiente em relação à profundidade
        'num_leaves': trial.suggest_int('num_leaves', 10, 2 ** max_depth - 1),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Validação cruzada mais robusta com 7 splits
    tscv = TimeSeriesSplit(n_splits=7)
    scores_rmse = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(20, verbose=False)])  # Aumentado o early stopping

        preds = model.predict(X_val)
        score = root_mean_squared_error(y_val, preds)
        scores_rmse.append(score)

        trial.report(score, i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return sum(scores_rmse) / len(scores_rmse)


def save_reports(report_path: str, metrics: dict, params: dict):
    """
    Salva os resultados do treinamento em arquivos JSON e Markdown.
    """
    print("\n6. Salvando relatórios de performance...")
    os.makedirs(report_path, exist_ok=True)

    json_path = os.path.join(report_path, 'resultados_modelo.json')
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'hyperparameters': params
    }
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    print(f"   - Relatório JSON salvo em: {json_path}")

    md_path = os.path.join(report_path, 'relatorio_modelo.md')
    md_content = f"""
# Relatório de Performance do Modelo de Estabilidade Fiscal

- **Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Métricas de Avaliação (Conjunto de Teste)

| Métrica | Valor |
|---|---|
| R² (R-squared) | {metrics['r2']:.4f} |
| RMSE (Root Mean Squared Error) | {metrics['rmse']:.4f} |
| MAE (Mean Absolute Error) | {metrics['mae']:.4f} |

## Hiperparâmetros Otimizados (Optuna)

```json
{json.dumps(params, indent=4)}
```
"""
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"   - Relatório Markdown salvo em: {md_path}")


def train_model(data_path: str, model_output_path: str, report_output_path: str):
    """
    Orquestra o processo de treinamento: carrega dados, otimiza hiperparâmetros com Optuna,
    treina o modelo final, avalia e salva os artefatos e relatórios.
    """
    print("--- Iniciando o processo de treinamento com Calibração Final de Alta Performance ---")
    df = pd.read_csv(data_path)

    print(f"\n1. Aplicando One-Hot Encoding na coluna '{ENTITY_COLUMN}'...")
    df_encoded = pd.get_dummies(df, columns=[ENTITY_COLUMN], prefix=ENTITY_COLUMN)

    df_sanitized = sanitize_feature_names(df_encoded)

    train_df = df_sanitized[df_sanitized['year'] < SPLIT_YEAR]
    test_df = df_sanitized[df_sanitized['year'] >= SPLIT_YEAR]

    if train_df.empty or test_df.empty:
        raise ValueError(f"O conjunto de treino ou teste está vazio após a divisão no ano {SPLIT_YEAR}.")

    print(f"\n2. Dados divididos: {len(train_df)} amostras de treino, {len(test_df)} amostras de teste.")

    X_train = train_df.drop(columns=['target', 'year'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target', 'year'])
    y_test = test_df['target']

    print("\n3. Iniciando a busca de hiperparâmetros com Optuna (Busca Intensiva)...")
    study_objective = functools.partial(objective, X=X_train, y=y_train)

    sampler = optuna.samplers.TPESampler(seed=42)
    # Aumentado o n_warmup_steps devido ao maior número de splits
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
                                sampler=sampler)

    # Busca mais ampla com 300 trials
    study.optimize(study_objective, n_trials=300, timeout=1800)  # Timeout de 30 minutos

    print(f"\n   -> Otimização concluída. Número de trials finalizados: {len(study.trials)}")

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print(f"   -> Trials completos: {len(complete_trials)}, Trials podados: {len(pruned_trials)}")

    print("   -> Melhor trial encontrado:")
    best_trial = study.best_trial
    print(f"      - Valor (RMSE médio na validação cruzada): {best_trial.value:.4f}")
    print("      - Melhores Hiperparâmetros:")
    for key, value in best_trial.params.items():
        print(f"        - {key}: {value}")

    print("\n4. Treinando o modelo final com os melhores hiperparâmetros em todo o conjunto de treino...")
    best_params = best_trial.params
    best_params['random_state'] = 42
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'rmse'

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    final_metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}

    print(f'\n5. Métricas de Avaliação Finais (no conjunto de teste):')
    print(f'   - RMSE: {final_metrics["rmse"]:.4f}')
    print(f'   - MAE:  {final_metrics["mae"]:.4f}')
    print(f'   - R²:   {final_metrics["r2"]:.4f}')

    save_reports(report_output_path, final_metrics, best_params)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_model, model_output_path)
    print(f"\n--- Modelo otimizado com Optuna salvo com sucesso em: {model_output_path} ---")

