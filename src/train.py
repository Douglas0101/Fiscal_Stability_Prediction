import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import mlflow
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import config
from . import data_processing

def gerar_relatorios(modelo, metrics, feature_names, X_test_data,
                     output_dir=config.REPORTS_DIR):
    """Gera relatórios de performance e análise em formatos JSON e Markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # Extrair parâmetros do modelo
    params = modelo.named_steps['model'].get_params()

    # Análise SHAP
    explainer = shap.TreeExplainer(modelo.named_steps['model'])
    shap_values = explainer.shap_values(X_test_data)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    report_data = {
        'model_name': type(modelo.named_steps['model']).__name__,
        'model_performance': metrics,
        'hyperparameters': params,
        'feature_importance': feature_importance_df.to_dict(orient='records')
    }

    # Salvar JSON
    with open(config.JSON_REPORT_FILE, 'w') as f:
        json.dump(report_data, f, indent=4, default=lambda x: str(x))
    print(f"Relatório JSON salvo em: {config.JSON_REPORT_FILE}")

    # Gerar Markdown
    md_content = f"""# Relatório do Modelo: {report_data['model_name']}

## 1. Resumo da Performance

- **R-quadrado (R²):** {metrics['r2']:.4f}
- **Erro Médio Absoluto (MAE):** {metrics['mae']:.4f}
- **Raiz do Erro Quadrático Médio (RMSE):** {metrics['rmse']:.4f}

## 2. Hiperparâmetros do Modelo Final

```json
{json.dumps(params, indent=4, default=lambda x: str(x))}
```

## 3. Importância das Features (Top 10)

| Rank | Feature | Importância (SHAP) |
|:----:|:--------|:-------------------|
"""
    for i, row in enumerate(feature_importance_df.head(10).itertuples(), 1):
        md_content += f"| {i} | {row.feature} | {row.importance:.4f} |
"

    with open(config.MD_REPORT_FILE, 'w') as f:
        f.write(md_content)
    print(f"Relatório Markdown salvo em: {config.MD_REPORT_FILE}")


def run_training():
    """Executa o pipeline completo de treinamento, otimização e avaliação."""

    mlflow.set_experiment("Previsao_Estabilidade_Fiscal")

    with mlflow.start_run() as run:
        print(f"Iniciando run do MLflow: {run.info.run_id}")
        mlflow.log_param("test_split_year", config.TEST_SPLIT_YEAR)

        # 1. Processamento de Dados
        df = data_processing.pipeline_completo_de_dados()
        # X e y são definidos aqui para uso no SHAP e avaliação
        X = df.drop(columns=config.COLS_TO_DROP)
        y = df[config.TARGET_VARIABLE]

        X_train = df[df["year"] < config.TEST_SPLIT_YEAR].drop(
            columns=config.COLS_TO_DROP
        )
        y_train = df[df["year"] < config.TEST_SPLIT_YEAR][config.TARGET_VARIABLE]
        X_test = df[df["year"] >= config.TEST_SPLIT_YEAR].drop(
            columns=config.COLS_TO_DROP
        )
        y_test = df[df["year"] >= config.TEST_SPLIT_YEAR][config.TARGET_VARIABLE]

        # 2. Definição dos Pipelines e Grid de Hiperparâmetros
        pipelines = {
            "RandomForest": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=config.RANDOM_STATE)),
                ]
            ),
            "XGBoost": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", XGBRegressor(random_state=config.RANDOM_STATE)),
                ]
            ),
        }
        param_grids = {
            "RandomForest": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20],
                "model__min_samples_leaf": [2, 4],
            },
            "XGBoost": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        }

        # 3. Treinamento e Otimização
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_model = None
        best_model_name = ""

        for model_name in pipelines.keys():
            print(f"\n--- Treinando e otimizando {model_name} ---")
            grid_search = GridSearchCV(
                pipelines[model_name],
                param_grids[model_name],
                cv=tscv,
                scoring="r2",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                print(
                    f"Novo melhor modelo encontrado: {model_name} com R² de {best_score:.4f}"
                )

        # 4. Avaliação, Log e Salvamento
        print(f"\n--- Avaliação Final do Modelo Campeão: {best_model_name} ---")
        y_pred = best_model.predict(X_test)
        final_metrics = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        # Log no MLflow
        mlflow.log_params(best_model.named_steps["model"].get_params())
        mlflow.log_metrics(final_metrics)
        mlflow.set_tag("best_model", best_model_name)

        # Salvar e logar o modelo como artefato
        mlflow.sklearn.log_model(best_model, "model")
        print("Modelo e resultados registrados no MLflow.")

        # Geração de Relatórios
        gerar_relatorios(
            modelo=best_model,
            metrics=final_metrics,
            feature_names=X_test.columns,
            X_test_data=X_test # Passando X_test para a função
        )

if __name__ == "__main__":
    run_training()

