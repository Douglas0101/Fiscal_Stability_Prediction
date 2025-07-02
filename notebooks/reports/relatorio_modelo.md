
# Relatório de Performance do Modelo de Estabilidade Fiscal

- **Data de Geração:** 2025-07-02 21:55:54

## Métricas de Avaliação (Conjunto de Teste)

| Métrica | Valor |
|---|---|
| R² (R-squared) | 0.9699 |
| RMSE (Root Mean Squared Error) | 6.8400 |
| MAE (Mean Absolute Error) | 3.9149 |

## Hiperparâmetros Otimizados (Optuna)

```json
{
    "max_depth": 9,
    "n_estimators": 500,
    "learning_rate": 0.026948022473899044,
    "num_leaves": 95,
    "min_child_samples": 7,
    "subsample": 0.9744427686266666,
    "colsample_bytree": 0.9828160165372797,
    "random_state": 42,
    "objective": "regression_l1",
    "metric": "rmse"
}
```
