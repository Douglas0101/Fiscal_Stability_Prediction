
# Relatório de Modelo - Predição de Estabilidade Fiscal

## Resumo do Projeto
Este relatório documenta o processo de treinamento e avaliação de um modelo de machine learning para prever a Dívida Pública (`Public Debt (% of GDP)`).

## Métricas de Avaliação
O modelo foi avaliado no conjunto de teste (anos >= 2021).

| Métrica | Valor |
|---|---|
| Mean Absolute Error (MAE) | 23.08 |
| Root Mean Squared Error (RMSE) | 29.03 |
| R-squared (R²) | -0.50 |

## Análise
- O **MAE** de 23.08 indica que, em média, as previsões do modelo erram em 23.08 pontos percentuais do PIB.
- O **R²** de -0.50 sugere que o modelo consegue explicar aproximadamente -50.3% da variância na dívida pública.

## Importância das Features
O gráfico abaixo mostra as features que mais influenciaram as previsões do modelo.

![Feature Importance](reports/feature_importance.png)
*(Nota: Salve o gráfico de importância das features como 'feature_importance.png' no diretório 'reports/')*
