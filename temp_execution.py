import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import shap

# Configurações de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# Carregamento dos dados
diretorio_raiz_projeto = '/home/douglas-souza/PycharmProjects/Fiscal_Stability_Prediction'
caminho_dados = os.path.join(diretorio_raiz_projeto, 'data', '04_features', 'featured_data.csv')
df = pd.read_csv(caminho_dados)

# Correção de valores infinitos
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"Dataset carregado com sucesso de: {caminho_dados}")
print(f"Dimensões do dataset após correção: {df.shape}")

# Análise Exploratória de Dados (EDA)
print("Estatísticas descritivas das colunas numéricas:")
print(df.describe().round(2))

# Matriz de correlação
plt.figure(figsize=(18, 12))
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlação', fontsize=16)
plt.show()

# Pré-processamento e Pipeline de Modelagem
TARGET = 'Public Debt (% of GDP)'
FEATURES = [col for col in df.columns if col not in [TARGET, 'country_name', 'country_id']]
X = df[FEATURES]
y = df[TARGET]

ano_divisao = 2021
X_train = X[X['year'] < ano_divisao].drop('year', axis=1)
y_train = y[X['year'] < ano_divisao]
X_test = X[X['year'] >= ano_divisao].drop('year', axis=1)
y_test = y[X['year'] >= ano_divisao]

# Otimização e Treinamento do Modelo Final
pipeline_rf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_leaf': [1, 2, 4]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(pipeline_rf, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

# Avaliação Final e Análise de Erros
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Visualização: Predições vs. Valores Reais
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
plt.title('Predições vs. Valores Reais')
plt.xlabel('Valores Reais')
plt.ylabel('Predições')
plt.show()

# Análise de Interpretabilidade com SHAP
explainer = shap.TreeExplainer(best_model.named_steps['model'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Simulação de Cenários de Estresse
def simular_cenario_estresse(pais, modelo, df_historico, anos_simulacao, choque_gdp):
    # Função para recalcular features para um novo ano
    def proximo_ano_features(df_ano_anterior, pred_divida, choque_gdp):
        novo_ano = df_ano_anterior.copy()
        novo_ano['year'] += 1
        novo_ano['Public Debt (% of GDP)_lag1'] = pred_divida
        novo_ano['GDP Growth (% Annual)_lag1'] = novo_ano['GDP Growth (% Annual)']
        novo_ano['Inflation (CPI %)_lag1'] = novo_ano['Inflation (CPI %)']
        novo_ano['GDP Growth (% Annual)'] = choque_gdp
        return novo_ano

    # Isolar dados e preparar para simulação
    dados_pais = df_historico[df_historico['country_name'] == pais].sort_values(by='year')
    ultimo_ano_dados = dados_pais.iloc[-1:]
    
    # Simular cenários
    resultados = []
    cenarios = {'Base': ultimo_ano_dados['GDP Growth (% Annual)'].values[0], 'Estresse': choque_gdp}
    for nome_cenario, valor_choque in cenarios.items():
        df_sim = ultimo_ano_dados.copy()
        for _ in range(anos_simulacao):
            pred = modelo.predict(df_sim[X_train.columns])[0]
            resultados.append([pais, df_sim['year'].values[0] + 1, nome_cenario, pred])
            df_sim = proximo_ano_features(df_sim, pred, valor_choque)

    # Plotar resultados
    df_resultados = pd.DataFrame(resultados, columns=['País', 'Ano', 'Cenário', 'Dívida (% PIB)'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_resultados, x='Ano', y='Dívida (% PIB)', hue='Cenário', marker='o')
    plt.title(f'Simulação de Cenário de Estresse para {pais}')
    plt.ylabel('Dívida Pública Prevista (% do PIB)')
    plt.grid(True)
    plt.show()

# Executar simulação para o Brasil com um choque de -2% no PIB
simular_cenario_estresse('Brazil', best_model, df, 3, -2.0)