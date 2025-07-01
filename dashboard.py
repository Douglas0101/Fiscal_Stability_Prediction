# =============================================================================
# DASHBOARD DE FORECASTING DE ESTABILIDADE FISCAL (VERSÃO FINAL)
# Projeto: Predição de Estabilidade Fiscal e Risco Soberano
# =============================================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import re
from src import config

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Forecasting Fiscal",
    page_icon="🔮",
    layout="wide"
)


# --- Funções de Carregamento (com cache para performance) ---
@st.cache_data
def load_model():
    """Carrega o modelo treinado."""
    try:
        model = joblib.load(config.MODEL_FILE)
        return model
    except FileNotFoundError:
        return None


@st.cache_data
def load_data():
    """Carrega os dados com features e os dados brutos."""
    try:
        df_featured = pd.read_csv(config.FEATURED_DATA_FILE)
        df_raw = pd.read_csv(config.RAW_DATA_FILE)
        return df_featured, df_raw
    except FileNotFoundError:
        return None, None


def generate_future_forecasts(country_df, model, years_to_forecast, sanitized_cols_map, model_feature_names):
    """
    Gera previsões futuras de forma recursiva.
    """
    last_known_data = country_df.iloc[[-1]].copy()
    future_predictions = []
    sanitized_to_original = {v: k for k, v in sanitized_cols_map.items()}

    for year_offset in range(1, years_to_forecast + 1):
        input_data = last_known_data.copy()
        target_sanitized = sanitized_cols_map[config.TARGET_VARIABLE]

        # =====================================================================
        # CORREÇÃO CRÍTICA: Garante que o dataframe de previsão (X_predict)
        # tenha exatamente as mesmas colunas que o modelo espera.
        # Isso resolve o erro de incompatibilidade de features.
        # =====================================================================
        X_predict = input_data[model_feature_names]

        prediction = model.predict(X_predict)[0]
        future_predictions.append(
            {'year': last_known_data['year'].iloc[0] + year_offset, 'Previsão Futura': prediction})

        # ATUALIZAÇÃO RECURSIVA
        new_row = last_known_data.copy()
        new_row['year'] += 1
        new_row[target_sanitized] = prediction

        # Atualiza lags
        for col_s, col_o in sanitized_to_original.items():
            if col_o in config.LAG_FEATURES:
                # Usa os nomes sanitizados para as operações
                lag1_s = f'{col_s}_lag1'
                if lag1_s in new_row.columns:
                    new_row[lag1_s] = last_known_data[col_s]
                for p in config.LAG_PERIODS:
                    if p > 1:
                        lag_p_s = f'{col_s}_lag{p}'
                        prev_lag_s = f'{col_s}_lag{p - 1}'
                        if lag_p_s in new_row.columns:
                            new_row[lag_p_s] = last_known_data[prev_lag_s]

        last_known_data = new_row

    return pd.DataFrame(future_predictions)


# --- Carregamento Inicial ---
model = load_model()
df_featured, df_raw = load_data()

# --- Título e Descrição do Dashboard ---
st.title("🔮 Dashboard de Forecasting de Estabilidade Fiscal")
st.markdown("...")  # Descrição omitida para brevidade

# --- Verificação de Erros ---
if model is None or df_featured is None or df_raw is None:
    st.error("**Erro Crítico:** Artefatos do modelo não encontrados. Execute o pipeline `main.py` primeiro.")
else:
    # --- Barra Lateral de Filtros ---
    st.sidebar.header("Controlos de Forecasting")
    country_map = df_raw[['country_id', 'country_name']].drop_duplicates().set_index('country_id')['country_name']
    country_name_list = country_map.sort_values().unique()
    selected_country_name = st.sidebar.selectbox("Selecione um País:", options=country_name_list)
    selected_country_id = country_map[country_map == selected_country_name].index[0]

    years_to_forecast = st.sidebar.slider("Anos a Prever no Futuro:", 1, 10, 5)

    # --- Lógica de Previsão e Exibição ---
    st.header(f"Análise e Forecasting para: {selected_country_name}")

    country_data_hist = df_featured[df_featured['country_id'] == selected_country_id].copy()

    if country_data_hist.empty:
        st.warning("Não foram encontrados dados suficientes para este país.")
    else:
        # Pega a lista de features diretamente do modelo treinado
        model_feature_names = model.feature_name_

        # Prepara dados históricos para previsão e visualização
        sanitized_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in country_data_hist.columns}
        country_data_hist.rename(columns=sanitized_cols, inplace=True)
        target_sanitized = sanitized_cols[config.TARGET_VARIABLE]

        # Garante que X_hist também tenha as colunas corretas
        X_hist = country_data_hist[model_feature_names]
        country_data_hist['Previsão Histórica'] = model.predict(X_hist)

        # Gera previsões futuras
        future_df = generate_future_forecasts(country_data_hist, model, years_to_forecast, sanitized_cols,
                                              model_feature_names)

        # --- Visualizações ---
        latest_year = future_df['year'].max()
        latest_prediction = future_df['Previsão Futura'].iloc[-1]
        st.metric(
            label=f"Previsão da Dívida Pública para {latest_year}",
            value=f"{latest_prediction:.2f} % do PIB",
            help="Esta é uma previsão recursiva. A incerteza aumenta para anos mais distantes."
        )

        plot_df_hist = country_data_hist[['year', target_sanitized, 'Previsão Histórica']]
        plot_df_hist = plot_df_hist.rename(columns={target_sanitized: 'Dívida Real'})

        fig = px.line(plot_df_hist, x='year', y=['Dívida Real', 'Previsão Histórica'], template='plotly_white')
        fig.add_scatter(x=future_df['year'], y=future_df['Previsão Futura'], mode='lines', name='Previsão Futura',
                        line=dict(dash='dot'))

        fig.update_layout(
            title=f'Dívida Pública para {selected_country_name}: Histórico e Previsão Futura',
            xaxis_title='Ano',
            yaxis_title='Dívida Pública (% do PIB)',
            legend_title_text=''
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver dados detalhados da previsão futura"):
            st.dataframe(future_df.set_index('year'))

