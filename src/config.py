# src/config.py

import os

# --- Definição de Caminhos Base ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# --- Caminhos dos Dados ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# CORREÇÃO: Aponta para o nome de arquivo correto 'world_bank_data_2025.csv'
RAW_DATA_PATH = os.path.join(DATA_DIR, '01_raw', 'world_bank_data_2025.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, '02_processed', 'processed_data.csv')
FEATURED_DATA_PATH = os.path.join(DATA_DIR, '04_features', 'featured_data.csv')

# --- Caminhos do Modelo e Relatórios ---
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fiscal_stability_model.joblib')

NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
REPORTS_PATH = os.path.join(NOTEBOOKS_DIR, 'reports')
RESULTS_JSON_PATH = os.path.join(REPORTS_PATH, 'resultados_modelo.json')
REPORT_MD_PATH = os.path.join(REPORTS_PATH, 'relatorio_modelo.md')

# --- Configurações da Engenharia de Features ---
TARGET_COLUMN = 'Public Debt (% of GDP)'
ENTITY_COLUMN = 'country_name' # Coluna para agrupar dados por entidade (país)
FEATURES_TO_LAG = [
    'Public Debt (% of GDP)',
    'GDP Growth (%)',
    'Inflation Rate (%)',
    'Unemployment Rate (%)',
    'Government Budget Balance (% of GDP)',
    'Current Account Balance (% of GDP)'
]

# --- Configurações do Modelo ---
SPLIT_YEAR = 2021
