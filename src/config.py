# Arquivo de Configuração para o Projeto de Estabilidade Fiscal

# --- Caminhos de Arquivos ---
DATA_DIR = "data"
RAW_DATA_FILE = f"{DATA_DIR}/01_raw/world_bank_data_2025.csv"
PROCESSED_DATA_FILE = f"{DATA_DIR}/02_processed/processed_data.csv"
FINAL_DATA_FILE = f"{DATA_DIR}/03_final/final_data.csv"
FEATURED_DATA_FILE = f"{DATA_DIR}/04_features/featured_data.csv"

MODELS_DIR = "models"
MODEL_FILE = f"{MODELS_DIR}/fiscal_stability_model.joblib"

REPORTS_DIR = "reports"
JSON_REPORT_FILE = f"{REPORTS_DIR}/resultados_modelo.json"
MD_REPORT_FILE = f"{REPORTS_DIR}/relatorio_modelo.md"

# --- Parâmetros de Modelagem ---
TARGET_VARIABLE = "Public Debt (% of GDP)"
TEST_SPLIT_YEAR = 2021
RANDOM_STATE = 42

# --- Listas de Features ---
# Colunas a serem removidas antes da modelagem
COLS_TO_DROP = [TARGET_VARIABLE, "country_name", "country_id", "year"]

# Features para criar lags
LAG_FEATURES = [
    "Public Debt (% of GDP)",
    "GDP Growth (% Annual)",
    "Inflation (CPI %)",
    "Interest Rate (Real, %)",
    "Current Account Balance (% GDP)",
]

# Features para criar estatísticas móveis
ROLLING_FEATURES = ["GDP Growth (% Annual)", "Inflation (CPI %)"]
