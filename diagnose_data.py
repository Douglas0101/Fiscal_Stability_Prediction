
import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('/home/douglas-souza/PycharmProjects/Fiscal_Stability_Prediction/data/02_processed/processed_data.csv')
    print("Arquivo 'processed_data.csv' carregado com sucesso.")
    print(f"Shape do DataFrame: {df.shape}")
except FileNotFoundError:
    print("Erro: Arquivo 'processed_data.csv' não encontrado.")
    exit()

# --- Verificação 1: Valores Infinitos ---
numeric_cols = df.select_dtypes(include=np.number).columns
inf_check = df[numeric_cols].isin([np.inf, -np.inf]).sum()
inf_cols = inf_check[inf_check > 0]

if not inf_cols.empty:
    print("\n--- [FALHA] Verificação de Valores Infinitos ---")
    print("As seguintes colunas contêm valores infinitos:")
    print(inf_cols)
else:
    print("\n--- [SUCESSO] Verificação de Valores Infinitos: Nenhuma coluna com valores infinitos encontrada. ---")

# --- Verificação 2: Valores Ausentes (NaN) na Variável Alvo ---
target_col = 'fiscal_stability_index'
if target_col in df.columns:
    nan_in_target = df[target_col].isnull().sum()
    if nan_in_target > 0:
        print(f"\n--- [FALHA] Verificação de NaN na Variável Alvo ('{target_col}') ---")
        print(f"Encontrados {nan_in_target} valores ausentes na coluna alvo.")
    else:
        print(f"\n--- [SUCESSO] Verificação de NaN na Variável Alvo: Nenhuma valor ausente encontrado em '{target_col}'. ---")
else:
    print(f"\n--- [AVISO] A coluna alvo '{target_col}' não foi encontrada. ---")


# --- Verificação 3: Variância Zero em Colunas Numéricas ---
variances = df[numeric_cols].var()
zero_variance_cols = variances[variances == 0].index.tolist()

if zero_variance_cols:
    print("\n--- [FALHA] Verificação de Variância Zero ---")
    print("As seguintes colunas numéricas possuem variância zero:")
    print(zero_variance_cols)
else:
    print("\n--- [SUCESSO] Verificação de Variância Zero: Nenhuma coluna numérica com variância zero encontrada. ---")

# --- Verificação 4: Valores Ausentes (NaN) em todo o DataFrame ---
nan_check = df.isnull().sum()
nan_cols = nan_check[nan_check > 0]

if not nan_cols.empty:
    print("\n--- [AVISO] Verificação Geral de Valores Ausentes (NaN) ---")
    print("As seguintes colunas contêm valores ausentes (NaN):")
    print(nan_cols)
else:
    print("\n--- [SUCESSO] Verificação Geral de Valores Ausentes (NaN): Nenhum NaN encontrado no DataFrame. ---")
