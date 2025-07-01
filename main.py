# main.py

import os
# A importação 'import pandas as pd' foi removida por não ser utilizada neste arquivo.
from src.data_loader import load_data
from src.data_processing import process_data
from src.feature_engineering import create_temporal_features
from src.train import train_model
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURED_DATA_PATH, MODEL_PATH, REPORTS_PATH,
    TARGET_COLUMN, FEATURES_TO_LAG, ENTITY_COLUMN
)


def main():
    """
    Orquestra o pipeline de ponta a ponta.
    """
    # Etapa 1: Carregar Dados Brutos
    print("Etapa 1: Carregando dados brutos...")
    raw_data = load_data(RAW_DATA_PATH)

    # Etapa 2: Processar Dados
    print("\nEtapa 2: Processando dados...")
    processed_data = process_data(raw_data)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Dados processados salvos em: {PROCESSED_DATA_PATH}")

    # --- Verificação de Sanidade ---
    if processed_data.empty:
        raise ValueError(
            "DataFrame vazio após o processamento inicial. Verifique o arquivo de dados brutos e os nomes das colunas.")

    # Etapa 3: Engenharia de Features
    print("\nEtapa 3: Iniciando a engenharia de features...")
    # Confirma que a função chamada é 'create_temporal_features'
    featured_data = create_temporal_features(
        df=processed_data,
        target_col=TARGET_COLUMN,
        lag_features=FEATURES_TO_LAG,
        entity_col=ENTITY_COLUMN
    )
    os.makedirs(os.path.dirname(FEATURED_DATA_PATH), exist_ok=True)
    featured_data.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Dados com features salvas em: {FEATURED_DATA_PATH}")

    # Etapa 4: Treinamento do Modelo
    print("\nEtapa 4: Iniciando o treinamento do modelo...")
    train_model(
        data_path=FEATURED_DATA_PATH,
        model_output_path=MODEL_PATH,
        report_output_path=REPORTS_PATH
    )
    print("\n--- Pipeline concluído com sucesso! ---")


if __name__ == '__main__':
    main()
