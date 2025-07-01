# =============================================================================
# SCRIPT PRINCIPAL DE ORQUESTRAÇÃO (main.py) - VERSÃO FINAL E CORRIGIDA
# Projeto: Predição de Estabilidade Fiscal e Risco Soberano
# =============================================================================
# Este script serve como ponto de entrada para executar todo o pipeline de ML.
#
# CORREÇÃO CRÍTICA: O código abaixo adiciona o diretório raiz do projeto
# ao sys.path, resolvendo o 'ModuleNotFoundError' para 'src'.
# =============================================================================

import os
import sys

# Adiciona o diretório raiz do projeto ao Python Path
# Isso garante que os módulos dentro de 'src' possam ser encontrados
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Agora, as importações do 'src' funcionarão corretamente
from src import config
from src.data_loader import load_data
from src.data_processing import process_data
from src.feature_engineering import engineer_features
from src.train import train_model

def run_pipeline():
    """
    Executa o pipeline completo de machine learning.
    """
    print("=============================================")
    print("🚀 INICIANDO O PIPELINE DE ESTABILIDADE FISCAL 🚀")
    print("=============================================")

    # --- Passo 1: Carregar Dados ---
    print("\n[PASSO 1/4] Carregando dados brutos...")
    try:
        raw_df = load_data(config.RAW_DATA_FILE)
        print(f"✅ Dados carregados com sucesso. Shape: {raw_df.shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar os dados: {e}")
        return

    # --- Passo 2: Pré-processamento e Limpeza ---
    print("\n[PASSO 2/4] Pré-processando e limpando os dados...")
    try:
        processed_df = process_data(raw_df)
        processed_df.to_csv(config.PROCESSED_DATA_FILE, index=False)
        print(f"✅ Dados processados e salvos em '{config.PROCESSED_DATA_FILE}'. Shape: {processed_df.shape}")
    except Exception as e:
        print(f"❌ Erro no processamento dos dados: {e}")
        return

    # --- Passo 3: Engenharia de Features ---
    print("\n[PASSO 3/4] Executando engenharia de features...")
    try:
        featured_df = engineer_features(processed_df)
        featured_df.to_csv(config.FEATURED_DATA_FILE, index=False)
        print(f"✅ Features criadas e salvas em '{config.FEATURED_DATA_FILE}'. Shape: {featured_df.shape}")
    except Exception as e:
        print(f"❌ Erro na engenharia de features: {e}")
        return

    # --- Passo 4: Treinamento e Avaliação do Modelo ---
    print("\n[PASSO 4/4] Treinando e avaliando o modelo...")
    try:
        train_model(featured_df)
        print("✅ Modelo treinado e avaliado com sucesso!")
        print(f"Modelo salvo em: '{config.MODEL_FILE}'")
        print(f"Relatórios gerados em: '{config.REPORTS_DIR}'")
    except Exception as e:
        print(f"❌ Erro no treinamento do modelo: {e}")
        return

    print("\n=============================================")
    print("🎉 PIPELINE CONCLUÍDO COM SUCESSO! 🎉")
    print("=============================================")


if __name__ == '__main__':
    # Cria os diretórios necessários (se não existirem) a partir do config
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(config.FEATURED_DATA_FILE), exist_ok=True)

    run_pipeline()
