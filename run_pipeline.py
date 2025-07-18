# -*- coding: utf-8 -*-
"""
Script Orquestrador para o Pipeline de Previsão de Estabilidade Fiscal.

Este script serve como ponto de entrada único para executar todo o pipeline de
Machine Learning, desde o pré-processamento dos dados brutos até o treino do
modelo final.

Passos:
1.  Executa o script de criação da variável alvo para gerar o dataset final.
2.  Invoca o script de treino do classificador com o modelo especificado.
"""
import logging
import os
import subprocess
import sys

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_command(command: list):
    """Executa um comando no terminal e verifica por erros."""
    logger.info(f"Executando comando: {' '.join(command)}")
    try:
        # Usamos sys.executable para garantir que estamos usando o mesmo interpretador Python
        subprocess.run([sys.executable] + command, check=True)
        logger.info(f"Comando '{command[0]}' concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ocorreu um erro ao executar o script '{command[0]}'.")
        logger.error(f"Código de saída: {e.returncode}")
        sys.exit(1)  # Termina a execução se um passo falhar
    except FileNotFoundError:
        logger.error(f"Erro: O script '{command[0]}' não foi encontrado. Verifique o caminho.")
        sys.exit(1)


def main():
    """
    Ponto de entrada principal para orquestrar o pipeline.
    """
    # --- Passo 1: Pré-processamento e Criação da Variável Alvo ---
    # Este passo garante que 'data/03_final/final_data.csv' terá a coluna alvo.
    # NOTA: Ajuste os caminhos de entrada e saída conforme a sua lógica real.
    # Assumindo que 'create_target.py' lê de 'processed_data' e salva em 'final_data'.

    # Primeiro, vamos garantir que o script `create_target.py` é executável
    # e que ele faz o que esperamos. Vamos criar uma versão simplificada dele
    # que apenas adiciona a coluna alvo.

    # (O ideal é que você tenha um script de processamento de dados completo,
    # mas para resolver o problema agora, vamos focar em criar o alvo)

    create_target_script_path = "src/create_target.py"
    run_command([create_target_script_path])

    # --- Passo 2: Treinar o Modelo ---
    # Agora que temos a certeza que os dados estão prontos, treinamos o modelo.
    # O modelo a ser treinado é passado como argumento para este script.

    if len(sys.argv) < 2 or sys.argv[1] not in ['lgbm', 'xgb', 'ebm']:
        logger.error("Uso: python run_pipeline.py <model_name>")
        logger.error("Modelos disponíveis: lgbm, xgb, ebm")
        sys.exit(1)

    model_name = sys.argv[1]

    train_script_path = "src/train.py"
    train_command = [
        train_script_path,
        "--model", model_name
    ]
    run_command(train_command)

    logger.info("Pipeline completo executado com sucesso!")


if __name__ == "__main__":
    main()
