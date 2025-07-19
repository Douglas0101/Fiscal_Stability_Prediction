# -*- coding: utf-8 -*-
"""
Script Orquestrador para o Pipeline de Previsão de Estabilidade Fiscal.
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


# --- FUNÇÃO ATUALIZADA ---
def run_command(command: list):
    """
    Executa um comando de módulo Python e verifica por erros.
    Usa o flag '-m' para garantir que os caminhos de importação funcionem corretamente.
    """
    # Converte o caminho do script (ex: "src/train.py") para o formato de módulo (ex: "src.train")
    script_path = command[0]
    module_path = script_path.replace('.py', '').replace(os.path.sep, '.')

    # Monta o novo comando com o flag -m
    full_command = [sys.executable, "-m", module_path] + command[1:]

    logger.info(f"Executando comando de módulo: {' '.join(full_command)}")
    try:
        subprocess.run(full_command, check=True)
        logger.info(f"Módulo '{module_path}' concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ocorreu um erro ao executar o módulo '{module_path}'.")
        logger.error(f"Código de saída: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Erro: O script para o módulo '{module_path}' não foi encontrado. Verifique o caminho.")
        sys.exit(1)


def main():
    """
    Ponto de entrada principal para orquestrar o pipeline.
    """
    # --- Passo 1: Pré-processamento e Criação da Variável Alvo ---
    # Este script agora será executado como um módulo para manter a consistência.
    # (Supondo que você tenha um script `src/create_target.py`)
    create_target_script_path = "src/create_target.py"
    run_command([create_target_script_path])

    # --- Passo 2: Treinar o Modelo ---
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