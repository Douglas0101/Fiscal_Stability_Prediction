# ==============================================================================
# ORQUESTRADOR DO PIPELINE DE PREVISÃO DE ESTABILIDADE FISCAL
# ------------------------------------------------------------------------------
# Versão com carregamento explícito de variáveis de ambiente para robustez.
# ==============================================================================

import argparse
import logging
import os
from dotenv import load_dotenv

# --- CARREGAMENTO EXPLÍCITO DAS VARIÁVEIS DE AMBIENTE ---
# Esta é a correção crucial. `load_dotenv()` procura por um ficheiro .env
# na raiz do projeto e o carrega no ambiente ANTES de qualquer outra coisa.
# Isto garante que as configurações estarão disponíveis quando `src.config` for importado.
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)
# ---------------------------------------------------------

# Agora que o ambiente está preparado, podemos importar nossos módulos com segurança.
from src.config import settings
from src.logger_config import setup_logging
from src.data_processing import process_data
from src.train import train_model

# Configura o logging no início da execução
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """
    Função principal que analisa os argumentos da linha de comando e
    executa a ação correspondente.
    """
    parser = argparse.ArgumentParser(
        description="Orquestrador do Pipeline de Previsão de Estabilidade Fiscal."
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Comandos disponíveis")

    parser_process = subparsers.add_parser("process-data", help="Executa o pipeline de pré-processamento de dados.")
    parser_process.set_defaults(func=run_data_processing)

    parser_train = subparsers.add_parser("train-model", help="Executa o pipeline de treinamento do modelo.")
    parser_train.set_defaults(func=run_model_training)

    args = parser.parse_args()
    args.func(args)


def run_data_processing(args):
    """Executa a etapa de processamento de dados."""
    logger.info("=" * 50)
    logger.info("INICIANDO O PIPELINE DE PROCESSAMENTO DE DADOS")
    logger.info("=" * 50)
    try:
        input_path = os.path.join(settings.paths.RAW_DATA_PATH, "world_bank_data_2025.csv")
        output_path = os.path.join(settings.paths.PROCESSED_DATA_PATH, "processed_data.csv")
        process_data(input_path=input_path, output_path=output_path)
        logger.info("Pipeline de processamento de dados concluído com sucesso.")
    except Exception as e:
        logger.error(f"Falha no pipeline de processamento de dados: {e}", exc_info=True)


def run_model_training(args):
    """Executa a etapa de treinamento do modelo."""
    logger.info("=" * 50)
    logger.info("INICIANDO O PIPELINE DE TREINAMENTO DO MODELO")
    logger.info("=" * 50)
    try:
        input_path = os.path.join(settings.paths.PROCESSED_DATA_PATH, "processed_data.csv")
        train_model(data_path=input_path)
        logger.info("Pipeline de treinamento do modelo concluído com sucesso.")
    except Exception as e:
        logger.error(f"Falha no pipeline de treinamento do modelo: {e}", exc_info=True)


if __name__ == "__main__":
    main()
