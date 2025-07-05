import sys
import os
import logging
import argparse

# Adiciona a pasta raiz do projeto ao caminho de busca do Python
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logger_config import setup_logging
from src.data_processing import process_data
from src.train import train_model
from src.create_target import generate_target_variable
from src.config import settings

setup_logging()
logger = logging.getLogger(__name__)

def run_target_creation():
    logger.info("==================================================")
    logger.info("INICIANDO O PIPELINE DE CRIAÇÃO DA VARIÁVEL ALVO")
    logger.info("==================================================")
    try:
        generate_target_variable(
            input_path=settings.RAW_DATA_PATH,
            output_path=settings.RAW_DATA_WITH_TARGET_PATH
        )
        logger.info("Pipeline de criação da variável alvo concluído com sucesso.")
    except Exception as e:
        logger.error(f"Falha no pipeline de criação da variável alvo: {e}", exc_info=True)

def run_data_processing():
    logger.info("==================================================")
    logger.info("INICIANDO O PIPELINE DE PROCESSAMENTO DE DADOS")
    logger.info("==================================================")
    try:
        process_data(
            input_path=settings.RAW_DATA_WITH_TARGET_PATH,
            output_path=settings.PROCESSED_DATA_PATH
        )
        logger.info("Pipeline de processamento de dados concluído com sucesso.")
    except Exception as e:
        logger.error(f"Falha no pipeline de processamento de dados: {e}", exc_info=True)

def run_model_training(model_choice: str):
    logger.info("==================================================")
    logger.info(f"INICIANDO O PIPELINE DE TREINAMENTO PARA O MODELO: {model_choice.upper()}")
    logger.info("==================================================")
    try:
        train_model(data_path=settings.PROCESSED_DATA_PATH, model_choice=model_choice)
        logger.info(f"Pipeline de treinamento do modelo {model_choice.upper()} concluído com sucesso.")
    except Exception as e:
        logger.error(f"Falha no pipeline de treinamento do modelo: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning para Estabilidade Fiscal.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Comandos disponíveis')

    parser_create_target = subparsers.add_parser('create-target', help='Cria a variável alvo nos dados brutos.')
    parser_create_target.set_defaults(func=lambda args: run_target_creation())

    parser_process = subparsers.add_parser('process-data', help='Executa o pipeline de processamento de dados.')
    parser_process.set_defaults(func=lambda args: run_data_processing())

    parser_train = subparsers.add_parser('train-model', help='Executa o pipeline de treinamento do modelo.')
    parser_train.add_argument(
        '--model',
        type=str,
        default='rf',
        choices=['rf', 'lgbm', 'xgb', 'pytorch'],
        help='Selecione o modelo para treinar: rf, lgbm, xgb, pytorch'
    )
    parser_train.set_defaults(func=lambda args: run_model_training(args.model))

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
