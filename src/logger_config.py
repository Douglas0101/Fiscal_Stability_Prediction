import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from src.config import settings

# Define o formato do log
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)"

def setup_logging():
    """
    Configura o sistema de logging para a aplicação.
    """
    # Define o nível de logging com base na configuração (DEBUG ou INFO)
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Obtém o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove handlers existentes para evitar duplicação de logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Handler para exibir logs no console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(stream_handler)

    # Handler para salvar logs em um arquivo com rotação diária
    file_handler = TimedRotatingFileHandler("app.log", when="midnight", interval=1, backupCount=7)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    logging.info("Logging configurado com sucesso.")

# Como usar:
# Em outros arquivos, basta obter o logger com `logging.getLogger(__name__)`
# Exemplo:
#
# import logging
# from src.logger_config import setup_logging
#
# setup_logging() # Chame isso uma vez no ponto de entrada da sua aplicação (ex: main.py)
#
# logger = logging.getLogger(__name__)
#
# def my_function():
#     logger.info("Executando a função.")
#     try:
#         result = 1 / 0
#     except ZeroDivisionError:
#         logger.error("Ocorreu um erro de divisão por zero!", exc_info=True)

