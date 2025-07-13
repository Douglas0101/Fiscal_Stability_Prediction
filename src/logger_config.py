# src/logger_config.py
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Configura e retorna um logger com o nome especificado.
    Evita adicionar handlers duplicados se o logger já tiver sido configurado.
    """
    # Obtém o logger para o módulo específico (ex: 'src.api')
    logger = logging.getLogger(name)

    # Configura o logger apenas se ele ainda não tiver handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Cria um handler para enviar os logs para o terminal
        handler = logging.StreamHandler(sys.stdout)

        # Define o formato da mensagem de log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Adiciona o handler ao logger
        logger.addHandler(handler)

        # Evita que os logs sejam propagados para o logger raiz, prevenindo duplicação
        logger.propagate = False

    return logger
