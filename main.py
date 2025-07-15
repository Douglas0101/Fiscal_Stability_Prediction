# main.py
import subprocess
import time

from src.logger_config import get_logger
from src.config import AppConfig

logger = get_logger(__name__)


def run_command(command, service_name):
    """Executa um comando no terminal e regista o output."""
    logger.info(f"A executar comando para o serviço '{service_name}': {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        for line in process.stdout:
            logger.info(f"[{service_name}] {line.strip()}")
        for line in process.stderr:
            logger.error(f"[{service_name}-ERROR] {line.strip()}")

        process.wait()

        if process.returncode == 0:
            logger.info(f"Comando para '{service_name}' executado com sucesso.")
            return True
        else:
            logger.error(f"Erro ao executar comando para '{service_name}'. Código de saída: {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"Uma excepção ocorreu ao executar o comando para '{service_name}': {e}")
        return False


def main():
    """Orquestrador principal para o pipeline de ML."""
    config = AppConfig()

    logger.info("--- INICIANDO ORQUESTRADOR DO PIPELINE DE ML ---")

    if not run_command(["docker-compose", "up", "-d", "--build"], "docker-compose up"):
        logger.critical("Falha ao iniciar os serviços Docker. A abortar o pipeline.")
        return

    logger.info("Serviços Docker iniciados. A aguardar 15 segundos para a estabilização...")
    time.sleep(15)

    logger.info("--- A iniciar o pipeline de processamento de dados ---")
    if not run_command(["docker-compose", "exec", "api", "python", config.data_processing_script_path],
                       "data-processing"):
        logger.critical("Falha no pipeline de processamento de dados. A abortar o pipeline.")
        return

    logger.info(f"--- A iniciar o treino para os modelos: {config.models_to_run} ---")
    for model_name in config.models_to_run:
        logger.info(f"--- A treinar o modelo: {model_name} ---")
        train_command = ["docker-compose", "exec", "api", "python", config.train_script_path, model_name]
        if not run_command(train_command, f"model-training-{model_name}"):
            logger.error(f"Falha no treino do modelo {model_name}. A continuar com os próximos...")

    logger.info("--- A reiniciar o serviço da API para carregar os modelos treinados ---")
    if not run_command(["docker-compose", "restart", "api"], "api-restart"):
        logger.critical("Falha ao reiniciar a API.")
        return

    logger.info("--- PIPELINE DE ML CONCLUÍDO COM SUCESSO! ---")
    logger.info(f"Dashboard disponível em http://localhost:8501")
    logger.info(f"API disponível em http://localhost:8000/docs")
    logger.info(f"MLflow UI disponível em http://localhost:5000")


if __name__ == "__main__":
    main()
