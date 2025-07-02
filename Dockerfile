# =======================================================
# DOCKERFILE PARA O SERVIÇO DE BACKEND (PYTHON) - FINAL
# Local: Raiz do projeto (./Dockerfile)
# =======================================================

# --- ESTÁGIO 1: Instalação de Dependências ---
FROM python:3.9-slim-buster AS python-deps
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# --- ESTÁGIO 2: Aplicação Final ---
FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN addgroup --system app && adduser --system --group app
COPY --from=python-deps /usr/src/app/wheels /wheels
COPY --from=python-deps /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copia todo o código-fonte da aplicação para o diretório de trabalho.
COPY . .

# --- CORREÇÃO DEFINITIVA ---
# Copia a pasta 'models' (que está na raiz do projeto) para o local
# onde a API espera encontrá-la dentro do contêiner (/app/src/models).
COPY models /app/src/models

# Define o usuário 'app' como proprietário de todos os arquivos, incluindo os modelos.
RUN chown -R app:app /app

# Muda para o usuário não-root para maior segurança.
USER app

# Expõe a porta em que a API irá rodar.
EXPOSE 8000

# Verificação de saúde: Garante que a API está operacional.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Comando de inicialização definitivo para a API.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
