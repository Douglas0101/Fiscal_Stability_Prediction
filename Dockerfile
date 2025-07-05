# ==============================================================================
# DOCKERFILE OTIMIZADO - CPU-ONLY & MULTI-STAGE BUILD
# ------------------------------------------------------------------------------
# Esta versão força a instalação de uma versão do PyTorch exclusiva para CPU,
# reduzindo drasticamente o tempo de build e o tamanho da imagem final.
# ==============================================================================

# --- ESTÁGIO 1: Builder ---
# Este estágio instala as dependências em um ambiente temporário.
FROM python:3.9-slim-buster AS builder

# Define o diretório de trabalho
WORKDIR /opt/venv

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Cria e ativa um ambiente virtual
RUN python -m venv .
ENV PATH="/opt/venv/bin:$PATH"

# Atualiza o pip
RUN pip install --upgrade pip

# --- OTIMIZAÇÃO DE VELOCIDADE ---
# Instala o PyTorch (CPU-only) separadamente. Esta é a etapa mais demorada.
# Ao isolá-la, garantimos que ela só será executada se a versão do torch mudar.
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Copia e instala o resto das dependências.
# Esta camada será cacheada enquanto o requirements.txt não for alterado.
COPY requirements.txt .
RUN pip install -r requirements.txt


# --- ESTÁGIO 2: Final ---
# Este estágio cria a imagem final, copiando apenas o necessário.
FROM python:3.9-slim-buster AS final

WORKDIR /app

# Cria um usuário não-root para segurança
RUN addgroup --system app && adduser --system --group app

# Copia o ambiente virtual com TODAS as dependências já instaladas
COPY --from=builder /opt/venv /opt/venv

# Copia o código-fonte da aplicação
COPY ./src ./src
COPY ./data ./data
COPY ./models ./models

# Define as permissões
RUN chown -R app:app /app
USER app

# Define o path para o ambiente virtual
ENV PATH="/opt/venv/bin:$PATH"

# Expõe a porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
